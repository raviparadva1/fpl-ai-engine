import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import sys
import json
import time

sys.path.append(r"d:\Fpl_winning")
from src.optimization.multi_gw import MultiGWOptimizer

def prepare_historical_test_environment():
    df = pd.read_parquet(r"d:\Fpl_winning\data\processed\features_targets.parquet")
    
    # Load Models (trained in live_pipeline)
    features = [c for c in df.columns if c.endswith('_last_1') or c.endswith('_last_3') or c.endswith('_last_5') or c.endswith('_last_10') or c.endswith('_std')]
    if 'was_home' in df.columns:
        features.append('was_home')
        df['was_home'] = df['was_home'].astype(int)
        
    train = df[df['season'] < '2026-27'].dropna(subset=['target_points']).copy()
    clf = lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1)
    clf.fit(train[features], train['target_played'])
    
    train_played = train[train['target_played'] == 1]
    reg = lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
    reg.fit(train_played[features], train_played['target_points'])
    
    return df, features, clf, reg

def build_horizon_gws(df, current_gw, current_season, horizon, features, clf, reg):
    # Simulates predict_8gw_horizon purely from historical data
    horizon_gws = []
    
    for t in range(horizon):
        gw = current_gw + t
        # In historical, we can actually fetch the true features for the upcoming GW to get exact point-in-time info
        gw_df = df[(df['season'] == current_season) & (df['GW'] == gw)].copy()
        if gw_df.empty:
            break
            
        gw_df['prob_play'] = clf.predict_proba(gw_df[features])[:, 1]
        gw_df['xp_if_play'] = reg.predict(gw_df[features])
        gw_df['xP'] = (gw_df['prob_play'] * gw_df['xp_if_play']).clip(lower=0)
        
        gw_df['value'] = gw_df['value'].fillna(5.0)
        gw_df['P10'] = (gw_df['xP'] - 1.5).clip(lower=0)
        gw_df['P50'] = gw_df['xP']
        gw_df['P90'] = gw_df['xP'] + 3.0
        
        horizon_gws.append(gw_df)
        
    return horizon_gws

def run_phase12():
    print("Starting Phase 12 Strategy Matrix Backtest...")
    sys.path.append(r"d:\Fpl_winning")
    from src.api.live_pipeline import LiveFPLPipeline
    p = LiveFPLPipeline()
    p.train_or_load_models()
    # Need fixtures and teams to pass to predict_8gw_horizon
    _, teams, _, fixtures = p.fetch_live_data()
    
    # We will sample 3 representative GWs per season to map the matrix space efficiently 
    # without running an 8-hour process, simulating the cross-matrix metrics precisely.
    test_gws = [5, 15, 25] 
    
    # Get the raw historical data for target points
    hist_df = pd.read_parquet(r"d:\Fpl_winning\data\processed\features_targets.parquet")
    
    seasons = ['2024-25', '2025-26']
    horizons = [1, 3, 5, 8]
    strategies = [
        {'mode': 'neutral', 'type': 'expected_points'},
        {'mode': 'neutral', 'type': 'risk_adjusted'},
        {'mode': 'attack', 'type': 'expected_points'},
        {'mode': 'defend', 'type': 'expected_points'}
    ] 
    
    results = []
    
    for season in seasons:
        for gw in test_gws:
            # Baseline initial squad (must be valid to avoid forcing -20 hits in t=0)
            prev_gw = hist_df[(hist_df['season'] == season) & (hist_df['GW'] == gw-1)].copy()
            if prev_gw.empty:
                continue
            prev_gw = prev_gw.sort_values('total_points', ascending=False)
            
            initial_squad = []
            selected_pos = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
            pos_req = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
            team_counts = {}
            
            pos_map = {
                'Goalkeeper': 'GK', 'Defender': 'DEF', 'Midfielder': 'MID', 'Forward': 'FWD',
                'G': 'GK', 'D': 'DEF', 'M': 'MID', 'F': 'FWD',
                'CB': 'DEF', 'FB': 'DEF', 'CM': 'MID', 'AM': 'MID', 'WM': 'MID', 'CF': 'FWD', 'W': 'FWD'
            }
            
            for _, row in prev_gw.iterrows():
                raw_pos = row['position']
                pos = pos_map.get(raw_pos, raw_pos) # Try map, else keep original
                
                if pos not in selected_pos: # Skip invalid pos entirely
                    continue
                    
                tm = row['team']
                if selected_pos[pos] < pos_req[pos]:
                    if team_counts.get(tm, 0) < 3:
                        initial_squad.append(row['player_code'])
                        selected_pos[pos] += 1
                        team_counts[tm] = team_counts.get(tm, 0) + 1
                if len(initial_squad) == 15:
                    break
                    
            bank = 100.0 - prev_gw[prev_gw['player_code'].isin(initial_squad)]['value'].sum()
            if bank < 0: bank = 0.0
            
            # Base df for current GW
            live_df = hist_df[(hist_df['season'] == season) & (hist_df['GW'] == gw)].copy()
            if live_df.empty:
                continue
                
            live_df['availability'] = 'a'
            if 'was_home' not in live_df.columns:
                live_df['was_home'] = 0
            
            horizon_gws = p.predict_8gw_horizon(live_df, gw, fixtures, teams)
            
            for h in horizons:
                for strat in strategies:
                    # Run MultiGW Optimizer
                    opt = MultiGWOptimizer(horizon=h, bank=bank, free_transfers=1)
                    
                    # Temporarily override max_time for the matrix test to prove feasibility/optimality capturing
                    
                    opt.build_model(horizon_gws, initial_squad, strat)
                    
                    start_t = time.time()
                    res = opt.solve()
                    solve_time = time.time() - start_t
                    
                    # Evaluate on ACTUAL target points
                    if res['status'] in ['OPTIMAL', 'FEASIBLE']:
                        actual_gw_df = hist_df[(hist_df['season'] == season) & (hist_df['GW'] == gw)]
                        xi_actuals = actual_gw_df[actual_gw_df['player_code'].isin(res['xi'])]
                        
                        gross_pts = xi_actuals['target_points'].sum()
                        
                        cap_pts = 0
                        if res['cap'] in xi_actuals['player_code'].values:
                            cap_pts = actual_gw_df[actual_gw_df['player_code'] == res['cap']]['target_points'].values[0]
                        
                        net_pts = gross_pts + cap_pts - (res['hits'] * 4)
                        
                        bench_codes = [p for p in (res['buy'] if res['buy'] else initial_squad) if p not in res['xi']]
                        bench_pts = actual_gw_df[actual_gw_df['player_code'].isin(bench_codes)]['target_points'].sum()
                        
                        dgw_players = xi_actuals[xi_actuals['target_minutes'] > 90]['player_code'].tolist()
                        
                    else:
                        gross_pts = net_pts = cap_pts = bench_pts = 0
                        dgw_players = []
                        
                    results.append({
                        'season': season,
                        'gw': gw,
                        'horizon': h,
                        'strategy': strat['mode'],
                        'status': res['status'],
                        'solve_time': solve_time,
                        'gross_points': gross_pts,
                        'net_points': net_pts,
                        'cap_points': cap_pts,
                        'bench_points': bench_pts,
                        'hits': res.get('hits', 0),
                        'transfers': len(res.get('buy', [])),
                        'dgw_players_count': len(dgw_players)
                    })
                    print(f"{season} GW{gw} H{h} {strat['mode']}: {res['status']} | Net Pts: {net_pts:.1f}")

    out_df = pd.DataFrame(results)
    out_df.to_csv(r"d:\Fpl_winning\data\processed\phase12_matrix.csv", index=False)
    print("Matrix Complete!")

if __name__ == '__main__':
    run_phase12()
