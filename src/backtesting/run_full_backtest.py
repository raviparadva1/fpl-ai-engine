import pandas as pd
import numpy as np
import lightgbm as lgb
from ortools.sat.python import cp_model
import json
import time
import os
import sys

def solve_gw(player_data, current_squad, bank, ft, horizon, strategy):
    model = cp_model.CpModel()
    players = player_data.to_dict('records')
    n = len(players)
    
    sq = [model.NewBoolVar(f"sq_{i}") for i in range(n)]
    xi = [model.NewBoolVar(f"xi_{i}") for i in range(n)]
    cap = [model.NewBoolVar(f"cap_{i}") for i in range(n)]
    buy = [model.NewBoolVar(f"buy_{i}") for i in range(n)]
    sell = [model.NewBoolVar(f"sell_{i}") for i in range(n)]
    hits = model.NewIntVar(0, 15, "hits")
    
    current_map = {p['player_code']: 1 if p['player_code'] in current_squad else 0 for p in players}
    
    model.Add(sum(sq) == 15)
    model.Add(sum(xi) == 11)
    model.Add(sum(cap) == 1)
    
    for i in range(n):
        model.Add(xi[i] <= sq[i])
        model.Add(cap[i] <= xi[i])
        model.Add(sq[i] == current_map[players[i]['player_code']] + buy[i] - sell[i])
        
    model.Add(sum(buy) == sum(sell))
    
    pos_limits = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        p_idx = [i for i, p in enumerate(players) if p['position'] == pos]
        model.Add(sum(sq[i] for i in p_idx) == pos_limits[pos])
        
    cost = sum(buy[i] * int(players[i]['value'] * 10) for i in range(n))
    gain = sum(sell[i] * int(players[i]['value'] * 10) for i in range(n))
    model.Add(cost <= int(bank * 10) + gain)
    
    diff = model.NewIntVar(-15, 15, "diff")
    model.Add(diff == sum(buy) - ft)
    model.AddMaxEquality(hits, [0, diff])
    
    obj_expr = 0
    for i in range(n):
        val = players[i]['xP']
        
        if strategy == 'risk_adjusted':
            val = val - 0.2 * (players[i]['P90'] - players[i]['P10'])
        elif strategy == 'attack':
            own_pct = players[i].get('selected', 0) / 10000000.0
            val = val + 0.5 * val * (1 - own_pct)
        elif strategy == 'defend':
            own_pct = players[i].get('selected', 0) / 10000000.0
            val = val - 0.3 * val * (1 - own_pct)
            
        int_val = int(val * 1000)
        obj_expr += (xi[i] + cap[i]) * int_val
        
    obj_expr -= (hits * 4000)
    model.Maximize(obj_expr)
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 0.5
    status = solver.Solve(model)
    
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        new_squad = [players[i]['player_code'] for i in range(n) if solver.Value(sq[i]) == 1]
        xi_codes = [players[i]['player_code'] for i in range(n) if solver.Value(xi[i]) == 1]
        cap_code = [players[i]['player_code'] for i in range(n) if solver.Value(cap[i]) == 1][0]
        n_hits = solver.Value(hits)
        new_bank = bank + (solver.Value(gain) - solver.Value(cost)) / 10.0
        return new_squad, xi_codes, cap_code, n_hits, new_bank
    return current_squad, current_squad[:11], current_squad[0], 0, bank

def run_backtest():
    print("Starting Comprehensive Historical Backtest (24-25 and 25-26)...")
    
    df = pd.read_parquet(r"d:\Fpl_winning\data\processed\features_targets.parquet")
    features = [c for c in df.columns if c.endswith('_last_1') or c.endswith('_last_3') or c.endswith('_last_5') or c.endswith('_last_10') or c.endswith('_std')]
    
    train = df[df['season'] < '2024-25'].dropna(subset=['target_points']).copy()
    test = df[df['season'].isin(['2024-25', '2025-26'])].copy()
    
    print("Training ML Models...")
    clf = lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1)
    clf.fit(train[features], train['target_played'])
    
    train_played = train[train['target_played'] == 1]
    reg = lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
    reg.fit(train_played[features], train_played['target_points'])
    
    test['prob_play'] = clf.predict_proba(test[features])[:, 1]
    test['xp_if_play'] = reg.predict(test[features])
    test['xP'] = (test['prob_play'] * test['xp_if_play']).clip(lower=0)
    
    test['P10'] = (test['xP'] - 1.5).clip(lower=0)
    test['P90'] = test['xP'] + 3.0
    
    strategies = [
        {'name': '1-GW Horizon', 'horizon': 1, 'type': 'expected_points'},
        {'name': '3-GW Horizon', 'horizon': 3, 'type': 'expected_points'},
        {'name': '5-GW Horizon', 'horizon': 5, 'type': 'expected_points'},
        {'name': '8-GW Horizon', 'horizon': 8, 'type': 'expected_points'},
        {'name': '5-GW Risk-Adjusted', 'horizon': 5, 'type': 'risk_adjusted'},
        {'name': '5-GW Attack (Trailing)', 'horizon': 5, 'type': 'attack'},
        {'name': '5-GW Defend (Leading)', 'horizon': 5, 'type': 'defend'}
    ]
    
    results = {}
    
    for strat in strategies:
        print(f"\nSimulating Strategy: {strat['name']}...")
        total_pts = 0
        total_hits = 0
        cap_pts = 0
        gw_count = 0
        
        # Test subset to get empirical results quickly
        for season in ['2024-25']:
            season_data = test[test['season'] == season]
            gws = sorted(season_data['GW'].unique())[:15]  # Limit to 15 GWs for rapid testing

            
            # Initial Squad for the season
            gw1 = season_data[season_data['GW'] == gws[0]]
            if gw1.empty: continue
            
            gk = gw1[gw1['position'] == 'GK'].head(2)['player_code'].tolist()
            dfc = gw1[gw1['position'] == 'DEF'].head(5)['player_code'].tolist()
            mid = gw1[gw1['position'] == 'MID'].head(5)['player_code'].tolist()
            fwd = gw1[gw1['position'] == 'FWD'].head(3)['player_code'].tolist()
            squad = gk + dfc + mid + fwd
            
            bank = 0.0
            ft = 1
            
            for gw in gws:
                current_gw_data = season_data[season_data['GW'] == gw].copy()
                
                if strat['horizon'] > 1:
                    future_data = season_data[(season_data['GW'] >= gw) & (season_data['GW'] < gw + strat['horizon'])]
                    future_xp = future_data.groupby('player_code')['xP'].sum().reset_index()
                    future_xp.rename(columns={'xP': 'multi_xP'}, inplace=True)
                    current_gw_data = current_gw_data.merge(future_xp, on='player_code', how='left')
                    current_gw_data['xP'] = current_gw_data['multi_xP']
                
                current_gw_data.fillna(0, inplace=True)
                
                prev_squad = list(squad)
                squad, xi, cap, hits, bank = solve_gw(current_gw_data, squad, bank, ft, strat['horizon'], strat['type'])
                
                actual_gw = season_data[season_data['GW'] == gw]
                
                if xi:
                    xi_data = actual_gw[actual_gw['player_code'].isin(xi)]
                    cap_data = actual_gw[actual_gw['player_code'] == cap]
                    
                    squad_pts = xi_data['target_points'].sum()
                    cap_bonus = cap_data['target_points'].sum() if not cap_data.empty else 0
                    
                    gw_net = squad_pts + cap_bonus - (hits * 4)
                    
                    total_pts += gw_net
                    cap_pts += cap_bonus
                    total_hits += hits
                    gw_count += 1
                    
                transfers_made = 15 - len(set(squad).intersection(set(prev_squad)))
                ft = min(5, max(1, ft + 1 - transfers_made))
                
        results[strat['name']] = {
            'Total Points': total_pts,
            'Total Hits': total_hits,
            'Captain Points': cap_pts,
            'Avg GW Points': total_pts / max(1, gw_count)
        }
        print(f"Result -> Total: {total_pts:.1f}, Avg: {total_pts/max(1,gw_count):.1f}, Hits: {total_hits}")
        
    with open(r'd:\Fpl_winning\data\processed\backtest_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    run_backtest()
