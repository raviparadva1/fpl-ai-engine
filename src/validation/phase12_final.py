import sys, json, time, os
sys.path.append(r"d:\Fpl_winning")
import pandas as pd
from src.optimization.multi_gw import MultiGWOptimizer
from src.api.live_pipeline import LiveFPLPipeline

def run_matrix():
    hist_df = pd.read_parquet(r"d:\Fpl_winning\data\processed\features_targets.parquet")
    p = LiveFPLPipeline()
    p.train_or_load_models()
    
    test_cases = {
        '2024-25': [15],
        '2025-26': [15]
    }
    
    horizons = [1, 3, 5, 8]
    configs = {
        'Pure xP': {'mode': 'neutral', 'type': 'expected_points', 'risk_penalty': 0.0},
        'Neutral': {'mode': 'neutral', 'type': 'expected_points', 'risk_penalty': 0.1},
        'Risk-adjusted': {'mode': 'neutral', 'type': 'expected_points', 'risk_penalty': 0.2},
        'Attack': {'mode': 'attack', 'type': 'expected_points', 'risk_penalty': 0.0},
        'Defend': {'mode': 'defend', 'type': 'expected_points', 'risk_penalty': 0.0}
    }
    
    results = []
    
    pos_map = {
        'Goalkeeper': 'GK', 'Defender': 'DEF', 'Midfielder': 'MID', 'Forward': 'FWD',
        'G': 'GK', 'D': 'DEF', 'M': 'MID', 'F': 'FWD', 'GKP': 'GK',
        'CB': 'DEF', 'FB': 'DEF', 'CM': 'MID', 'AM': 'MID', 'WM': 'MID', 'CF': 'FWD', 'W': 'FWD'
    }
    
    _, teams_data, _, fixtures = p.fetch_live_data()
    
    # We will use the dynamically provided fixtures per season
    for season, gws in test_cases.items():
        for gw in gws:
            print(f"--- Running {season} GW {gw} ---", flush=True)
            prev_gw = hist_df[(hist_df['season'] == season) & (hist_df['GW'] == gw-1)].copy()
            if prev_gw.empty: continue
            prev_gw = prev_gw.sort_values('total_points', ascending=False)
            
            # Build valid initial squad
            initial_squad = []
            selected_pos = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
            pos_req = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
            team_counts = {}
            for _, row in prev_gw.iterrows():
                pos = pos_map.get(row['position'], row['position'])
                if pos not in selected_pos: continue
                tm = row['team']
                if selected_pos[pos] < pos_req[pos] and team_counts.get(tm, 0) < 3:
                    initial_squad.append(row['player_code'])
                    selected_pos[pos] += 1
                    team_counts[tm] = team_counts.get(tm, 0) + 1
                if len(initial_squad) == 15: break
                    
            bank = max(0.0, 100.0 - prev_gw[prev_gw['player_code'].isin(initial_squad)]['value'].sum())
            
            live_df = hist_df[(hist_df['season'] == season) & (hist_df['GW'] == gw)].copy()
            if live_df.empty: continue
            live_df['availability'] = 'a'
            if 'was_home' not in live_df.columns: live_df['was_home'] = 0
            
            print("Predicting 8GW horizon...", flush=True)
            full_horizon_gws = p.predict_8gw_horizon(live_df, gw, fixtures, teams_data)
            horizon_gws = {h: full_horizon_gws[:h] for h in horizons}
                
            actual_points = dict(zip(live_df['player_code'], live_df['total_points']))
            
            for h in horizons:
                for cfg_name, cfg in configs.items():
                    opt = MultiGWOptimizer(horizon=h, bank=bank, free_transfers=1)
                    opt.build_model(horizon_gws[h], initial_squad, cfg)
                    
                    t0 = time.time()
                    res = opt.solve()
                    runtime = time.time() - t0
                    
                    row_res = {
                        'season': season,
                        'gw': gw,
                        'horizon': h,
                        'strategy': cfg_name,
                        'status': res['status'],
                        'runtime': runtime
                    }
                    
                    if res['status'] in ['OPTIMAL', 'FEASIBLE']:
                        xi = res['xi']
                        cap = res['cap']
                        
                        # Fix legality check by using recognized_initial
                        recognized_initial = [opt.players[i]['player_code'] for i in range(len(opt.players)) if opt.initial_sq[i] == 1]
                        sq = set(recognized_initial) - set(res.get('sell', [])) | set(res.get('buy', []))
                        
                        is_legal = (len(sq) == 15) and (len(xi) == 11)
                        
                        actual_xi_pts = sum(actual_points.get(c, 0) for c in xi)
                        actual_cap_pts = actual_points.get(cap, 0)
                        gross = actual_xi_pts + actual_cap_pts
                        hits = res.get('hits', 0)
                        net = gross - (hits * 4)
                        bench = list(sq - set(xi))
                        bench_pts = sum(actual_points.get(c, 0) for c in bench)
                        
                        row_res.update({
                            'gross_points': gross,
                            'net_points': net,
                            'hits': hits,
                            'captain_points': actual_cap_pts,
                            'bench_points': bench_pts,
                            'transfers_made': len(res.get('buy', [])),
                            'legal_squad': is_legal
                        })
                    else:
                        row_res.update({
                            'gross_points': 0, 'net_points': 0, 'hits': 0,
                            'captain_points': 0, 'bench_points': 0, 'transfers_made': 0,
                            'legal_squad': False
                        })
                        
                    results.append(row_res)
                    print(f"H={h} | {cfg_name} | {res['status']} | Runtime: {runtime:.2f}s | Net: {row_res['net_points']}", flush=True)
                    
    df_res = pd.DataFrame(results)
    df_res.to_json(r"d:\Fpl_winning\data\processed\phase12_final_results.json", orient='records', indent=4)
    
    with open(r"d:\Fpl_winning\data\processed\phase12_final_report.md", "w", encoding='utf-8') as f:
        f.write("# Phase 12 Final Validation Report\n\n")
        
        f.write("## Overall Strategy Performance (Averages per GW)\n")
        strat_agg = df_res[df_res['status'].isin(['OPTIMAL', 'FEASIBLE'])].groupby(['horizon', 'strategy']).agg({
            'net_points': 'mean',
            'gross_points': 'mean',
            'hits': 'mean',
            'bench_points': 'mean',
            'transfers_made': 'mean'
        }).round(2).reset_index()
        f.write(strat_agg.to_markdown(index=False) + "\n\n")
        
        f.write("## Solver Reliability\n")
        status_agg = df_res.groupby(['horizon', 'strategy', 'status']).size().unstack(fill_value=0).reset_index()
        f.write(status_agg.to_markdown(index=False) + "\n\n")
        
        f.write("## Legality Check\n")
        legal_agg = df_res.groupby('horizon')['legal_squad'].mean().reset_index()
        legal_agg['legal_squad'] = (legal_agg['legal_squad'] * 100).astype(str) + "%"
        f.write(legal_agg.to_markdown(index=False) + "\n\n")
        
        f.write("## Runtime Analysis (Seconds)\n")
        time_agg = df_res.groupby('horizon')['runtime'].mean().round(2).reset_index()
        f.write(time_agg.to_markdown(index=False) + "\n\n")

if __name__ == '__main__':
    run_matrix()
