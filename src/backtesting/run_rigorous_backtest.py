import pandas as pd
import numpy as np
import lightgbm as lgb
from ortools.sat.python import cp_model
import json
import time

def solve_gw(player_data, current_squad, bank, ft, horizon, strategy_config):
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
    strategy_type = strategy_config['type']
    
    for i in range(n):
        val = players[i]['xP']
        
        if strategy_type == 'risk_adjusted':
            val -= strategy_config.get('risk_coeff', 0.2) * (players[i]['P90'] - players[i]['P10'])
        elif strategy_type == 'attack':
            own_pct = players[i].get('selected', 0) / 10000000.0
            val += strategy_config.get('attack_coeff', 0.5) * val * (1 - own_pct)
        elif strategy_type == 'defend':
            own_pct = players[i].get('selected', 0) / 10000000.0
            val -= strategy_config.get('defend_coeff', 0.3) * val * (1 - own_pct)
            
        int_val = int(val * 1000)
        obj_expr += (xi[i] + cap[i]) * int_val
        
    obj_expr -= (hits * 4000)
    model.Maximize(obj_expr)
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5.0 # Increased significantly to prevent UNKNOWN
    
    start_time = time.time()
    status = solver.Solve(model)
    runtime = time.time() - start_time
    
    status_str = "UNKNOWN"
    if status == cp_model.OPTIMAL: status_str = "OPTIMAL"
    elif status == cp_model.FEASIBLE: status_str = "FEASIBLE"
    elif status == cp_model.INFEASIBLE: status_str = "INFEASIBLE"
    
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        new_squad = [players[i]['player_code'] for i in range(n) if solver.Value(sq[i]) == 1]
        xi_codes = [players[i]['player_code'] for i in range(n) if solver.Value(xi[i]) == 1]
        cap_code = [players[i]['player_code'] for i in range(n) if solver.Value(cap[i]) == 1][0]
        n_hits = solver.Value(hits)
        transfers_made = sum(solver.Value(buy[i]) for i in range(n))
        new_bank = bank + (solver.Value(gain) - solver.Value(cost)) / 10.0
        obj_val = solver.ObjectiveValue() / 1000.0
        return new_squad, xi_codes, cap_code, n_hits, transfers_made, new_bank, status_str, runtime, obj_val
        
    return current_squad, current_squad[:11], current_squad[0], 0, 0, bank, status_str, runtime, 0.0

def run_rigorous_backtest():
    df = pd.read_parquet(r"d:\Fpl_winning\data\processed\features_targets.parquet")
    features = [c for c in df.columns if c.endswith('_last_1') or c.endswith('_last_3') or c.endswith('_last_5') or c.endswith('_last_10') or c.endswith('_std')]
    
    train = df[df['season'] < '2024-25'].dropna(subset=['target_points']).copy()
    test = df[df['season'].isin(['2024-25', '2025-26'])].copy()
    
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
        {'name': '1-GW xP', 'horizon': 1, 'type': 'expected_points'},
        {'name': '3-GW xP', 'horizon': 3, 'type': 'expected_points'},
        {'name': '5-GW xP', 'horizon': 5, 'type': 'expected_points'},
        {'name': '8-GW xP', 'horizon': 8, 'type': 'expected_points'},
        {'name': '5-GW Risk (Coeff=0.2)', 'horizon': 5, 'type': 'risk_adjusted', 'risk_coeff': 0.2},
        {'name': '5-GW Attack (Coeff=0.5)', 'horizon': 5, 'type': 'attack', 'attack_coeff': 0.5},
        {'name': '5-GW Defend (Coeff=0.3)', 'horizon': 5, 'type': 'defend', 'defend_coeff': 0.3}
    ]
    
    final_report = []
    
    for season in ['2024-25', '2025-26']:
        season_data = test[test['season'] == season]
        gws = sorted(season_data['GW'].unique())
        
        gw1 = season_data[season_data['GW'] == gws[0]]
        if gw1.empty: continue
        
        # Consistent exact starting state
        gk = gw1[gw1['position'] == 'GK'].head(2)['player_code'].tolist()
        dfc = gw1[gw1['position'] == 'DEF'].head(5)['player_code'].tolist()
        mid = gw1[gw1['position'] == 'MID'].head(5)['player_code'].tolist()
        fwd = gw1[gw1['position'] == 'FWD'].head(3)['player_code'].tolist()
        init_squad = gk + dfc + mid + fwd
        
        for strat in strategies:
            squad = list(init_squad)
            bank = 0.0
            ft = 1
            
            s_gross = 0
            s_net = 0
            s_hits_pts = 0
            s_hits_cnt = 0
            s_transfers = 0
            s_cap = 0
            s_bench = 0
            status_counts = {'OPTIMAL': 0, 'FEASIBLE': 0, 'UNKNOWN': 0}
            
            for gw in gws:
                current_gw_data = season_data[season_data['GW'] == gw].copy()
                
                if strat['horizon'] > 1:
                    future_data = season_data[(season_data['GW'] >= gw) & (season_data['GW'] < gw + strat['horizon'])]
                    future_xp = future_data.groupby('player_code')['xP'].sum().reset_index()
                    future_xp.rename(columns={'xP': 'multi_xP'}, inplace=True)
                    current_gw_data = current_gw_data.merge(future_xp, on='player_code', how='left')
                    current_gw_data['xP'] = current_gw_data['multi_xP']
                
                current_gw_data.fillna(0, inplace=True)
                
                squad, xi, cap, hits, transfers_made, bank, status_str, runtime, obj_val = solve_gw(current_gw_data, squad, bank, ft, strat['horizon'], strat)
                
                status_counts[status_str] = status_counts.get(status_str, 0) + 1
                
                actual_gw = season_data[season_data['GW'] == gw]
                
                if xi:
                    xi_data = actual_gw[actual_gw['player_code'].isin(xi)]
                    cap_data = actual_gw[actual_gw['player_code'] == cap]
                    bench_data = actual_gw[(actual_gw['player_code'].isin(squad)) & (~actual_gw['player_code'].isin(xi))]
                    
                    squad_pts = xi_data['target_points'].sum()
                    cap_bonus = cap_data['target_points'].sum() if not cap_data.empty else 0
                    bench_pts = bench_data['target_points'].sum()
                    
                    gross = squad_pts + cap_bonus
                    net = gross - (hits * 4)
                    
                    s_gross += gross
                    s_net += net
                    s_hits_pts += (hits * 4)
                    s_hits_cnt += hits
                    s_transfers += transfers_made
                    s_cap += cap_bonus
                    s_bench += bench_pts
                    
                ft = min(5, max(1, ft + 1 - transfers_made))
                
            final_report.append({
                'Season': season,
                'Strategy': strat['name'],
                'Gross Points': float(s_gross),
                'Net Points': float(s_net),
                'Avg Net/GW': float(s_net / len(gws)),
                'Transfers': int(s_transfers),
                'Hits': int(s_hits_cnt),
                'Captain Points': float(s_cap),
                'Bench Points': float(s_bench),
                'OPTIMAL_cnt': status_counts.get('OPTIMAL', 0),
                'FEASIBLE_cnt': status_counts.get('FEASIBLE', 0),
                'UNKNOWN_cnt': status_counts.get('UNKNOWN', 0)
            })
            
    with open(r'd:\Fpl_winning\data\processed\rigorous_results.json', 'w') as f:
        json.dump(final_report, f, indent=4)

if __name__ == "__main__":
    run_rigorous_backtest()
