import pandas as pd
import lightgbm as lgb
from ortools.sat.python import cp_model
import time

def test_solver(horizon, time_limit, current_gw_data):
    model = cp_model.CpModel()
    players = current_gw_data.to_dict('records')
    n = len(players)
    
    sq = [model.NewBoolVar(f"sq_{i}") for i in range(n)]
    xi = [model.NewBoolVar(f"xi_{i}") for i in range(n)]
    cap = [model.NewBoolVar(f"cap_{i}") for i in range(n)]
    buy = [model.NewBoolVar(f"buy_{i}") for i in range(n)]
    sell = [model.NewBoolVar(f"sell_{i}") for i in range(n)]
    hits = model.NewIntVar(0, 15, "hits")
    
    # Mock current squad (first 15 valid)
    current_squad = [p['player_code'] for p in players[:15]]
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
    model.Add(cost <= gain)
    
    diff = model.NewIntVar(-15, 15, "diff")
    model.Add(diff == sum(buy) - 1)
    model.AddMaxEquality(hits, [0, diff])
    
    obj_expr = 0
    for i in range(n):
        int_val = int(players[i]['multi_xP'] * 1000)
        obj_expr += (xi[i] + cap[i]) * int_val
        
    obj_expr -= (hits * 4000)
    model.Maximize(obj_expr)
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    
    start = time.time()
    status = solver.Solve(model)
    elapsed = time.time() - start
    
    status_str = "UNKNOWN"
    if status == cp_model.OPTIMAL: status_str = "OPTIMAL"
    elif status == cp_model.FEASIBLE: status_str = "FEASIBLE (TIMEOUT)"
    elif status == cp_model.INFEASIBLE: status_str = "INFEASIBLE"
    
    obj_val = solver.ObjectiveValue() / 1000.0 if status in [cp_model.OPTIMAL, cp_model.FEASIBLE] else 0
    return status_str, elapsed, obj_val

def run():
    df = pd.read_parquet(r"d:\Fpl_winning\data\processed\features_targets.parquet")
    test = df[df['season'] == '2024-25'].copy()
    test['xP'] = np.random.uniform(1, 6, len(test)) # Mock xP for speed test
    
    gw = 1
    for horizon in [5, 8]:
        future_data = test[(test['GW'] >= gw) & (test['GW'] < gw + horizon)]
        future_xp = future_data.groupby('player_code')['xP'].sum().reset_index()
        future_xp.rename(columns={'xP': 'multi_xP'}, inplace=True)
        current_gw_data = test[test['GW'] == gw].merge(future_xp, on='player_code', how='left').fillna(0)
        
        for limit in [0.5, 5.0]:
            status, elapsed, obj = test_solver(horizon, limit, current_gw_data)
            print(f"Horizon: {horizon}GW | Time Limit: {limit}s | Status: {status} | Time: {elapsed:.2f}s | Obj: {obj:.2f}")

if __name__ == "__main__":
    import numpy as np
    run()
