import sys
sys.path.append(r"d:\Fpl_winning")
import pandas as pd
from ortools.sat.python import cp_model
from src.optimization.multi_gw import MultiGWOptimizer
from src.api.live_pipeline import LiveFPLPipeline

hist_df = pd.read_parquet(r"d:\Fpl_winning\data\processed\features_targets.parquet")
p = LiveFPLPipeline()
p.train_or_load_models()
_, teams, _, fixtures = p.fetch_live_data()

season = '2024-25'
gw = 5
prev_gw = hist_df[(hist_df['season'] == season) & (hist_df['GW'] == gw-1)].copy()
prev_gw = prev_gw.sort_values('total_points', ascending=False)
        
initial_squad = []
selected_pos = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
pos_req = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
team_counts = {}

pos_map = {
    'Goalkeeper': 'GK', 'Defender': 'DEF', 'Midfielder': 'MID', 'Forward': 'FWD',
    'G': 'GK', 'D': 'DEF', 'M': 'MID', 'F': 'FWD', 'GKP': 'GK',
    'CB': 'DEF', 'FB': 'DEF', 'CM': 'MID', 'AM': 'MID', 'WM': 'MID', 'CF': 'FWD', 'W': 'FWD'
}

for _, row in prev_gw.iterrows():
    raw_pos = row['position']
    pos = pos_map.get(raw_pos, raw_pos)
    if pos not in selected_pos: continue
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

live_df = hist_df[(hist_df['season'] == season) & (hist_df['GW'] == gw)].copy()
live_df['availability'] = 'a'
if 'was_home' not in live_df.columns:
    live_df['was_home'] = 0

horizon_gws = p.predict_8gw_horizon(live_df, gw, fixtures, teams)

opt = MultiGWOptimizer(horizon=8, bank=bank, free_transfers=1)
config = {'mode': 'neutral', 'type': 'expected_points', 'risk_penalty': 0.0}

opt.build_model(horizon_gws, initial_squad, config)

hint_sq = opt.initial_sq
hint_xi = [0] * len(opt.players)
pos_added = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
for i in range(len(opt.players)):
    if hint_sq[i] == 1:
        pos = opt.players[i]['position']
        if pos == 'GK' and pos_added['GK'] < 1:
            hint_xi[i] = 1; pos_added['GK'] += 1
        elif pos == 'DEF' and pos_added['DEF'] < 3:
            hint_xi[i] = 1; pos_added['DEF'] += 1
        elif pos == 'MID' and pos_added['MID'] < 4:
            hint_xi[i] = 1; pos_added['MID'] += 1
        elif pos == 'FWD' and pos_added['FWD'] < 3:
            hint_xi[i] = 1; pos_added['FWD'] += 1

for t in range(opt.H):
    cap_selected = False
    for i in range(len(opt.players)):
        opt.model.AddHint(opt.buy[t][i], 0)
        opt.model.AddHint(opt.sell[t][i], 0)
        opt.model.AddHint(opt.sq[t][i], hint_sq[i])
        opt.model.AddHint(opt.xi[t][i], hint_xi[i])
        if hint_xi[i] == 1 and not cap_selected:
            opt.model.AddHint(opt.cap[t][i], 1)
            cap_selected = True
        else:
            opt.model.AddHint(opt.cap[t][i], 0)

import time
t0 = time.time()
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 2.0
solver.parameters.log_search_progress = True
status = solver.Solve(opt.model)
t1 = time.time()
print(f"Status with AddHint: {solver.StatusName(status)}")
print(f"Solve runtime: {t1-t0:.4f}s")
print(f"Objective: {solver.ObjectiveValue()}")
