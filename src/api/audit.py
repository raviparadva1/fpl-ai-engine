import sys, os
import pandas as pd
import numpy as np
sys.path.append(r'd:\Fpl_winning')
from src.api.live_pipeline import LiveFPLPipeline
from src.optimization.multi_gw import MultiGWOptimizer

print("Initializing Pipeline...")
p = LiveFPLPipeline()
p.train_or_load_models()
print("Fetching Live Data for Team 1471257...")
elements, teams, current_gw, fixtures = p.fetch_live_data()
team_info = p.fetch_team_data(1471257, elements)

squad_codes, bank = team_info

live_df = p.generate_live_features(elements, teams)
live_df['prob_play'] = p.model_clf.predict_proba(live_df[p.features])[:, 1]
live_df['xp_if_play'] = p.model_reg.predict(live_df[p.features])
live_df['xP'] = live_df['prob_play'] * live_df['xp_if_play']

print(f"Current Squad Codes: {squad_codes}")
print(f"Bank: {bank}")

horizon_gws = p.predict_8gw_horizon(live_df, current_gw, fixtures, teams)

# Evaluate BASE squad (No transfers)
# We can evaluate base squad by forcing 0 transfers, or just let the optimizer return it
opt = MultiGWOptimizer(horizon=8, bank=bank, free_transfers=0)
opt.build_model(horizon_gws, squad_codes, {'mode': 'neutral', 'type': 'expected_points'})
res_base = opt.solve()
# If res_base['buy'] is empty, it's the base objective.
base_objective = res_base.get('objective_value', 0.0)
print(f"Base Squad Optimizer Objective (With up to 1 transfer): {base_objective:.2f}")

recommendations = []
banned_buys = []

for i in range(5):
    # Ban previously found buys by heavily penalizing their xP
    temp_horizon = []
    for t_df in horizon_gws:
        t_df_copy = t_df.copy()
        t_df_copy.loc[t_df_copy['player_code'].isin(banned_buys), 'xP'] = -100.0
        temp_horizon.append(t_df_copy)
        
    opt = MultiGWOptimizer(horizon=8, bank=bank, free_transfers=1)
    opt.build_model(temp_horizon, squad_codes, {'mode': 'neutral', 'type': 'expected_points'})
    res = opt.solve()
    
    if res.get('buy') and res.get('sell'):
        buy_code = res['buy'][0]
        sell_code = res['sell'][0]
        recommendations.append((sell_code, buy_code, res.get('objective_value', 0.0)))
        banned_buys.append(buy_code)
    else:
        break

print("\n=== LIVE RECOMMENDATION AUDIT ===")
for idx, (sell_code, buy_code, obj_val) in enumerate(recommendations):
    sell_player = live_df[live_df['player_code'] == sell_code].iloc[0]
    buy_player = live_df[live_df['player_code'] == buy_code].iloc[0]
    
    print(f"\nRECOMMENDATION {idx+1}: SELL {sell_player['name']} -> BUY {buy_player['name']}")
    print(f"Optimizer Objective: {obj_val:.2f} (Gain over Base: +{obj_val - base_objective:.2f})")
    
    for p_type, p_row in [('OUTGOING', sell_player), ('INCOMING', buy_player)]:
        print(f"\n  [{p_type}] {p_row['name']} ({p_row['team']}) - £{p_row['value']}m")
        print(f"  Classification: {p_row['player_history_type']} | Flags: {p_row['validation_flags']}")
        
        # Raw Input Features
        print(f"  Raw Inputs -> mins_last_3: {p_row.get('minutes_last_3', 0):.1f}, xG: {p_row.get('xG', 0):.2f}, xA: {p_row.get('xA', 0):.2f}")
        
        # ML outputs
        print(f"  ML Prob of Playing: {p_row['prob_play']:.4f}")
        print(f"  ML xP (if plays):   {p_row['xp_if_play']:.4f}")
        print(f"  Final GW4 xP:       {p_row['xP']:.4f}")
        
        # 8-GW projection sum
        sum_8gw = sum([t_df[t_df['player_code'] == p_row['player_code']]['xP'].values[0] for t_df in horizon_gws])
        print(f"  8-GW Projected Sum: {sum_8gw:.2f}")
