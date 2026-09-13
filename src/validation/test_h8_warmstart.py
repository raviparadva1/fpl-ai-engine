import sys
sys.path.append(r"d:\Fpl_winning")
import pandas as pd
import time
from src.optimization.multi_gw import MultiGWOptimizer
from src.api.live_pipeline import LiveFPLPipeline

def run_test():
    hist_df = pd.read_parquet(r"d:\Fpl_winning\data\processed\features_targets.parquet")
    
    p = LiveFPLPipeline()
    p.train_or_load_models()
    
    test_cases = [
        ('2024-25', 5),
        ('2024-25', 15),
        ('2025-26', 10)
    ]
    
    _, teams, _, fixtures = p.fetch_live_data()
    
    for season, gw in test_cases:
        print(f"\n--- Testing {season} GW {gw} (H=8) ---")
        
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
        if live_df.empty:
            continue
            
        live_df['availability'] = 'a'
        if 'was_home' not in live_df.columns:
            live_df['was_home'] = 0
            
        horizon_gws = p.predict_8gw_horizon(live_df, gw, fixtures, teams)
        
        opt = MultiGWOptimizer(horizon=8, bank=bank, free_transfers=1)
        config = {'mode': 'neutral', 'type': 'expected_points', 'risk_penalty': 0.0}
        
        opt.build_model(horizon_gws, initial_squad, config)
        
        t0 = time.time()
        res = opt.solve()
        t1 = time.time()
        
        print(f"Status: {res['status']}")
        if res['status'] in ['OPTIMAL', 'FEASIBLE']:
            print(f"Runtime: {t1-t0:.2f} seconds")
            print(f"Objective Value: {res['projected_obj']:.2f}")
            print(f"Transfers (Buy): {len(res['buy'])}")
            
            recognized_initial = [opt.players[i]['player_code'] for i in range(len(opt.players)) if opt.initial_sq[i] == 1]
            new_squad = set(recognized_initial) - set(res['sell']) | set(res['buy'])
            is_15 = len(new_squad) == 15
            is_11_xi = len(res['xi']) == 11
            is_1_cap = len([res['cap']]) == 1
            legal = is_15 and is_11_xi and is_1_cap
            print(f"Squad Legal (15 sq, 11 xi, 1 cap): {legal}")
            
if __name__ == '__main__':
    run_test()
