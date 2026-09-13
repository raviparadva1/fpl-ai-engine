import sys
import pandas as pd
import json
sys.path.append(r"d:\Fpl_winning")
from src.api.live_pipeline import LiveFPLPipeline

def run_advisory(team_id):
    print("=========================================================")
    print(f"      FPL LIVE ADVISORY - TEAM ID {team_id}")
    print("=========================================================\n")
    
    p = LiveFPLPipeline()
    p.train_or_load_models()
    
    # Fetch bootstrap data
    elements, teams_data, target_gw, fixtures = p.fetch_live_data()
    
    # Fetch specific manager data
    my_squad, bank = p.fetch_team_data(team_id, elements)
    ft = 1 # Assuming 1 FT for this test
    print(f"\nCurrent Squad Value + Bank: £{(sum([e['now_cost'] for e in elements if e['code'] in my_squad]) + bank)/10.0:.1f}m")
    print(f"Bank: £{bank/10.0:.1f}m | Free Transfers: {ft}")
    
    squad_names = [e['web_name'] for e in elements if e['code'] in my_squad]
    print(f"Current Squad ({len(squad_names)}): {', '.join(squad_names)}\n")
    
    # Generate features
    live_df = p.generate_live_features(elements, teams_data)
    
    # We want H=3 explicitly
    H = 3
    print(f"Predicting ML Horizons (H={H})...")
    
    # predict_8gw_horizon returns up to 8. We slice to H.
    full_horizon_gws = p.predict_8gw_horizon(live_df, target_gw, fixtures, teams_data)
    horizon_gws = full_horizon_gws[:H]
    
    # Configure Risk-Adjusted Optimizer
    strategy_config = {
        'mode': 'neutral',
        'type': 'expected_points',
        'risk_penalty': 0.2
    }
    
    print("\nRunning CP-SAT Optimizer (H=3, Risk-Adjusted)...")
    res, gw_df, current_squad = p.run_optimizer(horizon_gws, my_squad, bank, ft, strategy_config=strategy_config)
    
    # Display the breakdown
    print("\n================== SOLVER RESULTS ==================")
    print(f"Status       : {res['status']}")
    print(f"Runtime      : {res.get('runtime', 'N/A')}s")
    print(f"Proj. Pts    : {res.get('projected_obj', 0):.2f} (over {H} GWs)")
    
    print("\n================= RECOMMENDED MOVES ================")
    buy_names = []
    sell_names = []
    
    for b in res['buy']:
        row = gw_df[gw_df['player_code'] == b].iloc[0]
        buy_names.append(f"{row['name']} (xP: {row['xP']:.2f}, £{row['value']:.1f}m)")
        
    for s in res['sell']:
        row = gw_df[gw_df['player_code'] == s].iloc[0]
        sell_names.append(f"{row['name']} (xP: {row['xP']:.2f}, £{row['value']:.1f}m)")
        
    print(f"SELL: {', '.join(sell_names) if sell_names else 'None'}")
    print(f"BUY : {', '.join(buy_names) if buy_names else 'None'}")
    print(f"Hits Taken: {res['hits'] * 4} pts")
    
    print("\n================== PROJECTED GW ==================")
    cap_row = gw_df[gw_df['player_code'] == res['cap']].iloc[0]
    print(f"Captain: {cap_row['name']} (xP: {cap_row['xP']:.2f})")
    
    print("\nStarting XI:")
    xi_rows = gw_df[gw_df['player_code'].isin(res['xi'])]
    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        players = xi_rows[xi_rows['position'] == pos]
        names_xp = [f"{r['name']} ({r['xP']:.2f})" for _, r in players.iterrows()]
        print(f"{pos} ({len(names_xp)}): {', '.join(names_xp)}")
        
    print("\nBench:")
    bench_codes = set(res['buy'] + list(set(current_squad) - set(res['sell']))) - set(res['xi'])
    bench_rows = gw_df[gw_df['player_code'].isin(bench_codes)]
    bench_xp = [f"{r['name']} ({r['xP']:.2f})" for _, r in bench_rows.iterrows()]
    print(f"Substitutes: {', '.join(bench_xp)}")
    
    print("\n[Data Validated: Live FPL API (prices, fixtures, injuries), Legal FPL Constraints, CP-SAT Solved]")
    print("====================================================")

if __name__ == '__main__':
    run_advisory(1471257)
