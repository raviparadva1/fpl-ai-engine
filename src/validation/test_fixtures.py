import sys
import pandas as pd
import numpy as np

sys.path.append(r"d:\Fpl_winning")
from src.api.live_pipeline import LiveFPLPipeline

def test_fixtures():
    print("Testing BGW and DGW functionality in predict_8gw_horizon...")
    p = LiveFPLPipeline()
    p.train_or_load_models()
    
    # 1. Fetch real data just to get base data structures
    elements, teams_data, current_gw, _ = p.fetch_live_data()
    live_df = p.generate_live_features(elements, teams_data)
    
    team_name_to_id = {t['name']: t['id'] for t in teams_data}
    arsenal_id = team_name_to_id.get("Arsenal", 1)
    chelsea_id = team_name_to_id.get("Chelsea", 6)
    
    target_gw = 5
    # Create mock fixtures for Target GW 5
    mock_fixtures = [
        # Arsenal has NO fixtures in GW5 (BGW)
        
        # Chelsea has TWO fixtures in GW5 (DGW)
        {
            'event': target_gw, 
            'team_h': chelsea_id, 'team_a': 99, 
            'team_h_difficulty': 2, 'team_a_difficulty': 5
        },
        {
            'event': target_gw, 
            'team_h': 98, 'team_a': chelsea_id, 
            'team_h_difficulty': 5, 'team_a_difficulty': 4
        }
    ]
    
    # Run the prediction using these mock fixtures
    horizon_gws = p.predict_8gw_horizon(live_df, target_gw, mock_fixtures, teams_data)
    gw5_df = horizon_gws[0]
    
    # Check Arsenal (BGW)
    arsenal_players = gw5_df[gw5_df['team'] == 'Arsenal']
    arsenal_xp = arsenal_players['xP'].sum()
    print(f"\nArsenal GW5 Total xP (BGW): {arsenal_xp}")
    assert arsenal_xp == 0.0, "BGW must have strictly 0 xP!"
    
    # Check Chelsea (DGW)
    chelsea_players = gw5_df[gw5_df['team'] == 'Chelsea']
    chelsea_xp = chelsea_players['xP'].sum()
    print(f"Chelsea GW5 Total xP (DGW): {chelsea_xp}")
    
    # Run a normal SGW comparison
    mock_fixtures_sgw = [
        {
            'event': target_gw, 
            'team_h': chelsea_id, 'team_a': 99, 
            'team_h_difficulty': 2, 'team_a_difficulty': 5
        }
    ]
    horizon_gws_sgw = p.predict_8gw_horizon(live_df, target_gw, mock_fixtures_sgw, teams_data)
    gw5_df_sgw = horizon_gws_sgw[0]
    
    chelsea_sgw_xp = gw5_df_sgw[gw5_df_sgw['team'] == 'Chelsea']['xP'].sum()
    print(f"Chelsea GW5 Total xP (SGW): {chelsea_sgw_xp}")
    
    assert chelsea_xp > chelsea_sgw_xp, "DGW xP must be greater than SGW xP!"
    
    # Math Verification:
    # First fixture: Home, FDR 2 -> base_mult = (7-2)/4 = 1.25. home_adv = 1.1 -> 1.375? (Wait, I haven't seen the exact math running yet.)
    # Second fixture: Away, FDR 4 -> base_mult = (7-4)/4 = 0.75. home_adv = 0.9 -> ...
    
    print("\nSUCCESS: All fixture tests passed!")

if __name__ == '__main__':
    test_fixtures()
