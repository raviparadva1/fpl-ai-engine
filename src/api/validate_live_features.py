import sys
import pandas as pd
import numpy as np
import os

sys.path.append(r'd:\Fpl_winning')
from src.api.live_pipeline import LiveFPLPipeline

def generate_validation_report():
    print("Initializing Pipeline...")
    p = LiveFPLPipeline()
    p.train_or_load_models()
    
    print("Fetching Live Data...")
    elements, teams, target_gw = p.fetch_live_data()
    
    print("Generating Features with Smart Imputation...")
    live_df = p.generate_live_features(elements, teams)
    
    print("Running LightGBM Predictions...")
    live_df['prob_play'] = p.model_clf.predict_proba(live_df[p.features])[:, 1]
    live_df['xp_if_play'] = p.model_reg.predict(live_df[p.features])
    
    # Zero out injured/suspended
    live_df.loc[live_df['availability'].isin(['i', 's']), 'prob_play'] = 0.0
    live_df.loc[live_df['availability'] == 'd', 'prob_play'] *= 0.25
    
    live_df['final_xP'] = live_df['prob_play'] * live_df['xp_if_play']
    
    print("Building Validation Report...")
    report_cols = [
        'name', 'id', 'player_code', 'validation_flags', 'hist_season', 'hist_gw', 
        'team', 'position', 'availability', 'minutes', 'starts', 'total_points', 
        'points_per_90', 'xG', 'xA', 'xGI', 'xGC',
        'prob_play', 'xp_if_play', 'final_xP'
    ]
    
    report_df = live_df[report_cols].sort_values('final_xP', ascending=False)
    report_path = r'd:\Fpl_winning\data\processed\validation_report.csv'
    report_df.to_csv(report_path, index=False)
    print(f"Report saved to {report_path}")
    
    # Print summary
    flag_counts = report_df['validation_flags'].value_counts()
    print("\n--- VALIDATION SUMMARY ---")
    print(flag_counts)
    
    print("\n--- TOP 5 PREDICTIONS ---")
    print(report_df[['name', 'validation_flags', 'hist_gw', 'final_xP']].head(5).to_string(index=False))
    
    print("\n--- GROSS & BELLOUMI TEST ---")
    gb_df = report_df[report_df['name'].str.contains('Groß|Groß|Belloumi', case=False, na=False)]
    print(gb_df[['name', 'validation_flags', 'hist_gw', 'prob_play', 'xp_if_play', 'final_xP']].to_string(index=False))

if __name__ == '__main__':
    generate_validation_report()
