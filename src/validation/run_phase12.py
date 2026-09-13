import os
import json
from src.backtesting.engine import FPLBacktester

def run_experiment_matrix():
    horizons = [1, 3, 5, 8]
    strategies = [
        {'mode': 'neutral', 'type': 'expected_points'},
        {'mode': 'attack', 'type': 'expected_points'}
    ]
    
    # We will test over a fast subset of GWs in 2025-26
    start_gw = 5
    end_gw = 15
    
    results = []
    
    for h in horizons:
        for strat in strategies:
            print(f"\n--- Running Backtest: Horizon {h}, Strategy {strat['mode']} ---")
            backtester = FPLBacktester(
                start_season='2025-26', start_gw=start_gw, end_gw=end_gw,
                initial_bank=100.0,
                horizon=h,
                strategy_config=strat
            )
            stats = backtester.run()
            stats['horizon'] = h
            stats['strategy'] = strat['mode']
            results.append(stats)
            
    with open(r'd:\Fpl_winning\data\processed\phase12_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Matrix Complete!")

if __name__ == '__main__':
    run_experiment_matrix()
