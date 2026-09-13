import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
import lightgbm as lgb
import json
import os

def evaluate(model, name, X_t, y_t):
    preds = model.predict(X_t)
    mae = mean_absolute_error(y_t, preds)
    rmse = np.sqrt(mean_squared_error(y_t, preds))
    corr = np.corrcoef(y_t, preds)[0,1]
    rank_corr = spearmanr(y_t, preds)[0]
    return {'MAE': mae, 'RMSE': rmse, 'Corr': corr, 'Rank_Corr': rank_corr}

def train_models():
    print("Loading data...")
    df = pd.read_parquet(r"d:\Fpl_winning\data\processed\features_targets.parquet")
    
    # Feature columns (only those computed strictly from historical data)
    features = [c for c in df.columns if c.endswith('_last_1') or c.endswith('_last_3') or 
                c.endswith('_last_5') or c.endswith('_last_10') or c.endswith('_std')]
    
    # Fill NAs with 0 for models that don't handle them naturally (like Ridge)
    # LightGBM handles NAs naturally.
    df_clean = df.dropna(subset=['target_points']).copy()
    
    # Walk-forward splits
    print("Splitting data (Walk-Forward)...")
    train = df_clean[df_clean['season'].isin(['2021-22', '2022-23'])]
    val = df_clean[df_clean['season'] == '2023-24']
    test = df_clean[df_clean['season'].isin(['2024-25', '2025-26'])]
    
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    X_train, y_train = train[features], train['target_points']
    X_val, y_val = val[features], val['target_points']
    X_test, y_test = test[features], test['target_points']

    results = {}
    
    print("1. Baseline (Simple Average of Last 3)")
    baseline_preds = test['pts_last_3'].fillna(0)
    results['Baseline (Last 3)'] = {
        'MAE': mean_absolute_error(y_test, baseline_preds),
        'RMSE': np.sqrt(mean_squared_error(y_test, baseline_preds)),
        'Corr': np.corrcoef(y_test, baseline_preds)[0,1],
        'Rank_Corr': spearmanr(y_test, baseline_preds)[0]
    }
    
    print("2. Ridge Regression Baseline")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train.fillna(0), y_train)
    results['Ridge'] = evaluate(ridge, 'Ridge', X_test.fillna(0), y_test)
    
    print("3. LightGBM (Direct Regression)")
    lgb_reg = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05, random_state=42)
    lgb_reg.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    results['LightGBM Direct'] = evaluate(lgb_reg, 'LightGBM Direct', X_test, y_test)
    
    print("4. LightGBM Two-Stage Hurdle Model")
    # Stage 1: Predict if playing > 0 min
    y_train_play = train['target_played']
    y_val_play = val['target_played']
    lgb_clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
    lgb_clf.fit(X_train, y_train_play, eval_set=[(X_val, y_val_play)])
    
    # Stage 2: Predict points given playing
    train_played = train[train['target_played'] == 1]
    val_played = val[val['target_played'] == 1]
    lgb_reg_2 = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    lgb_reg_2.fit(train_played[features], train_played['target_points'], eval_set=[(val_played[features], val_played['target_points'])])
    
    # Combine
    prob_play = lgb_clf.predict_proba(X_test)[:, 1]
    xp_if_play = lgb_reg_2.predict(X_test)
    preds_two_stage = prob_play * xp_if_play
    
    results['LightGBM Two-Stage'] = {
        'MAE': mean_absolute_error(y_test, preds_two_stage),
        'RMSE': np.sqrt(mean_squared_error(y_test, preds_two_stage)),
        'Corr': np.corrcoef(y_test, preds_two_stage)[0,1],
        'Rank_Corr': spearmanr(y_test, preds_two_stage)[0]
    }
    
    # Save results
    out_file = r'd:\Fpl_winning\data\processed\ml_results.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\n--- Final ML Evaluation on Holdout Test Set (24-25, 25-26) ---")
    for model, metrics in results.items():
        print(f"Model: {model}")
        print(f"  MAE:  {metrics['MAE']:.3f}")
        print(f"  RMSE: {metrics['RMSE']:.3f}")
        print(f"  Corr: {metrics['Corr']:.3f}")
        print(f"  Rank: {metrics['Rank_Corr']:.3f}\n")

if __name__ == "__main__":
    train_models()
