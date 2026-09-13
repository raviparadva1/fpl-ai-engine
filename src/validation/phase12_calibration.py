import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from sklearn.metrics import mean_absolute_error, brier_score_loss

print("Starting Phase 12 Calibration Audit...")

# 1. Load Data
df = pd.read_parquet(r"d:\Fpl_winning\data\processed\features_targets.parquet")
# Split Train/Test
train = df[df['season'] == '2024-25'].copy()
test = df[df['season'] == '2025-26'].copy()

features = [c for c in df.columns if c.endswith('_last_1') or c.endswith('_last_3') or c.endswith('_last_5') or c.endswith('_last_10') or c.endswith('_std')]

# Train Models
clf = lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1)
clf.fit(train[features], train['target_played'])

train_played = train[train['target_played'] == 1]
reg = lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
reg.fit(train_played[features], train_played['target_points'])

# 2. Predict on Test Set
test['prob_play_pred'] = clf.predict_proba(test[features])[:, 1]
test['xp_if_play_pred'] = reg.predict(test[features])
test['played_actual'] = (test['target_minutes'] > 0).astype(int)
test['xp_pred'] = test['prob_play_pred'] * test['xp_if_play_pred']

# 3. Calibration - Probability of Playing
print("\n--- CLASSIFICATION CALIBRATION (Prob Play) ---")
brier = brier_score_loss(test['played_actual'], test['prob_play_pred'])
print(f"Brier Score: {brier:.4f} (Closer to 0 is better)")

test['prob_bin'] = pd.cut(test['prob_play_pred'], bins=10)
calibration = test.groupby('prob_bin')['played_actual'].agg(['mean', 'count']).reset_index()
print(calibration.to_string())

# 4. Calibration - Expected Points (for those who played)
print("\n--- REGRESSION CALIBRATION (xP vs Actual Points) ---")
played_mask = test['played_actual'] == 1
mae_played = mean_absolute_error(test.loc[played_mask, 'target_points'], test.loc[played_mask, 'xp_if_play_pred'])
print(f"MAE (Played Only): {mae_played:.2f} points")

test['xp_bin'] = pd.qcut(test.loc[played_mask, 'xp_if_play_pred'], q=5, duplicates='drop')
xp_calib = test[played_mask].groupby('xp_bin')[['xp_if_play_pred', 'target_points']].mean().reset_index()
print(xp_calib.to_string())

# 5. Segment Performance (New Transfers vs Established)
print("\n--- PERFORMANCE BY PLAYER SEGMENT ---")
# To approximate team transfers in the test set, we check if their GW1 team matches their previous season team.
# For simplicity, we just look at the MAE of prediction vs actual across different cohorts.
test['abs_err'] = np.abs(test['xp_pred'] - test['target_points'])
print(f"Global MAE (All Predictions): {test['abs_err'].mean():.2f} points")

# Let's find players who appeared for the first time in 25-26
train_players = set(train['player_code'])
test['is_new_pl'] = ~test['player_code'].isin(train_players)

print(f"MAE for Established PL Players: {test[~test['is_new_pl']]['abs_err'].mean():.2f}")
print(f"MAE for New PL Players (25-26 arrivals): {test[test['is_new_pl']]['abs_err'].mean():.2f}")

# DGW / BGW Analysis
print("\n--- DGW / BGW ANALYSIS ---")
dgw_mask = test['target_minutes'] > 90
bgw_mask = test['target_minutes'] == 0 # Note: this mixes benched players with actual BGWs, but let's approximate
print(f"Mean Predicted xP for DGW Players (>90 target mins): {test.loc[dgw_mask, 'xp_pred'].mean():.2f} vs Actual {test.loc[dgw_mask, 'target_points'].mean():.2f}")

