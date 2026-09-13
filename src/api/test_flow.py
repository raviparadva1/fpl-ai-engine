import sys
sys.path.append(r'd:\Fpl_winning')
from src.api.live_pipeline import LiveFPLPipeline
p = LiveFPLPipeline()
p.train_or_load_models()
e, t, gw = p.fetch_live_data()
squad, bank = p.fetch_team_data(1471257, e)
print(f'15 Players retrieved: {len(squad) == 15}')
print(f'Correct Bank: {bank}')
print(f'Current GW Target: {gw}')

live_df = p.generate_live_features(e, t)
horizon_gws = p.predict_8gw_horizon(live_df, gw)
res, gw_df, current_squad = p.run_optimizer(horizon_gws, squad, bank, 1)

print('--- Live Recommendation for 1471257 ---')
buy_names = gw_df[gw_df['player_code'].isin(res['buy'])]['name'].tolist()
sell_names = gw_df[gw_df['player_code'].isin(res['sell'])]['name'].tolist()
print(f'Transfers: BUY {buy_names} | SELL {sell_names}')
print(f'Hits: {res.get("hits", 0)}')
cap_name = gw_df[gw_df['player_code'] == res.get('cap')]['name'].values[0] if res.get('cap') else "N/A"
print(f'Captain: {cap_name}')
print('Success!')
