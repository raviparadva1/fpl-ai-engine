import sys
sys.path.append(r'd:\Fpl_winning')
from src.api.live_pipeline import LiveFPLPipeline
p = LiveFPLPipeline()
p.train_or_load_models()
e, t, gw = p.fetch_live_data()
df = p.generate_live_features(e, t)
h = p.predict_8gw_horizon(df, gw)
pool = h[0].sort_values('selected', ascending=False)
my_squad = []
team_counts = {}
for pos, limit in [('GK', 2), ('DEF', 5), ('MID', 5), ('FWD', 3)]:
    count = 0
    for _, player in pool[pool['position'] == pos].iterrows():
        if count >= limit: break
        t = player['team']
        if team_counts.get(t, 0) < 3:
            my_squad.append(player['player_code'])
            team_counts[t] = team_counts.get(t, 0) + 1
            count += 1
print(f"Squad Length: {len(my_squad)}")
print(f"Team Counts: {team_counts}")
