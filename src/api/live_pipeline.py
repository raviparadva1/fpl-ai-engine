import requests
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime
import sys
import os

sys.path.append(r"d:\Fpl_winning")
from src.optimization.multi_gw import MultiGWOptimizer

class LiveFPLPipeline:
    def __init__(self):
        self.api_url = "https://fantasy.premierleague.com/api"
        self.last_updated = None
        self.model_clf = None
        self.model_reg = None
        
    def train_or_load_models(self):
        print("[1/6] Loading historical data and pre-trained ML models...")
        df = pd.read_parquet(r"d:\Fpl_winning\data\processed\features_targets.parquet")
        self.features = [c for c in df.columns if c.endswith('_last_1') or c.endswith('_last_3') or c.endswith('_last_5') or c.endswith('_last_10') or c.endswith('_std')]
        if 'was_home' in df.columns:
            self.features.append('was_home')
            df['was_home'] = df['was_home'].astype(int)
        
        train = df[df['season'] < '2026-27'].dropna(subset=['target_points']).copy()
        
        self.model_clf = lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1)
        self.model_clf.fit(train[self.features], train['target_played'])
        
        train_played = train[train['target_played'] == 1]
        self.model_reg = lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
        self.model_reg.fit(train_played[self.features], train_played['target_points'])

    def fetch_live_data(self):
        print("[2/6] Fetching live data from FPL API...")
        try:
            res = requests.get(f"{self.api_url}/bootstrap-static/", timeout=10)
            res.raise_for_status()
            data = res.json()
            self.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            events = pd.DataFrame(data['events'])
            current_gw = events[events['is_current'] == True]['id'].values
            next_gw = events[events['is_next'] == True]['id'].values
            target_gw = next_gw[0] if len(next_gw) > 0 else (current_gw[0] if len(current_gw) > 0 else 1)
            
            fixtures_res = requests.get(f"{self.api_url}/fixtures/", timeout=10)
            fixtures_res.raise_for_status()
            fixtures = fixtures_res.json()
            
            print(f"      -> Successfully fetched. Target GW: {target_gw}")
            return data['elements'], data['teams'], target_gw, fixtures
            
        except Exception as e:
            print(f"CRITICAL ERROR: FPL API unavailable or changed. {e}")
            sys.exit(1)

    def generate_live_features(self, elements, teams_data):
        print("[3/6] Mapping live API data to ML feature schema...")
        df = pd.DataFrame(elements)
        teams = {t['id']: t['name'] for t in teams_data}
        
        pos_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        
        live_df = pd.DataFrame()
        live_df['player_code'] = df['code']
        live_df['id'] = df['id']
        live_df['name'] = df['web_name']
        live_df['position'] = df['element_type'].map(pos_map)
        live_df['team'] = df['team'].map(teams)
        live_df['value'] = df['now_cost'] / 10.0
        live_df['selected'] = df['selected_by_percent'].astype(float)
        live_df['availability'] = df['status']
        live_df['minutes'] = df['minutes'].astype(int)
        live_df['starts'] = df.get('starts', 0).astype(int)
        live_df['total_points'] = df['total_points'].astype(int)
        live_df['points_per_90'] = np.where(live_df['minutes'] > 0, (live_df['total_points'] / live_df['minutes']) * 90, 0.0)
        live_df['xG'] = df.get('expected_goals', 0.0).astype(float)
        live_df['xA'] = df.get('expected_assists', 0.0).astype(float)
        live_df['xGI'] = df.get('expected_goal_involvements', 0.0).astype(float)
        live_df['xGC'] = df.get('expected_goals_conceded', 0.0).astype(float)
        
        # Load actual historical rolling features from the parquet file
        hist_df = pd.read_parquet(r"d:\Fpl_winning\data\processed\features_targets.parquet")
        latest_features = hist_df.sort_values(['season', 'GW']).drop_duplicates('player_code', keep='last')
        
        # Calculate how much history they actually have (new signings will have <= 3 rows if just added this season)
        hist_counts = hist_df.groupby('player_code').size().reset_index(name='hist_rows')
        latest_features = pd.merge(latest_features, hist_counts, on='player_code', how='left')
        
        # Merge live players with their latest historical features
        live_df = pd.merge(live_df, latest_features[['player_code', 'season', 'GW', 'team', 'hist_rows'] + self.features], 
                           on='player_code', how='left', indicator='_merge', suffixes=('', '_hist'))
                           
        # Load ingested foreign league priors
        try:
            foreign_priors = pd.read_csv(r"d:\Fpl_winning\data\processed\non_pl_priors.csv")
        except FileNotFoundError:
            foreign_priors = pd.DataFrame()
            
        history_types = []
        flags = []
        
        for i, row in live_df.iterrows():
            player_flags = []
            
            # 1. Determine Player History Type
            if row['_merge'] == 'both' and row.get('hist_rows', 0) > 3:
                if row['team'] == row['team_hist']:
                    hist_type = 'PL_HISTORY'
                else:
                    hist_type = 'PL_TEAM_TRANSFER'
                    player_flags.append('TEAM_CHANGED_UPDATED_CONTEXT')
            else:
                # Distinguish foreign transfers from academy new players via FPL price heuristic
                # We apply this even if they have 1-3 rows in the parquet (e.g. newly ingested GW1 rows for summer signings)
                thresholds = {'GK': 4.5, 'DEF': 4.5, 'MID': 5.0, 'FWD': 5.5}
                if row['value'] > thresholds.get(row['position'], 4.5):
                    hist_type = 'NON_PL_TRANSFER'
                else:
                    hist_type = 'NEW_PLAYER'
                    
                if row['_merge'] == 'both':
                    player_flags.append('OVERRODE_MINIMAL_HISTORY')
                    
            history_types.append(hist_type)
            
            if pd.isna(row['xG']):
                player_flags.append('MISSING_STATS')
                
            # 2. Smart Imputation & Normalized Priors
            missing_features = False
            for f in self.features:
                if pd.isna(row.get(f, np.nan)) or (hist_type in ['NON_PL_TRANSFER', 'NEW_PLAYER']):
                    # Force overwrite the feature if they are categorized as a new transfer
                    missing_features = True
                    if 'value' in f:
                        live_df.at[i, f] = row['value']
                    elif hist_type == 'NON_PL_TRANSFER':
                        # Check ingested foreign data first
                        foreign_match = foreign_priors[foreign_priors['name'].str.contains(row['name'], case=False, na=False)] if not foreign_priors.empty else pd.DataFrame()
                        if not foreign_match.empty:
                            tax = foreign_match['league_tax'].values[0]
                            prev_mins = foreign_match['prev_league_minutes'].values[0]
                            
                            if prev_mins > 0:
                                # Normalize per 90 and apply tax
                                goals_90 = (foreign_match['goals'].values[0] / prev_mins) * 90 * tax
                                assists_90 = (foreign_match['assists'].values[0] / prev_mins) * 90 * tax
                                xg_90 = (foreign_match['xg'].values[0] / prev_mins) * 90 * tax
                                xa_90 = (foreign_match['xa'].values[0] / prev_mins) * 90 * tax
                                
                                if 'minutes' in f: live_df.at[i, f] = min(90.0, (prev_mins / 38.0)) # Approx per game
                                elif 'goals' in f: live_df.at[i, f] = goals_90 * (3 if '3' in f else 5) # Scale to rolling window
                                elif 'assists' in f: live_df.at[i, f] = assists_90 * (3 if '3' in f else 5)
                                elif 'xG' in f: live_df.at[i, f] = xg_90 * (3 if '3' in f else 5)
                                elif 'xA' in f: live_df.at[i, f] = xa_90 * (3 if '3' in f else 5)
                                elif 'points' in f: live_df.at[i, f] = (goals_90 * 4 + assists_90 * 3 + 2) * tax * (3 if '3' in f else 5)
                                else: live_df.at[i, f] = 0.0
                            else:
                                live_df.at[i, f] = 0.0
                                
                            player_flags.append('FOREIGN_DATA_INGESTED')
                        else:
                            # Fallback if no scout data provided
                            live_df.at[i, f] = 0.0 
                    else:
                        live_df.at[i, f] = 0.0 # New academy players get 0
            
            if missing_features:
                if hist_type == 'NON_PL_TRANSFER' and 'FOREIGN_DATA_INGESTED' not in player_flags:
                    player_flags.append('NO_FOREIGN_DATA_FOUND')
                elif hist_type == 'NEW_PLAYER':
                    player_flags.append('ACADEMY_PRIOR_APPLIED')
                    
            flags.append(" | ".join(set(player_flags)) if player_flags else 'CLEAN')
            
        live_df['player_history_type'] = history_types
        live_df['validation_flags'] = flags
        
        live_df.rename(columns={'season': 'hist_season', 'GW': 'hist_gw'}, inplace=True)
            
        return live_df

    def predict_8gw_horizon(self, live_df, target_gw, fixtures, teams_data):
        print("[4/6] Generating 8-GW ML predictions dynamically using actual FPL fixtures...")
        team_name_to_id = {t['name']: t['id'] for t in teams_data}
        horizon_gws = []
        for t in range(8):
            gw = target_gw + t
            gw_df = live_df.copy()
            gw_df['xP'] = 0.0
            if 'was_home' in self.features:
                gw_df['was_home'] = 0
            
            gw_fixtures = [f for f in fixtures if f['event'] == gw]
            
            for t_name, t_id in team_name_to_id.items():
                t_fixtures = [f for f in gw_fixtures if f['team_h'] == t_id or f['team_a'] == t_id]
                team_mask = (gw_df['team'] == t_name)
                
                if len(t_fixtures) == 0 or team_mask.sum() == 0:
                    pass # BGW or no players for this team
                else:
                    agg_xP = np.zeros(team_mask.sum())
                    for f in t_fixtures:
                        is_home = (f['team_h'] == t_id)
                        if 'was_home' in self.features:
                            gw_df.loc[team_mask, 'was_home'] = int(is_home)
                        
                        prob_play = self.model_clf.predict_proba(gw_df.loc[team_mask, self.features])[:, 1]
                        xp_if_play = self.model_reg.predict(gw_df.loc[team_mask, self.features])
                        
                        avail = gw_df.loc[team_mask, 'availability']
                        prob_play[avail.isin(['i', 's', 'u', 'n'])] = 0.0
                        prob_play[avail == 'd'] *= 0.25
                        
                        base_xP = np.clip(prob_play * xp_if_play, a_min=0, a_max=None)
                        
                        fdr = f['team_h_difficulty'] if is_home else f['team_a_difficulty']
                        fdr_multiplier = (7.0 - fdr) / 4.0 
                        home_adv = 1.1 if is_home else 0.9
                        
                        agg_xP += (base_xP * fdr_multiplier * home_adv)
                        
                    gw_df.loc[team_mask, 'xP'] = agg_xP
            
            out_cols = ['id', 'player_code', 'name', 'position', 'team', 'value', 'selected', 'availability', 
                        'minutes', 'starts', 'total_points', 'points_per_90', 'xG', 'xA', 'xGI', 'xGC',
                        'player_history_type', 'validation_flags', 'xP']
            gw_df = gw_df[[c for c in out_cols if c in gw_df.columns]]
            
            gw_df['P10'] = (gw_df['xP'] - 1.5).clip(lower=0)
            gw_df['P50'] = gw_df['xP']
            gw_df['P90'] = gw_df['xP'] + 3.0
            
            horizon_gws.append(gw_df)
            
        return horizon_gws

    def fetch_team_data(self, team_id, elements):
        print(f"Fetching real team data for Team ID: {team_id}...")
        try:
            # Find the most recent GW to get the active squad (includes current active GWs)
            res = requests.get(f"{self.api_url}/bootstrap-static/", timeout=10)
            events = pd.DataFrame(res.json()['events'])
            visible_events = events[(events['finished'] == True) | (events['is_current'] == True)]
            
            if not visible_events.empty:
                last_gw = visible_events['id'].max()
            else:
                last_gw = 1 # Fallback if season hasn't started
                
            team_url = f"{self.api_url}/entry/{team_id}/event/{last_gw}/picks/"
            print(f"DEBUG: Requesting {team_url}")
            team_res = requests.get(team_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            print(f"DEBUG: HTTP Status {team_res.status_code}")
            
            if team_res.status_code != 200:
                print(f"DEBUG: API Error Body: {team_res.text}")
                raise ValueError(f"FPL API Error {team_res.status_code} for URL: {team_url}")
                
            team_data = team_res.json()
            
            # Fetch chip history & calculate dynamic FTs
            history_url = f"{self.api_url}/entry/{team_id}/history/"
            hist_res = requests.get(history_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            chips_played = []
            dynamic_ft = 1
            if hist_res.status_code == 200:
                hist_data = hist_res.json()
                chips_played = [c['name'] for c in hist_data.get('chips', [])]
                
                # Calculate rolled free transfers (up to 5)
                h = hist_data.get('current', [])
                chips_by_event = {c['event']: c['name'] for c in hist_data.get('chips', [])}
                
                ft = 1
                if len(h) > 1:
                    for gw in h[1:]:
                        gw_num = gw['event']
                        if chips_by_event.get(gw_num) in ['wildcard', 'freehit']:
                            transfers_made = 0
                        else:
                            transfers_made = gw['event_transfers']
                        
                        ft = max(0, ft - transfers_made)
                        ft = min(5, ft + 1)
                dynamic_ft = ft
            
            # Map FPL element IDs to our player_code (Opta ID) format
            elements_df = pd.DataFrame(elements)
            id_to_code = dict(zip(elements_df['id'], elements_df['code']))
            
            squad_codes = [id_to_code[p['element']] for p in team_data['picks']]
            bank = team_data['entry_history']['bank'] / 10.0
            
            return squad_codes, bank, chips_played, dynamic_ft
        except Exception as e:
            print(f"Failed to fetch team: {e}")
            raise e

    def run_optimizer(self, horizon_gws, my_squad, bank, ft, strategy_config=None):
        print("[5/6] Initializing CP-SAT Optimizer (Multi-GW)...")
        if strategy_config is None:
            strategy_config = {'mode': 'neutral', 'type': 'expected_points'}
            
        H = len(horizon_gws)
        opt = MultiGWOptimizer(horizon=H, bank=bank, free_transfers=ft)
        opt.build_model(horizon_gws, my_squad, strategy_config)
        res = opt.solve()
        return res, horizon_gws[0], my_squad

    def display_recommendation(self, res, gw_df, current_squad):
        print("\n[6/6] LIVE PIPELINE COMPLETE. VALIDATING RECOMMENDATION:")
        print(f"Data Timestamp: {self.last_updated}")
        
        if res['status'] not in ['OPTIMAL', 'FEASIBLE']:
            print("ERROR: Optimizer failed to find a valid FPL squad.")
            sys.exit(1)
            
        buy_names = gw_df[gw_df['player_code'].isin(res['buy'])]['name'].tolist()
        sell_names = gw_df[gw_df['player_code'].isin(res['sell'])]['name'].tolist()
        xi_df = gw_df[gw_df['player_code'].isin(res['xi'])]
        cap_name = gw_df[gw_df['player_code'] == res['cap']]['name'].values[0]
        
        # Verify Constraints
        assert len(res['xi']) == 11, "XI must be exactly 11 players."
        assert len(res['buy']) == len(res['sell']), "Buys must equal sells."
        
        pos_counts = xi_df['position'].value_counts().to_dict()
        assert pos_counts.get('GK', 0) == 1, "Must have exactly 1 GK in XI."
        assert pos_counts.get('DEF', 0) >= 3, "Must have at least 3 DEFs in XI."
        assert pos_counts.get('FWD', 0) >= 1, "Must have at least 1 FWD in XI."
        
        print("\n================ LIVE FPL RECOMMENDATION ================")
        print(f"Transfers : BUY {buy_names} | SELL {sell_names}")
        print(f"Hits Taken: {res['hits']}")
        print(f"Captain   : {cap_name}")
        print("\nStarting XI:")
        for pos in ['GK', 'DEF', 'MID', 'FWD']:
            players = xi_df[xi_df['position'] == pos]['name'].tolist()
            print(f"{pos} ({len(players)}): {', '.join(players)}")
        print("=========================================================\n")
        print("SUCCESS: Valid FPL squad generated. All constraints respected.")

if __name__ == "__main__":
    pipeline = LiveFPLPipeline()
    pipeline.train_or_load_models()
    elements, teams, target_gw = pipeline.fetch_live_data()
    live_df = pipeline.generate_live_features(elements, teams)
    horizon_gws = pipeline.predict_8gw_horizon(live_df, target_gw)
    res, gw_df, current_squad = pipeline.run_optimizer(horizon_gws)
    pipeline.display_recommendation(res, gw_df, current_squad)
