import requests
import pandas as pd

class LiveFPLAPI:
    def __init__(self):
        self.base_url = "https://fantasy.premierleague.com/api/"
        
    def fetch_current_data(self):
        print("Connecting to live 2026/27 FPL API...")
        res = requests.get(f"{self.base_url}bootstrap-static/")
        if res.status_code == 200:
            data = res.json()
            players = pd.DataFrame(data['elements'])
            events = pd.DataFrame(data['events'])
            teams = pd.DataFrame(data['teams'])
            
            # Identify current gameweek
            current_gw = events[events['is_current'] == True]
            next_gw = events[events['is_next'] == True]
            
            print(f"Successfully fetched live data. Next GW: {next_gw['id'].values[0] if not next_gw.empty else 'N/A'}")
            
            return {
                'players': players,
                'teams': teams,
                'events': events
            }
        else:
            raise ConnectionError(f"Failed to fetch FPL data: {res.status_code}")
            
    def fetch_live_fixtures(self):
        res = requests.get(f"{self.base_url}fixtures/")
        if res.status_code == 200:
            return pd.DataFrame(res.json())
        return None
