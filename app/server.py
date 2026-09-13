import sys
import os
sys.path.append(r"d:\Fpl_winning")

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from src.api.live_pipeline import LiveFPLPipeline
from src.optimization.multi_gw import MultiGWOptimizer

app = FastAPI(title="FPL AI Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline = LiveFPLPipeline()
pipeline.train_or_load_models()

from typing import Optional


@app.get("/api/players")
def get_players():
    try:
        elements, _, _, _ = pipeline.fetch_live_data()
        return [{"id": e['id'], "name": e['web_name']} for e in elements]
    except Exception as e:
        return []

@app.get("/api/optimize/{team_id}")
def optimize_team(team_id: int, mode: str = 'neutral', sell_override: Optional[str] = None, buy_override: Optional[str] = None, ft_override: int = None):
    try:
        elements, teams_data, target_gw, fixtures = pipeline.fetch_live_data()
        my_squad, bank, chips_played, ft = pipeline.fetch_team_data(team_id, elements)
        
        if ft_override is not None:
            ft = ft_override
        
        
        # Determine Chip availability
        # Wildcard resets at GW19. So if they played 'wildcard' in GW<19 and we are <19, they don't have it.
        wc_available = 'wildcard' not in chips_played # simplified
        fh_available = 'freehit' not in chips_played
        bb_available = 'bboost' not in chips_played
        tc_available = '3xc' not in chips_played
        
        # Team value calculation
        squad_value = sum([e['now_cost'] for e in elements if e['code'] in my_squad])
        total_value = (squad_value + bank) / 10.0
        
        live_df = pipeline.generate_live_features(elements, teams_data)
        
        # Apply Manual UI Overrides (Early Transfers)
        if sell_override and buy_override:
            import unicodedata
            def norm(s): 
                return unicodedata.normalize('NFKD', str(s)).encode('ASCII', 'ignore').decode('utf-8').lower()
            
            norm_sell = norm(sell_override)
            norm_buy = norm(buy_override)
            
            live_df['norm_name'] = live_df['name'].apply(norm)
            
            sell_match = live_df[live_df['norm_name'].str.contains(norm_sell, na=False)]
            buy_match = live_df[live_df['norm_name'].str.contains(norm_buy, na=False)]
            
            if not sell_match.empty and not buy_match.empty:
                sell_code = sell_match.iloc[0]['player_code']
                buy_code = buy_match.iloc[0]['player_code']
                sell_price = sell_match.iloc[0]['value']
                buy_price = buy_match.iloc[0]['value']
                
                if sell_code in my_squad:
                    my_squad = [c for c in my_squad if c != sell_code]
                    my_squad.append(buy_code)
                    bank = bank + sell_price - buy_price
                    ft = max(0, ft - 1)  # Consume an FT since they mocked a transfer
                    print(f"Applied Override: Sold {sell_match.iloc[0]['name']}, Bought {buy_match.iloc[0]['name']}. New Bank: {bank}")
            else:
                print(f"Override failed! Sell Match Empty: {sell_match.empty}, Buy Match Empty: {buy_match.empty}")
        full_horizon_gws = pipeline.predict_8gw_horizon(live_df, target_gw, fixtures, teams_data)
        
        # Hardcode 8-GW horizon for true long-term EV, but cap to length of full_horizon_gws
        horizon_gws = full_horizon_gws[:8]
        
        risk_penalty = 0.2 if mode == 'neutral' else 0.0
        strategy_config = {'mode': mode, 'type': 'expected_points', 'risk_penalty': risk_penalty}
        
        print(f"DEBUG IN SERVER: ft_override={ft_override}, final ft={ft}")
        opt = MultiGWOptimizer(horizon=len(horizon_gws), bank=bank, free_transfers=ft)
        opt.build_model(horizon_gws, my_squad, strategy_config)
        res = opt.solve()
        
        # Format the output for the frontend
        gw_df = horizon_gws[0]
        

        # --- Generate Next Fixture Strings ---
        team_id_to_name = {t['id']: t['name'] for t in teams_data}
        team_id_to_short = {t['id']: t['short_name'] for t in teams_data}
        
        target_gw_fixtures = [f for f in fixtures if f['event'] == target_gw]
        team_next_fixture = {}
        team_next_fdr = {}
        
        for t_id, t_name in team_id_to_name.items():
            t_fix = [f for f in target_gw_fixtures if f['team_h'] == t_id or f['team_a'] == t_id]
            if not t_fix:
                team_next_fixture[t_name] = "BLANK"
                team_next_fdr[t_name] = 0
            else:
                f = t_fix[0] # Take first match
                if f['team_h'] == t_id:
                    team_next_fixture[t_name] = f"{team_id_to_short[f['team_a']]} (H)"
                    team_next_fdr[t_name] = f['team_h_difficulty']
                else:
                    team_next_fixture[t_name] = f"{team_id_to_short[f['team_h']]} (A)"
                    team_next_fdr[t_name] = f['team_a_difficulty']
        # ------------------------------------
        
        buy_players = []
        for b in res.get('buy', []):
            row = gw_df[gw_df['player_code'] == b].iloc[0]
            buy_players.append({
                'name': row['name'], 'team': row['team'], 'pos': row['position'],
                'price': row['value'], 'xP': round(row['xP'], 2)
            })
            
        sell_players = []
        for s in res.get('sell', []):
            row = gw_df[gw_df['player_code'] == s].iloc[0]
            sell_players.append({
                'name': row['name'], 'team': row['team'], 'pos': row['position'],
                'price': row['value'], 'xP': round(row['xP'], 2)
            })
            
        starting_xi = []
        for p in res.get('xi', []):
            row = gw_df[gw_df['player_code'] == p].iloc[0]
            is_cap = (p == res.get('cap'))
            is_vc = (p == res.get('vcap'))
            starting_xi.append({
                'name': row['name'], 'team': row['team'], 'pos': row['position'],
                'price': row['value'], 'xP': round(row['xP'], 2),
                'is_captain': is_cap, 'is_vice': is_vc,
                'next_fixture': team_next_fixture.get(row['team'], 'BLANK'),
                'next_fdr': team_next_fdr.get(row['team'], 0)
            })
            
        bench = []
        bench_codes = set(res.get('buy', []) + list(set(my_squad) - set(res.get('sell', [])))) - set(res.get('xi', []))
        for p in bench_codes:
            row = gw_df[gw_df['player_code'] == p].iloc[0]
            bench.append({
                'name': row['name'], 'team': row['team'], 'pos': row['position'],
                'xP': round(row['xP'], 2),
                'next_fixture': team_next_fixture.get(row['team'], 'BLANK'),
                'next_fdr': team_next_fdr.get(row['team'], 0)
            })
            
        # Format Bench Order (GK, Sub 1, Sub 2, Sub 3)
        gks = [p for p in bench if p['pos'] == 'GK']
        outfielders = [p for p in bench if p['pos'] != 'GK']
        outfielders.sort(key=lambda x: x['xP'], reverse=True)
        
        formatted_bench = []
        if gks:
            formatted_bench.append({'order': 'SUB GK', **gks[0]})
        for i, p in enumerate(outfielders):
            formatted_bench.append({'order': f'SUB {i+1}', **p})

        # CHIP AI LOGIC
        ai_chip = "NONE"
        hits_taken = res.get('hits', 0)
        
        bench_xp = sum([p['xP'] for p in bench])
        cap_xp = max([p['xP'] for p in starting_xi if p.get('is_captain')], default=0)
        
        if hits_taken >= 3 and wc_available:
            ai_chip = "WILDCARD"
        elif bench_xp > 12.0 and bb_available:
            ai_chip = "BENCH BOOST"
        elif cap_xp > 9.0 and tc_available:
            ai_chip = "TRIPLE CAPTAIN"
            
        return {
            'ai_chip': ai_chip,
            'target_gw': int(target_gw),
            'team_value': round(total_value, 1),
            'bank': round(bank, 1),
            'ft': ft,
            'status': res['status'],
            'projected_points': round(res.get('projected_obj', 0), 2),
            'hits': res.get('hits', 0) * 4,
            'buy': buy_players,
            'sell': sell_players,
            'starting_xi': starting_xi,
            'bench': formatted_bench
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def serve_frontend():
    return FileResponse(r"d:\Fpl_winning\app\frontend\index.html")

# Serve static files if needed
app.mount("/static", StaticFiles(directory=r"d:\Fpl_winning\app\frontend"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
