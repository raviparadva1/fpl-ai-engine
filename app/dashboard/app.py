import streamlit as st
import pandas as pd
import numpy as np
import sys
import json
from datetime import datetime

# Path setup to access backend modules
sys.path.append(r"d:\Fpl_winning")
from src.api.live_pipeline import LiveFPLPipeline

# Configuration
st.set_page_config(page_title="FPL AI Optimizer", page_icon="⚽", layout="wide", initial_sidebar_state="expanded")

# --- Custom Premium CSS ---
st.markdown("""
<style>
    :root {
        --bg-color: #0d1117;
        --card-bg: #161b22;
        --accent: #00ff87;
        --accent-dark: #00b35e;
        --text-primary: #c9d1d9;
        --text-muted: #8b949e;
        --border-color: #30363d;
    }
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-primary);
    }
    .metric-card {
        background-color: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }
    .metric-value { font-size: 28px; font-weight: 800; color: #ffffff; }
    .metric-label { font-size: 12px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; }
    
    .pitch-container {
        background: linear-gradient(180deg, #1d4d29 0%, #255e34 100%);
        border: 2px solid #ffffff;
        border-radius: 8px;
        padding: 20px;
        position: relative;
        overflow: hidden;
    }
    .pitch-line-center {
        position: absolute; top: 50%; left: 0; width: 100%; height: 2px; background: rgba(255,255,255,0.4);
    }
    .pitch-circle {
        position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
        width: 100px; height: 100px; border: 2px solid rgba(255,255,255,0.4); border-radius: 50%;
    }
    .player-card {
        background-color: rgba(0,0,0,0.6); border: 1px solid var(--accent); border-radius: 6px;
        padding: 5px; text-align: center; font-size: 12px; color: #fff; margin: 5px;
    }
    .player-name { font-weight: bold; color: #fff; }
    .player-stats { color: var(--accent); font-size: 10px; }
    .captain-badge { background: #ffaa00; color: #000; padding: 2px 4px; border-radius: 4px; font-size: 10px; font-weight: bold; }
    
    .buy-action { color: #00ff87; font-weight: bold; }
    .sell-action { color: #ff4444; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- Backend Caching & Integration ---
@st.cache_resource
def get_pipeline():
    pipeline = LiveFPLPipeline()
    pipeline.train_or_load_models()
    return pipeline

@st.cache_data(ttl=3600) # Cache for 1 hour
def fetch_and_predict(_pipeline):
    elements, teams, target_gw, fixtures = _pipeline.fetch_live_data()
    live_df = _pipeline.generate_live_features(elements, teams)
    horizon_gws = _pipeline.predict_8gw_horizon(live_df, target_gw, fixtures, teams)
    return elements, teams, target_gw, live_df, horizon_gws, fixtures

@st.cache_data(show_spinner=False)
def run_strategy_optimizer(_pipeline, horizon_gws, strategy_config, squad_codes, bank, ft, horizon=3):
    _pipeline.model_clf # Ensure pipeline is loaded
    from src.optimization.multi_gw import MultiGWOptimizer
    
    # We slice the horizon_gws list down to the selected horizon
    sliced_gws = horizon_gws[:horizon]
    opt_instance = MultiGWOptimizer(horizon=horizon, bank=bank, free_transfers=ft)
    opt_instance.build_model(sliced_gws, squad_codes, strategy_config)
    return opt_instance.solve(), squad_codes

# --- Authentication / Setup Screen ---
if 'team_id' not in st.session_state:
    st.markdown('<div style="text-align: center; margin-top: 100px;">', unsafe_allow_html=True)
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/f/f2/Premier_League_Logo.svg/1200px-Premier_League_Logo.svg.png", width=150)
    st.title("FPL AI Optimizer")
    st.markdown("### Connect Your Actual FPL Team")
    st.markdown("Enter your FPL Team ID to generate strictly personalized 8-GW optimization recommendations.")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        team_id_input = st.text_input("FPL Team ID:", placeholder="e.g. 1234567")
        if st.button("Connect Team", use_container_width=True):
            if team_id_input.isdigit():
                st.session_state.team_id = int(team_id_input)
                st.rerun()
            else:
                st.error("Please enter a valid numeric Team ID.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# --- Load Data ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/f/f2/Premier_League_Logo.svg/1200px-Premier_League_Logo.svg.png", width=100)
st.sidebar.title("Live Control Panel")
st.sidebar.markdown(f"**Connected Team ID:** `{st.session_state.team_id}`")
if st.sidebar.button("Disconnect Team"):
    del st.session_state.team_id
    st.cache_data.clear()
    st.rerun()

pipeline = get_pipeline()

if st.sidebar.button("🔄 Force Refresh API Data"):
    st.cache_data.clear()
    if 'neutral_res' in st.session_state: del st.session_state.neutral_res
    if 'my_squad_cache' in st.session_state: del st.session_state.my_squad_cache
    
try:
    elements, teams, target_gw, live_df, horizon_gws, fixtures = fetch_and_predict(pipeline)
    st.sidebar.success(f"✅ Data Live: {pipeline.last_updated}")
except Exception as e:
    st.error("🚨 CRITICAL: Failed to connect to FPL API. Displaying stale/fallback data.")
    st.stop()
    
# Fetch Actual Team
try:
    if 'my_squad_cache' not in st.session_state:
        st.session_state.my_squad_cache, st.session_state.bank_cache = pipeline.fetch_team_data(st.session_state.team_id, elements)
except Exception as e:
    st.error(f"🚨 Could not fetch team data for ID {st.session_state.team_id}. Ensure it is correct.")
    st.exception(e) # Expose developer diagnostics
    if st.button("Back to Login"):
        del st.session_state.team_id
        st.rerun()
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("Strategy Overrides")
user_horizon = st.sidebar.select_slider("Planning Horizon (GWs)", options=[1, 3, 5, 8], value=3)
user_ft = st.sidebar.number_input("Available Free Transfers", min_value=1, max_value=5, value=1)
user_bank = st.sidebar.number_input("Current Bank (£m)", min_value=0.0, max_value=100.0, value=st.session_state.bank_cache, step=0.1)

# Evaluate default 3-GW Risk-Adjusted strategy
if 'neutral_res' not in st.session_state:
    with st.spinner(f"Optimizing {user_horizon}-GW Branch-and-Bound Tree..."):
        st.session_state.neutral_res, st.session_state.current_squad = run_strategy_optimizer(
            pipeline, horizon_gws, {'mode': 'neutral', 'type': 'expected_points', 'risk_penalty': 0.2}, 
            st.session_state.my_squad_cache, user_bank, user_ft, horizon=user_horizon
        )

res = st.session_state.neutral_res
current_squad = st.session_state.current_squad
gw_df = horizon_gws[0]

# --- UI Sections ---
st.title(f"Gameweek {target_gw} Analytics")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Overview", "🏟️ My Team", "🔄 Transfer Planner", 
    "🔍 Player Explorer", "⚔️ Rival Analysis", "🎲 Simulation", "📈 Backtesting"
])

# 1. Overview
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Target GW</div><div class="metric-value">GW {target_gw}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Projected Next GW</div><div class="metric-value">{res.get("projected_obj", 0):.1f} xP</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Transfer Hits</div><div class="metric-value">-{res.get("hits", 0) * 4}</div></div>', unsafe_allow_html=True)
    with col4:
        cap_name = gw_df[gw_df['player_code'] == res.get('cap')]['name'].values[0] if res.get('cap') else "N/A"
        st.markdown(f'<div class="metric-card"><div class="metric-label">Recommended Captain</div><div class="metric-value">{cap_name}</div></div>', unsafe_allow_html=True)
        
    st.divider()
    st.subheader("Automated Recommendation")
    if res['status'] in ['OPTIMAL', 'FEASIBLE']:
        buy_names = gw_df[gw_df['player_code'].isin(res['buy'])]['name'].tolist()
        sell_names = gw_df[gw_df['player_code'].isin(res['sell'])]['name'].tolist()
        st.success(f"**Action Plan**: SELL {sell_names} ➔ BUY {buy_names}")
    else:
        st.error("Optimizer failed to find a feasible solution.")

# 2. My Team (Pitch)
with tab2:
    st.subheader("Your Current Actual Squad")
    actual_squad_df = gw_df[gw_df['player_code'].isin(st.session_state.my_squad_cache)]
    st.dataframe(actual_squad_df[['name', 'position', 'team', 'value', 'xP']].sort_values('position'), use_container_width=True)
    
    st.divider()
    st.subheader("AI Optimized Starting XI (Recommendation)")
    st.markdown("**(Information Source: LIVE FPL API & 8-GW ML Projections)**")
    
    if res['status'] in ['OPTIMAL', 'FEASIBLE']:
        xi_df = gw_df[gw_df['player_code'].isin(res['xi'])]
        
        st.markdown('<div class="pitch-container"><div class="pitch-line-center"></div><div class="pitch-circle"></div>', unsafe_allow_html=True)
        
        for pos in ['GK', 'DEF', 'MID', 'FWD']:
            players = xi_df[xi_df['position'] == pos]
            cols = st.columns(len(players) if len(players) > 0 else 1)
            for i, (_, p) in enumerate(players.iterrows()):
                with cols[i]:
                    cap_tag = '<span class="captain-badge">C</span>' if p['player_code'] == res['cap'] else ''
                    st.markdown(f"""
                    <div class="player-card">
                        <div class="player-name">{p['name']} {cap_tag}</div>
                        <div class="player-stats">{p['team']} | £{p['value']:.1f}m</div>
                        <div class="player-stats">AI xP: {p['xP']:.1f}</div>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# 3. Transfer Planner
with tab3:
    st.subheader("8-GW Optimization Roadmap")
    
    # Analyze Transfer Module (Safe Architecture)
    st.markdown("Click below to run the 8-Gameweek CP-SAT Optimizer against your current squad.")
    if st.button("💡 Analyze Transfers", type="primary"):
        with st.spinner("Calculating optimal 8-GW trajectory..."):
            res, gw_df, current_squad = pipeline.run_optimizer(horizon_gws, st.session_state.my_squad_cache, st.session_state.bank_cache, 1)
            
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### <span class='sell-action'>SELL (Outgoing)</span>", unsafe_allow_html=True)
            sell_df = gw_df[gw_df['player_code'].isin(res.get('sell', []))]
            st.dataframe(sell_df[['name', 'position', 'team', 'value', 'xP']], use_container_width=True)
        with col2:
            st.markdown("### <span class='buy-action'>BUY (Incoming)</span>", unsafe_allow_html=True)
            buy_df = gw_df[gw_df['player_code'].isin(res.get('buy', []))]
            st.dataframe(buy_df[['name', 'position', 'team', 'value', 'xP']], use_container_width=True)
            
        st.divider()
        
        st.subheader("💡 Recommendation Details")
        
        if res.get('buy') and res.get('sell'):
            out_name = ", ".join(sell_df['name'].tolist())
            in_name = ", ".join(buy_df['name'].tolist())
            
            # Calculate expected gain over 8 GWs
            out_8gw_xp = sum([t_df[t_df['player_code'].isin(res['sell'])]['xP'].sum() for t_df in horizon_gws])
            in_8gw_xp = sum([t_df[t_df['player_code'].isin(res['buy'])]['xP'].sum() for t_df in horizon_gws])
            expected_gain = in_8gw_xp - out_8gw_xp
            hit_cost = res.get('hits', 0) * 4
            net_gain = expected_gain - hit_cost
            
            confidence = "High" if net_gain > 5.0 else ("Medium" if net_gain > 2.0 else "Low")
            
            st.info(f"**RECOMMENDATION**\n\n**SELL**: {out_name}\n\n**BUY**: {in_name}\n\n"
                    f"**Expected gain**: +{net_gain:.1f} over 8 GWs (Gross: +{expected_gain:.1f}, Hits: -{hit_cost})\n\n"
                    f"**Transfer cost**: {hit_cost} pts\n\n"
                    f"**Confidence**: {confidence}\n\n"
                    "*You should review these metrics before manually executing this transfer in the official FPL app.*")
        else:
            st.success("**RECOMMENDATION**: Roll your transfer. No moves significantly improve the 8-GW projection.")
            
        st.divider()
        with st.expander("🔍 Deep Dive: Model Diagnostics"):
            st.markdown("Detailed breakdown of the underlying data driving the optimizer's decision.")
            
            display_cols = ['id', 'name', 'player_history_type', 'validation_flags', 'value', 'selected', 'minutes', 'starts', 'total_points', 'points_per_90', 'xG', 'xA', 'xGI', 'xGC', 'xP']
            
            if not sell_df.empty and not buy_df.empty:
                st.markdown(f"#### Outgoing Player(s) Metrics")
                st.dataframe(sell_df[display_cols], use_container_width=True)
                
                st.markdown(f"#### Incoming Player(s) Metrics")
                st.dataframe(buy_df[display_cols], use_container_width=True)
                
                st.markdown("#### 8-Gameweek Projection Profile (Base xP)")
                # Reconstruct the 8-GW trajectory for the chart
                proj_data = []
                for t in range(8):
                    t_df = horizon_gws[t]
                    for p_code in res.get('sell', []) + res.get('buy', []):
                        p_name = t_df[t_df['player_code'] == p_code]['name'].values[0]
                        p_xp = t_df[t_df['player_code'] == p_code]['xP'].values[0]
                        proj_data.append({'Gameweek': f"GW+{t+1}", 'Player': p_name, 'Projected Points': p_xp})
                
                proj_df = pd.DataFrame(proj_data)
                st.line_chart(proj_df, x='Gameweek', y='Projected Points', color='Player')

# 4. Player Explorer
with tab4:
    st.subheader("Global Player Database (Point-in-Time Predictions)")
    search = st.text_input("🔍 Search Player")
    
    display_df = gw_df[['name', 'player_history_type', 'validation_flags', 'position', 'team', 'value', 'selected', 'availability', 'minutes', 'starts', 'total_points', 'points_per_90', 'xG', 'xA', 'xGI', 'xGC', 'xP', 'P10', 'P50', 'P90']]
    if search:
        display_df = display_df[display_df['name'].str.contains(search, case=False)]
        
    st.dataframe(display_df.sort_values('xP', ascending=False).head(50), use_container_width=True)

# 5. Rival Analysis
with tab5:
    st.subheader("Rival Strategy Selector")
    strategy = st.radio("Select Mini-League Context:", ["Risk-Adjusted (Default)", "Pure xP (Neutral)", "Attack (Trailing, Coeff=0.5)", "Defend (Leading, Coeff=0.3)"])
    
    if st.button("Recalculate Strategy"):
        strat_conf = {'mode': 'neutral', 'type': 'expected_points', 'risk_penalty': 0.2}
        if "Pure" in strategy: strat_conf = {'mode': 'neutral', 'type': 'expected_points', 'risk_penalty': 0.0}
        if "Attack" in strategy: strat_conf = {'mode': 'attack', 'type': 'expected_points', 'risk_penalty': 0.0}
        if "Defend" in strategy: strat_conf = {'mode': 'defend', 'type': 'expected_points', 'risk_penalty': 0.0}
        
        with st.spinner(f"Optimizing {user_horizon}-GW Branch-and-Bound Tree..."):
            new_res, _ = run_strategy_optimizer(
                pipeline, horizon_gws, strat_conf, 
                st.session_state.my_squad_cache, user_bank, user_ft, horizon=user_horizon
            )
        st.session_state.neutral_res = new_res
        st.rerun()

# 6. Simulation
with tab6:
    st.subheader("Monte Carlo Uncertainty Bounds")
    st.write("Visualizing the confidence intervals for the top 10 projected players.")
    mc_top = gw_df.sort_values('xP', ascending=False).head(10)
    
    chart_data = mc_top[['name', 'P10', 'xP', 'P90']].set_index('name')
    st.bar_chart(chart_data)

# 7. Backtesting
with tab7:
    st.subheader("Rigorous Historical Backtest (2024-25 & 2025-26)")
    try:
        with open(r'd:\Fpl_winning\data\processed\rigorous_results.json', 'r') as f:
            backtest_res = json.load(f)
            
        bt_df = pd.DataFrame(backtest_res)
        st.dataframe(bt_df[['Season', 'Strategy', 'Net Points', 'Gross Points', 'Hits', 'OPTIMAL_cnt']], use_container_width=True)
    except Exception:
        st.warning("Historical backtest results not found. Run the rigorous backtest script to populate.")
