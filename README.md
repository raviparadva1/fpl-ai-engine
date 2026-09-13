# 🏆 Autonomous FPL AI Engine

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-orange.svg)
![Optimization](https://img.shields.io/badge/Optimization-Google%20OR--Tools-red.svg)

A production-grade, full-stack Fantasy Premier League (FPL) Artificial Intelligence that autonomously manages your FPL team using Machine Learning and Combinatorial Optimization.

This engine connects to the official FPL APIs, predicts future player performance using a Random Forest model, and uses Google's CP-SAT solver to mathematically calculate the absolute optimal transfer sequences across an 8-Gameweek time horizon.

## ✨ Features
- **Live FPL API Integration:** Automatically fetches live player prices, injury statuses, fixtures, and your exact bank balance and accrued free transfers.
- **Machine Learning Predictor:** Generates dynamic Expected Points (xP) by analyzing historical metrics (xG, xA, ICT index) against opponent Fixture Difficulty Ratings (FDR).
- **Mathematical Optimizer:** Uses Google OR-Tools (CP-SAT) to solve millions of transfer permutations over an 8-week horizon. It perfectly respects FPL constraints: budget caps, maximum of 3 players per team, formation limits, and calculates the exact value of -4pt hits.
- **Autonomous Chip Strategy:** Monitors your available chips and calculates the mathematical expected value (EV) of deploying a Wildcard or Bench Boost versus saving them.
- **Esports-Style Dashboard:** A highly responsive, sleek dark-mode UI built with Tailwind CSS. It features dynamic FT counters, autocomplete manual transfer overrides, and color-coded FDR fixture badges.

## 🏗️ Architecture

The pipeline operates in three distinct layers:

1. **The Live Data & ML Pipeline (`src/api/live_pipeline.py`)**
   - Ingests the current game state directly from FPL servers.
   - Cleans and transforms the data into a feature matrix.
   - Feeds the matrix into a pre-trained Random Forest Regressor to output an 8-Gameweek rolling xP prediction for every active player.

2. **The Optimization Engine (`src/optimization/multi_gw.py`)**
   - Converts the FPL ruleset into absolute mathematical constraints (Linear Programming).
   - Maximizes a specific objective function (Expected Points) minus the penalty of transfer hits.
   - Uses warm-start heuristics to ensure the solver always finds a baseline valid squad within 15 seconds.

3. **The Web Application (`app/server.py` & `app/frontend/index.html`)**
   - A FastAPI backend bridging the Python engine to a frontend interface.
   - Allows users to inject "Manual Overrides" (forcing the AI to evaluate the impact of a specific transfer they want to make).

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- The following pip packages: `fastapi uvicorn pandas scikit-learn ortools requests`

### Installation & Execution
1. Clone the repository.
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
*(If you don't have a requirements file yet, just run `pip install fastapi uvicorn pandas scikit-learn ortools requests`)*

3. Run the FastAPI backend server:
```bash
python -m uvicorn app.server:app --host 0.0.0.0 --port 8000
```
4. Open your web browser and navigate to `http://localhost:8000`.
5. Enter your **FPL Team ID**, select your risk strategy, and click **SOLVE**.

## 🧠 Why an 8-Gameweek Horizon?
Optimizing for just the next Gameweek leads to "greedy" transfer behavior (buying a player for one good game, then needing to sell them immediately). By optimizing across an 8-Gameweek horizon, the AI intrinsically understands the value of long-term holds and fixture swings, behaving exactly like a veteran elite FPL manager.
