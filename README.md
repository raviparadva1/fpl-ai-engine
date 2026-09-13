# Autonomous FPL AI Engine

A production-grade, full-stack Fantasy Premier League Artificial Intelligence. 

This engine automatically connects to the official FPL APIs, predicts future player performance using a Machine Learning Random Forest model, and uses Google's CP-SAT solver (Combinatorial Optimization) to mathematically calculate the absolute best transfers across an 8-Gameweek horizon.

## Features
- **Live FPL API Integration:** Automatically fetches live player prices, injury statuses, fixtures, and your exact bank balance/free transfers.
- **Machine Learning Predictor:** Generates dynamic Expected Points (xP) based on underlying historical metrics (xG, xA) and opponent Fixture Difficulty Ratings (FDR).
- **CP-SAT Optimizer:** Solves millions of transfer permutations over 8 weeks to maximize your total score, perfectly respecting FPL constraints (budget, 3-players-per-team, -4pt hits).
- **Autonomous Chip Strategy:** Monitors your available chips and calculates the optimal mathematical moment to deploy Wildcards and Bench Boosts.
- **SaaS Dashboard:** A highly responsive, dark-mode frontend built with Tailwind CSS, featuring live editable Free Transfers, autocomplete manual overrides, and color-coded FDR fixtures.

## Getting Started

### Prerequisites
- Python 3.9+
- `pip install fastapi uvicorn pandas scikit-learn ortools requests`

### Running the Server
1. Clone the repository.
2. Run the FastAPI backend:
```bash
python -m uvicorn app.server:app --host 0.0.0.0 --port 8000
```
3. Open `http://localhost:8000` in your web browser.
4. Enter your FPL Team ID, set your strategy, and click **SOLVE**.

## Architecture
- `src/api/live_pipeline.py`: Handles FPL API data ingestion and Machine Learning inference.
- `src/optimization/multi_gw.py`: The Google OR-Tools CP-SAT math engine.
- `app/server.py`: The FastAPI backend bridging the Python engine to the web.
- `app/frontend/index.html`: The UI dashboard.
