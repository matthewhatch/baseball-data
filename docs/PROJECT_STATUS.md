# Project Status: MLB Game Outcome Prediction System

**Last Updated:** February 2026  
**Status:** ✅ PRODUCTION-READY (Phase 1 Complete)

---

## Quick Overview

A complete machine learning pipeline that predicts MLB game outcomes with 54.63% accuracy. Includes data collection, model training, API deployment, and automated daily retraining.

---

## What's Working

### ✅ Data Pipeline
- **Source:** statsapi.mlb.com REST API (free, no auth required)
- **Dataset:** 11,486 games across 2020-2024 seasons
- **Fields:** game_id, date, home_team, away_team, home_score, away_score, gameType, venue, status, doubleheader
- **Location:** `data/raw/` (games_2020.csv through games_2024.csv)
- **Status:** Fully scraped, validated, ready to use

### ✅ Feature Engineering
- **9 Core Features:** home_historical_wr, home_field_advantage, home_avg_score, home_avg_runs_allowed, away_historical_wr, away_field_advantage, away_avg_score, away_avg_runs_allowed, home_team_coded
- **Rolling Features:** Added recently (last 5, 10, 20 game rolling win rates)
- **Processing:** `src/train_model.py` (375+ lines, production-ready)

### ✅ Machine Learning Models (3 Trained)
- **Best Model:** Gradient Boosting (54.63% accuracy) ⭐
- **Alternative:** Random Forest (54.01% accuracy)
- **Baseline:** Logistic Regression (54.05% accuracy)
- **Train/Test Split:** 2020-2023 (8,848 games) / 2024 (2,570 games)
- **Files:** `models/gb_model.pkl`, `models/rf_model.pkl`, `models/lr_model.pkl`
- **Metadata:** `models/feature_names.pkl`, `models/scaler.pkl`

### ✅ Prediction System
- **Module:** `src/predict.py` (178 lines)
- **Outputs:** Winner prediction + confidence score + win probabilities for both teams
- **Error Handling:** Robust NaN handling for missing team stats
- **Sample Predictions:** Included for testing

### ✅ API (FastAPI)
- **File:** `api.py` (~350 lines)
- **Endpoints:**
  - `POST /predict` - Get prediction for a game
  - `GET /health` - Health check
  - `GET /teams` - List all teams
  - `GET /stats/{team}` - Get team statistics
  - `GET /docs` - Swagger UI documentation
- **Status:** Production-ready, tested
- **Run:** `python api.py` (port 8000)

### ✅ Automation (GitHub Actions)
- **File:** `.github/workflows/train.yml`
- **Triggers:** Daily 2 AM UTC, manual `workflow_dispatch`, push to main
- **Steps:**
  1. Fetch latest game data
  2. Run exploratory analysis
  3. Train all 3 models
  4. Extract performance metrics
  5. Save models and metadata
  6. Commit changes to repo
  7. Email notification on errors
- **Performance:** First run ~3-5 min, cached runs ~30-60 sec (10x faster with model/data caching)
- **Status:** Fully automated, no manual intervention needed

### ✅ Local Database (DuckDB)
- **Location:** `duckdb_poc/baseball.duckdb` (~50MB)
- **Setup:** `duckdb_poc/duckdb_setup.py` (fully automated)
- **Components:**
  - Data loader (CSV → DuckDB)
  - Model store (versioning + metadata)
  - Prediction query (SQL + model inference)
  - Config management
- **Status:** Fully functional, proven in testing
- **Guide:** `README_DUCKDB.md` (comprehensive)

### ✅ Docker Containerization
- **File:** `Dockerfile` (multi-stage, optimized)
- **Build:** `docker build -t baseball-predictor .`
- **Run:** `docker run -p 8000:8000 baseball-predictor`
- **Status:** Production-ready, health checks included

### ✅ Experiment Tracking (MLflow)
- **Module:** `mlflow_tracker.py` (~150 lines)
- **Features:** Log runs, track metrics, model registry, compare experiments
- **Status:** Ready to integrate into train_model.py

### ✅ Documentation
- README.md - Main project overview
- MLOPS_GUIDE.md - Complete setup instructions
- USE_CASES.md - 12 real-world prediction scenarios
- src/README.md - Source code guide
- models/README.md - Model details
- ingestion/README.md - Data pipeline guide
- README_DUCKDB.md - Local database setup
- ORGANIZATION.md - Code structure
- PROJECT_STRUCTURE.md - Directory layout

### ✅ Code Quality
- Professional directory structure (`src/`, `data/`, `models/`, `docs/`, `archive/`)
- Legacy code archived (not deleted)
- Lean dependencies (20 essential packages, no bloat)
- Version control (GitHub: matthewhatch/baseball-data)

---

## Architecture at a Glance

```
statsapi.mlb.com (REST API)
    ↓
data/raw/*.csv (11,486 games)
    ↓
src/train_model.py (Feature engineering + training)
    ↓
models/*.pkl (GB model + metadata)
    ↓
┌─────────────────┬──────────────────┐
│   FastAPI       │   GitHub Actions │
│   (api.py)      │   (train.yml)    │
└─────────────────┴──────────────────┘
    ↓                    ↓
Local API        Daily retraining
(port 8000)      (automated)
    ↓
Docker (Containerized)
```

---

## Dependencies (Cleaned & Lean)

```
pandas>=1.5.0           # Data manipulation
numpy>=1.23.0           # Numerical computing
scikit-learn>=1.2.0     # ML models
fastapi>=0.104.0        # REST API framework
uvicorn>=0.24.0         # ASGI server
pydantic>=2.0.0         # Data validation
duckdb>=0.9.0           # Local database
mlflow>=2.8.0           # Experiment tracking
requests>=2.31.0        # HTTP library
beautifulsoup4>=4.12.0  # Web scraping
python-dotenv>=1.0.0    # Environment variables
```

---

## How to Use

### 1. Quick Start (Local)
```bash
# Set up environment
python -m venv env
source env/bin/activate
pip install -r requirements.txt

# Train model
python src/train_model.py

# Make predictions
python src/predict.py
```

### 2. Run API Locally
```bash
python api.py
# Visit http://localhost:8000/docs for interactive docs
```

### 3. Run in Docker
```bash
docker build -t baseball-predictor .
docker run -p 8000:8000 baseball-predictor
```

### 4. Automated Daily Training (GitHub Actions)
- Commits to `main` automatically trigger workflow
- Daily schedule: 2 AM UTC
- Models automatically saved and committed back

### 5. Use DuckDB Locally
```bash
cd duckdb_poc
python duckdb_setup.py        # Initialize database
python duckdb_predict.py      # Make predictions
```

---

## Current Metrics

| Metric | Value |
|--------|-------|
| **Best Model Accuracy** | 54.63% (GB) |
| **Baseline (Home Win)** | 52.9% |
| **Improvement** | +1.73 percentage points |
| **Training Set Size** | 8,848 games |
| **Test Set Size** | 2,570 games |
| **Total Games** | 11,486 |
| **API Response Time** | ~50-100ms |
| **GitHub Actions Run Time** | 30-60 sec (cached) |
| **Model Size** | 1.2 MB (GB) |

---

## What's NOT Done (Phase 2+)

- [ ] Advanced features (pitcher stats, weather, travel distance)
- [ ] Production cloud deployment (AWS, GCP, Heroku)
- [ ] Dashboard/UI (Streamlit, Dash)
- [ ] Real-time monitoring and alerts
- [ ] dbt data pipeline (if needed for feature complexity)
- [ ] Deep learning models (LSTM time series)
- [ ] Advanced ensemble methods

---

## Next Steps (User to Choose)

See [docs/PROJECT_NEXT_STEPS.md](PROJECT_NEXT_STEPS.md) for detailed options and recommendations.
