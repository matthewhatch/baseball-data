# MLOps Setup Guide

Complete MLOps pipeline for baseball predictions using GitHub Actions, Docker, FastAPI, and MLflow.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ GitHub Repository                                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Code          Workflows           Models    Data          │
│  (src/)    → .github/workflows/ → models/  ← data/raw/     │
│                   ↓                                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ GitHub Actions CI/CD Pipeline                       │   │
│  │ • Daily model retraining                            │   │
│  │ • Auto-commit model updates                         │   │
│  │ • Test + Deploy                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                   ↓                                         │
└─────────────────────┼───────────────────────────────────────┘
                      │
        ┌─────────────┴──────────────┐
        ↓                            ↓
   ┌─────────────┐          ┌──────────────┐
   │ Docker      │          │ FastAPI      │
   │ Image       │          │ (api.py)     │
   │ (reproducible)         │ (REST API)   │
   └─────────────┘          └──────────────┘
        ↓                            ↓
   ┌─────────────┐          ┌──────────────┐
   │ Deploy to   │          │ /predict     │
   │ Cloud       │          │ /health      │
   │ (production)│          │ /teams       │
   └─────────────┘          └──────────────┘
                             
        MLflow Tracking (mlflow_tracker.py)
        • Log experiments
        • Track metrics
        • Model registry
        • Compare runs
```

---

## 1. Automation (GitHub Actions)

**File**: `.github/workflows/train.yml`

### What it does:
- Runs on schedule (daily at 2 AM UTC)
- Fetches fresh data
- Trains models
- Commits updates back to repo
- Logs metrics

### Setup:
1. Push code to GitHub
2. Workflow runs automatically on schedule
3. Models stay up-to-date without manual work

### To customize:
Edit `.github/workflows/train.yml`:
```yaml
schedule:
  - cron: '0 2 * * *'  # Change timing here
```

### View workflow runs:
```
GitHub → Actions tab → "ML Model Training Pipeline"
```

---

## 2. Reproducibility (Docker)

**Files**: `Dockerfile`, `.dockerignore`

### What it does:
- Packages entire application (code + dependencies)
- Runs identically on any machine
- Multi-stage build for small images

### Build the image:
```bash
docker build -t baseball-ml:latest .
```

### Run predictions:
```bash
docker run baseball-ml:latest python -m src.predict
```

### Run training:
```bash
docker run -v $(pwd)/models:/app/models baseball-ml:latest python -m src.train_model
```

### Deploy to cloud:
```bash
# Push to Docker Hub
docker tag baseball-ml:latest yourname/baseball-ml:latest
docker push yourname/baseball-ml:latest

# Deploy to Heroku, AWS, Google Cloud, etc.
```

### Benefits:
- ✓ Works on Mac, Linux, Windows identically
- ✓ No "works on my machine" problems
- ✓ Easy cloud deployment
- ✓ Reproducible environment

---

## 3. Deployment (FastAPI)

**File**: `api.py`

### What it does:
- REST API for predictions
- Health checks
- Team info endpoints
- Auto-generated documentation

### Install dependencies:
```bash
pip install fastapi uvicorn
```

### Run locally:
```bash
python api.py
```

Then visit: `http://localhost:8000`

### API endpoints:

**1. Health Check**
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model": "gradient_boosting",
  "version": "1.0.0"
}
```

**2. Make Prediction**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "LAD",
    "away_team": "NYY"
  }'
```

Response:
```json
{
  "home_team": "LAD",
  "away_team": "NYY",
  "predicted_winner": "HOME",
  "home_win_prob": 0.643,
  "away_win_prob": 0.357,
  "confidence": 64.3
}
```

**3. Get Teams**
```bash
curl http://localhost:8000/teams
```

**4. Team Stats**
```bash
curl http://localhost:8000/stats/LAD
```

**5. API Docs** (Interactive)
```
http://localhost:8000/docs  # Swagger UI
http://localhost:8000/redoc # ReDoc
```

### Deploy to Heroku:
```bash
# Create requirements.txt with fastapi, uvicorn
pip freeze > requirements.txt

# Create Procfile
echo "web: uvicorn api:app --host 0.0.0.0 --port $PORT" > Procfile

# Deploy
heroku create baseball-predictions
git push heroku main
```

### Deploy with Docker:
```bash
docker build -t baseball-api:latest .
docker run -p 8000:8000 baseball-api:latest uvicorn api:app --host 0.0.0.0
```

---

## 4. Experiment Tracking (MLflow)

**File**: `mlflow_tracker.py`

### What it does:
- Track model training runs
- Log hyperparameters
- Log metrics
- Compare experiments
- Model registry

### Install:
```bash
pip install mlflow
```

### Integrate with training:
```python
from mlflow_tracker import MLflowTracker

tracker = MLflowTracker("baseball-predictions")

# During training:
tracker.log_training_run(
    model_type="GradientBoosting",
    params={'n_estimators': 100, 'max_depth': 5},
    metrics={'accuracy': 0.5463, 'precision': 0.552},
    model=gb_model,
    features=['home_historical_wr', 'away_historical_wr', ...]
)
```

### View experiments:
```bash
mlflow ui
```

Then visit: `http://localhost:5000`

### Features:
- View all runs
- Compare metrics side-by-side
- Download best model
- Reproduce runs
- Model registry for versioning

---

## Complete Workflow

### Day 1: Setup
```bash
# Clone repo, install
git clone <repo>
cd baseball-data
pip install -r requirements.txt

# Test each component
python -m src.eda            # EDA works
python -m src.train_model    # Training works
python api.py                # API works
```

### Day 2: Containerize
```bash
# Build Docker image
docker build -t baseball-ml:v1 .

# Test in container
docker run baseball-ml:v1 python -m src.predict
```

### Day 3: Deploy
```bash
# Push to Docker Hub
docker push yourname/baseball-ml:v1

# Deploy to cloud (Heroku example)
heroku create my-baseball-api
heroku container:push web
heroku container:release web
```

### Ongoing: Automate
```bash
# Commit to GitHub
git push origin main

# GitHub Actions runs automatically:
# 1. Fetches fresh data daily
# 2. Retrains models
# 3. Commits updates
# 4. Runs tests

# View in GitHub → Actions tab
```

### Monitoring: Track Experiments
```bash
# Start MLflow UI
mlflow ui

# View all training runs, metrics, models
# Compare runs side-by-side
# Download best model
```

---

## Testing Integration

### Add tests to GitHub Actions:
```yaml
- name: Run tests
  run: |
    pip install pytest
    pytest tests/
```

### Example test:
```python
# tests/test_api.py
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict():
    response = client.post("/predict", json={
        "home_team": "LAD",
        "away_team": "NYY"
    })
    assert response.status_code == 200
    assert "predicted_winner" in response.json()
```

---

## Production Checklist

- [ ] GitHub Actions workflow set up and tested
- [ ] Docker image builds successfully
- [ ] API runs and responds correctly
- [ ] MLflow tracking configured
- [ ] Tests pass locally and in CI/CD
- [ ] Models committed to repo (or artifact storage)
- [ ] Secrets (.env credentials) not committed
- [ ] API deployed to cloud
- [ ] Health checks configured
- [ ] Monitoring/logging set up

---

## Cost Estimate

| Service | Cost | Notes |
|---------|------|-------|
| GitHub Actions | Free | 2,000 min/month free |
| Docker Hub | Free | Public repos |
| Heroku | $7/mo | Starter dyno |
| MLflow | Free | Self-hosted |
| **Total** | **~$7/mo** | Very affordable |

---

## Next Steps

1. **Push to GitHub** - Enable Actions
2. **Build Docker image** - Test locally
3. **Deploy API** - Heroku or cloud
4. **Set up MLflow** - Track experiments
5. **Add monitoring** - Logs, alerts
6. **Create tests** - Ensure reliability

---

## Resources

- [GitHub Actions Docs](https://docs.github.com/en/actions)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [Production ML Systems](https://martinfowler.com/articles/cd4ml.html)

---

**Status**: Ready for production  
**Last Updated**: February 6, 2026
