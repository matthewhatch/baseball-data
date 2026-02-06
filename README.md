# MLB Game Outcome Prediction Model

A machine learning system to predict MLB game outcomes (home win vs away win) using 5 years of historical game data (2020-2024).

## Quick Start

```bash
# 1. Fetch new game data (optional)
python -m src.scraper

# 2. Explore the data
python -m src.eda

# 3. Train models
python -m src.train_model

# 4. Make predictions
python -m src.predict
```

## Project Organization

```
baseball-data/
├── src/                    # Main source code
│   ├── scraper.py         # Fetch data from statsapi.mlb.com
│   ├── eda.py             # Data exploration & analysis
│   ├── train_model.py     # Model training pipeline
│   └── predict.py         # Game predictions
├── data/
│   └── raw/               # CSV files (11,486 games)
├── models/                # Trained models & scalers
├── docs/                  # Documentation
├── archive/               # Deprecated code & logs
└── main.py                # Entry point
```

**See [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for detailed organization.**

## Features

✅ **Data Acquisition** - Scrapes 11,486 MLB games from official statsapi.mlb.com  
✅ **Exploration** - Analyzes distributions, team performance, home field advantage  
✅ **Feature Engineering** - Historical win rates, scoring averages, recent form  
✅ **Three Models** - Logistic Regression, Random Forest, Gradient Boosting  
✅ **Best Accuracy** - 54.6% on 2024 test set (vs 50% random baseline)  
✅ **Predictions** - Real-time game outcome forecasts with confidence scores  

## Model Performance

| Model | Accuracy | Precision | Recall | ROC AUC |
|-------|----------|-----------|--------|---------|
| Logistic Regression | 54.05% | 54.11% | 75.88% | 0.564 |
| Random Forest | 54.01% | 54.78% | 65.62% | 0.545 |
| **Gradient Boosting** ⭐ | **54.63%** | **55.67%** | **62.10%** | **0.550** |

**Test Set**: 2,570 games from 2024  
**Training Set**: 8,848 games from 2020-2023

## Data

- **11,486 total games** across 5 MLB seasons
- **10 fields**: game_id, date, home_team, away_team, home_score, away_score, gameType, venue, status, doubleheader
- **30 MLB teams** represented
- **Home win rate**: 52.9% (statistically significant advantage)

### Usage

#### Get EDA Summary
- `date`: YYYY-MM-DD format
- `home_team`: 3-letter team code (NYY, BOS, LAD, etc.)
- `away_team`: 3-letter team code
- `home_score`: Integer final score
- `away_score`: Integer final score

Create files for each season: `games_2020.csv`, `games_2021.csv`, etc.

### 3. Run Training

```bash
python train.py
```

This will:
1. Load all historical seasons
2. Calculate rolling team statistics (wins %, runs avg, etc.)
3. Split data into train/validation/test sets (respecting temporal order)
4. Train an XGBoost model
5. Evaluate on test set
6. Save the trained model to `./models/`

## Model Architecture

### Features Engineered
- **Team Win %**: Rolling 10-game win percentage for each team
- **Runs Average**: Rolling 10-game average runs scored
- **Runs Allowed**: Rolling 10-game average runs allowed by opponent
- **Home/Away Stats**: Separate calculations for home and away games

### Models Available
- **XGBoost** (default): Fast, gradient boosted trees
- **Random Forest**: Ensemble of decision trees

### Train/Validation/Test Split
- **Temporal split** preserves time order (critical for time series)
- 80% training, 10% validation, 10% test
- Ensures model doesn't use future data to predict the past

## Output

After training completes:

```
TEST SET METRICS:
  Accuracy:  0.5847    (% correct predictions)
  Precision: 0.5912    (% of predicted wins that are correct)
  Recall:    0.5847    (% of actual wins that were predicted)
  F1 Score:  0.5878    (harmonic mean of precision & recall)
```

**Top features**: Shows which team stats are most predictive.

## Usage

### Make Predictions

```python
from warehouse.model import BaseballPredictionModel
import pandas as pd

# Load trained model
model = BaseballPredictionModel(model_type="xgboost")
model.load("./models/baseball_model_xgb.pkl")

# Prepare features (same format as training)
new_games = pd.DataFrame({
    'home_team_win_pct': [0.55],
    'away_team_win_pct': [0.50],
    # ... add all features used during training
})

# Make predictions
predictions = model.predict(new_games)
probabilities = model.predict_proba(new_games)
```

## Next Steps

1. **Get historical data**: Download from Baseball Reference, Kaggle, or MLB API
2. **Tune hyperparameters**: Modify learning rate, tree depth, etc. in `warehouse/model.py`
3. **Try ensemble models**: Combine XGBoost + Random Forest predictions
4. **Add advanced features**: Pitcher stats, weather, rest days, injury reports
5. **Deploy**: Create API endpoint for real-time predictions

## Notes

- Model predicts **home team win probability** (binary classification)
- Uses only **past game statistics** (no future data leakage)
- Baseline: ~55% accuracy (better than random 50%)
- Historical data quality matters significantly

## Dependencies

- `pandas`: Data manipulation
- `numpy`: Numerical computation
- `scikit-learn`: ML utilities (scaling, metrics)
- `xgboost`: Gradient boosting
- `matplotlib`, `seaborn`: Visualization
