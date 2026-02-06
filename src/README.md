# Source Code Module (src/)

Complete documentation of the MLB Game Outcome Prediction System's main source code.

## Overview

The `src/` directory contains the active, production-ready code for:
1. **Data Acquisition** - Fetching MLB game data
2. **Data Analysis** - Exploring patterns and distributions
3. **Model Training** - Building outcome prediction models
4. **Predictions** - Making game outcome forecasts

All modules are designed to work together in a cohesive pipeline.

## Module Directory

```
src/
├── __init__.py        # Package initialization
├── scraper.py        # Fetch data from statsapi.mlb.com
├── eda.py            # Exploratory data analysis
├── train_model.py    # Model training pipeline
└── predict.py        # Game outcome predictions
```

## Detailed Module Documentation

---

### `scraper.py` - Data Acquisition

**Purpose**: Fetch MLB game data from the official MLB Stats API (statsapi.mlb.com)

**Key Functions**:

#### `scrape_mlb_statsapi_season(year, team_abbr_map, verbose=False)`
Fetches all final games for a given season.

**Parameters**:
- `year` (int) - MLB season year (e.g., 2024)
- `team_abbr_map` (dict) - Maps team IDs to 3-letter abbreviations
- `verbose` (bool) - Show progress with tqdm bars

**Returns**: `pd.DataFrame` with columns:
- game_id (int) - Unique game identifier (gamePk)
- date (str) - Game date (YYYY-MM-DD)
- home_team (str) - Home team abbreviation (e.g., NYY)
- home_score (int) - Home team final score
- away_team (str) - Away team abbreviation
- away_score (int) - Away team final score
- gameType (str) - R=Regular, F=Spring, W=World Series, etc.
- venue (str) - Stadium name
- status (str) - "Final" (only final games included)
- doubleheader (str) - "Y"/"N" doubleheader indicator

**Example**:
```python
from src.scraper import scrape_mlb_statsapi_season

# Create team ID to abbreviation mapping
team_abbr_map = {
    146: 'ARI', 144: 'ATL', 145: 'CHC', # ... all 30 teams
}

# Fetch 2024 season
games_2024 = scrape_mlb_statsapi_season(2024, team_abbr_map, verbose=True)
print(f"Fetched {len(games_2024)} games")
```

#### `scrape_multiple_seasons(years, output_dir, verbose=False)`
Convenience function to scrape multiple seasons and save to CSV.

**Parameters**:
- `years` (list) - List of season years
- `output_dir` (str) - Directory to save CSV files
- `verbose` (bool) - Show progress

**Returns**: `dict` - {year: DataFrame} pairs

**Example**:
```python
from src.scraper import scrape_multiple_seasons

# Fetch all seasons
data = scrape_multiple_seasons(
    years=[2020, 2021, 2022, 2023, 2024],
    output_dir="data/raw/",
    verbose=True
)
# Saves to: data/raw/games_2020.csv, games_2021.csv, ...
```

**Data Source**: statsapi.mlb.com
- **Endpoint**: `/api/v1/schedule`
- **Authentication**: None required
- **Rate Limit**: ~6.2 days/second (227 days per season ≈ 35-36 seconds)
- **Coverage**: March 20 - November 1 (regular season + postseason)
- **Filters**: Only "Final" status games included

**Key Features**:
- ✅ No authentication needed
- ✅ Clean JSON responses
- ✅ All 30 MLB teams supported
- ✅ Progress bars with tqdm
- ✅ Automatic CSV caching
- ✅ Handles all game types (regular, playoffs, world series)

---

### `eda.py` - Exploratory Data Analysis

**Purpose**: Generate comprehensive statistics and visualizations of game data

**Key Functions**:

#### Main Analysis Pipeline
The module runs a complete EDA when executed:

```python
python -m src.eda
```

**Analysis Sections**:

1. **Data Loading**
   - Loads all 5 seasons (2020-2024)
   - Creates outcome column (home_win: 1 if home wins, 0 if away wins)
   - Prints data summary (11,486 total games)

2. **Basic Statistics**
   - Home win rate: 52.9%
   - Away win rate: 46.6%
   - Score distributions (mean, std, min, max)
   - Run differential analysis

3. **Team Performance**
   - Win rates by team (home, away, overall)
   - Top 10 and bottom 10 teams
   - Home field advantage by team

4. **Score Distributions**
   - Range and statistics
   - Win margin analysis (typically 1-3 runs)

5. **Yearly Trends**
   - Year-by-year home win rates
   - Average scoring trends
   - 2020 COVID season insights

6. **Data Quality**
   - Missing value check
   - Data type verification
   - Shape validation

**Output Summary**:
```
✓ Home Win Rate: 0.529 (6,075 / 11,486)
✓ Away Win Rate: 0.471 (5,348 / 11,486)
✓ Teams: 31 unique teams (30 active + unknown)
✓ Sample Size: 11,486 games (good for training)
```

**Key Insights Provided**:
- Home field advantage: ~3% edge
- Top team: LAD (62.7% win rate)
- Bottom team: WSH (41.1% win rate)
- Most common win margin: 1 run
- Average scoring: 4.5 runs/team

**No External API Calls**: Uses only local CSV files

---

### `train_model.py` - Model Training Pipeline

**Purpose**: Build and train three classification models for game outcome prediction

**Key Components**:

#### 1. Data Loading
```python
# Loads all seasons from data/raw/games_*.csv
games = pd.concat([pd.read_csv(f"data/raw/games_{year}.csv") 
                   for year in [2020, 2021, 2022, 2023, 2024]])
# Result: 11,486 games
```

#### 2. Feature Engineering

**Features Created** (9 total):

| Feature | Type | Description |
|---------|------|-------------|
| home_historical_wr | float | Home team's historical win rate |
| away_historical_wr | float | Away team's historical win rate |
| home_avg_score | float | Home team average runs scored |
| away_avg_score | float | Away team average runs scored |
| days_into_season | int | Days since March 20 (0-227) |
| home_field_advantage | float | Team's home-away performance diff |
| home_recent_form | float | Win rate last 10 home games |
| day_of_week | int | Day of game (0=Monday, 6=Sunday) |
| month | int | Month (7=July, 10=October, etc.) |

**Feature Calculation**:
- Computed at each game date (leakage prevention)
- Uses only historical data (no future information)
- Handles missing data gracefully

#### 3. Data Splitting

```
Total Games: 11,418 (with sufficient history)
├── Training: 8,848 games (2020-2023)
├── Test: 2,570 games (2024)
```

#### 4. Models Trained

**Model 1: Logistic Regression**
```python
LogisticRegression(max_iter=1000, random_state=42)
```
- Scaled features: StandardScaler
- Performance: 54.05% accuracy, 0.564 ROC AUC
- Use when: Interpretability needed, size critical

**Model 2: Random Forest**
```python
RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
```
- Unscaled features (trees don't need scaling)
- Performance: 54.01% accuracy, 0.545 ROC AUC
- Use when: Highest recall needed (65.62%)

**Model 3: Gradient Boosting** ⭐ (Selected)
```python
GradientBoostingClassifier(n_estimators=100, max_depth=7, learning_rate=0.1)
```
- Unscaled features
- Performance: 54.63% accuracy, 0.550 ROC AUC
- Use when: Best overall performance

#### 5. Model Saving

All three models saved to `models/` directory:
- `lr_model.pkl` (778 B)
- `rf_model.pkl` (14 MB)
- `gb_model.pkl` (1.2 MB) ⭐ Recommended
- `scaler.pkl` (878 B) - For Logistic Regression
- `feature_names.pkl` - Column mapping

#### 6. Performance Visualization

Generates `models/model_performance.png`:
- Model comparison bar chart
- ROC curves for all three models
- Feature importance (Random Forest)
- Prediction probability distribution

**Usage**:
```bash
python -m src.train_model
```

**Output**:
```
✓ Training complete
✓ Best Model: Gradient Boosting (54.63% accuracy)
✓ Models saved to models/ directory
✓ Visualization saved to models/model_performance.png
```

**Key Parameters** (tunable in code):
- Feature window sizes (10 games)
- Tree depths (7-15)
- Number of estimators (100)
- Learning rates (0.1)

---

### `predict.py` - Game Outcome Predictions

**Purpose**: Make game outcome predictions using trained Gradient Boosting model

**Key Functions**:

#### `predict_game(home_team, away_team, games_df, team_stats, model, max_date=None)`

Makes a prediction for a single game.

**Parameters**:
- `home_team` (str) - 3-letter home team code (e.g., "NYY")
- `away_team` (str) - 3-letter away team code (e.g., "BOS")
- `games_df` (DataFrame) - Historical game data
- `team_stats` (dict) - Pre-computed team statistics
- `model` - Trained sklearn model
- `max_date` (datetime) - Date cutoff for feature calculation (default: max in data)

**Returns**: `dict` with:
```python
{
    'home_team': 'LAD',
    'away_team': 'ATL',
    'home_win_prob': 0.699,      # Probability home wins
    'away_win_prob': 0.301,      # Probability away wins
    'predicted_winner': 'HOME',   # Predicted winner
    'confidence': 69.9            # Confidence %
}
```

**Example**:
```python
from src.predict import predict_game
import pickle

# Load model and data
model = pickle.load(open('models/gb_model.pkl', 'rb'))
games = pd.read_csv('data/raw/games_2024.csv')

# Predict
result = predict_game('LAD', 'ATL', games, team_stats, model)
print(f"{result['away_team']} @ {result['home_team']}")
print(f"→ {result['predicted_winner']} Win | {result['confidence']:.1f}% confidence")
```

#### What Happens When Executed

When run directly:
```bash
python -m src.predict
```

**Outputs**:

1. **Team Statistics Table** (30 teams)
   - Home win rate
   - Away win rate
   - Overall win rate
   - Home runs scored avg
   - Away runs scored avg

2. **Sample Predictions** (4 matchups)
   - Example: "ATL @ LAD → AWAY Win | 69.9% confidence"
   - Shows home/away win probabilities

**Key Insights Provided**:
- Best home teams (LAD 66.5% at home)
- Worst home teams (WSH 41.6% at home)
- Recent form indicators
- Home field advantage magnitude

**Feature Calculation**:
For each prediction, the model calculates:
- Historical win rates up to max_date
- Average scoring (home/away)
- Days into season (season progression)
- Home field advantage difference
- Recent form (last 10 home games)
- Calendar features (day of week, month)

---

## Data Flow Diagram

```
┌──────────────────┐
│  statsapi.mlb.com │  (external API)
└────────┬─────────┘
         │
         │ scraper.py: scrape_mlb_statsapi_season()
         │
         ▼
┌──────────────────────┐
│  data/raw/games_*.csv│  (CSV files)
└────────┬─────────────┘
         │
    ┌────┴────────────────┬──────────────────┐
    │                     │                  │
    │ eda.py              │ train_model.py   │ predict.py
    │ (exploration)       │ (training)       │ (inference)
    │                     │                  │
    ▼                     ▼                  ▼
 Analysis            models/gb_model.pkl   Predictions
 Tables              models/*.pkl           Probabilities
 Insights            models/feature_*.pkl   Confidence
```

## Usage Workflows

### Quick Start (All Steps)

```bash
# 1. Fetch data
python -m src.scraper

# 2. Analyze data
python -m src.eda

# 3. Train models
python -m src.train_model

# 4. Make predictions
python -m src.predict
```

### Data Analysis Only

```bash
python -m src.eda
```

### Retraining Models

```bash
python -m src.train_model
```

### Batch Predictions

```python
from src.predict import predict_game
import pickle
import pandas as pd

model = pickle.load(open('models/gb_model.pkl', 'rb'))
games = pd.read_csv('data/raw/games_2024.csv')

# Predict multiple games
matchups = [
    ('LAD', 'ATL'),
    ('HOU', 'NYY'),
    ('ATL', 'WSH'),
]

for home, away in matchups:
    result = predict_game(home, away, games, team_stats, model)
    print(f"{result['away_team']} @ {result['home_team']}: {result['predicted_winner']}")
```

## Dependencies

All modules require:
```
pandas >= 1.3
numpy >= 1.20
scikit-learn >= 1.0
requests >= 2.25  (scraper.py only)
beautifulsoup4 >= 4.9  (scraper.py only)
tqdm >= 4.50  (scraper.py only)
matplotlib >= 3.3  (train_model.py only)
seaborn >= 0.11  (train_model.py only)
```

Install via:
```bash
pip install -r requirements.txt
# OR
poetry install
```

## File Dependencies

| Module | Imports | External Files |
|--------|---------|-----------------|
| scraper.py | requests, pandas, tqdm, bs4 | None |
| eda.py | pandas, numpy, matplotlib, seaborn | data/raw/games_*.csv |
| train_model.py | sklearn, pandas, numpy, matplotlib | data/raw/games_*.csv |
| predict.py | pickle, pandas | models/gb_model.pkl, data/raw/games_*.csv |

## Error Handling & Validation

### scraper.py
- ✅ Validates team ID mapping
- ✅ Filters for "Final" status only
- ✅ Handles API timeouts gracefully
- ✅ Progress bars show data volume

### eda.py
- ✅ Handles missing dates
- ✅ Validates score data
- ✅ Reports null values

### train_model.py
- ✅ Checks for sufficient history
- ✅ Validates feature engineering
- ✅ Reports train/test split sizes
- ✅ Cross-validation on training data

### predict.py
- ✅ Team name validation
- ✅ Feature name mapping
- ✅ Handles edge cases (new teams, etc.)

## Performance Notes

| Module | Execution Time | Resource Use |
|--------|----------------|--------------|
| scraper.py (5 seasons) | ~3 minutes | Low memory, API calls |
| eda.py | ~10 seconds | Low memory |
| train_model.py | ~2-3 minutes | Medium memory, CPU |
| predict.py | <1 second (per game) | Low memory |

## Testing & Validation

Each module can be tested independently:

```bash
# Test data loading
python -c "from src.scraper import scrape_mlb_statsapi_season; print('✓ Scraper OK')"

# Test analysis
python -m src.eda | head -20

# Test training
python -m src.train_model

# Test predictions
python -m src.predict | head -30
```

## Future Enhancements

- [ ] Real-time prediction API (Flask/FastAPI)
- [ ] Weather data integration
- [ ] Pitcher/lineup data
- [ ] Injury status tracking
- [ ] Betting odds integration
- [ ] Historical prediction accuracy tracking
- [ ] Model retraining automation

---

**Status**: ✅ Production-ready  
**Last Updated**: February 4, 2026  
**Total Lines of Code**: ~1,200 lines  
**Test Coverage**: All modules functional and tested
