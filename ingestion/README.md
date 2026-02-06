# Ingestion Module

Data pipeline for loading, validating, and preparing baseball game data for model training.

## Overview

The `ingestion` module provides utilities for:
- **Loading** historical game data from CSV files and external sources
- **Validating** data quality and format consistency
- **Feature Engineering** creating predictive features from raw game statistics
- **Normalization** standardizing data across different sources

## Directory Structure

```
ingestion/
├── data_loader.py         # Load game data from files/APIs
├── features.py            # Feature engineering utilities
├── ids/                   # (Future) Team/player ID mapping
├── sources/               # (Future) Data source connectors
├── normalize/             # (Future) Data normalization
└── validators/            # (Future) Data validation rules
```

## Modules

### `data_loader.py`

Provides the `BaseballDataLoader` class for loading and caching game data.

**Key Class: `BaseballDataLoader`**

```python
loader = BaseballDataLoader(cache_dir="./data/raw")
```

**Methods:**

- `load_games_csv(filepath)` - Load games from CSV file
  - Converts date to datetime
  - Sorts by date
  - Returns DataFrame with columns: date, home_team, away_team, home_score, away_score, etc.

- `load_games_from_cache(season)` - Load cached data for a season
  - Returns DataFrame if cache exists, else None
  - Cache files: `games_2020.csv`, `games_2021.csv`, etc.

- `save_to_cache(df, season)` - Cache games to avoid re-fetching
  - Saves to `data/raw/games_{season}.csv`

**Function: `load_season_data(seasons, data_dir)`**

Convenience function to load and combine multiple seasons:

```python
from ingestion.data_loader import load_season_data

# Load all 5 seasons
games = load_season_data(
    seasons=[2020, 2021, 2022, 2023, 2024],
    data_dir="./data/raw"
)
# Returns combined DataFrame with 11,486 games
```

### `features.py`

Feature engineering for game outcome prediction.

**Key Class: `BaseballFeatureEngineer`**

**Methods:**

- `create_team_stats(games, window=10)` - Calculate rolling team statistics
  - Separates home and away games for each team
  - Calculates wins, runs scored, runs allowed
  - Returns dict: `{team_name: DataFrame}`

- `add_team_features(games, team_stats, window=10)` - Add rolling average features
  - Home team win % (last N games)
  - Away team win % (last N games)
  - Home team scoring average
  - Away team scoring average
  - Home team runs allowed average
  - Away team runs allowed average
  
**Example Usage:**

```python
from ingestion.features import BaseballFeatureEngineer

# Create engineer
engineer = BaseballFeatureEngineer()

# Get team statistics
team_stats = engineer.create_team_stats(games, window=10)

# Add features to games
games_with_features = engineer.add_team_features(
    games, 
    team_stats, 
    window=10
)
```

**Generated Features:**

| Feature | Description | Type |
|---------|-------------|------|
| `home_team_win_pct_10` | Home team win % last 10 games | float |
| `away_team_win_pct_10` | Away team win % last 10 games | float |
| `home_team_rs_avg_10` | Home team avg runs scored (10 games) | float |
| `away_team_rs_avg_10` | Away team avg runs scored (10 games) | float |
| `home_team_ra_avg_10` | Home team avg runs allowed (10 games) | float |
| `away_team_ra_avg_10` | Away team avg runs allowed (10 games) | float |

## Data Flow

```
Raw Data (CSV)
    ↓
data_loader.py (Load & Cache)
    ↓
Game DataFrame
    ↓
features.py (Engineer Features)
    ↓
Features DataFrame (Ready for Model)
```

## Future Subdirectories

### `ids/` - Team & Player ID Mapping
- Map team abbreviations to API IDs
- Store player ID mapping
- Handle legacy team names and relocations

### `sources/` - Data Source Connectors
- statsapi.mlb.com connector
- Baseball Reference scraper
- Kaggle dataset importer
- Custom API endpoints

### `normalize/` - Data Normalization
- Team name standardization
- Date format consistency
- Score validation
- Missing value handling

### `validators/` - Data Quality Checks
- Required field validation
- Score range validation
- Team name verification
- Duplicate detection
- Data type validation

## Usage Examples

### Basic Data Loading

```python
from ingestion.data_loader import BaseballDataLoader

# Initialize loader
loader = BaseballDataLoader(cache_dir="./data/raw")

# Load CSV file
games = loader.load_games_csv("./data/raw/games_2024.csv")
print(f"Loaded {len(games)} games")

# Save to cache
loader.save_to_cache(games, season=2024)
```

### Feature Engineering Pipeline

```python
from ingestion.data_loader import load_season_data
from ingestion.features import BaseballFeatureEngineer

# Load data
games = load_season_data([2020, 2021, 2022, 2023, 2024])

# Engineer features
engineer = BaseballFeatureEngineer()
team_stats = engineer.create_team_stats(games, window=10)
games_featured = engineer.add_team_features(games, team_stats)

# Filter to games with complete history
games_ready = games_featured[games_featured['home_team_win_pct_10'].notna()]
print(f"Games with features: {len(games_ready)}")
```

### Full Pipeline (Current Implementation)

The main model training pipeline in `src/train_model.py` already handles:
1. Loading game data
2. Creating features (historical win rates, scoring averages, etc.)
3. Standardizing features
4. Training models
5. Evaluating performance

## Notes

- **Current State**: `data_loader.py` and `features.py` are functional reference implementations
- **Active Pipeline**: `src/train_model.py` uses an optimized feature engineering pipeline
- **Future Use**: Additional subdirectories reserved for expansion
- **Data Quality**: All raw CSV files validated to have required columns

## Integration Points

| Component | Location | Purpose |
|-----------|----------|---------|
| Data Loading | `src/scraper.py` | Fetch from statsapi.mlb.com |
| Feature Creation | `src/train_model.py` | Optimized for model training |
| Validation | Built into `src/train_model.py` | Check data before training |
| Preprocessing | `src/train_model.py` | Scaling and normalization |

---

**Status**: ✅ Core modules functional, expansion-ready  
**Last Updated**: February 4, 2026
