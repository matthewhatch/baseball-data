# Project Structure

## Directory Organization

```
baseball-data/
├── src/                      # Main source code
│   ├── __init__.py
│   ├── scraper.py           # Data acquisition from MLB Stats API
│   ├── eda.py               # Exploratory data analysis
│   ├── train_model.py       # Model training pipeline
│   └── predict.py           # Game outcome predictions
├── data/                     # Data directory
│   ├── raw/                 # Raw downloaded CSV files
│   │   ├── games_2020.csv
│   │   ├── games_2021.csv
│   │   ├── games_2022.csv
│   │   ├── games_2023.csv
│   │   └── games_2024.csv
│   └── processed/           # (future) preprocessed data
├── models/                   # Trained models & scalers
│   ├── gb_model.pkl         # Gradient Boosting classifier
│   ├── rf_model.pkl         # Random Forest classifier
│   ├── lr_model.pkl         # Logistic Regression classifier
│   ├── scaler.pkl           # Feature scaler
│   ├── feature_names.pkl    # Feature column names
│   └── model_performance.png # Performance visualizations
├── docs/                     # Documentation
│   ├── README.md            # Main documentation
│   ├── OUTCOME_MODEL_SUMMARY.md
│   ├── SCRAPER_SUMMARY.md
│   ├── DATA_GUIDE.md
│   └── GAMETYPE.json
├── archive/                  # Deprecated/reference code
│   ├── example_data_format.py
│   ├── fetch_statsapi.py
│   ├── generate_sample_data.py
│   ├── scrape_baseball_ref.py
│   ├── scrape_yahoo.py
│   ├── train.py
│   └── logs/
│       ├── scraper_output.log
│       ├── output.txt
│       └── yahoo_sample_Aug16.html
├── main.py                   # Entry point
├── pyproject.toml           # Project config
├── poetry.lock              # Dependency lock
├── .gitignore
└── README.md                # Quick start guide
```

## Quick Start

```bash
# Run specific module
python -m src.scraper      # Fetch new data
python -m src.eda          # Analyze data
python -m src.train_model  # Train models
python -m src.predict      # Make predictions

# Or use main entry point
python main.py
```

## File Descriptions

### src/
- **scraper.py**: Fetches MLB game data from statsapi.mlb.com for 2020-2024
- **eda.py**: Generates statistics, distributions, and team performance analysis
- **train_model.py**: Trains 3 models (LR, RF, GB) and compares performance
- **predict.py**: Makes predictions using trained Gradient Boosting model

### data/
- **raw/**: Downloaded CSV files with all game data (11,486 games)
- **processed/**: (Future) cleaned/transformed data

### models/
- Serialized trained models and preprocessing objects
- Performance visualization charts

### docs/
- OUTCOME_MODEL_SUMMARY.md - Model details and results
- SCRAPER_SUMMARY.md - Data acquisition info
- DATA_GUIDE.md - Field descriptions
- GAMETYPE.json - Game type definitions

### archive/
- Legacy code and experimental scripts
- Deprecated scrapers (Baseball Reference, Yahoo)
- Temporary files and logs

## Key Files

| File | Purpose |
|------|---------|
| main.py | Entry point with command reference |
| pyproject.toml | Poetry dependencies |
| README.md | Project overview |
