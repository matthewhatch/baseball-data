# DuckDB MLOps POC - README

**Local, free, fast MLOps platform using DuckDB.**

## What is DuckDB?

- **In-process SQL database** (no server, no installation)
- **Instant setup** (pip install duckdb)
- **SQL + Python integration**
- Perfect for development, testing, and learning
- Free and open-source

## Architecture

```
┌─────────────────────────────────────┐
│ LOCAL MACHINE                       │
├─────────────────────────────────────┤
│                                     │
│  CSV Data         Trained Models    │
│  (../data/raw)    (../models/)      │
│       │                 │           │
│       └────────┬────────┘           │
│                │                    │
│     ┌──────────▼──────────┐        │
│     │   duckdb_poc/       │        │
│     │  (This POC code)    │        │
│     └──────────┬──────────┘        │
│                │                    │
│     ┌──────────▼──────────┐        │
│     │  baseball.duckdb    │        │
│     │  (Local database)   │        │
│     └─────────────────────┘        │
│                                     │
└─────────────────────────────────────┘
```

## Quick Start

### 1. Install

```bash
cd duckdb_poc
pip install -r requirements.txt
```

### 2. Setup

```bash
python duckdb_setup.py
```

This will:
- Create DuckDB database
- Load 11,486 games from CSV
- Initialize model registry
- Verify data

### 3. Use

```bash
# Explore data
python duckdb_data_loader.py

# Manage models
python duckdb_model_store.py

# Make predictions
python duckdb_predict.py
```

## Files

### `duckdb_config.py`
Configuration and paths
- Database location: `baseball.duckdb`
- Table names
- Models directory

### `duckdb_data_loader.py`
Load and explore data
```bash
python duckdb_data_loader.py
```

**Features:**
- Load CSV → DuckDB (11,486 games)
- Create tables
- Data verification
- Statistics summary

**Usage:**
```python
from duckdb_data_loader import DuckDBDataLoader

loader = DuckDBDataLoader()
loader.connect()
loader.create_tables()
loader.load_csv_files('../data/raw')
loader.verify_data()
loader.get_stats()
loader.close()
```

### `duckdb_model_store.py`
Model registry and versioning
```bash
python duckdb_model_store.py
```

**Features:**
- Register trained models
- Version tracking
- Metadata storage
- Load/retrieve models
- Find best model

**Usage:**
```python
from duckdb_model_store import DuckDBModelStore

store = DuckDBModelStore()
store.connect()

# Register model
model_id = store.upload_model(
    model_path='../models/gb_model.pkl',
    model_name='gb_model',
    model_type='GradientBoosting',
    version='1.0.0',
    accuracy=0.5463,
    features=['home_historical_wr', ...],
    trained_records=8848
)

# List models
store.list_models()

# Get best
best, path = store.get_best_model()

# Load model
model = store.load_model(model_id)

store.close()
```

### `duckdb_predict.py`
Make predictions
```bash
python duckdb_predict.py
```

**Features:**
- Query team stats from DuckDB
- Make game predictions
- Log predictions with confidence
- Track accuracy
- Recent predictions

**Usage:**
```python
from duckdb_predict import DuckDBPredictions

predictor = DuckDBPredictions('../models/gb_model.pkl')
predictor.connect()

# Predict
result = predictor.predict_game('LAD', 'NYY')

# Log
predictor.log_prediction(result, actual_winner='HOME')

# Check accuracy
predictor.get_prediction_accuracy()

# Recent predictions
predictor.get_recent_predictions()

predictor.close()
```

### `duckdb_setup.py`
Automated setup
```bash
python duckdb_setup.py
```

## Data Schema

### GAMES
```sql
game_id VARCHAR PRIMARY KEY
date DATE
home_team VARCHAR
away_team VARCHAR
home_score INTEGER
away_score INTEGER
game_type VARCHAR
venue VARCHAR
status VARCHAR
double_header BOOLEAN
-- 11,486 rows
```

### MODELS_METADATA
```sql
model_id VARCHAR PRIMARY KEY
model_name VARCHAR
model_type VARCHAR
version VARCHAR
accuracy DECIMAL(5,4)
created_at TIMESTAMP
trained_on_records INTEGER
features VARCHAR (JSON)
file_path VARCHAR
```

### PREDICTIONS_LOG
```sql
prediction_id VARCHAR PRIMARY KEY
home_team VARCHAR
away_team VARCHAR
predicted_winner VARCHAR
confidence DECIMAL(5,4)
home_win_prob DECIMAL(5,4)
away_win_prob DECIMAL(5,4)
model_id VARCHAR
predicted_at TIMESTAMP
actual_winner VARCHAR  -- NULL until game played
game_date DATE
```

## Example Workflow

### Load Data
```bash
python duckdb_data_loader.py
```

Output:
```
✓ Connected to DuckDB
✓ Table games created
✓ Table models_metadata created
✓ Table predictions_log created

Found 5 CSV files

Loading games_2020.csv...
  ✓ Loaded 984 rows
Loading games_2021.csv...
  ✓ Loaded 2,621 rows
...

✓ Total rows loaded: 11,486

✓ Data verification: 11,486 rows in DuckDB

Sample data:
  LAD 2020-07-23  4 - COL 2
  BOS 2020-07-23  7 - BAL 4
  ...

📊 Data Statistics:
  Total games: 11,486
  Unique teams: 30
  Years covered: 5
  Home avg score: 4.51
  Away avg score: 4.47
  Home win rate: 52.89%
```

### Register Models
```python
from duckdb_model_store import DuckDBModelStore

store = DuckDBModelStore()
store.connect()

store.upload_model(
    model_path='../models/gb_model.pkl',
    model_name='gb_model',
    model_type='GradientBoosting',
    version='1.0.0',
    accuracy=0.5463,
    features=['home_historical_wr', 'away_historical_wr', ...],
    trained_records=8848
)

store.list_models()
store.close()
```

Output:
```
📦 Registered Models (1 total):
────────────────────────────────────────────────────────────────────────────────
Model ID                                 Type                 Accuracy  Records
────────────────────────────────────────────────────────────────────────────────
gb_model_v1.0.0_a7f3c2b1                GradientBoosting       54.63%   8,848
```

### Make Predictions
```python
from duckdb_predict import DuckDBPredictions

predictor = DuckDBPredictions('../models/gb_model.pkl')
predictor.connect()

result = predictor.predict_game('LAD', 'NYY')
predictor.log_prediction(result, actual_winner='HOME')
predictor.get_prediction_accuracy()

predictor.close()
```

Output:
```
✓ Model loaded: ../models/gb_model.pkl
✓ Connected to DuckDB

🎯 Prediction Result:
  NYY @ LAD
  Winner: HOME
  Confidence: 64.3%
  Home Win Prob: 64.30%
  Away Win Prob: 35.70%

✓ Prediction logged: f7e3c9a1-...

📊 Prediction Accuracy:
  Total predictions: 1
  Correct: 1
  Accuracy: 100.00%
```

## Common Queries

### Query team stats
```sql
SELECT 
  home_team as team,
  COUNT(*) as games,
  SUM(CASE WHEN home_score > away_score THEN 1 ELSE 0 END) as wins,
  AVG(home_score) as avg_score
FROM games
GROUP BY home_team
ORDER BY wins DESC;
```

### Check predictions
```sql
SELECT 
  home_team, away_team, predicted_winner, confidence,
  actual_winner, CASE WHEN predicted_winner = actual_winner THEN '✓' ELSE '✗' END
FROM predictions_log
WHERE actual_winner IS NOT NULL
ORDER BY predicted_at DESC
LIMIT 10;
```

### Model comparison
```sql
SELECT 
  model_id, model_type, accuracy,
  created_at, trained_on_records
FROM models_metadata
ORDER BY accuracy DESC;
```

## Advantages vs Snowflake

| Feature | DuckDB | Snowflake |
|---------|--------|-----------|
| Cost | Free | ~$4/credit |
| Setup | 1 minute | Account + config |
| Speed | Instant | Startup time |
| Learning | Perfect | Production |
| Offline | ✓ Works | ✗ Cloud only |
| Scale | GB-TB | TB-PB |

## Next Steps

1. **Learn SQL** - Write queries on baseball data
2. **Explore** - Analyze team performance patterns
3. **Integrate** - Add to your applications
4. **Extend** - Add more features/tables
5. **Migrate** - Move to Snowflake when ready

## Tips

- **Speed**: DuckDB runs in-process, very fast for local data
- **Development**: Perfect for iterating without cloud costs
- **Testing**: Test MLOps workflows before production
- **Sharing**: Export data/models to CSV/Parquet
- **Scaling**: Graduate to Snowflake when data grows

## Troubleshooting

**"Module not found: duckdb"**
```bash
pip install -r requirements.txt
```

**"CSV files not found"**
- Run from `duckdb_poc/` directory
- CSV files should be in `../data/raw/`

**"Database locked"**
- Only one process can write at a time
- Close other Python processes
- DuckDB creates `.tmp` files during writes

**"Memory error with large data"**
- DuckDB is in-process, uses system RAM
- 11,486 games = ~50MB, should fit easily
- For larger data, use Snowflake

## References

- [DuckDB Documentation](https://duckdb.org/docs/)
- [SQL Tutorial](https://duckdb.org/docs/sql/introduction)
- [Python API](https://duckdb.org/docs/api/python/overview)
- [Parquet Support](https://duckdb.org/docs/data/parquet)

---

**Database Location**: `duckdb_poc/baseball.duckdb` (~50MB)  
**Models Directory**: `duckdb_poc/models/` (stores .pkl files)  
**Created**: February 6, 2026
