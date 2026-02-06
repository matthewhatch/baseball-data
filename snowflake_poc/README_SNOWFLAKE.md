# Snowflake MLOps POC

Proof of Concept for using Snowflake as MLOps platform for baseball predictions.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ LOCAL MACHINE                                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CSV Data              Trained Models                          │
│  (data/raw/)           (models/)                                │
│       │                    │                                    │
│       └────────┬──────────┘                                     │
│                │                                                │
│     ┌──────────▼───────────┐                                   │
│     │   snowflake_poc/     │                                   │
│     │  (This POC code)     │                                   │
│     └──────────┬───────────┘                                   │
│                │                                                │
└────────────────┼────────────────────────────────────────────────┘
                 │
                 │ (Upload via SQL/Stages)
                 │
┌────────────────▼────────────────────────────────────────────────┐
│ SNOWFLAKE CLOUD                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Database: BASEBALL_DB                                   │  │
│  │                                                          │  │
│  │  Tables:                                                │  │
│  │  • GAMES (11,486 rows)                                 │  │
│  │  • MODELS_METADATA (model registry)                    │  │
│  │  • PREDICTIONS_LOG (audit trail)                       │  │
│  │                                                          │  │
│  │  Stages:                                                │  │
│  │  • @MODELS_STAGE (model artifacts .pkl)               │  │
│  │  • @DATA_STAGE (backup data)                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Setup

### 1. Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Credentials

Create `.env` file in this directory:

```bash
cp .env.example .env
# Edit .env with your Snowflake credentials
```

Find your Account ID:
1. Login to Snowflake
2. Check URL: `https://<account_id>.snowflakecomputing.com`
3. Use only `<account_id>` (e.g., `xy12345.us-east-1`)

### 3. Create Snowflake Resources

If you don't have them:
1. Create Database: `CREATE DATABASE BASEBALL_DB;`
2. Create Warehouse: `CREATE WAREHOUSE COMPUTE_WH;`
3. Create Stages:
   ```sql
   CREATE STAGE @MODELS_STAGE DIRECTORY = (ENABLE = TRUE);
   CREATE STAGE @DATA_STAGE DIRECTORY = (ENABLE = TRUE);
   ```

### 4. Run Setup

```bash
python snowflake_setup.py
```

This will:
- Create schema and tables
- Load CSV data from `../data/raw/`
- Initialize model registry
- Verify data integrity

## Files

### `snowflake_config.py`
Connection configuration and constants
- Load credentials from `.env`
- Define table/stage names
- Setup instructions

### `snowflake_data_loader.py`
Load CSV data → Snowflake
```bash
python snowflake_data_loader.py
```
- Creates schema/tables
- Uploads CSV files (11,486 games)
- Verifies data integrity

### `snowflake_model_store.py`
Model registry and versioning
```bash
python snowflake_model_store.py
```
- Upload models to Snowflake Stages
- Register metadata (accuracy, features, version)
- List/download models
- Get best performing model

### `snowflake_predict.py`
Make predictions using Snowflake data
```bash
python snowflake_predict.py
```
- Query team stats from Snowflake
- Make predictions using loaded model
- Log predictions with confidence scores
- Calculate prediction accuracy

### `snowflake_setup.py`
Full automated setup
```bash
python snowflake_setup.py
```

## Usage

### Load Data

```python
from snowflake_data_loader import SnowflakeDataLoader

loader = SnowflakeDataLoader()
loader.connect()
loader.create_schema()
loader.create_tables()
loader.load_csv_files('../data/raw')
loader.verify_data()
loader.close()
```

### Manage Models

```python
from snowflake_model_store import SnowflakeModelStore

store = SnowflakeModelStore()
store.connect()

# Register a model
model_id = store.upload_model(
    model_path='../models/gb_model.pkl',
    model_name='gb_model',
    model_type='GradientBoosting',
    version='1.0.0',
    accuracy=0.5463,
    features=['home_historical_wr', 'away_historical_wr', ...],
    trained_records=8848
)

# List all models
store.list_models()

# Get best model
best = store.get_best_model()

store.close()
```

### Make Predictions

```python
from snowflake_predict import SnowflakePredictions

predictor = SnowflakePredictions('../models/gb_model.pkl')
predictor.connect()

# Predict game
result = predictor.predict_game('LAD', 'NYY')

# Log prediction
predictor.log_prediction(result, actual_winner='HOME')

# Check accuracy
predictor.get_prediction_accuracy()

predictor.close()
```

## MLOps Workflow

### 1. **Data Ingestion**
```
Local CSV → Snowflake GAMES table
Status: ✓ Complete (snowflake_data_loader.py)
```

### 2. **Model Training**
```
Local Training (src/train_model.py)
    ↓
Register in Snowflake (snowflake_model_store.py)
    ↓
Upload to @MODELS_STAGE
```

### 3. **Predictions**
```
Query Team Stats from GAMES table
    ↓
Load model from @MODELS_STAGE
    ↓
Make prediction
    ↓
Log to PREDICTIONS_LOG table
```

### 4. **Monitoring**
```
Query PREDICTIONS_LOG
    ↓
Calculate accuracy vs actual outcomes
    ↓
Track model performance over time
```

## Data Schema

### GAMES Table
```sql
CREATE TABLE GAMES (
    GAME_ID VARCHAR,
    DATE DATE,
    HOME_TEAM VARCHAR,
    AWAY_TEAM VARCHAR,
    HOME_SCORE INTEGER,
    AWAY_SCORE INTEGER,
    GAME_TYPE VARCHAR,
    VENUE VARCHAR,
    STATUS VARCHAR,
    DOUBLE_HEADER BOOLEAN
);
-- 11,486 rows (2020-2024)
```

### MODELS_METADATA Table
```sql
CREATE TABLE MODELS_METADATA (
    MODEL_ID VARCHAR PRIMARY KEY,
    MODEL_NAME VARCHAR,
    MODEL_TYPE VARCHAR,
    VERSION VARCHAR,
    ACCURACY DECIMAL(5,4),
    CREATED_AT TIMESTAMP_NTZ,
    TRAINED_ON_RECORDS INTEGER,
    FEATURES ARRAY,
    LOCATION VARCHAR  -- Stage path
);
```

### PREDICTIONS_LOG Table
```sql
CREATE TABLE PREDICTIONS_LOG (
    PREDICTION_ID VARCHAR PRIMARY KEY,
    HOME_TEAM VARCHAR,
    AWAY_TEAM VARCHAR,
    PREDICTED_WINNER VARCHAR,
    CONFIDENCE DECIMAL(5,4),
    HOME_WIN_PROB DECIMAL(5,4),
    AWAY_WIN_PROB DECIMAL(5,4),
    MODEL_ID VARCHAR,
    PREDICTED_AT TIMESTAMP_NTZ,
    ACTUAL_WINNER VARCHAR,  -- NULL until game is played
    GAME_DATE DATE
);
```

## Next Steps

1. **Testing**
   - Verify all connections work
   - Test data loading
   - Test predictions

2. **Enhancements**
   - Add Snowflake Tasks for scheduled retraining
   - Create Snowflake Notebooks for analysis
   - Set up Snowflake alerts for low accuracy
   - Add feature store tables

3. **Production**
   - Deploy to Snowflake's native Python (UDFs)
   - Use Snowpipe for continuous data ingestion
   - Implement CI/CD for model versioning
   - Set up Snowflake dashboards

4. **Integration**
   - Connect to Snowflake Cortex for LLM features
   - Use dbt for data transformations
   - Stream predictions to downstream systems

## Troubleshooting

### Connection Errors
```
✗ Connection failed: ...
→ Check .env file exists and is correct
→ Verify Snowflake account ID format
→ Ensure user has ACCOUNTADMIN role
```

### Data Loading Issues
```
✗ Data directory not found
→ Run from snowflake_poc/ directory
→ CSV files should be in ../data/raw/
```

### Stage Upload Errors
```
✗ Stage does not exist
→ Create stages manually in Snowflake:
   CREATE STAGE @MODELS_STAGE;
   CREATE STAGE @DATA_STAGE;
```

## Cost Considerations

**Snowflake Pricing (on-demand):**
- Compute: ~$4/credit/hour
- Storage: ~$40/TB/month
- This POC uses minimal resources (~0.1 credits for setup)

**Cost Optimization:**
- Use Snowflake Free Trial (10 credits free)
- Use X-Large warehouse for bulk operations only
- Store models in stages (cheap storage)
- Clean up unused tables

## References

- [Snowflake Python Connector](https://docs.snowflake.com/en/developer-guide/python-connector)
- [Snowflake Stages](https://docs.snowflake.com/en/user-guide/data-load-local-file-system-create-stage)
- [Snowflake Model Registry](https://docs.snowflake.com/en/user-guide/ml-model-registry)
- [MLOps Best Practices](https://docs.snowflake.com/en/guides/mloops)

## License

Same as main project (baseball-data)
