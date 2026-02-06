"""
DuckDB Configuration
Local MLOps database for baseball predictions
"""

import os
from pathlib import Path

# Database location
DB_PATH = Path(__file__).parent / 'baseball.duckdb'

# Tables
GAMES_TABLE = 'games'
MODELS_METADATA_TABLE = 'models_metadata'
PREDICTIONS_LOG_TABLE = 'predictions_log'

# Create directory if needed
DB_PATH.parent.mkdir(exist_ok=True)

print(f"DuckDB database: {DB_PATH}")
