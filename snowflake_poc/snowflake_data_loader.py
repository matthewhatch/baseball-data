"""
Data Loading: CSV -> Snowflake
Load baseball games data from local CSV files into Snowflake
"""

import pandas as pd
from pathlib import Path
from snowflake.connector import connect
from sqlalchemy import create_engine
from snowflake_config import SnowflakeConfig


class SnowflakeDataLoader:
    """Load CSV data into Snowflake"""
    
    def __init__(self):
        self.config = SnowflakeConfig()
        self.connection = None
        self.engine = None
    
    def connect(self):
        """Establish Snowflake connection"""
        try:
            self.connection = connect(**self.config.get_connection_params())
            print("✓ Connected to Snowflake")
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            raise
    
    def create_connection_string(self):
        """Create SQLAlchemy connection string"""
        params = self.config.get_connection_params()
        connection_string = (
            f"snowflake://{params['user']}:{params['password']}@"
            f"{params['account']}/{params['database']}/{params['schema']}?"
            f"warehouse={params['warehouse']}&role={params['role']}"
        )
        return connection_string
    
    def create_schema(self):
        """Create database and schema if they don't exist"""
        cursor = self.connection.cursor()
        try:
            # Create database
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.config.DATABASE}")
            print(f"✓ Database {self.config.DATABASE} ready")
            
            # Create schema
            cursor.execute(
                f"CREATE SCHEMA IF NOT EXISTS {self.config.DATABASE}.{self.config.SCHEMA}"
            )
            print(f"✓ Schema {self.config.SCHEMA} ready")
            
            # Create stages for models and data
            cursor.execute(
                f"CREATE STAGE IF NOT EXISTS {self.config.MODELS_STAGE} "
                f"DIRECTORY = (ENABLE = TRUE)"
            )
            cursor.execute(
                f"CREATE STAGE IF NOT EXISTS {self.config.DATA_STAGE} "
                f"DIRECTORY = (ENABLE = TRUE)"
            )
            print("✓ Stages created")
            
        finally:
            cursor.close()
    
    def create_tables(self):
        """Create tables for games data"""
        cursor = self.connection.cursor()
        try:
            # Games table
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.config.GAMES_TABLE} (
                    GAME_ID VARCHAR,
                    DATE DATE,
                    HOME_TEAM VARCHAR,
                    AWAY_TEAM VARCHAR,
                    HOME_SCORE INTEGER,
                    AWAY_SCORE INTEGER,
                    GAME_TYPE VARCHAR,
                    VENUE VARCHAR,
                    STATUS VARCHAR,
                    DOUBLE_HEADER BOOLEAN,
                    PRIMARY KEY (GAME_ID)
                )
            """)
            print(f"✓ Table {self.config.GAMES_TABLE} created")
            
            # Models metadata table
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.config.MODELS_METADATA_TABLE} (
                    MODEL_ID VARCHAR PRIMARY KEY,
                    MODEL_NAME VARCHAR,
                    MODEL_TYPE VARCHAR,
                    VERSION VARCHAR,
                    ACCURACY DECIMAL(5,4),
                    CREATED_AT TIMESTAMP_NTZ,
                    TRAINED_ON_RECORDS INTEGER,
                    FEATURES ARRAY,
                    LOCATION VARCHAR
                )
            """)
            print(f"✓ Table {self.config.MODELS_METADATA_TABLE} created")
            
            # Predictions log table
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.config.PREDICTIONS_LOG_TABLE} (
                    PREDICTION_ID VARCHAR PRIMARY KEY,
                    HOME_TEAM VARCHAR,
                    AWAY_TEAM VARCHAR,
                    PREDICTED_WINNER VARCHAR,
                    CONFIDENCE DECIMAL(5,4),
                    HOME_WIN_PROB DECIMAL(5,4),
                    AWAY_WIN_PROB DECIMAL(5,4),
                    MODEL_ID VARCHAR,
                    PREDICTED_AT TIMESTAMP_NTZ,
                    ACTUAL_WINNER VARCHAR,
                    GAME_DATE DATE
                )
            """)
            print(f"✓ Table {self.config.PREDICTIONS_LOG_TABLE} created")
            
        finally:
            cursor.close()
    
    def load_csv_files(self, data_dir='../data/raw'):
        """Load all CSV files into Snowflake"""
        data_path = Path(data_dir)
        
        if not data_path.exists():
            print(f"✗ Data directory not found: {data_path}")
            return
        
        csv_files = list(data_path.glob('games_*.csv'))
        print(f"\nFound {len(csv_files)} CSV files")
        
        try:
            engine = create_engine(self.create_connection_string())
            
            total_rows = 0
            for csv_file in sorted(csv_files):
                print(f"\nLoading {csv_file.name}...")
                
                # Read CSV
                df = pd.read_csv(csv_file)
                print(f"  - Rows: {len(df)}")
                
                # Normalize column names
                df.columns = df.columns.str.upper()
                
                # Load to Snowflake
                df.to_sql(
                    self.config.GAMES_TABLE.lower(),
                    engine,
                    if_exists='append',
                    index=False,
                    method='multi',
                    chunksize=1000
                )
                
                total_rows += len(df)
                print(f"  ✓ Loaded to Snowflake")
            
            print(f"\n✓ Total rows loaded: {total_rows}")
            
        except Exception as e:
            print(f"✗ Load failed: {e}")
            raise
        finally:
            engine.dispose()
    
    def verify_data(self):
        """Verify data was loaded correctly"""
        cursor = self.connection.cursor()
        try:
            cursor.execute(f"SELECT COUNT(*) as row_count FROM {self.config.GAMES_TABLE}")
            result = cursor.fetchone()
            count = result[0]
            print(f"\n✓ Data verification: {count:,} rows in Snowflake")
            
            # Sample query
            cursor.execute(f"""
                SELECT HOME_TEAM, AWAY_TEAM, HOME_SCORE, AWAY_SCORE, DATE 
                FROM {self.config.GAMES_TABLE}
                LIMIT 5
            """)
            print("\nSample data:")
            for row in cursor.fetchall():
                print(f"  {row}")
            
        finally:
            cursor.close()
    
    def close(self):
        """Close connection"""
        if self.connection:
            self.connection.close()
            print("\n✓ Connection closed")


if __name__ == '__main__':
    # Usage example
    loader = SnowflakeDataLoader()
    
    try:
        loader.connect()
        loader.create_schema()
        loader.create_tables()
        loader.load_csv_files()
        loader.verify_data()
    finally:
        loader.close()
