"""
DuckDB Data Loader
Load baseball games data from CSV files into DuckDB
"""

import pandas as pd
import duckdb
from pathlib import Path
from duckdb_config import DB_PATH, GAMES_TABLE


class DuckDBDataLoader:
    """Load CSV data into DuckDB"""
    
    def __init__(self):
        self.db_path = DB_PATH
        self.conn = None
    
    def connect(self):
        """Connect to DuckDB"""
        self.conn = duckdb.connect(str(self.db_path))
        print(f"✓ Connected to DuckDB: {self.db_path}")
    
    def create_tables(self):
        """Create tables"""
        try:
            # Games table
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {GAMES_TABLE} (
                    game_id VARCHAR,
                    date DATE,
                    home_team VARCHAR,
                    away_team VARCHAR,
                    home_score INTEGER,
                    away_score INTEGER,
                    game_type VARCHAR,
                    venue VARCHAR,
                    status VARCHAR,
                    double_header BOOLEAN,
                    PRIMARY KEY (game_id)
                )
            """)
            print(f"✓ Table {GAMES_TABLE} created")
            
            # Models metadata table
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS models_metadata (
                    model_id VARCHAR PRIMARY KEY,
                    model_name VARCHAR,
                    model_type VARCHAR,
                    version VARCHAR,
                    accuracy DECIMAL(5,4),
                    created_at TIMESTAMP,
                    trained_on_records INTEGER,
                    features VARCHAR,  -- JSON array as string
                    file_path VARCHAR
                )
            """)
            print("✓ Table models_metadata created")
            
            # Predictions log table
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS predictions_log (
                    prediction_id VARCHAR PRIMARY KEY,
                    home_team VARCHAR,
                    away_team VARCHAR,
                    predicted_winner VARCHAR,
                    confidence DECIMAL(5,4),
                    home_win_prob DECIMAL(5,4),
                    away_win_prob DECIMAL(5,4),
                    model_id VARCHAR,
                    predicted_at TIMESTAMP,
                    actual_winner VARCHAR,
                    game_date DATE
                )
            """)
            print("✓ Table predictions_log created")
            
        except Exception as e:
            print(f"✗ Table creation failed: {e}")
            raise
    
    def load_csv_files(self, data_dir='../data/raw'):
        """Load all CSV files into DuckDB"""
        data_path = Path(data_dir)
        
        if not data_path.exists():
            print(f"✗ Data directory not found: {data_path.absolute()}")
            return False
        
        csv_files = list(data_path.glob('games_*.csv'))
        print(f"\nFound {len(csv_files)} CSV files")
        
        if not csv_files:
            print("✗ No CSV files found")
            return False
        
        try:
            total_rows = 0
            for csv_file in sorted(csv_files):
                print(f"\nLoading {csv_file.name}...")
                
                # Read and normalize
                df = pd.read_csv(csv_file)
                df.columns = df.columns.str.lower()
                
                # Insert into DuckDB
                self.conn.from_df(df).insert_into(GAMES_TABLE)
                
                rows = len(df)
                total_rows += rows
                print(f"  ✓ Loaded {rows} rows")
            
            print(f"\n✓ Total rows loaded: {total_rows:,}")
            return True
            
        except Exception as e:
            print(f"✗ Load failed: {e}")
            return False
    
    def verify_data(self):
        """Verify data was loaded correctly"""
        try:
            result = self.conn.execute(
                f"SELECT COUNT(*) as row_count FROM {GAMES_TABLE}"
            ).fetchall()
            
            count = result[0][0]
            print(f"\n✓ Data verification: {count:,} rows in DuckDB")
            
            # Sample query
            print("\nSample data:")
            samples = self.conn.execute(f"""
                SELECT home_team, away_team, home_score, away_score, date 
                FROM {GAMES_TABLE}
                LIMIT 5
            """).fetchall()
            
            for row in samples:
                print(f"  {row[0]} {row[4]:>10} {row[2]:>2} - {row[1]} {row[3]:>2}")
            
            return True
            
        except Exception as e:
            print(f"✗ Verification failed: {e}")
            return False
    
    def get_stats(self):
        """Get data statistics"""
        try:
            stats = self.conn.execute(f"""
                SELECT
                    COUNT(*) as total_games,
                    COUNT(DISTINCT home_team) as unique_teams,
                    COUNT(DISTINCT DATE_TRUNC('year', date)) as years,
                    AVG(home_score) as avg_home_score,
                    AVG(away_score) as avg_away_score,
                    SUM(CASE WHEN home_score > away_score THEN 1 ELSE 0 END)::FLOAT / 
                    COUNT(*) as home_win_rate
                FROM {GAMES_TABLE}
            """).fetchall()
            
            total, teams, years, home_avg, away_avg, home_wr = stats[0]
            
            print("\n📊 Data Statistics:")
            print(f"  Total games: {total:,}")
            print(f"  Unique teams: {teams}")
            print(f"  Years covered: {int(years)}")
            print(f"  Home avg score: {home_avg:.2f}")
            print(f"  Away avg score: {away_avg:.2f}")
            print(f"  Home win rate: {home_wr:.2%}")
            
        except Exception as e:
            print(f"✗ Stats failed: {e}")
    
    def close(self):
        """Close connection"""
        if self.conn:
            self.conn.close()
            print("\n✓ Connection closed")


if __name__ == '__main__':
    loader = DuckDBDataLoader()
    
    try:
        loader.connect()
        loader.create_tables()
        loader.load_csv_files()
        loader.verify_data()
        loader.get_stats()
    finally:
        loader.close()
