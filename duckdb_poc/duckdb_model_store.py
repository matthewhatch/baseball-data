"""
DuckDB Model Store
Store and manage ML models locally in DuckDB
"""

import pickle
import json
from datetime import datetime
from pathlib import Path
import duckdb
import hashlib
from duckdb_config import DB_PATH, MODELS_METADATA_TABLE


class DuckDBModelStore:
    """Store and manage ML models in DuckDB"""
    
    def __init__(self, models_dir='./models'):
        self.db_path = DB_PATH
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.conn = None
    
    def connect(self):
        """Connect to DuckDB"""
        self.conn = duckdb.connect(str(self.db_path))
        print(f"✓ Connected to DuckDB")
    
    def upload_model(self, model_path, model_name, model_type, 
                    version, accuracy, features, trained_records):
        """
        Register a trained model
        
        Args:
            model_path: Local path to .pkl file
            model_name: Name (e.g., 'gb_model')
            model_type: Type (e.g., 'GradientBoosting')
            version: Version string (e.g., '1.0.0')
            accuracy: Model accuracy (0-1)
            features: List of feature names
            trained_records: Records used for training
        """
        
        if not Path(model_path).exists():
            print(f"✗ Model file not found: {model_path}")
            return False
        
        try:
            # Copy model to models directory
            src = Path(model_path)
            file_hash = hashlib.md5(open(src, 'rb').read()).hexdigest()[:8]
            model_id = f"{model_name}_v{version}_{file_hash}"
            
            dest = self.models_dir / f"{model_id}.pkl"
            with open(src, 'rb') as f_in:
                with open(dest, 'wb') as f_out:
                    f_out.write(f_in.read())
            
            print(f"✓ Model stored: {dest}")
            
            # Register metadata
            self.conn.execute(f"""
                INSERT INTO {MODELS_METADATA_TABLE}
                (model_id, model_name, model_type, version, accuracy, 
                 created_at, trained_on_records, features, file_path)
                VALUES 
                ('{model_id}', '{model_name}', '{model_type}', '{version}', 
                 {accuracy}, '{datetime.utcnow().isoformat()}', {trained_records},
                 '{json.dumps(features)}', '{str(dest)}')
            """)
            
            print(f"✓ Model registered: {model_id}")
            return model_id
            
        except Exception as e:
            print(f"✗ Upload failed: {e}")
            return False
    
    def list_models(self):
        """List all registered models"""
        try:
            models = self.conn.execute(f"""
                SELECT model_id, model_name, model_type, version, accuracy, 
                       created_at, trained_on_records
                FROM {MODELS_METADATA_TABLE}
                ORDER BY created_at DESC
            """).fetchall()
            
            print(f"\n📦 Registered Models ({len(models)} total):")
            print("-" * 80)
            print(f"{'Model ID':<40} {'Type':<20} {'Accuracy':<10} {'Records':<10}")
            print("-" * 80)
            
            for model in models:
                model_id, name, mtype, version, accuracy, created, records = model
                print(f"{model_id:<40} {mtype:<20} {accuracy:<10.2%} {records:<10,}")
            
            return models
            
        except Exception as e:
            print(f"✗ List failed: {e}")
            return []
    
    def get_model_info(self, model_id):
        """Get detailed info about a specific model"""
        try:
            result = self.conn.execute(f"""
                SELECT *
                FROM {MODELS_METADATA_TABLE}
                WHERE model_id = '{model_id}'
            """).fetchall()
            
            if result:
                row = result[0]
                print(f"\n📋 Model: {model_id}")
                print(f"  Name: {row[1]}")
                print(f"  Type: {row[2]}")
                print(f"  Version: {row[3]}")
                print(f"  Accuracy: {row[4]:.2%}")
                print(f"  Created: {row[5]}")
                print(f"  Training records: {row[6]:,}")
                print(f"  Path: {row[8]}")
                return row
            else:
                print(f"✗ Model not found: {model_id}")
                return None
            
        except Exception as e:
            print(f"✗ Get info failed: {e}")
            return None
    
    def load_model(self, model_id):
        """Load a model from storage"""
        try:
            result = self.conn.execute(f"""
                SELECT file_path FROM {MODELS_METADATA_TABLE}
                WHERE model_id = '{model_id}'
            """).fetchall()
            
            if not result:
                print(f"✗ Model not found: {model_id}")
                return None
            
            file_path = result[0][0]
            
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
            
            print(f"✓ Model loaded: {model_id}")
            return model
            
        except Exception as e:
            print(f"✗ Load failed: {e}")
            return None
    
    def get_best_model(self, model_type=None):
        """Get the best performing model"""
        try:
            if model_type:
                where_clause = f"WHERE model_type = '{model_type}'"
            else:
                where_clause = ""
            
            result = self.conn.execute(f"""
                SELECT model_id, model_name, model_type, version, accuracy, file_path
                FROM {MODELS_METADATA_TABLE}
                {where_clause}
                ORDER BY accuracy DESC
                LIMIT 1
            """).fetchall()
            
            if result:
                model_id, name, mtype, version, accuracy, path = result[0]
                print(f"\n🏆 Best Model: {model_id}")
                print(f"  Type: {mtype}")
                print(f"  Accuracy: {accuracy:.2%}")
                return model_id, path
            else:
                print("✗ No models found")
                return None, None
            
        except Exception as e:
            print(f"✗ Get best failed: {e}")
            return None, None
    
    def delete_model(self, model_id):
        """Delete a model"""
        try:
            # Get file path first
            result = self.conn.execute(f"""
                SELECT file_path FROM {MODELS_METADATA_TABLE}
                WHERE model_id = '{model_id}'
            """).fetchall()
            
            if result:
                file_path = result[0][0]
                
                # Delete from database
                self.conn.execute(f"""
                    DELETE FROM {MODELS_METADATA_TABLE}
                    WHERE model_id = '{model_id}'
                """)
                
                # Delete file
                if Path(file_path).exists():
                    Path(file_path).unlink()
                
                print(f"✓ Model deleted: {model_id}")
                return True
            else:
                print(f"✗ Model not found: {model_id}")
                return False
            
        except Exception as e:
            print(f"✗ Delete failed: {e}")
            return False
    
    def close(self):
        """Close connection"""
        if self.conn:
            self.conn.close()
            print("\n✓ Connection closed")


if __name__ == '__main__':
    store = DuckDBModelStore()
    
    try:
        store.connect()
        store.list_models()
        
        best = store.get_best_model()
        if best[0]:
            store.get_model_info(best[0])
        
    finally:
        store.close()
