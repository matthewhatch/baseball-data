"""
Model Storage: Store and Retrieve ML Models in Snowflake
Manages model artifacts, versioning, and metadata
"""

import pickle
import json
from datetime import datetime
from pathlib import Path
from snowflake.connector import connect
import hashlib
from snowflake_config import SnowflakeConfig


class SnowflakeModelStore:
    """Store and manage ML models in Snowflake"""
    
    def __init__(self):
        self.config = SnowflakeConfig()
        self.connection = None
    
    def connect(self):
        """Connect to Snowflake"""
        try:
            self.connection = connect(**self.config.get_connection_params())
            print("✓ Connected to Snowflake")
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            raise
    
    def upload_model(self, model_path, model_name, model_type, 
                    version, accuracy, features, trained_records):
        """
        Upload a trained model to Snowflake Stage and register metadata
        
        Args:
            model_path: Local path to .pkl file
            model_name: Name of the model (e.g., 'gb_model')
            model_type: Type (e.g., 'GradientBoosting')
            version: Version string (e.g., '1.0.0')
            accuracy: Model accuracy (0-1)
            features: List of feature names
            trained_records: Number of records used for training
        """
        
        if not Path(model_path).exists():
            print(f"✗ Model file not found: {model_path}")
            return False
        
        cursor = self.connection.cursor()
        try:
            # Generate model ID
            file_hash = hashlib.md5(
                open(model_path, 'rb').read()
            ).hexdigest()[:8]
            model_id = f"{model_name}_v{version}_{file_hash}"
            
            # Upload to stage
            stage_path = f"@MODELS_STAGE/{model_name}/v{version}/{Path(model_path).name}"
            
            # Put file to stage
            cursor.execute(f"PUT file://{model_path} {stage_path} AUTO_COMPRESS = FALSE")
            print(f"✓ Model uploaded to {stage_path}")
            
            # Register in metadata table
            cursor.execute(f"""
                INSERT INTO {self.config.MODELS_METADATA_TABLE}
                (MODEL_ID, MODEL_NAME, MODEL_TYPE, VERSION, ACCURACY, 
                 CREATED_AT, TRAINED_ON_RECORDS, FEATURES, LOCATION)
                VALUES 
                ('{model_id}', '{model_name}', '{model_type}', '{version}', 
                 {accuracy}, '{datetime.utcnow().isoformat()}', {trained_records},
                 ARRAY_CONSTRUCT({', '.join([f"'{f}'" for f in features])}),
                 '{stage_path}')
            """)
            print(f"✓ Model metadata registered: {model_id}")
            
            return model_id
            
        except Exception as e:
            print(f"✗ Upload failed: {e}")
            return False
        finally:
            cursor.close()
    
    def list_models(self):
        """List all registered models"""
        cursor = self.connection.cursor()
        try:
            cursor.execute(f"""
                SELECT MODEL_ID, MODEL_NAME, MODEL_TYPE, VERSION, ACCURACY, 
                       CREATED_AT, TRAINED_ON_RECORDS
                FROM {self.config.MODELS_METADATA_TABLE}
                ORDER BY CREATED_AT DESC
            """)
            
            models = cursor.fetchall()
            print(f"\nRegistered Models ({len(models)} total):")
            print("-" * 80)
            print(f"{'Model ID':<40} {'Type':<20} {'Accuracy':<10} {'Records':<10}")
            print("-" * 80)
            
            for model in models:
                model_id, name, mtype, version, accuracy, created, records = model
                print(f"{model_id:<40} {mtype:<20} {accuracy:<10.2%} {records:<10,}")
            
            return models
            
        finally:
            cursor.close()
    
    def get_model_info(self, model_id):
        """Get detailed info about a specific model"""
        cursor = self.connection.cursor()
        try:
            cursor.execute(f"""
                SELECT *
                FROM {self.config.MODELS_METADATA_TABLE}
                WHERE MODEL_ID = '{model_id}'
            """)
            
            result = cursor.fetchone()
            if result:
                print(f"\nModel: {model_id}")
                print(f"  Type: {result[2]}")
                print(f"  Version: {result[3]}")
                print(f"  Accuracy: {result[4]:.2%}")
                print(f"  Created: {result[5]}")
                print(f"  Training Records: {result[6]:,}")
                print(f"  Location: {result[8]}")
                return result
            else:
                print(f"✗ Model not found: {model_id}")
                return None
            
        finally:
            cursor.close()
    
    def download_model(self, model_id, output_dir='./'):
        """Download a model from Snowflake Stage"""
        cursor = self.connection.cursor()
        try:
            # Get model location
            cursor.execute(f"""
                SELECT LOCATION FROM {self.config.MODELS_METADATA_TABLE}
                WHERE MODEL_ID = '{model_id}'
            """)
            
            result = cursor.fetchone()
            if not result:
                print(f"✗ Model not found: {model_id}")
                return False
            
            stage_path = result[0]
            
            # Download from stage
            output_path = Path(output_dir) / f"{model_id}.pkl"
            cursor.execute(f"GET {stage_path} file://{output_path}")
            
            print(f"✓ Downloaded to {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"✗ Download failed: {e}")
            return False
        finally:
            cursor.close()
    
    def get_best_model(self, model_type=None):
        """Get the best performing model (highest accuracy)"""
        cursor = self.connection.cursor()
        try:
            if model_type:
                where_clause = f"WHERE MODEL_TYPE = '{model_type}'"
            else:
                where_clause = ""
            
            cursor.execute(f"""
                SELECT MODEL_ID, MODEL_NAME, MODEL_TYPE, VERSION, ACCURACY, LOCATION
                FROM {self.config.MODELS_METADATA_TABLE}
                {where_clause}
                ORDER BY ACCURACY DESC
                LIMIT 1
            """)
            
            result = cursor.fetchone()
            if result:
                model_id, name, mtype, version, accuracy, location = result
                print(f"\nBest Model: {model_id}")
                print(f"  Type: {mtype}")
                print(f"  Accuracy: {accuracy:.2%}")
                print(f"  Location: {location}")
                return model_id
            else:
                print("✗ No models found")
                return None
            
        finally:
            cursor.close()
    
    def close(self):
        """Close connection"""
        if self.connection:
            self.connection.close()
            print("\n✓ Connection closed")


if __name__ == '__main__':
    # Usage example
    store = SnowflakeModelStore()
    
    try:
        store.connect()
        
        # List all models
        store.list_models()
        
        # Get best model
        best = store.get_best_model()
        
        # Get model details
        if best:
            store.get_model_info(best)
        
    finally:
        store.close()
