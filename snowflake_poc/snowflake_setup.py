"""
Snowflake MLOps POC Setup Script
Automated setup for POC
"""

from snowflake_data_loader import SnowflakeDataLoader
from snowflake_model_store import SnowflakeModelStore


def setup_poc():
    """Run full POC setup"""
    
    print("\n" + "="*80)
    print("SNOWFLAKE MLOps POC - SETUP")
    print("="*80)
    
    # Step 1: Data Loading
    print("\n[STEP 1] Loading CSV data to Snowflake...")
    print("-" * 80)
    
    loader = SnowflakeDataLoader()
    try:
        loader.connect()
        loader.create_schema()
        loader.create_tables()
        loader.load_csv_files()
        loader.verify_data()
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False
    finally:
        loader.close()
    
    # Step 2: Model Store
    print("\n[STEP 2] Setting up Model Store...")
    print("-" * 80)
    
    store = SnowflakeModelStore()
    try:
        store.connect()
        store.list_models()
    except Exception as e:
        print(f"✗ Model store failed: {e}")
        return False
    finally:
        store.close()
    
    print("\n" + "="*80)
    print("✓ POC SETUP COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("  1. python snowflake_data_loader.py   # Verify data loaded")
    print("  2. python snowflake_model_store.py   # Manage models")
    print("  3. python snowflake_predict.py       # Make predictions")
    print("\n")
    
    return True


if __name__ == '__main__':
    setup_poc()
