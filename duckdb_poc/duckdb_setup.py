"""
DuckDB MLOps POC Setup Script
"""

from duckdb_data_loader import DuckDBDataLoader
from duckdb_model_store import DuckDBModelStore


def setup_poc():
    """Run full POC setup"""
    
    print("\n" + "="*80)
    print("DUCKDB MLOps POC - SETUP")
    print("="*80)
    
    # Step 1: Data Loading
    print("\n[STEP 1] Loading CSV data to DuckDB...")
    print("-" * 80)
    
    loader = DuckDBDataLoader()
    try:
        loader.connect()
        loader.create_tables()
        loader.load_csv_files()
        loader.verify_data()
        loader.get_stats()
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False
    finally:
        loader.close()
    
    # Step 2: Model Store
    print("\n[STEP 2] Setting up Model Store...")
    print("-" * 80)
    
    store = DuckDBModelStore()
    try:
        store.connect()
        models = store.list_models()
        if not models:
            print("  (No models registered yet)")
    except Exception as e:
        print(f"✗ Model store failed: {e}")
        return False
    finally:
        store.close()
    
    print("\n" + "="*80)
    print("✓ POC SETUP COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("  1. python duckdb_data_loader.py   # Verify data")
    print("  2. python duckdb_model_store.py   # Manage models")
    print("  3. python duckdb_predict.py       # Make predictions")
    print("\nDatabase: duckdb_poc/baseball.duckdb")
    print("\n")
    
    return True


if __name__ == '__main__':
    setup_poc()
