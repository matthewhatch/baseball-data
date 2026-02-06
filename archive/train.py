"""
Main training pipeline - orchestrates data loading, feature engineering, and model training.
"""
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.data_loader import load_season_data
from ingestion.features import BaseballFeatureEngineer
from warehouse.model import BaseballPredictionModel
import pandas as pd


def main():
    """
    Main training pipeline.
    
    Steps:
    1. Load historical game data
    2. Engineer features
    3. Train/validate/test split
    4. Train model
    5. Evaluate on test set
    """
    
    print("=" * 60)
    print("BASEBALL GAME PREDICTION MODEL")
    print("=" * 60)
    
    # Configuration
    seasons = [2020, 2021, 2022, 2023, 2024]  # Adjust based on available data
    data_dir = "./data/raw"
    model_dir = "./models"
    
    # Step 1: Load data
    print("\n[1] Loading historical game data...")
    try:
        games_df = load_season_data(seasons, data_dir)
        print(f"   Loaded {len(games_df)} games across {len(games_df['season'].unique())} seasons")
        print(f"   Date range: {games_df['date'].min().date()} to {games_df['date'].max().date()}")
    except FileNotFoundError as e:
        print(f"   ERROR: {e}")
        print(f"   Please ensure CSV files are in {data_dir} with format: games_YYYY.csv")
        return
    
    # Step 2: Feature engineering
    print("\n[2] Engineering features...")
    team_stats = BaseballFeatureEngineer.create_team_stats(games_df, window=10)
    games_with_features = BaseballFeatureEngineer.add_team_features(games_df, team_stats, window=10)
    print(f"   Created {len([c for c in games_with_features.columns if 'win_pct' in c])} feature columns")
    
    # Step 3: Prepare data
    print("\n[3] Preparing features and splitting data...")
    X, y = BaseballFeatureEngineer.prepare_features(games_with_features)
    print(f"   Total samples: {len(X)}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Target distribution: {(y.sum() / len(y) * 100):.1f}% home wins")
    
    # Temporal split (respecting time order)
    X_train, X_val, X_test, y_train, y_val, y_test = BaseballPredictionModel(
        model_type="xgboost"
    ).train_test_split_temporal(X, y, test_size=0.2, val_size=0.1)
    
    print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # Step 4: Train model
    print("\n[4] Training XGBoost model...")
    model = BaseballPredictionModel(model_type="xgboost")
    model.train(X_train, y_train, X_val, y_val)
    
    # Step 5: Evaluate
    print("\n[5] Evaluating model...")
    train_metrics = model.evaluate(X_train, y_train, dataset_name="Train")
    val_metrics = model.evaluate(X_val, y_val, dataset_name="Validation")
    test_metrics = model.evaluate(X_test, y_test, dataset_name="Test")
    
    # Feature importance
    print("\n[6] Feature Importance:")
    model.feature_importance(top_n=15)
    
    # Save model
    print("\n[7] Saving model...")
    Path(model_dir).mkdir(exist_ok=True)
    model.save(f"{model_dir}/baseball_model_xgb.pkl")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
