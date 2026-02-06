"""
Model training and validation pipeline for baseball game prediction.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
from pathlib import Path
import pickle


class BaseballPredictionModel:
    """Train and evaluate prediction models."""
    
    def __init__(self, model_type: str = "xgboost"):
        """
        Initialize model.
        
        Args:
            model_type: "xgboost" or "random_forest"
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
        if model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                random_state=42,
                early_stopping_rounds=20,
                verbose=False
            )
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            )
    
    def train_test_split_temporal(self, X: pd.DataFrame, y: pd.Series, 
                                 test_size: float = 0.2, val_size: float = 0.1):
        """
        Split data respecting temporal order (important for time series).
        
        Args:
            X: Features
            y: Target
            test_size: Fraction for test set
            val_size: Fraction for validation set
            
        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        n = len(X)
        test_start = int(n * (1 - test_size))
        val_start = int(test_start * (1 - val_size))
        
        X_train, y_train = X.iloc[:val_start], y.iloc[:val_start]
        X_val, y_val = X.iloc[val_start:test_start], y.iloc[val_start:test_start]
        X_test, y_test = X.iloc[test_start:], y.iloc[test_start:]
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: pd.DataFrame = None, y_val: pd.Series = None):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional, for early stopping)
            y_val: Validation target (optional)
        """
        self.feature_names = X_train.columns.tolist()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if self.model_type == "xgboost" and X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            eval_set = [(X_val_scaled, y_val)]
            self.model.fit(X_train_scaled, y_train, eval_set=eval_set, verbose=False)
        else:
            self.model.fit(X_train_scaled, y_train)
        
        print(f"Model trained with {len(X_train)} samples")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, dataset_name: str = "Test") -> dict:
        """
        Evaluate model performance.
        
        Returns:
            Dict with metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist()
        }
        
        print(f"\n{dataset_name} Set Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        
        return metrics
    
    def save(self, filepath: str):
        """Save model and scaler."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler, 'features': self.feature_names}, f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model and scaler."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['features']
        print(f"Model loaded from {filepath}")
    
    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance."""
        if self.model_type == "xgboost":
            importance = self.model.feature_importances_
        else:
            importance = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop {top_n} Important Features:")
        print(importance_df.head(top_n).to_string(index=False))
        
        return importance_df.head(top_n)
