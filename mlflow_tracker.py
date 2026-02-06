"""
MLflow experiment tracking integration
Track model training runs, metrics, and parameters
"""

import mlflow
import mlflow.sklearn
from pathlib import Path
import json


class MLflowTracker:
    """MLflow tracking wrapper"""
    
    def __init__(self, experiment_name="baseball-predictions"):
        """Initialize MLflow tracker"""
        self.experiment_name = experiment_name
        
        # Set tracking server (local by default)
        mlflow.set_tracking_uri("file:./mlruns")
        
        # Create or get experiment
        try:
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        except:
            experiment_id = mlflow.create_experiment(experiment_name)
        
        mlflow.set_experiment(experiment_name)
        print(f"✓ MLflow tracking: {experiment_name}")
    
    def log_training_run(self, model_type, params, metrics, model, features):
        """
        Log a training run to MLflow
        
        Args:
            model_type: Model type (e.g., 'GradientBoosting')
            params: Dictionary of hyperparameters
            metrics: Dictionary of metrics (accuracy, precision, etc.)
            model: Trained sklearn model
            features: List of feature names
        """
        
        with mlflow.start_run(run_name=model_type):
            # Log parameters
            for key, value in params.items():
                mlflow.log_param(key, value)
            
            # Log metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            
            # Log model
            mlflow.sklearn.log_model(
                model,
                artifact_path=model_type.lower(),
                registered_model_name=f"{model_type}-baseball"
            )
            
            # Log features
            mlflow.log_dict(
                {"features": features},
                "features.json"
            )
            
            # Log tags
            mlflow.set_tag("domain", "sports")
            mlflow.set_tag("data", "MLB")
            mlflow.set_tag("task", "classification")
            
            run = mlflow.active_run()
            print(f"\n✓ Logged run: {run.info.run_id}")
            return run.info.run_id
    
    def compare_runs(self):
        """Compare all runs in experiment"""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        if len(runs) == 0:
            print("No runs found")
            return
        
        print(f"\n📊 Runs in {self.experiment_name}:")
        print("-" * 100)
        
        for _, run in runs.iterrows():
            print(f"\nRun: {run['run_id'][:8]}...")
            print(f"  Model: {run.get('tags.mlflow.runName', 'N/A')}")
            
            # Print metrics
            for col in runs.columns:
                if col.startswith('metrics.'):
                    metric_name = col.replace('metrics.', '')
                    metric_value = run[col]
                    if metric_value == metric_value:  # Check for NaN
                        print(f"  {metric_name}: {metric_value:.4f}")
    
    def get_best_run(self, metric_name='accuracy'):
        """Get best run by metric"""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric_name} DESC"],
            max_results=1
        )
        
        if len(runs) == 0:
            print("No runs found")
            return None
        
        best_run = runs.iloc[0]
        print(f"\n🏆 Best run (by {metric_name}):")
        print(f"  Run ID: {best_run['run_id']}")
        print(f"  Model: {best_run.get('tags.mlflow.runName', 'N/A')}")
        
        for col in runs.columns:
            if col.startswith('metrics.'):
                metric = col.replace('metrics.', '')
                value = best_run[col]
                if value == value:  # Check for NaN
                    print(f"  {metric}: {value:.4f}")
        
        return best_run['run_id']


if __name__ == '__main__':
    # Example usage
    tracker = MLflowTracker("test-experiment")
    
    # Log example run
    params = {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1
    }
    
    metrics = {
        'accuracy': 0.5463,
        'precision': 0.5520,
        'recall': 0.5210,
        'f1': 0.5360
    }
    
    features = ['home_historical_wr', 'away_historical_wr', 'home_avg_score', 'away_avg_score']
    
    # This would need a real model, so we'll just print
    print("✓ MLflow setup complete")
    print("View with: mlflow ui")
