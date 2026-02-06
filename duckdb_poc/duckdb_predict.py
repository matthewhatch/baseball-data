"""
DuckDB Predictions
Make predictions using DuckDB as source of truth
"""

import pickle
import uuid
from datetime import datetime
import duckdb
import pandas as pd
from duckdb_config import DB_PATH, GAMES_TABLE, PREDICTIONS_LOG_TABLE


class DuckDBPredictions:
    """Make predictions using DuckDB data"""
    
    def __init__(self, model_path):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained .pkl model
        """
        self.db_path = DB_PATH
        self.conn = None
        self.model = None
        self.model_id = None
        
        # Load model
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✓ Model loaded: {model_path}")
        except FileNotFoundError:
            print(f"✗ Model not found: {model_path}")
            raise
    
    def connect(self):
        """Connect to DuckDB"""
        self.conn = duckdb.connect(str(self.db_path))
        print(f"✓ Connected to DuckDB")
    
    def get_team_stats(self, team):
        """Get historical stats for a team"""
        try:
            stats = self.conn.execute(f"""
                SELECT
                    '{team}' as team,
                    COUNT(*) as games_played,
                    SUM(CASE WHEN home_score > away_score THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN home_score <= away_score THEN 1 ELSE 0 END) as losses,
                    SUM(CASE WHEN home_score > away_score THEN 1 ELSE 0 END)::FLOAT / 
                    COUNT(*) as win_rate,
                    AVG(home_score) as avg_score,
                    AVG(away_score) as avg_opp_score
                FROM {GAMES_TABLE}
                WHERE home_team = '{team}'
            """).fetchall()
            
            if stats:
                team, games, wins, losses, wr, avg_score, avg_opp = stats[0]
                return {
                    'team': team,
                    'games': games,
                    'wins': wins,
                    'losses': losses,
                    'win_rate': wr,
                    'avg_score': avg_score,
                    'avg_opp_score': avg_opp
                }
            return None
            
        except Exception as e:
            print(f"✗ Get stats failed: {e}")
            return None
    
    def predict_game(self, home_team, away_team, model_id='local_model'):
        """
        Predict outcome of a game
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            model_id: Model identifier
            
        Returns:
            Dictionary with prediction results
        """
        
        try:
            # Get team stats
            home_stats = self.get_team_stats(home_team)
            away_stats = self.get_team_stats(away_team)
            
            if not home_stats or not away_stats:
                print(f"✗ Could not find stats for {home_team} vs {away_team}")
                return None
            
            # Handle NaN values with sensible defaults
            def safe_value(val, default=0.5):
                """Replace NaN/None with default"""
                if val is None or (isinstance(val, float) and val != val):  # NaN check
                    return default
                return val
            
            home_wr = safe_value(home_stats['win_rate'], 0.500)
            away_wr = safe_value(away_stats['win_rate'], 0.500)
            home_score = safe_value(home_stats['avg_score'], 4.5)
            away_score = safe_value(away_stats['avg_score'], 4.5)
            
            # Create feature vector (matching train_model.py)
            features = [
                home_wr,              # home_historical_wr
                away_wr,              # away_historical_wr
                home_score,           # home_avg_score
                away_score,           # away_avg_score
                0.5,                  # days_into_season (placeholder)
                0.08,                 # home_field_advantage (placeholder)
                home_wr,              # home_recent_form (placeholder)
                2,                    # day_of_week (placeholder)
                7                     # month (placeholder)
            ]
            
            # Validate no NaN in features
            if any(v != v for v in features):  # NaN check
                print(f"✗ Invalid features contain NaN: {features}")
                return None
            
            # Make prediction
            pred = self.model.predict([features])[0]
            proba = self.model.predict_proba([features])[0]
            
            result = {
                'home_team': home_team,
                'away_team': away_team,
                'predicted_winner': 'HOME' if pred else 'AWAY',
                'home_win_prob': float(proba[1]),
                'away_win_prob': float(proba[0]),
                'confidence': float(max(proba)) * 100,
                'model_id': model_id,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return result
            
        except Exception as e:
            print(f"✗ Prediction failed: {e}")
            return None
    
    def log_prediction(self, prediction, actual_winner=None):
        """Log prediction to DuckDB"""
        try:
            prediction_id = str(uuid.uuid4())
            
            self.conn.execute(f"""
                INSERT INTO {PREDICTIONS_LOG_TABLE}
                (prediction_id, home_team, away_team, predicted_winner, confidence,
                 home_win_prob, away_win_prob, model_id, predicted_at, actual_winner)
                VALUES
                ('{prediction_id}', '{prediction['home_team']}', '{prediction['away_team']}',
                 '{prediction['predicted_winner']}', {prediction['confidence']/100},
                 {prediction['home_win_prob']}, {prediction['away_win_prob']},
                 '{prediction['model_id']}', '{prediction['timestamp']}',
                 {f"'{actual_winner}'" if actual_winner else 'NULL'})
            """)
            
            print(f"✓ Prediction logged: {prediction_id}")
            return prediction_id
            
        except Exception as e:
            print(f"✗ Log failed: {e}")
            return None
    
    def get_prediction_accuracy(self, model_id=None):
        """Calculate accuracy of predictions"""
        try:
            where_clause = ""
            if model_id:
                where_clause = f"WHERE model_id = '{model_id}'"
            
            result = self.conn.execute(f"""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN predicted_winner = actual_winner THEN 1 ELSE 0 END) as correct,
                    SUM(CASE WHEN predicted_winner = actual_winner THEN 1 ELSE 0 END)::FLOAT /
                    COUNT(*) as accuracy
                FROM {PREDICTIONS_LOG_TABLE}
                WHERE actual_winner IS NOT NULL
                {where_clause}
            """).fetchall()
            
            if result:
                total, correct, accuracy = result[0]
                if total > 0:
                    print(f"\n📊 Prediction Accuracy:")
                    print(f"  Total predictions: {total}")
                    print(f"  Correct: {correct}")
                    print(f"  Accuracy: {accuracy:.2%}")
                    return accuracy
                else:
                    print("  No completed predictions yet")
                    return None
            
        except Exception as e:
            print(f"✗ Accuracy check failed: {e}")
            return None
    
    def get_recent_predictions(self, limit=10):
        """Get recent predictions"""
        try:
            results = self.conn.execute(f"""
                SELECT home_team, away_team, predicted_winner, confidence,
                       actual_winner, predicted_at
                FROM {PREDICTIONS_LOG_TABLE}
                ORDER BY predicted_at DESC
                LIMIT {limit}
            """).fetchall()
            
            print(f"\n📝 Recent Predictions (last {limit}):")
            print("-" * 80)
            for row in results:
                home, away, pred, conf, actual, ts = row
                actual_str = f" → {actual}" if actual else ""
                print(f"  {away} @ {home:<4} | Pred: {pred:<4} ({conf:>5.1f}%){actual_str}")
            
            return results
            
        except Exception as e:
            print(f"✗ Get predictions failed: {e}")
            return []
    
    def close(self):
        """Close connection"""
        if self.conn:
            self.conn.close()
            print("\n✓ Connection closed")


if __name__ == '__main__':
    try:
        predictor = DuckDBPredictions('../models/gb_model.pkl')
        predictor.connect()
        
        # Make prediction
        result = predictor.predict_game('LAD', 'NYY')
        if result:
            print("\n🎯 Prediction Result:")
            print(f"  {result['away_team']} @ {result['home_team']}")
            print(f"  Winner: {result['predicted_winner']}")
            print(f"  Confidence: {result['confidence']:.1f}%")
            print(f"  Home Win Prob: {result['home_win_prob']:.2%}")
            print(f"  Away Win Prob: {result['away_win_prob']:.2%}")
            
            # Log prediction
            predictor.log_prediction(result, actual_winner='HOME')
            
            # Check accuracy
            predictor.get_prediction_accuracy()
            
            # Recent predictions
            predictor.get_recent_predictions()
        
    finally:
        predictor.close()
