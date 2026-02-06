"""
Predictions using Snowflake as source of truth
Query data from Snowflake, make predictions, log results
"""

import pickle
import uuid
from datetime import datetime
from snowflake.connector import connect
from snowflake.connector.errors import ProgrammingError
import pandas as pd
from snowflake_config import SnowflakeConfig


class SnowflakePredictions:
    """Make predictions using Snowflake data"""
    
    def __init__(self, model_path):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained .pkl model
        """
        self.config = SnowflakeConfig()
        self.connection = None
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
        """Connect to Snowflake"""
        try:
            self.connection = connect(**self.config.get_connection_params())
            print("✓ Connected to Snowflake")
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            raise
    
    def get_team_stats(self, team):
        """Get historical stats for a team from Snowflake"""
        cursor = self.connection.cursor()
        try:
            cursor.execute(f"""
                SELECT 
                    TEAM,
                    GAMES_PLAYED,
                    WINS,
                    LOSSES,
                    WIN_RATE,
                    AVG_SCORE,
                    AVG_OPPONENT_SCORE
                FROM (
                    SELECT
                        HOME_TEAM as TEAM,
                        COUNT(*) as GAMES_PLAYED,
                        SUM(CASE WHEN HOME_SCORE > AWAY_SCORE THEN 1 ELSE 0 END) as WINS,
                        SUM(CASE WHEN HOME_SCORE <= AWAY_SCORE THEN 1 ELSE 0 END) as LOSSES,
                        WINS::FLOAT / GAMES_PLAYED as WIN_RATE,
                        AVG(HOME_SCORE) as AVG_SCORE,
                        AVG(AWAY_SCORE) as AVG_OPPONENT_SCORE
                    FROM {self.config.GAMES_TABLE}
                    WHERE HOME_TEAM = '{team}'
                    GROUP BY HOME_TEAM
                )
                LIMIT 1
            """)
            
            result = cursor.fetchone()
            if result:
                return {
                    'team': result[0],
                    'games': result[1],
                    'wins': result[2],
                    'losses': result[3],
                    'win_rate': result[4],
                    'avg_score': result[5],
                    'avg_opp_score': result[6]
                }
            return None
            
        finally:
            cursor.close()
    
    def predict_game(self, home_team, away_team, model_id='gb_model_v1.0'):
        """
        Predict outcome of a game using Snowflake data
        
        Args:
            home_team: Home team abbreviation (e.g., 'LAD')
            away_team: Away team abbreviation (e.g., 'NYY')
            model_id: Model to use for prediction
            
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
            
            # Handle NaN/None values with sensible defaults
            def safe_value(val, default=0.5):
                """Replace NaN/None with default"""
                if val is None or (isinstance(val, float) and val != val):  # NaN check
                    return default
                return val
            
            home_wr = safe_value(home_stats['win_rate'], 0.500)
            away_wr = safe_value(away_stats['win_rate'], 0.500)
            home_score = safe_value(home_stats['avg_score'], 4.5)
            away_score = safe_value(away_stats['avg_score'], 4.5)
            
            # Create feature vector
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
        """Log prediction to Snowflake"""
        cursor = self.connection.cursor()
        try:
            prediction_id = str(uuid.uuid4())
            
            cursor.execute(f"""
                INSERT INTO {self.config.PREDICTIONS_LOG_TABLE}
                (PREDICTION_ID, HOME_TEAM, AWAY_TEAM, PREDICTED_WINNER, CONFIDENCE,
                 HOME_WIN_PROB, AWAY_WIN_PROB, MODEL_ID, PREDICTED_AT, ACTUAL_WINNER)
                VALUES
                ('{prediction_id}', '{prediction['home_team']}', '{prediction['away_team']}',
                 '{prediction['predicted_winner']}', {prediction['confidence']/100},
                 {prediction['home_win_prob']}, {prediction['away_win_prob']},
                 '{prediction['model_id']}', '{prediction['timestamp']}',
                 {f"'{actual_winner}'" if actual_winner else 'NULL'})
            """)
            
            print(f"✓ Prediction logged: {prediction_id}")
            return prediction_id
            
        finally:
            cursor.close()
    
    def get_prediction_accuracy(self, model_id=None):
        """Calculate accuracy of predictions"""
        cursor = self.connection.cursor()
        try:
            where_clause = ""
            if model_id:
                where_clause = f"WHERE MODEL_ID = '{model_id}'"
            
            cursor.execute(f"""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN PREDICTED_WINNER = ACTUAL_WINNER THEN 1 ELSE 0 END) as correct,
                    SUM(CASE WHEN PREDICTED_WINNER = ACTUAL_WINNER THEN 1 ELSE 0 END)::FLOAT /
                    COUNT(*) as accuracy
                FROM {self.config.PREDICTIONS_LOG_TABLE}
                WHERE ACTUAL_WINNER IS NOT NULL
                {where_clause}
            """)
            
            result = cursor.fetchone()
            if result:
                total, correct, accuracy = result
                print(f"\nPrediction Accuracy:")
                print(f"  Total predictions: {total}")
                print(f"  Correct: {correct}")
                print(f"  Accuracy: {accuracy:.2%}")
                return accuracy
            
        finally:
            cursor.close()
    
    def close(self):
        """Close connection"""
        if self.connection:
            self.connection.close()
            print("\n✓ Connection closed")


if __name__ == '__main__':
    # Usage example
    try:
        predictor = SnowflakePredictions('../models/gb_model.pkl')
        predictor.connect()
        
        # Make prediction
        result = predictor.predict_game('LAD', 'NYY')
        if result:
            print("\nPrediction Result:")
            print(f"  {result['away_team']} @ {result['home_team']}")
            print(f"  Winner: {result['predicted_winner']}")
            print(f"  Confidence: {result['confidence']:.1f}%")
            print(f"  Home Win Prob: {result['home_win_prob']:.2%}")
            print(f"  Away Win Prob: {result['away_win_prob']:.2%}")
            
            # Log prediction
            predictor.log_prediction(result, actual_winner='HOME')
            
            # Check accuracy
            predictor.get_prediction_accuracy()
        
    finally:
        predictor.close()
