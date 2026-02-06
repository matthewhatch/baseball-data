"""
Feature engineering for baseball game prediction.
"""
import pandas as pd
import numpy as np
from typing import Tuple


class BaseballFeatureEngineer:
    """Create features for predicting game outcomes."""
    
    @staticmethod
    def create_team_stats(games: pd.DataFrame, window: int = 10) -> dict:
        """
        Calculate rolling team statistics.
        
        Args:
            games: DataFrame with columns: date, home_team, away_team, home_score, away_score
            window: Number of recent games for rolling stats
            
        Returns:
            Dict with team performance metrics
        """
        team_stats = {}
        
        for team in pd.concat([games['home_team'], games['away_team']]).unique():
            home_games = games[games['home_team'] == team].copy()
            away_games = games[games['away_team'] == team].copy()
            
            home_games['team_score'] = home_games['home_score']
            home_games['opponent_score'] = home_games['away_score']
            home_games['is_home'] = 1
            
            away_games['team_score'] = away_games['away_score']
            away_games['opponent_score'] = away_games['home_score']
            away_games['is_home'] = 0
            
            all_games = pd.concat([home_games, away_games], ignore_index=True).sort_values('date')
            
            all_games['win'] = (all_games['team_score'] > all_games['opponent_score']).astype(int)
            all_games['runs_diff'] = all_games['team_score'] - all_games['opponent_score']
            
            team_stats[team] = all_games
        
        return team_stats
    
    @staticmethod
    def add_team_features(games: pd.DataFrame, team_stats: dict, window: int = 10) -> pd.DataFrame:
        """
        Add rolling average features for both teams.
        
        Args:
            games: Original games DataFrame
            team_stats: Dict of team statistics from create_team_stats
            window: Rolling window size
            
        Returns:
            DataFrame with added features
        """
        df = games.copy()
        
        for team in df['home_team'].unique():
            if team not in team_stats:
                continue
            
            team_data = team_stats[team].copy()
            team_data = team_data.sort_values('date').reset_index(drop=True)
            
            # Rolling statistics
            team_data[f'{team}_win_pct'] = team_data['win'].rolling(window, min_periods=1).mean()
            team_data[f'{team}_runs_avg'] = team_data['team_score'].rolling(window, min_periods=1).mean()
            team_data[f'{team}_runs_allowed_avg'] = team_data['opponent_score'].rolling(window, min_periods=1).mean()
            
            # Map home team features
            home_stats = team_data[team_data['is_home'] == 1][['date', f'{team}_win_pct', f'{team}_runs_avg', f'{team}_runs_allowed_avg']].copy()
            home_stats.columns = ['date', f'home_{team}_win_pct', f'home_{team}_runs_avg', f'home_{team}_runs_allowed']
            
            away_stats = team_data[team_data['is_home'] == 0][['date', f'{team}_win_pct', f'{team}_runs_avg', f'{team}_runs_allowed_avg']].copy()
            away_stats.columns = ['date', f'away_{team}_win_pct', f'away_{team}_runs_avg', f'away_{team}_runs_allowed']
            
            # Merge home stats
            df = df.merge(home_stats, on='date', how='left')
            
            # Merge away stats
            df = df.merge(away_stats, on='date', how='left')
        
        return df
    
    @staticmethod
    def create_target(games: pd.DataFrame) -> pd.Series:
        """
        Create target variable: 1 if home team wins, 0 if home team loses.
        """
        return (games['home_score'] > games['away_score']).astype(int)
    
    @staticmethod
    def prepare_features(games: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for model training.
        
        Returns:
            (features_df, target_series)
        """
        df = games.copy()
        
        # Create target
        y = BaseballFeatureEngineer.create_target(df)
        
        # Select feature columns (excluding IDs, scores, and dates)
        feature_cols = [col for col in df.columns 
                       if col not in ['date', 'home_team', 'away_team', 'home_score', 'away_score']]
        
        X = df[feature_cols].fillna(0)
        
        return X, y
