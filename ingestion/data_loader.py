"""
Module for loading and fetching baseball game data.
"""
import pandas as pd
import requests
from pathlib import Path
from typing import Optional


class BaseballDataLoader:
    """Load historical baseball game data from various sources."""
    
    # Baseball Reference or similar API endpoints
    BASE_URL = "https://www.baseball-reference.com"
    
    def __init__(self, cache_dir: str = "./data/raw"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_games_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load games from a CSV file.
        
        Expected columns: date, home_team, away_team, home_score, away_score, etc.
        """
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date')
    
    def load_games_from_cache(self, season: int) -> Optional[pd.DataFrame]:
        """Load cached games for a given season."""
        cache_file = self.cache_dir / f"games_{season}.csv"
        if cache_file.exists():
            return pd.read_csv(cache_file)
        return None
    
    def save_to_cache(self, df: pd.DataFrame, season: int):
        """Cache games data to avoid re-fetching."""
        cache_file = self.cache_dir / f"games_{season}.csv"
        df.to_csv(cache_file, index=False)
        print(f"Saved {len(df)} games to {cache_file}")


def load_season_data(seasons: list[int], data_dir: str = "./data/raw") -> pd.DataFrame:
    """
    Load and combine game data for multiple seasons.
    
    Args:
        seasons: List of seasons to load (e.g., [2020, 2021, 2022, 2023])
        data_dir: Directory containing season CSV files
        
    Returns:
        Combined DataFrame with all games
    """
    dfs = []
    for season in seasons:
        filepath = Path(data_dir) / f"games_{season}.csv"
        if filepath.exists():
            df = pd.read_csv(filepath)
            df['season'] = season
            dfs.append(df)
    
    if not dfs:
        raise FileNotFoundError(f"No game data found in {data_dir}")
    
    combined = pd.concat(dfs, ignore_index=True)
    combined['date'] = pd.to_datetime(combined['date'])
    return combined.sort_values('date')
