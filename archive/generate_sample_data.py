"""
Generate realistic sample baseball data for testing the model.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

os.makedirs('./data/raw', exist_ok=True)

# Sample teams
TEAMS = ['NYY', 'BOS', 'LAD', 'SF', 'NYM', 'CHC', 'HOU', 'ATL', 'MIA', 'WSH',
         'PHI', 'STL', 'MIL', 'CIN', 'PIT', 'COL', 'ARI', 'SD', 'TEX', 'OAK',
         'TB', 'TOR', 'BAL', 'KC', 'MIN', 'CWS', 'DET', 'CLE', 'LAA', 'SEA']

def generate_season_data(year, num_games=1200):
    """Generate synthetic baseball season data."""
    np.random.seed(year)  # Reproducible
    
    games = []
    start_date = datetime(year, 3, 28)
    
    for i in range(num_games):
        date = start_date + timedelta(days=int(i * 162 / num_games))
        
        # Random teams
        teams = np.random.choice(TEAMS, 2, replace=False)
        home_team, away_team = teams[0], teams[1]
        
        # Realistic scores (most games 2-8 runs)
        home_score = np.random.poisson(4.2) + np.random.randint(0, 3)
        away_score = np.random.poisson(4.0) + np.random.randint(0, 3)
        
        games.append({
            'date': date.strftime('%Y-%m-%d'),
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score,
        })
    
    return pd.DataFrame(games)

# Generate 4 seasons of data
for year in [2021, 2022, 2023, 2024]:
    df = generate_season_data(year)
    filepath = f'./data/raw/games_{year}.csv'
    df.to_csv(filepath, index=False)
    print(f"✓ Created {filepath} ({len(df)} games)")

print("\nData generation complete!")
print("Now run: python train.py")
