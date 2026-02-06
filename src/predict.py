"""
Game Outcome Prediction - Make predictions for upcoming games
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
from pathlib import Path

print("=" * 70)
print("MLB GAME OUTCOME PREDICTOR")
print("=" * 70)

# Load models, scaler, and feature names
with open('models/gb_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/feature_names.pkl', 'rb') as f:
    feature_cols = pickle.load(f)

print("\n✓ Loaded trained Gradient Boosting model")

# Load historical data for context
data_dir = Path("data/raw")
dfs = []
for year in [2020, 2021, 2022, 2023, 2024]:
    df = pd.read_csv(data_dir / f"games_{year}.csv")
    df['year'] = year
    dfs.append(df)

games = pd.concat(dfs, ignore_index=True).reset_index(drop=True)
games['date'] = pd.to_datetime(games['date'])
games['home_win'] = (games['home_score'] > games['away_score']).astype(int)

# Compute team statistics
team_stats = {}
for team in games['home_team'].unique():
    team_stats[team] = {
        'home_wins': 0, 'home_games': 0,
        'away_wins': 0, 'away_games': 0,
        'total_runs_at_home': 0, 'total_runs_away': 0
    }

for idx, row in games.iterrows():
    home_team = row['home_team']
    away_team = row['away_team']
    
    team_stats[home_team]['home_games'] += 1
    team_stats[home_team]['total_runs_at_home'] += row['home_score']
    team_stats[away_team]['away_games'] += 1
    team_stats[away_team]['total_runs_away'] += row['away_score']
    
    if row['home_win']:
        team_stats[home_team]['home_wins'] += 1
    else:
        team_stats[away_team]['away_wins'] += 1

# Print team statistics
print("\n" + "=" * 70)
print("TEAM STATISTICS (2020-2024)")
print("=" * 70)

team_list = []
for team in sorted(team_stats.keys()):
    stats = team_stats[team]
    home_wr = stats['home_wins'] / max(stats['home_games'], 1)
    away_wr = stats['away_wins'] / max(stats['away_games'], 1)
    total_wr = (stats['home_wins'] + stats['away_wins']) / max(stats['home_games'] + stats['away_games'], 1)
    home_avg = stats['total_runs_at_home'] / max(stats['home_games'], 1)
    away_avg = stats['total_runs_away'] / max(stats['away_games'], 1)
    
    team_list.append({
        'Team': team,
        'Home_W%': f"{home_wr:.3f}",
        'Away_W%': f"{away_wr:.3f}",
        'Overall_W%': f"{total_wr:.3f}",
        'Home_RS': f"{home_avg:.2f}",
        'Away_RS': f"{away_avg:.2f}"
    })

team_df = pd.DataFrame(team_list)
print("\n" + team_df.to_string(index=False))

# Example predictions
print("\n" + "=" * 70)
print("EXAMPLE PREDICTIONS")
print("=" * 70)

def predict_game(home_team, away_team, games_df, team_stats, model, max_date=None):
    """
    Predict the outcome of a game
    """
    if max_date is None:
        max_date = games_df['date'].max()
    
    # Get team stats up to max_date
    recent_games = games_df[games_df['date'] <= max_date]
    
    h_stats = team_stats[home_team]
    a_stats = team_stats[away_team]
    
    # Historical win rates
    home_wr = h_stats['home_wins'] / max(h_stats['home_games'], 1)
    away_wr = a_stats['away_wins'] / max(a_stats['away_games'], 1)
    
    # Average scores
    home_avg_score = h_stats['total_runs_at_home'] / max(h_stats['home_games'], 1)
    away_avg_score = a_stats['total_runs_away'] / max(a_stats['away_games'], 1)
    
    # Days into season (assuming 2024 season at day 200)
    days_into_season = 200
    
    # Home field advantage
    h_home_wr = h_stats['home_wins'] / max(h_stats['home_games'], 1)
    h_away_wr = h_stats['away_wins'] / max(h_stats['away_games'], 1)
    a_home_wr = a_stats['home_wins'] / max(a_stats['home_games'], 1)
    a_away_wr = a_stats['away_wins'] / max(a_stats['away_games'], 1)
    hfa = h_home_wr - h_away_wr - (a_home_wr - a_away_wr)
    
    # Recent form (last 10 games at home)
    h_recent = recent_games[recent_games['home_team'] == home_team].tail(10)
    if len(h_recent) > 0:
        home_recent_form = h_recent['home_win'].mean()
    else:
        home_recent_form = 0.5
    
    # Day of week
    day_of_week = 2  # Wednesday default
    month = 7  # July default
    
    # Create feature vector as DataFrame with proper column names
    features = pd.DataFrame([{
        'home_historical_wr': home_wr,
        'away_historical_wr': away_wr,
        'home_avg_score': home_avg_score,
        'away_avg_score': away_avg_score,
        'days_into_season': days_into_season,
        'home_field_advantage': hfa,
        'home_recent_form': home_recent_form,
        'day_of_week': day_of_week,
        'month': month
    }])
    
    # Get prediction
    pred_proba = model.predict_proba(features)[0]
    pred = model.predict(features)[0]
    
    return {
        'home_team': home_team,
        'away_team': away_team,
        'home_win_prob': pred_proba[1],
        'away_win_prob': pred_proba[0],
        'predicted_winner': 'HOME' if pred == 1 else 'AWAY',
        'confidence': max(pred_proba) * 100
    }

# Make some example predictions
examples = [
    ('LAD', 'ATL'),
    ('HOU', 'NYY'),
    ('ATL', 'WSH'),
    ('NYM', 'MIA'),
    ('BOS', 'TB'),
]

print("\nSample Predictions:\n")
for home, away in examples:
    result = predict_game(home, away, games, team_stats, model)
    print(f"{result['away_team']:3s} @ {result['home_team']:3s}")
    print(f"  → {result['predicted_winner']:4s} Win | {result['confidence']:5.1f}% confidence")
    print(f"    Home: {result['home_win_prob']:.3f}, Away: {result['away_win_prob']:.3f}")
    print()

# Save prediction function
print("=" * 70)
print("✓ Predictor ready! Use predict_game() for custom predictions")
print("=" * 70)
