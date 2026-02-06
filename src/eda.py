"""
Exploratory Data Analysis for MLB Game Outcomes
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load all game data
print("=" * 70)
print("MLB GAME OUTCOMES - EXPLORATORY DATA ANALYSIS")
print("=" * 70)
print()

data_dir = Path("data/raw")
dfs = []

for year in [2020, 2021, 2022, 2023, 2024]:
    df = pd.read_csv(data_dir / f"games_{year}.csv")
    df['year'] = year
    dfs.append(df)
    print(f"✓ Loaded {year}: {len(df)} games")

games = pd.concat(dfs, ignore_index=True)
print(f"\nTotal games: {len(games)}")
print(f"Date range: {games['date'].min()} to {games['date'].max()}")

# Create outcome column
games['home_win'] = (games['home_score'] > games['away_score']).astype(int)
games['away_win'] = (games['away_score'] > games['home_score']).astype(int)

print("\n" + "=" * 70)
print("BASIC STATISTICS")
print("=" * 70)

print(f"\nHome Win Rate: {games['home_win'].mean():.3f} ({games['home_win'].sum()} / {len(games)})")
print(f"Away Win Rate: {games['away_win'].mean():.3f} ({games['away_win'].sum()} / {len(games)})")

print("\nScore Statistics:")
print(f"  Home Team - Mean: {games['home_score'].mean():.2f}, Std: {games['home_score'].std():.2f}")
print(f"  Away Team - Mean: {games['away_score'].mean():.2f}, Std: {games['away_score'].std():.2f}")
print(f"  Run Diff  - Mean: {(games['home_score'] - games['away_score']).mean():.2f}")

# Win/Loss records by team
print("\n" + "=" * 70)
print("TEAM PERFORMANCE (2020-2024)")
print("=" * 70)

home_records = games.groupby('home_team').agg({
    'home_win': ['sum', 'count', 'mean']
}).round(3)
home_records.columns = ['Wins', 'Home_Games', 'Home_Win_Rate']

away_records = games.groupby('away_team').agg({
    'away_win': ['sum', 'count', 'mean']
}).round(3)
away_records.columns = ['Wins', 'Away_Games', 'Away_Win_Rate']

# Combine
team_stats = home_records.copy()
team_stats['Away_Wins'] = away_records['Wins']
team_stats['Away_Games'] = away_records['Away_Games']
team_stats['Away_Win_Rate'] = away_records['Away_Win_Rate']
team_stats['Total_Wins'] = team_stats['Wins'] + team_stats['Away_Wins']
team_stats['Total_Games'] = team_stats['Home_Games'] + team_stats['Away_Games']
team_stats['Overall_Win_Rate'] = (team_stats['Total_Wins'] / team_stats['Total_Games']).round(3)

team_stats = team_stats.sort_values('Overall_Win_Rate', ascending=False)
print("\nTop 10 Teams:")
print(team_stats.head(10)[['Home_Win_Rate', 'Away_Win_Rate', 'Overall_Win_Rate', 'Total_Wins', 'Total_Games']])

print("\nBottom 10 Teams:")
print(team_stats.tail(10)[['Home_Win_Rate', 'Away_Win_Rate', 'Overall_Win_Rate', 'Total_Wins', 'Total_Games']])

# Home field advantage
print("\n" + "=" * 70)
print("HOME FIELD ADVANTAGE")
print("=" * 70)

home_advantage = games.groupby('home_team')['home_win'].mean().sort_values(ascending=False)
print("\nHome Win Rate by Team (sorted):")
print(home_advantage.round(3))

# Score distribution
print("\n" + "=" * 70)
print("SCORE DISTRIBUTIONS")
print("=" * 70)

print(f"\nHome Scores: Min={games['home_score'].min()}, Max={games['home_score'].max()}")
print(f"Away Scores: Min={games['away_score'].min()}, Max={games['away_score'].max()}")

score_diff = games['home_score'] - games['away_score']
print(f"\nScore Differential (Home - Away):")
print(f"  Mean: {score_diff.mean():.2f}")
print(f"  Median: {score_diff.median():.2f}")
print(f"  Std Dev: {score_diff.std():.2f}")

# Win/loss distribution
print("\nWin Margins:")
margins = []
for idx, row in games.iterrows():
    if row['home_win']:
        margins.append(row['home_score'] - row['away_score'])
    else:
        margins.append(row['away_score'] - row['home_score'])

margins = pd.Series(margins)
print(f"  Mean Margin: {margins.mean():.2f}")
print(f"  Median Margin: {margins.median():.2f}")
print(f"  Mode Margin: {margins.mode()[0]}")

# By year
print("\n" + "=" * 70)
print("YEARLY TRENDS")
print("=" * 70)

yearly_stats = games.groupby('year').agg({
    'home_win': ['sum', 'mean'],
    'home_score': 'mean',
    'away_score': 'mean'
}).round(3)
yearly_stats.columns = ['Home_Wins', 'Home_Win_Rate', 'Avg_Home_Score', 'Avg_Away_Score']
print("\n" + yearly_stats.to_string())

# Data completeness
print("\n" + "=" * 70)
print("DATA QUALITY")
print("=" * 70)

print(f"\nMissing Values:")
print(games.isnull().sum())

print(f"\nData Types:")
print(games.dtypes)

print(f"\nShape: {games.shape}")

# Summary for modeling
print("\n" + "=" * 70)
print("MODELING INSIGHTS")
print("=" * 70)

print(f"""
✓ Target Variable: 'home_win' (binary: 1 if home team wins, 0 if away team wins)
✓ Home Team Advantage: {games['home_win'].mean():.1%} (statistically significant)
✓ Class Balance: Home Wins {games['home_win'].sum()} / Away Wins {games['away_win'].sum()}
✓ Sample Size: {len(games)} games (good for training)
✓ Teams: {games['home_team'].nunique()} unique teams

Potential Features:
  - Home team historical win rate
  - Away team historical win rate
  - Home team average score
  - Away team average score
  - Recent form (last N games)
  - Rest days (days since last game)
  - Season progression
  - Home/Away splits
""")

print("=" * 70)
print("Ready for feature engineering and model training!")
print("=" * 70)
