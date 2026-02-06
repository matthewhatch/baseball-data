"""
Example: How to prepare your baseball data for the model.
"""
import pandas as pd
import os

# Example data structure needed
example_data = {
    'date': ['2024-03-28', '2024-03-29', '2024-03-30'],
    'home_team': ['NYY', 'NYY', 'BOS'],
    'away_team': ['BOS', 'BOS', 'NYY'],
    'home_score': [3, 5, 2],
    'away_score': [2, 3, 6],
}

example_df = pd.DataFrame(example_data)

# Create data directory structure
os.makedirs('./data/raw', exist_ok=True)

# Save example
example_df.to_csv('./data/raw/games_2024.csv', index=False)

print("Example data structure created at ./data/raw/games_2024.csv")
print("\nExpected format:")
print(example_df)
print("\nRequired columns:")
print("  - date: YYYY-MM-DD format")
print("  - home_team: Team abbreviation (e.g., NYY, BOS, LAD)")
print("  - away_team: Team abbreviation")
print("  - home_score: Integer score")
print("  - away_score: Integer score")
print("\nSave your data as: ./data/raw/games_YYYY.csv")
print("Then run: python train.py")
