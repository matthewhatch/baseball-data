"""
Game Outcome Prediction Model for MLB
Predicts whether the home team or away team wins
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("GAME OUTCOME PREDICTION - MODEL TRAINING")
print("=" * 70)
print()

# Load all game data
data_dir = Path("data/raw")
dfs = []

for year in [2020, 2021, 2022, 2023, 2024]:
    df = pd.read_csv(data_dir / f"games_{year}.csv")
    df['year'] = year
    dfs.append(df)

games = pd.concat(dfs, ignore_index=True).reset_index(drop=True)
games['date'] = pd.to_datetime(games['date'])
games['home_win'] = (games['home_score'] > games['away_score']).astype(int)

print(f"Loaded {len(games)} games from 2020-2024")

# Feature Engineering
print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)

features_df = games[['game_id', 'date', 'home_team', 'away_team', 'home_score', 'away_score', 'home_win', 'year']].copy()

# 1. Team Strength Features (historical win rates up to each date)
print("\n✓ Computing team strength features...")

team_stats = {}

for team in games['home_team'].unique():
    team_stats[team] = {'home_wins': 0, 'home_games': 0, 'away_wins': 0, 'away_games': 0}

home_wr = []
away_wr = []

for idx, row in features_df.iterrows():
    home_team = row['home_team']
    away_team = row['away_team']
    date = row['date']
    
    # Get historical stats BEFORE this game
    home_home_games = team_stats[home_team]['home_games']
    home_home_wins = team_stats[home_team]['home_wins']
    home_away_games = team_stats[home_team]['away_games']
    home_away_wins = team_stats[home_team]['away_wins']
    
    away_home_games = team_stats[away_team]['home_games']
    away_home_wins = team_stats[away_team]['home_wins']
    away_away_games = team_stats[away_team]['away_games']
    away_away_wins = team_stats[away_team]['away_wins']
    
    # Home team win rates
    home_home_wr = home_home_wins / max(home_home_games, 1)
    home_away_wr = home_away_wins / max(home_away_games, 1)
    
    # Away team win rates
    away_home_wr = away_home_wins / max(away_home_games, 1)
    away_away_wr = away_away_wins / max(away_away_games, 1)
    
    home_wr.append(home_home_wr)
    away_wr.append(away_away_wr)
    
    # Update stats after this game
    if row['home_win']:
        team_stats[home_team]['home_wins'] += 1
        team_stats[away_team]['away_games'] += 1
    else:
        team_stats[away_team]['away_wins'] += 1
        team_stats[home_team]['home_games'] += 1
    
    team_stats[home_team]['home_games'] += 1
    team_stats[away_team]['away_games'] += 1

features_df['home_historical_wr'] = home_wr
features_df['away_historical_wr'] = away_wr

# 2. Score features
print("✓ Adding score-based features...")
features_df['home_avg_score'] = games.groupby('home_team')['home_score'].expanding().mean().reset_index(drop=True)
features_df['away_avg_score'] = games.groupby('away_team')['away_score'].expanding().mean().reset_index(drop=True)

# 3. Season progression
print("✓ Adding season features...")
features_df['days_into_season'] = features_df.groupby('year')['date'].apply(
    lambda x: (x - x.min()).dt.days
).reset_index(drop=True)

# 4. Home field advantage by team
print("✓ Computing home field advantage...")
team_home_advantage = {}
for team in games['home_team'].unique():
    team_home_advantage[team] = {}
    team_home_advantage[team]['home_wins'] = 0
    team_home_advantage[team]['home_games'] = 0
    team_home_advantage[team]['away_wins'] = 0
    team_home_advantage[team]['away_games'] = 0

hfa = []
for idx, row in features_df.iterrows():
    home_team = row['home_team']
    away_team = row['away_team']
    
    # HFA = (home_win_rate - away_win_rate)
    h_home_wr = team_home_advantage[home_team]['home_wins'] / max(team_home_advantage[home_team]['home_games'], 1)
    h_away_wr = team_home_advantage[home_team]['away_wins'] / max(team_home_advantage[home_team]['away_games'], 1)
    a_home_wr = team_home_advantage[away_team]['home_wins'] / max(team_home_advantage[away_team]['home_games'], 1)
    a_away_wr = team_home_advantage[away_team]['away_wins'] / max(team_home_advantage[away_team]['away_games'], 1)
    
    hfa.append(h_home_wr - h_away_wr - (a_home_wr - a_away_wr))
    
    # Update
    if row['home_win']:
        team_home_advantage[home_team]['home_wins'] += 1
        team_home_advantage[away_team]['away_games'] += 1
    else:
        team_home_advantage[away_team]['away_wins'] += 1
        team_home_advantage[home_team]['home_games'] += 1
    
    team_home_advantage[home_team]['home_games'] += 1
    team_home_advantage[away_team]['away_games'] += 1

features_df['home_field_advantage'] = hfa

# 5. Recent form (last 10 games)
print("✓ Computing recent form...")
recent_wins = []
for idx, row in features_df.iterrows():
    home_team = row['home_team']
    date = row['date']
    
    # Get last 10 home games
    recent = features_df[(features_df['home_team'] == home_team) & (features_df['date'] < date)].tail(10)
    if len(recent) > 0:
        recent_wins.append(recent['home_win'].mean())
    else:
        recent_wins.append(0.5)

features_df['home_recent_form'] = recent_wins

# 6. Days of rest (simplified - using day of week)
print("✓ Adding rest/day features...")
features_df['day_of_week'] = features_df['date'].dt.dayofweek
features_df['month'] = features_df['date'].dt.month

# Remove games with insufficient history
train_df = features_df[features_df['home_historical_wr'] > 0].copy()
print(f"\nUsing {len(train_df)} games with sufficient history")

# Prepare features for modeling
print("\n" + "=" * 70)
print("MODEL TRAINING")
print("=" * 70)

feature_cols = [
    'home_historical_wr', 'away_historical_wr',
    'home_avg_score', 'away_avg_score',
    'days_into_season', 'home_field_advantage',
    'home_recent_form', 'day_of_week', 'month'
]

X = train_df[feature_cols].fillna(0.5)
y = train_df['home_win'].values

# Split data: use 2020-2023 for training, 2024 for testing
train_mask = train_df['year'] < 2024
test_mask = train_df['year'] == 2024

X_train = X[train_mask]
y_train = y[train_mask]
X_test = X[test_mask]
y_test = y[test_mask]

print(f"\nTrain set: {len(X_train)} games (2020-2023)")
print(f"Test set: {len(X_test)} games (2024)")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models
print("\n" + "-" * 70)
print("LOGISTIC REGRESSION")
print("-" * 70)

lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

print(f"Accuracy:  {accuracy_score(y_test, lr_pred):.4f}")
print(f"Precision: {precision_score(y_test, lr_pred):.4f}")
print(f"Recall:    {recall_score(y_test, lr_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, lr_pred):.4f}")
print(f"ROC AUC:   {roc_auc_score(y_test, lr_pred_proba):.4f}")

print("\nFeature Importance:")
for feat, coef in sorted(zip(feature_cols, lr_model.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {feat:25s}: {coef:7.4f}")

# Random Forest
print("\n" + "-" * 70)
print("RANDOM FOREST")
print("-" * 70)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]

print(f"Accuracy:  {accuracy_score(y_test, rf_pred):.4f}")
print(f"Precision: {precision_score(y_test, rf_pred):.4f}")
print(f"Recall:    {recall_score(y_test, rf_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, rf_pred):.4f}")
print(f"ROC AUC:   {roc_auc_score(y_test, rf_pred_proba):.4f}")

print("\nFeature Importance:")
for feat, imp in sorted(zip(feature_cols, rf_model.feature_importances_), key=lambda x: x[1], reverse=True):
    print(f"  {feat:25s}: {imp:7.4f}")

# Gradient Boosting
print("\n" + "-" * 70)
print("GRADIENT BOOSTING")
print("-" * 70)

gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=7, learning_rate=0.1)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_pred_proba = gb_model.predict_proba(X_test)[:, 1]

print(f"Accuracy:  {accuracy_score(y_test, gb_pred):.4f}")
print(f"Precision: {precision_score(y_test, gb_pred):.4f}")
print(f"Recall:    {recall_score(y_test, gb_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, gb_pred):.4f}")
print(f"ROC AUC:   {roc_auc_score(y_test, gb_pred_proba):.4f}")

print("\nFeature Importance:")
for feat, imp in sorted(zip(feature_cols, gb_model.feature_importances_), key=lambda x: x[1], reverse=True):
    print(f"  {feat:25s}: {imp:7.4f}")

# Model comparison
print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

models = {
    'Logistic Regression': (lr_model, lr_pred, lr_pred_proba, X_train_scaled, X_test_scaled),
    'Random Forest': (rf_model, rf_pred, rf_pred_proba, X_train, X_test),
    'Gradient Boosting': (gb_model, gb_pred, gb_pred_proba, X_train, X_test)
}

comparison_data = []
for name, (model, pred, pred_proba, X_tr, X_te) in models.items():
    acc = accuracy_score(y_test, pred)
    pre = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, pred_proba)
    
    # Cross validation on train set
    cv_score = cross_val_score(model, X_tr, y_train, cv=5, scoring='accuracy').mean()
    
    comparison_data.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': pre,
        'Recall': rec,
        'F1': f1,
        'ROC AUC': auc,
        'CV Score': cv_score
    })

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

# Best model
best_model_name = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']
print(f"\n✓ Best Model: {best_model_name}")

# Save models
import pickle
with open('models/lr_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
with open('models/rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
with open('models/gb_model.pkl', 'wb') as f:
    pickle.dump(gb_model, f)

# Save scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save feature names
with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)

print("\n✓ Models saved to models/ directory")

# Visualization
print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Model comparison
ax = axes[0, 0]
comparison_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1']].plot(kind='bar', ax=ax)
ax.set_title('Model Performance Comparison')
ax.set_ylabel('Score')
ax.legend(loc='best')
ax.grid(axis='y', alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 2. ROC Curves
ax = axes[0, 1]
for name, (model, _, pred_proba, _, X_te) in models.items():
    fpr, tpr, _ = roc_curve(y_test, pred_proba)
    auc = roc_auc_score(y_test, pred_proba)
    ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves')
ax.legend(loc='best')
ax.grid(alpha=0.3)

# 3. Feature importance (Random Forest)
ax = axes[1, 0]
importances = sorted(zip(feature_cols, rf_model.feature_importances_), key=lambda x: x[1], reverse=True)
feats, imps = zip(*importances)
ax.barh(range(len(feats)), imps)
ax.set_yticks(range(len(feats)))
ax.set_yticklabels(feats)
ax.set_xlabel('Importance')
ax.set_title('Feature Importance (Random Forest)')
ax.grid(axis='x', alpha=0.3)

# 4. Prediction distribution
ax = axes[1, 1]
ax.hist(gb_pred_proba[y_test == 1], bins=30, alpha=0.6, label='Home Wins', color='green')
ax.hist(gb_pred_proba[y_test == 0], bins=30, alpha=0.6, label='Away Wins', color='red')
ax.set_xlabel('Predicted Probability (Home Win)')
ax.set_ylabel('Frequency')
ax.set_title('Prediction Distribution (Gradient Boosting)')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('models/model_performance.png', dpi=100, bbox_inches='tight')
print("✓ Saved: models/model_performance.png")

print("\n" + "=" * 70)
print("✓ TRAINING COMPLETE")
print("=" * 70)
