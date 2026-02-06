# Use Cases - MLB Game Outcome Prediction System

Real-world scenarios and workflows for using the baseball data and prediction system.

## User Personas

### 1. Data Scientist / ML Engineer
Want to train models, experiment with features, and improve accuracy.

### 2. Baseball Fan / Analyst
Want predictions for upcoming games and team performance insights.

### 3. Developer / Engineer
Want to integrate predictions into an application or API.

### 4. Researcher
Want to analyze historical trends and team performance patterns.

---

## Use Case 1: Fresh Install & First Run

**Goal**: Get the system running and understand what it can do.

**Steps**:

```bash
# 1. Check the README
cat README.md

# 2. See quick reference
python QUICKREF.py

# 3. Fetch fresh data (optional - data is already in repo)
python -m src.scraper

# 4. Analyze the data
python -m src.eda

# 5. Make some predictions
python -m src.predict
```

**Output**: Understanding of data, models, and how to use them.

**Time**: 10-15 minutes

---

## Use Case 2: Data Exploration & Analysis

**Goal**: Understand team performance, home field advantage, seasonal trends.

**Persona**: Researcher, Sports Analyst

**Steps**:

```bash
# Run exploratory analysis
python -m src.eda
```

**Outputs**:
- Home win rate: 52.9% (statistically significant)
- Team rankings (best to worst)
- Year-over-year trends
- Score distributions
- Data quality summary

**What You Learn**:
- Which teams are strongest historically
- How much home field advantage matters
- Scoring patterns
- Data coverage and quality

**Use For**:
- Sports writing
- Statistical analysis
- Understanding team dynamics
- Historical trend analysis

---

## Use Case 3: Train Custom Models

**Goal**: Improve model accuracy with new features or better tuning.

**Persona**: Data Scientist, ML Engineer

**Steps**:

```python
# 1. Load the source code
python -m src.train_model

# 2. Or modify the code to:
#    - Add new features (pitcher stats, rest days, weather)
#    - Try different models (XGBoost, Neural Networks)
#    - Adjust hyperparameters
#    - Change train/test split
```

**Workflow**:

1. Read `src/train_model.py` to understand pipeline
2. Edit feature engineering section
3. Run to train new models
4. Check `models/model_performance.png` for results
5. Compare metrics in console output

**Example Improvements**:
```python
# Add pitcher performance features
features_df['home_pitcher_era'] = ...
features_df['away_pitcher_era'] = ...

# Add rest days
features_df['home_rest_days'] = ...
features_df['away_rest_days'] = ...

# Add weather
features_df['temperature'] = ...
features_df['wind_speed'] = ...
```

**Output**: New trained models in `models/`

---

## Use Case 4: Make Game Predictions

**Goal**: Predict outcomes for specific games.

**Persona**: Baseball Fan, Analyst, Bettor

**Steps**:

```bash
# See sample predictions
python -m src.predict

# Or programmatically:
```

```python
from src.predict import predict_game
import pandas as pd

# Load games and stats
games = pd.read_csv('data/raw/games_2024.csv')

# Make a prediction
result = predict_game(
    home_team='LAD',
    away_team='NYY',
    games_df=games,
    team_stats=team_stats,
    model=model
)

print(f"{result['away_team']} @ {result['home_team']}")
print(f"Prediction: {result['predicted_winner']}")
print(f"Confidence: {result['confidence']:.1f}%")
print(f"Win Probabilities: Home {result['home_win_prob']:.2%}, Away {result['away_win_prob']:.2%}")
```

**Output Example**:
```
NYY @ LAD
Prediction: HOME
Confidence: 64.3%
Win Probabilities: Home 64.3%, Away 35.7%
```

**Use For**:
- Personal predictions
- Casual betting
- Sports discussion
- Game analysis

---

## Use Case 5: Build a Prediction API

**Goal**: Create a web service that returns predictions.

**Persona**: Developer, Backend Engineer

**Implementation**:

```python
# app.py
from flask import Flask, jsonify, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model once at startup
with open('models/gb_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Create feature vector
    features = pd.DataFrame([{
        'home_historical_wr': data['home_wr'],
        'away_historical_wr': data['away_wr'],
        'home_avg_score': data['home_avg_score'],
        'away_avg_score': data['away_avg_score'],
        'days_into_season': data['days'],
        'home_field_advantage': data['hfa'],
        'home_recent_form': data['recent_form'],
        'day_of_week': data['day_of_week'],
        'month': data['month']
    }])
    
    # Predict
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    
    return jsonify({
        'prediction': 'HOME' if pred else 'AWAY',
        'home_win_prob': float(proba[1]),
        'away_win_prob': float(proba[0]),
        'confidence': float(max(proba)) * 100
    })

if __name__ == '__main__':
    app.run(debug=False, port=5000)
```

**Usage**:
```bash
python app.py
```

**API Endpoint**:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "home_wr": 0.55,
    "away_wr": 0.50,
    "home_avg_score": 4.5,
    "away_avg_score": 4.3,
    "days": 150,
    "hfa": 0.08,
    "recent_form": 0.60,
    "day_of_week": 2,
    "month": 7
  }'
```

**Response**:
```json
{
  "prediction": "HOME",
  "home_win_prob": 0.643,
  "away_win_prob": 0.357,
  "confidence": 64.3
}
```

---

## Use Case 6: Batch Predictions for Season

**Goal**: Predict all upcoming games for a season.

**Persona**: Analyst, Journalist

**Implementation**:

```python
import pandas as pd
from src.predict import predict_game

# Load upcoming games
upcoming = pd.read_csv('upcoming_games.csv')  # home_team, away_team

# Load historical data for features
games = pd.concat([
    pd.read_csv(f'data/raw/games_{y}.csv')
    for y in [2020, 2021, 2022, 2023, 2024]
], ignore_index=True)

# Make predictions for all
predictions = []
for _, game in upcoming.iterrows():
    pred = predict_game(
        game['home_team'],
        game['away_team'],
        games,
        model=model
    )
    predictions.append(pred)

# Save results
results_df = pd.DataFrame(predictions)
results_df.to_csv('season_predictions.csv', index=False)

# Analyze
print(f"Predicted HOME wins: {(results_df['predicted_winner']=='HOME').sum()}")
print(f"Predicted AWAY wins: {(results_df['predicted_winner']=='AWAY').sum()}")
print(f"Average confidence: {results_df['confidence'].mean():.1f}%")
```

**Output**: `season_predictions.csv` with:
- home_team, away_team
- predicted_winner
- confidence
- win probabilities

---

## Use Case 7: Model Performance Tracking

**Goal**: Monitor how well predictions perform over time.

**Persona**: Data Scientist, ML Engineer

**Implementation**:

```python
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

# Load model
with open('models/gb_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load test set from 2024
test_games = pd.read_csv('data/raw/games_2024.csv')
test_games['actual_winner'] = (test_games['home_score'] > test_games['away_score']).astype(int)

# Make predictions
predictions = []
for _, game in test_games.iterrows():
    # Feature engineering here
    pred = model.predict([features])[0]
    predictions.append(pred)

# Compare
accuracy = accuracy_score(test_games['actual_winner'], predictions)
print(f"Accuracy: {accuracy:.2%}")

# Analyze by team
by_team = pd.DataFrame({
    'home_team': test_games['home_team'],
    'prediction': predictions,
    'actual': test_games['actual_winner']
})

for team in by_team['home_team'].unique():
    team_data = by_team[by_team['home_team'] == team]
    acc = (team_data['prediction'] == team_data['actual']).mean()
    print(f"{team}: {acc:.2%}")
```

**Output**: Accuracy by team, season, month

---

## Use Case 8: Feature Importance Analysis

**Goal**: Understand which factors matter most for predictions.

**Persona**: Researcher, Analyst

**Implementation**:

```python
import pickle

# Load best model
with open('models/gb_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/feature_names.pkl', 'rb') as f:
    features = pickle.load(f)

# Get importance
importance = model.feature_importances_

# Display
for feat, imp in sorted(zip(features, importance), 
                        key=lambda x: x[1], reverse=True):
    print(f"{feat:30s}: {imp:6.2%}")
```

**Output**:
```
home_historical_wr           : 17.54%
home_field_advantage         : 16.05%
away_avg_score               : 15.71%
home_avg_score               : 15.53%
away_historical_wr           : 15.40%
days_into_season             : 9.20%
home_recent_form             : 4.86%
day_of_week                  : 4.12%
month                        : 1.60%
```

**Insights**:
- Team quality (win rate) most important
- Home field advantage matters significantly
- Scoring averages predictive
- Day/month have minimal impact

---

## Use Case 9: Compare Models

**Goal**: Understand trade-offs between different models.

**Persona**: Data Scientist, ML Engineer

**Implementation**:

```python
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load all 3 models
models = {
    'Logistic Regression': pickle.load(open('models/lr_model.pkl', 'rb')),
    'Random Forest': pickle.load(open('models/rf_model.pkl', 'rb')),
    'Gradient Boosting': pickle.load(open('models/gb_model.pkl', 'rb'))
}

# Load test data with features
X_test = ...  # Feature matrix
y_test = ...  # Actual outcomes

# Evaluate each
for name, model in models.items():
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    
    print(f"{name:25s}: Acc={acc:.2%}, Prec={prec:.2%}, Rec={rec:.2%}, F1={f1:.3f}")
```

**Decision Criteria**:
- **Logistic Regression**: Lightweight, interpretable, highest recall
- **Random Forest**: Balanced, robust, good for production
- **Gradient Boosting**: Best accuracy, good balance

---

## Use Case 10: Integrate with Existing System

**Goal**: Use predictions in a larger application.

**Persona**: Developer, Backend Engineer

**Implementation**:

```python
# betting_app.py
from src.predict import predict_game
import pandas as pd

class BettingOdds:
    def __init__(self, model_path='models/gb_model.pkl'):
        self.model = self._load_model(model_path)
        self.games = pd.concat([
            pd.read_csv(f'data/raw/games_{y}.csv')
            for y in [2020, 2021, 2022, 2023, 2024]
        ])
    
    def get_prediction(self, home_team, away_team):
        return predict_game(
            home_team, away_team,
            self.games, model=self.model
        )
    
    def recommend_bet(self, home_team, away_team, odds):
        pred = self.get_prediction(home_team, away_team)
        
        if pred['predicted_winner'] == 'HOME':
            # Calculate implied probability from odds
            implied_prob = 1 / odds['home']
            
            # If model probability > implied, it's undervalued
            if pred['home_win_prob'] > implied_prob:
                return {
                    'recommendation': 'BET_HOME',
                    'edge': pred['home_win_prob'] - implied_prob,
                    'confidence': pred['confidence']
                }
        
        return {'recommendation': 'SKIP', 'edge': 0}

# Usage
odds_engine = BettingOdds()
rec = odds_engine.recommend_bet('LAD', 'NYY', 
                                 {'home': 1.45, 'away': 2.80})
print(rec)
```

---

## Use Case 11: Data Quality & Validation

**Goal**: Ensure data integrity before modeling.

**Persona**: Data Engineer, QA Engineer

**Implementation**:

```python
import pandas as pd

# Load all data
games = pd.concat([
    pd.read_csv(f'data/raw/games_{y}.csv')
    for y in [2020, 2021, 2022, 2023, 2024]
])

# Validation checks
checks = {
    'No nulls': games.isnull().sum().sum() == 0,
    'Valid dates': pd.to_datetime(games['date'], errors='coerce').notna().all(),
    'Score >= 0': (games['home_score'] >= 0).all() and (games['away_score'] >= 0).all(),
    'Valid teams': games['home_team'].isin(['LAD', 'NYY', ...]).all(),
    'No duplicates': games.drop_duplicates().shape[0] == games.shape[0]
}

for check, result in checks.items():
    status = "✓" if result else "✗"
    print(f"{status} {check}")
```

---

## Use Case 12: Learning & Education

**Goal**: Understand ML pipeline from data to predictions.

**Persona**: Student, Beginner, Learner

**Path**:

1. **Read Documentation**
   - `README.md` - Overview
   - `src/README.md` - Code structure
   - `models/README.md` - Model details

2. **Explore Data**
   ```bash
   python -m src.eda
   ```

3. **Read Code**
   ```bash
   cat src/scraper.py  # How to fetch data
   cat src/eda.py      # How to analyze
   cat src/train_model.py  # How to train
   cat src/predict.py  # How to predict
   ```

4. **Experiment**
   - Modify features
   - Try different models
   - Compare results

5. **Build Your Own**
   - Create custom features
   - Train on different data
   - Deploy a model

---

## Quick Reference by Role

### Baseball Fan 🏟️
```bash
python -m src.eda       # See team stats
python -m src.predict   # Get predictions
```

### Data Scientist 🔬
```bash
python -m src.eda           # Analyze data
python -m src.train_model   # Experiment with models
# Edit src/train_model.py for custom features/models
```

### Developer 👨‍💻
```python
from src.predict import predict_game
result = predict_game(home_team, away_team, games, model)
```

### Researcher 📊
```bash
python -m src.eda      # Get statistics
# Use data in data/raw/ for analysis
```

### Analyst 📈
```bash
python -m src.predict  # Get predictions
# Batch predictions in upcoming_games.csv
```

---

## Summary

| Use Case | Time | Difficulty | Tools |
|----------|------|-----------|-------|
| Fresh Install | 10 min | Easy | CLI |
| Data Exploration | 5 min | Easy | CLI |
| Train Models | 30 min | Medium | Python |
| Make Predictions | 5 min | Easy | CLI/Python |
| Build API | 2 hours | Hard | Flask/FastAPI |
| Batch Predictions | 1 hour | Medium | Python |
| Performance Tracking | 1 hour | Medium | Python/Pandas |
| Feature Analysis | 30 min | Medium | Python |
| Model Comparison | 30 min | Medium | Python |
| System Integration | 4 hours | Hard | Python/Integration |
| Data Validation | 30 min | Medium | Python |
| Learning/Education | Varies | Varies | All |

---

**Last Updated**: February 4, 2026
