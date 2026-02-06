"""
FastAPI deployment for baseball predictions
Provides REST API for game outcome predictions
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pickle
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Baseball Predictions API",
    description="Predict MLB game outcomes",
    version="1.0.0"
)

# Data models
class PredictionRequest(BaseModel):
    """Request model for predictions"""
    home_team: str
    away_team: str
    
    class Config:
        example = {
            "home_team": "LAD",
            "away_team": "NYY"
        }


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    home_team: str
    away_team: str
    predicted_winner: str
    home_win_prob: float
    away_win_prob: float
    confidence: float
    
    class Config:
        example = {
            "home_team": "LAD",
            "away_team": "NYY",
            "predicted_winner": "HOME",
            "home_win_prob": 0.643,
            "away_win_prob": 0.357,
            "confidence": 64.3
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model: str
    version: str


# Global state
model = None
games_df = None

# Model paths
MODEL_PATH = Path(__file__).parent / 'models' / 'gb_model.pkl'
DATA_DIR = Path(__file__).parent / 'data' / 'raw'


@app.on_event("startup")
async def load_model():
    """Load model and data on startup"""
    global model, games_df
    
    try:
        # Load trained model
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"✓ Model loaded: {MODEL_PATH}")
        
        # Load games data for stats
        csv_files = list(DATA_DIR.glob('games_*.csv'))
        if csv_files:
            games_df = pd.concat([
                pd.read_csv(f) for f in sorted(csv_files)
            ], ignore_index=True)
            logger.info(f"✓ Loaded {len(games_df)} games from {len(csv_files)} files")
        else:
            logger.warning("⚠️ No game data files found")
    
    except Exception as e:
        logger.error(f"✗ Startup failed: {e}")
        raise


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    logger.info("Shutting down...")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model="gradient_boosting",
        version="1.0.0"
    )


@app.get("/", tags=["Info"])
async def root():
    """API information"""
    return {
        "name": "Baseball Predictions API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "teams": "/teams"
        }
    }


@app.get("/teams", tags=["Info"])
async def get_teams():
    """Get list of available teams"""
    if games_df is None:
        return {"teams": []}
    
    teams = sorted(games_df['home_team'].unique().tolist())
    return {
        "teams": teams,
        "count": len(teams)
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: PredictionRequest):
    """
    Predict game outcome
    
    **Parameters:**
    - `home_team`: Home team abbreviation (e.g., "LAD")
    - `away_team`: Away team abbreviation (e.g., "NYY")
    
    **Returns:**
    - Predicted winner (HOME or AWAY)
    - Win probabilities
    - Confidence score
    """
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if games_df is None:
        raise HTTPException(status_code=500, detail="Game data not available")
    
    try:
        # Validate teams
        home_team = request.home_team.upper()
        away_team = request.away_team.upper()
        
        if home_team == away_team:
            raise HTTPException(
                status_code=400,
                detail="Home and away teams must be different"
            )
        
        # Get team statistics
        home_stats = _get_team_stats(home_team, games_df)
        away_stats = _get_team_stats(away_team, games_df)
        
        if not home_stats or not away_stats:
            raise HTTPException(
                status_code=404,
                detail=f"Team statistics not found for {home_team} or {away_team}"
            )
        
        # Create feature vector
        features = _create_features(home_stats, away_stats)
        
        # Make prediction
        pred = model.predict([features])[0]
        proba = model.predict_proba([features])[0]
        
        return PredictionResponse(
            home_team=home_team,
            away_team=away_team,
            predicted_winner="HOME" if pred else "AWAY",
            home_win_prob=float(proba[1]),
            away_win_prob=float(proba[0]),
            confidence=float(max(proba)) * 100
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


def _get_team_stats(team: str, df: pd.DataFrame) -> dict:
    """Get team statistics"""
    try:
        team_data = df[df['home_team'] == team]
        
        if len(team_data) == 0:
            return None
        
        wins = (team_data['home_score'] > team_data['away_score']).sum()
        games = len(team_data)
        win_rate = wins / games if games > 0 else 0.5
        avg_score = team_data['home_score'].mean()
        avg_opp_score = team_data['away_score'].mean()
        
        return {
            'team': team,
            'games': games,
            'wins': wins,
            'win_rate': win_rate,
            'avg_score': avg_score,
            'avg_opp_score': avg_opp_score
        }
    except Exception as e:
        logger.error(f"Error getting stats for {team}: {e}")
        return None


def _create_features(home_stats: dict, away_stats: dict) -> list:
    """Create feature vector from team stats"""
    
    def safe_value(val, default=0.5):
        """Replace NaN/None with default"""
        if val is None or (isinstance(val, float) and val != val):
            return default
        return val
    
    return [
        safe_value(home_stats['win_rate'], 0.500),      # home_historical_wr
        safe_value(away_stats['win_rate'], 0.500),      # away_historical_wr
        safe_value(home_stats['avg_score'], 4.5),       # home_avg_score
        safe_value(away_stats['avg_score'], 4.5),       # away_avg_score
        0.5,                                              # days_into_season
        0.08,                                             # home_field_advantage
        safe_value(home_stats['win_rate'], 0.500),      # home_recent_form
        2,                                                # day_of_week
        7                                                 # month
    ]


@app.get("/stats/{team}", tags=["Stats"])
async def get_team_stats(team: str):
    """Get statistics for a specific team"""
    
    if games_df is None:
        raise HTTPException(status_code=500, detail="Game data not available")
    
    stats = _get_team_stats(team.upper(), games_df)
    
    if not stats:
        raise HTTPException(
            status_code=404,
            detail=f"Team not found: {team}"
        )
    
    return stats


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
