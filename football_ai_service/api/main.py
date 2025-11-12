"""
FastAPI Application - Football AI Prediction Service
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
from datetime import datetime
from typing import Optional

from api.models import (
    MatchInput,
    PredictionResponse,
    HealthResponse,
    ErrorResponse,
    ModelsListResponse
)
from api.prediction_service import PredictionService
from api.improved_prediction_service import ImprovedPredictionService
from core.utils import setup_logging


# Инициализация
logger = setup_logging()
app = FastAPI(
    title="Football AI Prediction Service",
    description="AI-powered football match predictions using ESPN data",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
prediction_service: Optional[PredictionService] = None
improved_prediction_service: Optional[ImprovedPredictionService] = None
start_time = time.time()


@app.on_event("startup")
async def startup_event():
    """Startup event - зареждане на модели"""
    global prediction_service, improved_prediction_service
    
    logger.info("=" * 70)
    logger.info("СТАРТИРАНЕ НА FOOTBALL AI PREDICTION SERVICE")
    logger.info("=" * 70)
    
    try:
        prediction_service = PredictionService()
        logger.info("✓ Prediction service инициализиран успешно")
    except Exception as e:
        logger.error(f"✗ Грешка при инициализация: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event"""
    logger.info("Спиране на Football AI Prediction Service...")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Грешка: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "service": "Football AI Prediction Service",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check endpoint
    
    Returns:
        Health status
    """
    if prediction_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    health = prediction_service.health_check()
    
    return HealthResponse(
        status="healthy" if health['models_loaded'] else "unhealthy",
        models_loaded=health['models_loaded'],
        version="1.0.0",
        uptime_seconds=time.time() - start_time
    )


@app.get("/models", response_model=ModelsListResponse, tags=["Models"])
async def list_models():
    """
    Списък на всички модели
    
    Returns:
        Информация за моделите
    """
    if prediction_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    info = prediction_service.get_model_info()
    
    return ModelsListResponse(
        models=info['models'],
        total_models=info['total_models']
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_match(match: MatchInput):
    """
    Прогноза за футболен мач
    
    Args:
        match: Информация за мача
    
    Returns:
        Пълна прогноза (1X2, OU2.5, BTTS, FII)
    
    Example:
        ```json
        {
            "home_team": "Manchester United",
            "away_team": "Liverpool",
            "league": "Premier League",
            "date": "2024-03-15"
        }
        ```
    """
    if prediction_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        logger.info(f"Prediction request: {match.home_team} vs {match.away_team}")
        
        # Prediction
        result = prediction_service.predict(
            home_team=match.home_team,
            away_team=match.away_team,
            league=match.league,
            date=match.date
        )
        
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/{home_team}/vs/{away_team}", response_model=PredictionResponse, tags=["Predictions"])
async def predict_match_get(
    home_team: str,
    away_team: str,
    league: Optional[str] = None,
    date: Optional[str] = None
):
    """
    Прогноза за футболен мач (GET endpoint)
    
    Args:
        home_team: Домакин
        away_team: Гост
        league: Лига (optional)
        date: Дата (optional)
    
    Returns:
        Пълна прогноза
    
    Example:
        `/predict/Manchester%20United/vs/Liverpool?league=Premier%20League`
    """
    if prediction_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        logger.info(f"Prediction request (GET): {home_team} vs {away_team}")
        
        result = prediction_service.predict(
            home_team=home_team,
            away_team=away_team,
            league=league,
            date=date
        )
        
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/improved", tags=["Predictions"])
async def predict_match_improved(match: MatchInput):
    """
    Подобрена прогноза за футболен мач с confidence scoring
    
    Тази версия предоставя:
    - Интелигентно търсене на отбори
    - Confidence scoring за качеството на данните
    - Подробни предупреждения за непознати отбори
    - Лигово-базирани default стойности
    
    Args:
        match: Данни за мача
    
    Returns:
        Пълна прогноза с metadata за качеството на данните
    """
    global improved_prediction_service
    
    # Lazy initialization на подобрения сервис
    if improved_prediction_service is None:
        try:
            logger.info("Инициализиране на ImprovedPredictionService...")
            improved_prediction_service = ImprovedPredictionService()
            logger.info("✓ ImprovedPredictionService инициализиран")
        except Exception as e:
            logger.error(f"✗ Грешка при инициализация на ImprovedPredictionService: {e}")
            raise HTTPException(status_code=503, detail="Improved service not available")
    
    try:
        logger.info(f"Improved prediction request: {match.home_team} vs {match.away_team}")
        
        # Подобрена прогноза с confidence
        result = improved_prediction_service.predict_with_confidence(
            home_team=match.home_team,
            away_team=match.away_team,
            league=match.league,
            date=match.date
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Improved prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/teams", tags=["Data"])
async def list_teams():
    """
    Списък на всички отбори в системата
    
    Returns:
        Списък с отбори и техните Elo ratings
    """
    if prediction_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    teams = []
    for team, data in prediction_service.elo_ratings.items():
        # Extract team ID from name (e.g., "Team_363" -> 363)
        team_id = team.split('_')[1] if '_' in team else team
        
        # Get real team name
        real_name = prediction_service.get_team_display_name(team)
        
        # Create display name with more info
        elo_rating = int(data['elo'])
        form_pct = int(data['form'] * 100)
        
        # Rank based on Elo
        if elo_rating >= 1900:
            tier = "Elite"
        elif elo_rating >= 1700:
            tier = "Strong"
        elif elo_rating >= 1500:
            tier = "Average"
        else:
            tier = "Weak"
        
        # Use real name if available, otherwise use Team ID
        if real_name != team:
            display_name = f"{real_name} (Elo: {elo_rating}, {tier})"
        else:
            display_name = f"Team {team_id} (Elo: {elo_rating}, {tier})"
        
        teams.append({
            'name': team,  # Original name for API
            'display_name': display_name,  # Human-readable name with real name
            'real_name': real_name if real_name != team else f"Team {team_id}",
            'team_id': team_id,
            'elo': data['elo'],
            'form': data['form'],
            'tier': tier
        })
    
    # Сортиране по Elo
    teams = sorted(teams, key=lambda x: x['elo'], reverse=True)
    
    return {
        'total_teams': len(teams),
        'teams': teams  # ВСИЧКИ отбори
    }


@app.get("/stats", tags=["Data"])
async def service_stats():
    """
    Статистики на сървиса
    
    Returns:
        Статистики
    """
    if prediction_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    health = prediction_service.health_check()
    
    return {
        'service': 'Football AI Prediction Service',
        'version': '1.0.0',
        'uptime_seconds': time.time() - start_time,
        'uptime_hours': (time.time() - start_time) / 3600,
        'models_loaded': health['num_models'],
        'teams_in_database': health['num_teams'],
        'features_used': len(prediction_service.feature_columns),
        'endpoints': {
            'health': '/health',
            'predict_post': '/predict',
            'predict_get': '/predict/{home_team}/vs/{away_team}',
            'models': '/models',
            'teams': '/teams',
            'stats': '/stats'
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Стартиране на FastAPI сървър...")
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=3000,
        log_level="info"
    )
