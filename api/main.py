"""
FastAPI Application - Football AI Prediction Service
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
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
from core.service_manager import (
    get_service_manager,
    get_prediction_service,
    get_improved_prediction_service,
    get_service_status
)
from core.utils import setup_logging


# Инициализация
logger = setup_logging()
start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager"""
    # Startup
    logger.info("=" * 70)
    logger.info("СТАРТИРАНЕ НА FOOTBALL AI PREDICTION SERVICE")
    logger.info("=" * 70)
    
    service_manager = get_service_manager()
    await service_manager.initialize()
    
    yield
    
    # Shutdown
    logger.info("Спиране на Football AI Prediction Service...")
    await service_manager.cleanup()


app = FastAPI(
    title="Football AI Prediction Service",
    description="AI-powered football match predictions using ESPN data",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Премахнахме глобалното състояние - сега използваме dependency injection


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
async def health_check(
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    """
    Health check endpoint
    
    Returns:
        Health status
    """
    health = prediction_service.health_check()
    
    return HealthResponse(
        status="healthy" if health['models_loaded'] else "unhealthy",
        models_loaded=health['models_loaded'],
        version="1.0.0",
        uptime_seconds=time.time() - start_time
    )


@app.get("/models", response_model=ModelsListResponse, tags=["Models"])
async def list_models(
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    """
    Списък на всички модели
    
    Returns:
        Информация за моделите
    """
    info = prediction_service.get_model_info()
    
    return ModelsListResponse(
        models=info['models'],
        total_models=info['total_models']
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_match(
    match: MatchInput,
    prediction_service: PredictionService = Depends(get_prediction_service)
):
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
    date: Optional[str] = None,
    prediction_service: PredictionService = Depends(get_prediction_service)
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
async def predict_match_improved(
    match: MatchInput,
    improved_service: ImprovedPredictionService = Depends(get_improved_prediction_service)
):
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
    try:
        logger.info(f"Improved prediction request: {match.home_team} vs {match.away_team}")
        
        # Подобрена прогноза с confidence
        result = improved_service.predict_with_confidence(
            home_team=match.home_team,
            away_team=match.away_team,
            league=match.league,
            date=match.date
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Improved prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/features/groups", tags=["Data"])
async def get_feature_groups():
    """
    Информация за feature групите и техните методи за попълване
    
    Returns:
        Подробна информация за всяка feature група
    """
    try:
        from core.feature_validator import FeatureValidator
        
        validator = FeatureValidator()
        groups_info = validator.get_feature_groups_info()
        
        return {
            "feature_groups": groups_info,
            "total_groups": len(groups_info),
            "description": "Feature групи с различни методи за валидиране и попълване"
        }
        
    except Exception as e:
        logger.error(f"Feature groups error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/services/status", tags=["General"])
async def get_services_status(
    status: dict = Depends(get_service_status)
):
    """
    Статус на всички services
    
    Returns:
        Подробна информация за състоянието на services
    """
    return {
        "service_manager": status,
        "uptime_seconds": time.time() - start_time,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/teams", tags=["Data"])
async def list_teams(
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    """
    Списък на всички отбори в системата
    
    Returns:
        Списък с отбори и техните Elo ratings
    """
    
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


@app.get("/teams/validate/{team_name}", tags=["Teams"])
async def validate_team_name(team_name: str, service: PredictionService = Depends(get_prediction_service)):
    """
    Валидира име на отбор и връща информация
    
    Args:
        team_name: Име на отбор за валидация
        
    Returns:
        Информация за валидацията
    """
    validation = service.validate_team(team_name)
    
    return {
        'team_name': team_name,
        'validation': validation,
        'timestamp': datetime.now().isoformat()
    }


@app.get("/teams/resolve/{team_name}", tags=["Teams"])
async def resolve_team_name(team_name: str, service: PredictionService = Depends(get_prediction_service)):
    """
    Резолва име на отбор към стандартизирано име
    
    Args:
        team_name: Име на отбор за резолване
        
    Returns:
        Резолвано име
    """
    resolved = service.resolve_team_name(team_name)
    similar = service.find_similar_teams(team_name, 5)
    
    return {
        'original_name': team_name,
        'resolved_name': resolved,
        'similar_teams': [{'name': name, 'similarity': score} for name, score in similar],
        'timestamp': datetime.now().isoformat()
    }


@app.get("/teams/search/{query}", tags=["Teams"])
async def search_teams(query: str, limit: int = 10, service: PredictionService = Depends(get_prediction_service)):
    """
    Търси отбори по име
    
    Args:
        query: Търсена дума/фраза
        limit: Максимален брой резултати
        
    Returns:
        Списък с подобни отбори
    """
    similar = service.find_similar_teams(query, limit)
    
    return {
        'query': query,
        'results': [{'name': name, 'similarity': score} for name, score in similar],
        'total_results': len(similar),
        'timestamp': datetime.now().isoformat()
    }


@app.get("/stats", tags=["Data"])
async def service_stats(service: PredictionService = Depends(get_prediction_service)):
    """
    Статистики на сървиса
    
    Returns:
        Статистики
    """
    health = service.health_check()
    
    return {
        'service': 'Football AI Prediction Service',
        'version': '1.0.0',
        'uptime_seconds': time.time() - start_time,
        'uptime_hours': (time.time() - start_time) / 3600,
        'models_loaded': health['num_models'],
        'teams_in_database': health['num_teams'],
        'team_resolver_loaded': health.get('team_resolver_loaded', False),
        'features_used': len(service.feature_columns),
        'endpoints': {
            'health': '/health',
            'predict_post': '/predict',
            'predict_get': '/predict/{home_team}/vs/{away_team}',
            'models': '/models',
            'teams': '/teams',
            'teams_validate': '/teams/validate/{team_name}',
            'teams_resolve': '/teams/resolve/{team_name}',
            'teams_search': '/teams/search/{query}',
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
