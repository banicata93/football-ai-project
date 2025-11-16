#!/usr/bin/env python3
"""
Scoreline API Endpoint

Provides REST API endpoint for scoreline probability predictions.
ADDITIVE - does not modify existing API structure.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.scoreline_engine import ScorelineProbabilityEngine
from core.data_loader import ESPNDataLoader
from core.utils import setup_logging


# Pydantic models for request/response
class ScorelineRequest(BaseModel):
    home_team: str
    away_team: str
    league: Optional[str] = None


class ScorelineMatrix(BaseModel):
    matrix: Dict[str, float]
    summary: Dict[str, float]
    features_used: Dict[str, float]
    engine_version: str
    matrix_size: str
    total_combinations: int


class ScorelineResponse(BaseModel):
    success: bool
    data: Optional[ScorelineMatrix] = None
    error: Optional[str] = None
    timestamp: str
    processing_time_ms: float


# Initialize router
router = APIRouter(prefix="/scoreline", tags=["scoreline"])

# Initialize components (lazy loading)
scoreline_engine = None
data_loader = None
logger = setup_logging()


def get_scoreline_engine() -> ScorelineProbabilityEngine:
    """Get scoreline engine (lazy loading)"""
    global scoreline_engine
    if scoreline_engine is None:
        try:
            scoreline_engine = ScorelineProbabilityEngine()
            logger.info("✅ Scoreline engine initialized")
        except Exception as e:
            logger.error(f"❌ Error initializing scoreline engine: {e}")
            raise HTTPException(status_code=500, detail="Scoreline engine initialization failed")
    return scoreline_engine


def get_data_loader() -> ESPNDataLoader:
    """Get data loader (lazy loading)"""
    global data_loader
    if data_loader is None:
        try:
            data_loader = ESPNDataLoader()
            logger.info("✅ Data loader initialized")
        except Exception as e:
            logger.error(f"❌ Error initializing data loader: {e}")
            raise HTTPException(status_code=500, detail="Data loader initialization failed")
    return data_loader


@router.get("/", response_model=ScorelineResponse)
async def predict_scoreline(
    home_team: str = Query(..., description="Home team name"),
    away_team: str = Query(..., description="Away team name"),
    league: Optional[str] = Query(None, description="League name (optional)")
) -> ScorelineResponse:
    """
    Predict scoreline probability distribution
    
    Returns complete matrix of scoreline probabilities (0-0, 1-0, etc.)
    plus aggregated metrics (1X2, BTTS, OU2.5).
    """
    start_time = datetime.now()
    
    try:
        logger.info(f"🎯 Scoreline prediction request: {home_team} vs {away_team}")
        
        # Validate input
        if not home_team or not away_team:
            raise HTTPException(status_code=400, detail="Home team and away team are required")
        
        if home_team.lower() == away_team.lower():
            raise HTTPException(status_code=400, detail="Home team and away team must be different")
        
        # Get components
        engine = get_scoreline_engine()
        loader = get_data_loader()
        
        # Load historical data
        df = loader.load_fixtures()
        if df is None or df.empty:
            logger.warning("⚠️ No historical data available")
            df = None
        
        # Get scoreline predictions
        result = engine.get_scoreline_probabilities(
            home_team=home_team,
            away_team=away_team,
            league=league,
            df=df
        )
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        # Build response
        scoreline_data = ScorelineMatrix(
            matrix=result['matrix'],
            summary=result['summary'],
            features_used=result.get('features_used', {}),
            engine_version=result.get('engine_version', 'scoreline_v1'),
            matrix_size=result.get('matrix_size', '5x5'),
            total_combinations=result.get('total_combinations', 25)
        )
        
        response = ScorelineResponse(
            success=True,
            data=scoreline_data,
            timestamp=end_time.isoformat(),
            processing_time_ms=round(processing_time, 2)
        )
        
        logger.info(f"✅ Scoreline prediction completed in {processing_time:.1f}ms")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in scoreline prediction: {e}")
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        return ScorelineResponse(
            success=False,
            error=str(e),
            timestamp=end_time.isoformat(),
            processing_time_ms=round(processing_time, 2)
        )


@router.post("/", response_model=ScorelineResponse)
async def predict_scoreline_post(request: ScorelineRequest) -> ScorelineResponse:
    """
    Predict scoreline probability distribution (POST method)
    
    Alternative POST endpoint for scoreline predictions.
    """
    return await predict_scoreline(
        home_team=request.home_team,
        away_team=request.away_team,
        league=request.league
    )


@router.get("/top/{n}", response_model=Dict[str, Any])
async def get_top_scorelines(
    n: int,
    home_team: str = Query(..., description="Home team name"),
    away_team: str = Query(..., description="Away team name"),
    league: Optional[str] = Query(None, description="League name (optional)")
) -> Dict[str, Any]:
    """
    Get top N most likely scorelines
    
    Returns the most probable scorelines with their probabilities.
    """
    try:
        logger.info(f"🎯 Top {n} scorelines request: {home_team} vs {away_team}")
        
        # Get full prediction
        full_response = await predict_scoreline(home_team, away_team, league)
        
        if not full_response.success:
            raise HTTPException(status_code=500, detail=full_response.error)
        
        # Extract and sort scorelines
        matrix = full_response.data.matrix
        sorted_scorelines = sorted(matrix.items(), key=lambda x: x[1], reverse=True)
        top_scorelines = sorted_scorelines[:n]
        
        # Calculate cumulative probability
        cumulative_prob = sum(prob for _, prob in top_scorelines)
        
        return {
            "success": True,
            "top_scorelines": [
                {"scoreline": scoreline, "probability": prob}
                for scoreline, prob in top_scorelines
            ],
            "cumulative_probability": cumulative_prob,
            "coverage_percentage": cumulative_prob * 100,
            "total_scorelines": len(matrix),
            "match": f"{home_team} vs {away_team}",
            "league": league,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting top scorelines: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary", response_model=Dict[str, Any])
async def get_scoreline_summary(
    home_team: str = Query(..., description="Home team name"),
    away_team: str = Query(..., description="Away team name"),
    league: Optional[str] = Query(None, description="League name (optional)")
) -> Dict[str, Any]:
    """
    Get scoreline summary metrics only
    
    Returns aggregated metrics (1X2, BTTS, OU2.5) without full matrix.
    """
    try:
        logger.info(f"🎯 Scoreline summary request: {home_team} vs {away_team}")
        
        # Get full prediction
        full_response = await predict_scoreline(home_team, away_team, league)
        
        if not full_response.success:
            raise HTTPException(status_code=500, detail=full_response.error)
        
        # Extract summary
        summary = full_response.data.summary
        
        return {
            "success": True,
            "summary": summary,
            "match": f"{home_team} vs {away_team}",
            "league": league,
            "engine_version": full_response.data.engine_version,
            "timestamp": datetime.now().isoformat(),
            "processing_time_ms": full_response.processing_time_ms
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting scoreline summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def scoreline_health_check() -> Dict[str, Any]:
    """
    Health check for scoreline endpoint
    
    Returns status of scoreline engine components.
    """
    try:
        # Check engine initialization
        engine = get_scoreline_engine()
        loader = get_data_loader()
        
        # Basic functionality test
        test_result = engine.get_scoreline_probabilities(
            home_team="Test_Team_A",
            away_team="Test_Team_B",
            league="Test_League",
            df=None
        )
        
        return {
            "status": "healthy",
            "scoreline_engine": "operational",
            "data_loader": "operational",
            "test_prediction": "successful",
            "engine_version": test_result.get('engine_version', 'unknown'),
            "matrix_size": test_result.get('matrix_size', 'unknown'),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Scoreline health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# Example usage function
def example_usage():
    """
    Example of how to use the scoreline endpoint
    """
    print("🎯 Scoreline API Endpoint Examples")
    print("=" * 50)
    
    print("\n1. GET Request:")
    print("   GET /scoreline?home_team=Manchester%20City&away_team=Liverpool&league=Premier%20League")
    
    print("\n2. POST Request:")
    print("   POST /scoreline")
    print("   Body: {")
    print('     "home_team": "Manchester City",')
    print('     "away_team": "Liverpool",')
    print('     "league": "Premier League"')
    print("   }")
    
    print("\n3. Top Scorelines:")
    print("   GET /scoreline/top/5?home_team=Arsenal&away_team=Chelsea")
    
    print("\n4. Summary Only:")
    print("   GET /scoreline/summary?home_team=Barcelona&away_team=Real%20Madrid")
    
    print("\n5. Health Check:")
    print("   GET /scoreline/health")
    
    print("\n📊 Example Response:")
    example_response = {
        "success": True,
        "data": {
            "matrix": {
                "0-0": 0.089,
                "1-0": 0.134,
                "0-1": 0.098,
                "1-1": 0.156,
                "2-0": 0.087,
                "2-1": 0.112
            },
            "summary": {
                "p_home": 0.456,
                "p_draw": 0.267,
                "p_away": 0.277,
                "btts_prob": 0.543,
                "over25_prob": 0.612
            },
            "engine_version": "scoreline_v1",
            "matrix_size": "5x5"
        },
        "timestamp": "2025-11-13T20:00:00",
        "processing_time_ms": 45.2
    }
    
    print(json.dumps(example_response, indent=2))


if __name__ == "__main__":
    example_usage()
