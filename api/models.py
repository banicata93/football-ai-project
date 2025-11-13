"""
Pydantic models за FastAPI
"""

from typing import Optional, Dict, List
from pydantic import BaseModel, Field
from datetime import datetime


class MatchInput(BaseModel):
    """Input за prediction request"""
    
    home_team: str = Field(..., description="Име на домакин")
    away_team: str = Field(..., description="Име на гост")
    league: Optional[str] = Field(None, description="Лига")
    date: Optional[str] = Field(None, description="Дата на мача (YYYY-MM-DD)")
    
    # Optional historical stats (ако са налични)
    home_elo: Optional[float] = Field(None, description="Home team Elo rating")
    away_elo: Optional[float] = Field(None, description="Away team Elo rating")
    home_form: Optional[float] = Field(None, description="Home team form")
    away_form: Optional[float] = Field(None, description="Away team form")
    
    class Config:
        json_schema_extra = {
            "example": {
                "home_team": "Manchester United",
                "away_team": "Liverpool",
                "league": "Premier League",
                "date": "2024-03-15"
            }
        }


class Prediction1X2(BaseModel):
    """1X2 prediction"""
    
    prob_home_win: float = Field(..., description="Вероятност за победа на домакина")
    prob_draw: float = Field(..., description="Вероятност за равенство")
    prob_away_win: float = Field(..., description="Вероятност за победа на госта")
    predicted_outcome: str = Field(..., description="Прогнозиран резултат (1/X/2)")
    confidence: float = Field(..., description="Увереност (0-1)")


class PredictionOU25(BaseModel):
    """Over/Under 2.5 prediction"""
    
    prob_over: float = Field(..., description="Вероятност за Over 2.5")
    prob_under: float = Field(..., description="Вероятност за Under 2.5")
    predicted_outcome: str = Field(..., description="Over или Under")
    confidence: float = Field(..., description="Увереност (0-1)")


class PredictionBTTS(BaseModel):
    """Both Teams To Score prediction"""
    
    prob_yes: float = Field(..., description="Вероятност за BTTS Yes")
    prob_no: float = Field(..., description="Вероятност за BTTS No")
    predicted_outcome: str = Field(..., description="Yes или No")
    confidence: float = Field(..., description="Увереност (0-1)")


class FIIScore(BaseModel):
    """Football Intelligence Index"""
    
    score: float = Field(..., description="FII Score (0-10)")
    confidence_level: str = Field(..., description="Low/Medium/High")
    components: Dict[str, float] = Field(..., description="FII компоненти")


class PredictionResponse(BaseModel):
    """Пълен prediction response"""
    
    match_info: Dict[str, str] = Field(..., description="Информация за мача")
    prediction_1x2: Prediction1X2
    prediction_ou25: PredictionOU25
    prediction_btts: PredictionBTTS
    fii: FIIScore
    model_versions: Dict[str, str] = Field(..., description="Версии на моделите")
    model_sources: Optional[Dict[str, str]] = Field(None, description="Източници на моделите")
    timestamp: str = Field(..., description="Timestamp на prediction")


class HealthResponse(BaseModel):
    """Health check response"""
    
    status: str = Field(..., description="healthy/unhealthy")
    models_loaded: bool = Field(..., description="Дали моделите са заредени")
    version: str = Field(..., description="API версия")
    uptime_seconds: float = Field(..., description="Uptime в секунди")


class ErrorResponse(BaseModel):
    """Error response"""
    
    error: str = Field(..., description="Error съобщение")
    detail: Optional[str] = Field(None, description="Детайли за грешката")
    timestamp: str = Field(..., description="Timestamp")


class ModelInfo(BaseModel):
    """Model information"""
    
    model_name: str
    version: str
    trained_date: str
    accuracy: Optional[float] = None
    metrics: Optional[Dict[str, float]] = None


class ModelsListResponse(BaseModel):
    """List of all models"""
    
    models: List[ModelInfo]
    total_models: int
