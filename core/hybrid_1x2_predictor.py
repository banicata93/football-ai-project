#!/usr/bin/env python3
"""
Hybrid 1X2 Predictor - Central inference orchestrator for Hybrid 1X2

Combines multiple prediction sources:
- ML binary models (home/draw/away)
- Scoreline matrix → derive P(H), P(X), P(A)
- Poisson v2 → derive P(H), P(X), P(A)
- Draw Specialist → single probability injection for Draw

ADDITIVE - does not modify existing model logic.
"""

import sys
import os
import yaml
import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.utils import setup_logging
from core.hybrid_integration import HybridIntegrator
from core.ml_utils import align_features
from core.scoreline_engine import ScorelineProbabilityEngine
from core.poisson_v2 import PoissonV2Model
from core.draw_predictor import DrawPredictor


class Hybrid1X2Predictor:
    """
    Central inference orchestrator for Hybrid 1X2 predictions
    
    Combines ML, Scoreline, Poisson, and Draw Specialist models
    with intelligent weighting and calibration.
    """
    
    def __init__(self, config_path: str = "config/hybrid_1x2_config.yaml"):
        """
        Initialize hybrid 1X2 predictor
        
        Args:
            config_path: Path to hybrid configuration file
        """
        self.config_path = Path(config_path)
        self.logger = setup_logging()
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize components
        self.hybrid_integrator = HybridIntegrator(self.config)
        self.ml_model = None
        self.ml_feature_list = None
        self.ml_calibrators = None
        self.scoreline_engine = None
        self.poisson_model = None
        self.draw_predictor = None
        
        # Load hybrid artifacts
        self.hybrid_calibrator = None
        self.hybrid_weights = None
        
        self._initialize_components()
        self._load_hybrid_artifacts()
        
        self.logger.info("🎯 Hybrid 1X2 Predictor initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load hybrid configuration"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                return config.get('hybrid_1x2_config', {})
            else:
                self.logger.warning(f"⚠️ Config file not found: {self.config_path}")
                return self._get_default_config()
        except Exception as e:
            self.logger.error(f"❌ Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'weights': {
                'ml': 0.45,
                'scoreline': 0.25,
                'poisson': 0.20,
                'draw_specialist': 0.10
            },
            'calibration': {
                'enabled': True,
                'method': 'temperature'
            },
            'leagues': {
                'use_per_league_weights': True
            },
            'fallback': {
                'use_v2_if_missing': True
            }
        }
    
    def _initialize_components(self):
        """Initialize all prediction components"""
        try:
            # Load ML 1X2 model
            self._load_ml_model()
            
            # Initialize Scoreline Engine
            self._initialize_scoreline_engine()
            
            # Initialize Poisson v2
            self._initialize_poisson_model()
            
            # Initialize Draw Specialist
            self._initialize_draw_predictor()
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing components: {e}")
    
    def _load_ml_model(self):
        """Load ML 1X2 model and calibrators"""
        try:
            model_dir = Path("models/model_1x2_v1")
            
            # Load model
            model_file = model_dir / "1x2_model.pkl"
            if model_file.exists():
                self.ml_model = joblib.load(model_file)
                self.logger.info("✅ ML 1X2 model loaded")
            
            # Load feature list
            feature_file = model_dir / "feature_list.json"
            if feature_file.exists():
                import json
                with open(feature_file, 'r') as f:
                    self.ml_feature_list = json.load(f)
                self.logger.info("✅ ML feature list loaded")
            
            # Load calibrators
            self._load_ml_calibrators(model_dir)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error loading ML model: {e}")
            self.ml_model = None
    
    def _load_ml_calibrators(self, model_dir: Path):
        """Load ML calibrators"""
        try:
            self.ml_calibrators = {}
            class_names = ['1', 'X', '2']
            
            for class_name in class_names:
                calibrator_file = model_dir / f"calibrator_{class_name}.pkl"
                if calibrator_file.exists():
                    self.ml_calibrators[class_name] = joblib.load(calibrator_file)
                else:
                    self.ml_calibrators[class_name] = None
            
            if all(cal is not None for cal in self.ml_calibrators.values()):
                self.logger.info("✅ ML calibrators loaded")
            else:
                self.logger.warning("⚠️ Some ML calibrators missing")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error loading ML calibrators: {e}")
            self.ml_calibrators = {}
    
    def _initialize_scoreline_engine(self):
        """Initialize Scoreline Engine"""
        try:
            self.scoreline_engine = ScorelineProbabilityEngine()
            self.logger.info("✅ Scoreline engine initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ Scoreline engine not available: {e}")
            self.scoreline_engine = None
    
    def _initialize_poisson_model(self):
        """Initialize Poisson v2 model"""
        try:
            self.poisson_model = PoissonV2Model()
            self.logger.info("✅ Poisson v2 model initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ Poisson v2 model not available: {e}")
            self.poisson_model = None
    
    def _initialize_draw_predictor(self):
        """Initialize Draw Specialist"""
        try:
            self.draw_predictor = DrawPredictor()
            self.logger.info("✅ Draw predictor initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ Draw predictor not available: {e}")
            self.draw_predictor = None
    
    def _load_hybrid_artifacts(self):
        """Load hybrid-specific artifacts"""
        try:
            hybrid_dir = Path("models/1x2_hybrid_v1")
            
            # Load hybrid calibrator
            calibrator_file = hybrid_dir / "hybrid_calibrator.pkl"
            if calibrator_file.exists():
                self.hybrid_calibrator = joblib.load(calibrator_file)
                self.logger.info("✅ Hybrid calibrator loaded")
            
            # Load hybrid weights
            weights_file = hybrid_dir / "hybrid_weights.json"
            if weights_file.exists():
                import json
                with open(weights_file, 'r') as f:
                    self.hybrid_weights = json.load(f)
                self.logger.info("✅ Hybrid weights loaded")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Hybrid artifacts not available: {e}")
    
    def predict_hybrid_1x2(self, 
                          features: pd.DataFrame,
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate hybrid 1X2 prediction
        
        Args:
            features: Match features DataFrame
            context: Context info (home_team, away_team, league, etc.)
            
        Returns:
            Hybrid 1X2 prediction with probabilities and metadata
        """
        try:
            self.logger.info(f"🎯 Generating hybrid 1X2 prediction")
            
            # Extract context
            home_team = context.get('home_team', 'Unknown')
            away_team = context.get('away_team', 'Unknown')
            league = context.get('league', 'default')
            
            # Get predictions from all sources
            ml_probs = self._get_ml_probabilities(features, league)
            scoreline_probs = self._get_scoreline_probabilities(home_team, away_team, league, context)
            poisson_probs = self._get_poisson_probabilities(home_team, away_team, context)
            draw_specialist_prob = self._get_draw_specialist_probability(home_team, away_team, league, context)
            
            # Combine probabilities using hybrid integrator
            combined_probs = self.hybrid_integrator.combine_probabilities(
                ml_probs=ml_probs,
                scoreline_probs=scoreline_probs,
                poisson_probs=poisson_probs,
                draw_specialist_prob=draw_specialist_prob,
                league=league,
                weights=self.hybrid_weights
            )
            
            # Apply calibration if enabled
            if self.config.get('calibration', {}).get('enabled', True):
                calibrated_probs = self._apply_hybrid_calibration(combined_probs)
            else:
                calibrated_probs = combined_probs
            
            # Build result
            result = {
                'prob_home_win': float(calibrated_probs.get('1', 1/3)),
                'prob_draw': float(calibrated_probs.get('X', 1/3)),
                'prob_away_win': float(calibrated_probs.get('2', 1/3)),
                'predicted_outcome': max(calibrated_probs, key=calibrated_probs.get),
                'confidence': float(max(calibrated_probs.values())),
                'calibrated': self.config.get('calibration', {}).get('enabled', True),
                'hybrid_used': True,
                'sources_used': {
                    'ml': ml_probs is not None,
                    'scoreline': scoreline_probs is not None,
                    'poisson': poisson_probs is not None,
                    'draw_specialist': draw_specialist_prob is not None
                },
                'components': {
                    'ml': ml_probs,
                    'scoreline': scoreline_probs,
                    'poisson': poisson_probs,
                    'draw_specialist': draw_specialist_prob
                },
                'weights_used': self.hybrid_integrator.get_effective_weights(league, self.hybrid_weights),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Hybrid 1X2 prediction completed")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error in hybrid 1X2 prediction: {e}")
            return self._fallback_prediction(features, context)
    
    def _get_ml_probabilities(self, features: pd.DataFrame, league: Optional[str]) -> Optional[Dict[str, float]]:
        """Get ML 1X2 probabilities"""
        try:
            if not self.ml_model or not self.ml_feature_list:
                return None
            
            # Align features
            X_1x2, _ = align_features(features, self.ml_feature_list, league)
            
            # Get raw predictions
            pred_raw = self.ml_model.predict_proba(X_1x2)[0]
            
            # Apply ML calibration if available
            if self.ml_calibrators:
                pred_calibrated = self._apply_ml_calibration(pred_raw.reshape(1, -1))[0]
            else:
                pred_calibrated = pred_raw
            
            return {
                '1': float(pred_calibrated[0]),
                'X': float(pred_calibrated[1]),
                '2': float(pred_calibrated[2])
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error getting ML probabilities: {e}")
            return None
    
    def _apply_ml_calibration(self, raw_probs: np.ndarray) -> np.ndarray:
        """Apply ML calibration"""
        try:
            calibrated_probs = np.zeros_like(raw_probs)
            class_names = ['1', 'X', '2']
            
            for i, class_name in enumerate(class_names):
                if self.ml_calibrators.get(class_name) is not None:
                    calibrated_probs[:, i] = self.ml_calibrators[class_name].predict(raw_probs[:, i])
                else:
                    calibrated_probs[:, i] = raw_probs[:, i]
            
            # Normalize
            row_sums = calibrated_probs.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            calibrated_probs = calibrated_probs / row_sums
            
            return calibrated_probs
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error in ML calibration: {e}")
            return raw_probs
    
    def _get_scoreline_probabilities(self, home_team: str, away_team: str, 
                                   league: str, context: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Get scoreline-derived 1X2 probabilities"""
        try:
            if not self.scoreline_engine:
                return None
            
            # Get historical data if available
            df = context.get('historical_data')
            if df is None:
                from core.data_loader import ESPNDataLoader
                data_loader = ESPNDataLoader()
                df = data_loader.load_fixtures()
            
            if df is None or df.empty:
                return None
            
            # Get scoreline probabilities
            scoreline_result = self.scoreline_engine.get_scoreline_probabilities(
                home_team, away_team, league, df
            )
            
            matrix = scoreline_result.get('matrix', {})
            if not matrix:
                return None
            
            # Derive 1X2 from matrix
            p_home = p_draw = p_away = 0.0
            
            for scoreline, prob in matrix.items():
                home_goals, away_goals = map(int, scoreline.split('-'))
                
                if home_goals > away_goals:
                    p_home += prob
                elif home_goals == away_goals:
                    p_draw += prob
                else:
                    p_away += prob
            
            # Normalize
            total = p_home + p_draw + p_away
            if total > 0:
                return {
                    '1': p_home / total,
                    'X': p_draw / total,
                    '2': p_away / total
                }
            
            return None
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error getting scoreline probabilities: {e}")
            return None
    
    def _get_poisson_probabilities(self, home_team: str, away_team: str, 
                                 context: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Get Poisson v2 1X2 probabilities"""
        try:
            if not self.poisson_model:
                return None
            
            # Get team IDs (simplified - would need proper team resolution)
            home_team_id = context.get('home_team_id', hash(home_team) % 1000)
            away_team_id = context.get('away_team_id', hash(away_team) % 1000)
            
            # Get Poisson prediction
            poisson_result = self.poisson_model.predict_match_probabilities(
                home_team_id, away_team_id
            )
            
            return {
                '1': poisson_result.get('prob_home_win', 1/3),
                'X': poisson_result.get('prob_draw', 1/3),
                '2': poisson_result.get('prob_away_win', 1/3)
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error getting Poisson probabilities: {e}")
            return None
    
    def _get_draw_specialist_probability(self, home_team: str, away_team: str, 
                                       league: str, context: Dict[str, Any]) -> Optional[float]:
        """Get draw specialist probability"""
        try:
            if not self.draw_predictor:
                return None
            
            # Get historical data if available
            df = context.get('historical_data')
            if df is None:
                from core.data_loader import ESPNDataLoader
                data_loader = ESPNDataLoader()
                df = data_loader.load_fixtures()
            
            if df is None or df.empty:
                return None
            
            # Get draw prediction
            draw_result = self.draw_predictor.predict_draw_probability(
                home_team=home_team,
                away_team=away_team,
                league=league,
                df=df,
                reference_date=datetime.now()
            )
            
            return draw_result.get('draw_probability')
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error getting draw specialist probability: {e}")
            return None
    
    def _apply_hybrid_calibration(self, probs: Dict[str, float]) -> Dict[str, float]:
        """Apply hybrid calibration"""
        try:
            if not self.hybrid_calibrator:
                return probs
            
            # Convert to array format expected by calibrator
            prob_array = np.array([[probs.get('1', 1/3), probs.get('X', 1/3), probs.get('2', 1/3)]])
            
            # Apply calibration (method depends on calibrator type)
            if hasattr(self.hybrid_calibrator, 'predict_proba'):
                calibrated = self.hybrid_calibrator.predict_proba(prob_array)[0]
            elif hasattr(self.hybrid_calibrator, 'predict'):
                calibrated = self.hybrid_calibrator.predict(prob_array)[0]
            else:
                # Temperature scaling or similar
                calibrated = self._apply_temperature_scaling(prob_array[0])
            
            return {
                '1': float(calibrated[0]),
                'X': float(calibrated[1]),
                '2': float(calibrated[2])
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error in hybrid calibration: {e}")
            return probs
    
    def _apply_temperature_scaling(self, probs: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Apply temperature scaling"""
        try:
            # Load temperature from config or use default
            temp = self.config.get('calibration', {}).get('temperature', temperature)
            
            # Apply temperature scaling
            scaled_logits = np.log(np.maximum(probs, 1e-10)) / temp
            
            # Softmax
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
            return exp_logits / np.sum(exp_logits)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error in temperature scaling: {e}")
            return probs
    
    def _fallback_prediction(self, features: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback to ML v2 if hybrid fails"""
        try:
            self.logger.warning("⚠️ Using fallback to ML v2")
            
            if self.ml_model and self.ml_feature_list:
                league = context.get('league')
                ml_probs = self._get_ml_probabilities(features, league)
                
                if ml_probs:
                    return {
                        'prob_home_win': ml_probs['1'],
                        'prob_draw': ml_probs['X'],
                        'prob_away_win': ml_probs['2'],
                        'predicted_outcome': max(ml_probs, key=ml_probs.get),
                        'confidence': max(ml_probs.values()),
                        'calibrated': True,
                        'hybrid_used': False,
                        'fallback_reason': 'hybrid_failed',
                        'timestamp': datetime.now().isoformat()
                    }
            
            # Ultimate fallback
            return {
                'prob_home_win': 1/3,
                'prob_draw': 1/3,
                'prob_away_win': 1/3,
                'predicted_outcome': 'X',
                'confidence': 1/3,
                'calibrated': False,
                'hybrid_used': False,
                'fallback_reason': 'all_models_failed',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error in fallback prediction: {e}")
            return {
                'prob_home_win': 1/3,
                'prob_draw': 1/3,
                'prob_away_win': 1/3,
                'predicted_outcome': 'X',
                'confidence': 1/3,
                'calibrated': False,
                'hybrid_used': False,
                'fallback_reason': 'fallback_failed',
                'timestamp': datetime.now().isoformat()
            }
    
    def is_available(self) -> bool:
        """Check if hybrid predictor is available"""
        return (self.ml_model is not None or 
                self.scoreline_engine is not None or 
                self.poisson_model is not None)
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all components"""
        return {
            'ml_model_loaded': self.ml_model is not None,
            'scoreline_engine_loaded': self.scoreline_engine is not None,
            'poisson_model_loaded': self.poisson_model is not None,
            'draw_predictor_loaded': self.draw_predictor is not None,
            'hybrid_calibrator_loaded': self.hybrid_calibrator is not None,
            'hybrid_weights_loaded': self.hybrid_weights is not None,
            'config_loaded': bool(self.config),
            'is_available': self.is_available()
        }


def main():
    """Test hybrid predictor"""
    print("🎯 Testing Hybrid 1X2 Predictor")
    print("=" * 50)
    
    # Initialize predictor
    predictor = Hybrid1X2Predictor()
    
    # Check status
    status = predictor.get_status()
    print(f"\n📊 Component Status:")
    for component, loaded in status.items():
        print(f"   {component}: {'✅' if loaded else '❌'}")
    
    # Test prediction (dummy data)
    features = pd.DataFrame({
        'home_elo': [1500],
        'away_elo': [1450],
        'home_form': [0.6],
        'away_form': [0.5]
    })
    
    context = {
        'home_team': 'Manchester City',
        'away_team': 'Liverpool',
        'league': 'Premier League'
    }
    
    try:
        result = predictor.predict_hybrid_1x2(features, context)
        
        print(f"\n🎯 Hybrid Prediction Result:")
        print(f"   Home: {result['prob_home_win']:.3f}")
        print(f"   Draw: {result['prob_draw']:.3f}")
        print(f"   Away: {result['prob_away_win']:.3f}")
        print(f"   Predicted: {result['predicted_outcome']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Hybrid Used: {result['hybrid_used']}")
        
        print(f"\n📊 Sources Used:")
        for source, used in result['sources_used'].items():
            print(f"   {source}: {'✅' if used else '❌'}")
        
        print("\n✅ Hybrid predictor test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    main()
