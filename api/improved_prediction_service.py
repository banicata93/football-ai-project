"""
Improved Prediction Service - Подобрена версия с по-добра обработка на непознати отбори
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
import json
from typing import Dict, Tuple, Optional, Any
from datetime import datetime

from core.utils import setup_logging
from core.ml_utils import get_feature_columns, prepare_features, align_features
from core.ensemble import EnsembleModel, FootballIntelligenceIndex
from core.team_resolver import TeamResolver
from core.scoreline_integration import ScorelineIntegrator
from core.hybrid_1x2 import Hybrid1X2Combiner


class ImprovedPredictionService:
    """
    Подобрен сървис за predictions с интелигентна обработка на отбори
    """
    
    def __init__(self):
        """Инициализация на подобрения сървис"""
        self.logger = setup_logging()
        self.models = {}
        self.feature_columns = []
        self.feature_lists = {}
        self.elo_ratings = {}
        self.team_stats = {}
        self.team_names = {}
        
        self._load_models()
        self._load_team_data()
        self._load_team_names()
        
        # Създаваме TeamResolver
        self.team_resolver = TeamResolver(
            self.elo_ratings, 
            self.team_names
        )
        
        # Създаваме Scoreline Integrator
        self.scoreline_integrator = ScorelineIntegrator()
        
        # Създаваме Hybrid 1X2 Combiner
        self.hybrid_1x2_combiner = Hybrid1X2Combiner()
        
        self.logger.info("ImprovedPredictionService инициализиран успешно")
    
    def _load_models(self):
        """Зареждане на всички модели (същото като оригинала)"""
        self.logger.info("Зареждане на модели...")
        
        try:
            # Poisson
            self.models['poisson'] = joblib.load('models/model_poisson_v1/poisson_model.pkl')
            
            # ML Models with feature lists (excluding BTTS - loaded separately)
            ml_models = {
                '1x2': 'models/model_1x2_v1',
                'ou25': 'models/model_ou25_v1'
            }
            
            for model_name, model_dir in ml_models.items():
                model_file = f"{model_dir}/{model_name}_model.pkl"
                self.models[model_name] = joblib.load(model_file)
                
                # Специално зареждане на калибратори за 1X2 модел
                if model_name == '1x2':
                    self._load_1x2_calibrators(model_dir)
                
                feature_list_file = f"{model_dir}/feature_list.json"
                try:
                    with open(feature_list_file, 'r') as f:
                        self.feature_lists[model_name] = json.load(f)
                    self.logger.info(f"✓ {model_name} model: {len(self.feature_lists[model_name])} features")
                except FileNotFoundError:
                    self.logger.warning(f"⚠ Feature list not found for {model_name}, using all features")
                    self.feature_lists[model_name] = get_feature_columns()
            
            # Load BTTS model (prioritize improved version)
            self._load_btts_models()
            
            # Зареждане на глобален OU2.5 калибратор
            self._load_global_ou25_calibrator()
            
            # Ensemble
            self.models['ensemble'] = joblib.load('models/ensemble_v1/ensemble_model.pkl')
            self.models['fii'] = joblib.load('models/ensemble_v1/fii_model.pkl')
            
            self.feature_columns = get_feature_columns()
            self.logger.info(f"Всички модели заредени успешно ({len(self.models)} модела)")
            
        except Exception as e:
            self.logger.error(f"Грешка при зареждане на модели: {e}")
            raise
    
    def _load_btts_models(self):
        """Зареждане на BTTS модели с приоритет на improved версията"""
        btts_loaded = False
        
        # Опит за зареждане на improved BTTS модел
        try:
            improved_btts_path = 'models/model_btts_improved/btts_model_improved.pkl'
            improved_features_path = 'models/model_btts_improved/feature_columns.json'
            
            if os.path.exists(improved_btts_path):
                self.models['btts'] = joblib.load(improved_btts_path)
                
                # Зарежда feature list за improved BTTS
                if os.path.exists(improved_features_path):
                    with open(improved_features_path, 'r') as f:
                        feature_data = json.load(f)
                        if isinstance(feature_data, dict) and 'features' in feature_data:
                            self.feature_lists['btts'] = feature_data['features']
                        else:
                            self.feature_lists['btts'] = feature_data
                
                self.logger.info(f"✓ Improved BTTS model зареден като основен с {len(self.feature_lists.get('btts', []))} features")
                btts_loaded = True
            else:
                self.logger.warning("⚠ Improved BTTS model файл не съществува")
                
        except Exception as e:
            self.logger.warning(f"⚠ Грешка при зареждане на improved BTTS: {e}")
        
        # Fallback към legacy BTTS модел ако improved не е зареден
        if not btts_loaded:
            try:
                legacy_btts_path = 'models/model_btts_v1/btts_model.pkl'
                legacy_features_path = 'models/model_btts_v1/feature_list.json'
                
                if os.path.exists(legacy_btts_path):
                    self.models['btts'] = joblib.load(legacy_btts_path)
                    
                    # Зарежда legacy feature list
                    if os.path.exists(legacy_features_path):
                        with open(legacy_features_path, 'r') as f:
                            self.feature_lists['btts'] = json.load(f)
                    
                    self.logger.warning(f"⚠ Fallback към legacy BTTS model с {len(self.feature_lists.get('btts', []))} features")
                else:
                    self.logger.error("❌ Нито improved, нито legacy BTTS модел не могат да се заредят")
                    self.models['btts'] = None
                    self.feature_lists['btts'] = []
                    
            except Exception as e:
                self.logger.error(f"❌ Грешка при зареждане на legacy BTTS: {e}")
                self.models['btts'] = None
                self.feature_lists['btts'] = []
    
    def _load_team_data(self):
        """Зареждане на team data (същото като оригинала)"""
        self.logger.info("Зареждане на team data...")
        
        try:
            df = pd.read_parquet("data/processed/test_final_predictions.parquet")
            
            home_teams = df.groupby('home_team_id').last()
            away_teams = df.groupby('away_team_id').last()
            
            for team_id, row in home_teams.iterrows():
                team_name = row.get('home_team', f'Team_{team_id}')
                self.elo_ratings[team_name] = {
                    'elo': row.get('home_elo_before', 1500),
                    'form': row.get('home_form_5', 0),
                    'goals_avg': row.get('home_goals_scored_avg_5', 1.5),
                    'xg_proxy': row.get('home_xg_proxy', 1.5),
                    'shooting_efficiency': row.get('home_shooting_efficiency', 0.3)
                }
            
            for team_id, row in away_teams.iterrows():
                team_name = row.get('away_team', f'Team_{team_id}')
                if team_name not in self.elo_ratings:
                    self.elo_ratings[team_name] = {
                        'elo': row.get('away_elo_before', 1500),
                        'form': row.get('away_form_5', 0),
                        'goals_avg': row.get('away_goals_scored_avg_5', 1.5),
                        'xg_proxy': row.get('away_xg_proxy', 1.5),
                        'shooting_efficiency': row.get('away_shooting_efficiency', 0.3)
                    }
            
            self.logger.info(f"Team data заредени за {len(self.elo_ratings)} отбора")
            
        except Exception as e:
            self.logger.warning(f"Не мога да заредя team data: {e}")
            self.elo_ratings = {}
    
    def _load_team_names(self):
        """Зареждане на реални имена на отборите (същото като оригинала)"""
        try:
            with open('models/team_names_mapping.json', 'r') as f:
                team_mapping = json.load(f)
            
            for team_id_str, info in team_mapping.items():
                team_id = int(team_id_str)
                team_key = f"Team_{team_id}"
                self.team_names[team_key] = info
            
            self.logger.info(f"✓ Реални имена заредени за {len(self.team_names)} отбора")
            
        except Exception as e:
            self.logger.warning(f"Не мога да заредя team names: {e}")
            self.team_names = {}
    
    def predict_with_confidence(
        self,
        home_team: str,
        away_team: str,
        league: Optional[str] = None,
        date: Optional[str] = None
    ) -> Dict:
        """
        Прогноза с confidence scoring и подробна информация за качеството на данните
        
        Args:
            home_team: Домакин
            away_team: Гост
            league: Лига
            date: Дата
        
        Returns:
            Dictionary с predictions и подробна metadata
        """
        self.logger.info(f"Improved prediction за: {home_team} vs {away_team}")
        
        # Получаваме данни за отборите чрез TeamResolver
        home_data, home_meta = self.team_resolver.get_team_data(home_team, league)
        away_data, away_meta = self.team_resolver.get_team_data(away_team, league)
        
        # Изчисляваме общ confidence score
        overall_confidence = min(home_meta['confidence'], away_meta['confidence'])
        
        # Събираме предупреждения
        warnings = []
        if 'warning' in home_meta:
            warnings.append(home_meta['warning'])
        if 'warning' in away_meta:
            warnings.append(away_meta['warning'])
        
        # Създаваме features с подобрените данни
        match_df = self._create_improved_features(
            home_team, away_team, home_data, away_data, league, date
        )
        
        # Получаваме team IDs за Poisson модела
        home_team_id, home_id_meta = self.team_resolver.get_team_id_for_poisson(home_team, league)
        away_team_id, away_id_meta = self.team_resolver.get_team_id_for_poisson(away_team, league)
        
        # Добавяме предупреждения за ID-тата
        if 'warning' in home_id_meta:
            warnings.append(home_id_meta['warning'])
        if 'warning' in away_id_meta:
            warnings.append(away_id_meta['warning'])
        
        # Правим predictions
        predictions = self._make_all_predictions(match_df, home_team_id, away_team_id, league)
        
        # Добавяме подробна metadata
        predictions['data_quality'] = {
            'overall_confidence': overall_confidence,
            'confidence_level': self._get_confidence_level(overall_confidence),
            'home_team': {
                'input_name': home_team,
                'resolved_name': home_meta.get('real_name', home_team),
                'data_source': home_meta['data_source'],
                'confidence': home_meta['confidence'],
                'search_method': home_meta['search_method']
            },
            'away_team': {
                'input_name': away_team,
                'resolved_name': away_meta.get('real_name', away_team),
                'data_source': away_meta['data_source'],
                'confidence': away_meta['confidence'],
                'search_method': away_meta['search_method']
            },
            'poisson_ids': {
                'home_id': home_team_id,
                'away_id': away_team_id,
                'home_method': home_id_meta['method'],
                'away_method': away_id_meta['method']
            },
            'warnings': warnings,
            'recommendation': self._get_recommendation(overall_confidence)
        }
        
        return predictions
    
    def predict_with_scoreline_enhancement(
        self,
        home_team: str,
        away_team: str,
        league: Optional[str] = None,
        date: Optional[str] = None
    ) -> Dict:
        """
        Enhanced predictions combining ML models with Scoreline Probability Engine
        
        Args:
            home_team: Домакин
            away_team: Гост
            league: Лига
            date: Дата
        
        Returns:
            Dictionary с enhanced predictions включително scoreline analysis
        """
        self.logger.info(f"Enhanced scoreline prediction за: {home_team} vs {away_team}")
        
        # Получаваме базовите ML predictions
        base_predictions = self.predict_with_confidence(home_team, away_team, league, date)
        
        try:
            # Зареждаме исторически данни за scoreline engine
            df = self._load_historical_data()
            
            # Подготвяме ML predictions в правилния формат за integration
            ml_predictions = self._format_ml_predictions_for_integration(base_predictions)
            
            # Получаваме enhanced predictions чрез scoreline integration
            enhanced_predictions = self.scoreline_integrator.get_enhanced_predictions(
                home_team=home_team,
                away_team=away_team,
                league=league,
                df=df,
                ml_predictions=ml_predictions
            )
            
            # Комбинираме с базовите predictions
            final_predictions = base_predictions.copy()
            final_predictions.update(enhanced_predictions)
            
            # Добавяме scoreline enhancement metadata
            final_predictions['enhancement_info'] = {
                'scoreline_integration_enabled': True,
                'scoreline_engine_available': self.scoreline_integrator.scoreline_engine is not None,
                'enhancement_timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Enhanced predictions completed for {home_team} vs {away_team}")
            return final_predictions
            
        except Exception as e:
            self.logger.error(f"❌ Error in scoreline enhancement: {e}")
            # Fallback to base predictions
            base_predictions['enhancement_info'] = {
                'scoreline_integration_enabled': False,
                'error': str(e),
                'fallback_reason': 'scoreline_enhancement_failed'
            }
            return base_predictions
    
    def _load_historical_data(self) -> pd.DataFrame:
        """Зарежда исторически данни за scoreline engine"""
        try:
            from core.data_loader import ESPNDataLoader
            data_loader = ESPNDataLoader()
            df = data_loader.load_fixtures()
            return df
        except Exception as e:
            self.logger.warning(f"⚠️ Could not load historical data: {e}")
            return pd.DataFrame()
    
    def _format_ml_predictions_for_integration(self, base_predictions: Dict) -> Dict:
        """Форматира ML predictions за scoreline integration"""
        try:
            ml_formatted = {}
            
            # 1X2 predictions
            if 'prediction_1x2' in base_predictions:
                pred_1x2 = base_predictions['prediction_1x2']
                ml_formatted['1x2'] = {
                    'p_home': pred_1x2['prob_home_win'],
                    'p_draw': pred_1x2['prob_draw'],
                    'p_away': pred_1x2['prob_away_win'],
                    'confidence': pred_1x2['confidence']
                }
            
            # BTTS predictions
            if 'prediction_btts' in base_predictions:
                pred_btts = base_predictions['prediction_btts']
                ml_formatted['btts'] = {
                    'p_yes': pred_btts['prob_yes'],
                    'p_no': pred_btts['prob_no'],
                    'confidence': pred_btts['confidence']
                }
            
            # OU2.5 predictions
            if 'prediction_ou25' in base_predictions:
                pred_ou25 = base_predictions['prediction_ou25']
                ml_formatted['ou25'] = {
                    'p_over': pred_ou25['prob_over'],
                    'p_under': pred_ou25['prob_under'],
                    'confidence': pred_ou25['confidence']
                }
            
            return ml_formatted
            
        except Exception as e:
            self.logger.error(f"❌ Error formatting ML predictions: {e}")
            return {}
    
    def _predict_1x2_hybrid(self, match_features: pd.DataFrame, 
                           home_team: str, away_team: str, league: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate hybrid 1X2 prediction combining multiple sources
        
        Args:
            match_features: Match features DataFrame
            home_team: Home team name
            away_team: Away team name
            league: League name
            
        Returns:
            Hybrid 1X2 prediction result
        """
        try:
            self.logger.info(f"🎯 Generating hybrid 1X2 prediction: {home_team} vs {away_team}")
            
            # 1. Get ML 1X2 probabilities (existing v2 model)
            ml_probs = self._get_ml_1x2_probabilities(match_features, league)
            
            # 2. Get scoreline-derived 1X2 probabilities
            scoreline_probs = self._get_scoreline_1x2_probabilities(home_team, away_team, league)
            
            # 3. Get draw specialist probability
            draw_specialist_prob = self._get_draw_specialist_probability(home_team, away_team, league)
            
            # 4. Get Poisson 1X2 probabilities
            poisson_probs = self._get_poisson_1x2_probabilities(home_team, away_team)
            
            # 5. Get Elo-based 1X2 probabilities (if available)
            elo_probs = self._get_elo_1x2_probabilities(home_team, away_team)
            
            # 6. Prepare meta features
            meta_features = {
                'league': league,
                'ml_confidence': ml_probs.get('confidence', 0.7),
                'scoreline_confidence': scoreline_probs.get('confidence', 0.7) if scoreline_probs else 0.5,
                'match_importance': 'normal'  # Could be enhanced with more logic
            }
            
            # 7. Combine using Hybrid 1X2 Combiner
            hybrid_result = self.hybrid_1x2_combiner.combine(
                ml_probs=ml_probs,
                scoreline_probs=scoreline_probs,
                draw_specialist_prob=draw_specialist_prob,
                poisson_probs=poisson_probs,
                elo_probs=elo_probs,
                meta_features=meta_features
            )
            
            self.logger.info(f"✅ Hybrid 1X2 prediction completed")
            return hybrid_result
            
        except Exception as e:
            self.logger.error(f"❌ Error in hybrid 1X2 prediction: {e}")
            # Fallback to ML only
            ml_probs = self._get_ml_1x2_probabilities(match_features, league)
            return {
                'hybrid_probs': ml_probs,
                'components': {'ml': ml_probs},
                'weights_used': {'ml': 1.0},
                'uncertainty': {'entropy': 0.0, 'confidence': 0.5},
                'fallback_reason': str(e)
            }
    
    def _get_ml_1x2_probabilities(self, match_features: pd.DataFrame, league: Optional[str]) -> Dict[str, float]:
        """Get ML 1X2 probabilities from existing model"""
        try:
            X_1x2, meta_1x2 = align_features(match_features, self.feature_lists['1x2'], league)
            pred_1x2_raw = self.models['1x2'].predict_proba(X_1x2)[0:1]
            pred_1x2 = self._apply_1x2_calibration(pred_1x2_raw)[0]
            
            return {
                '1': float(pred_1x2[0]),
                'X': float(pred_1x2[1]),
                '2': float(pred_1x2[2]),
                'confidence': float(np.max(pred_1x2))
            }
        except Exception as e:
            self.logger.warning(f"⚠️ Error getting ML 1X2 probabilities: {e}")
            return {'1': 1/3, 'X': 1/3, '2': 1/3, 'confidence': 0.33}
    
    def _get_scoreline_1x2_probabilities(self, home_team: str, away_team: str, 
                                       league: Optional[str]) -> Optional[Dict[str, float]]:
        """Get 1X2 probabilities from scoreline engine"""
        try:
            if not self.scoreline_integrator.scoreline_engine:
                return None
            
            df = self._load_historical_data()
            if df.empty:
                return None
            
            scoreline_result = self.scoreline_integrator.scoreline_engine.get_scoreline_probabilities(
                home_team, away_team, league, df
            )
            
            matrix = scoreline_result.get('matrix', {})
            if not matrix:
                return None
            
            # Derive 1X2 from scoreline matrix
            scoreline_1x2 = self.scoreline_integrator.derive_1x2_from_matrix(matrix)
            
            return {
                '1': scoreline_1x2['p_home'],
                'X': scoreline_1x2['p_draw'],
                '2': scoreline_1x2['p_away'],
                'confidence': scoreline_1x2['confidence']
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error getting scoreline 1X2 probabilities: {e}")
            return None
    
    def _get_draw_specialist_probability(self, home_team: str, away_team: str, 
                                       league: Optional[str]) -> Optional[float]:
        """Get draw specialist probability"""
        try:
            if not hasattr(self.scoreline_integrator, 'draw_predictor') or not self.scoreline_integrator.draw_predictor:
                return None
            
            df = self._load_historical_data()
            if df.empty:
                return None
            
            draw_result = self.scoreline_integrator.draw_predictor.predict_draw_probability(
                home_team, away_team, league or 'unknown', df
            )
            
            return draw_result.get('draw_probability')
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error getting draw specialist probability: {e}")
            return None
    
    def _get_poisson_1x2_probabilities(self, home_team: str, away_team: str) -> Optional[Dict[str, float]]:
        """Get 1X2 probabilities from Poisson model"""
        try:
            # Get team IDs for Poisson
            home_team_id, _ = self.team_resolver.get_team_id_for_poisson(home_team, None)
            away_team_id, _ = self.team_resolver.get_team_id_for_poisson(away_team, None)
            
            poisson_result = self.models['poisson'].predict_match_probabilities(
                home_team_id, away_team_id
            )
            
            return {
                '1': poisson_result.get('prob_home_win', 1/3),
                'X': poisson_result.get('prob_draw', 1/3),
                '2': poisson_result.get('prob_away_win', 1/3)
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error getting Poisson 1X2 probabilities: {e}")
            return None
    
    def _get_elo_1x2_probabilities(self, home_team: str, away_team: str) -> Optional[Dict[str, float]]:
        """Get 1X2 probabilities from Elo ratings"""
        try:
            # Get team data
            home_data, _ = self.team_resolver.get_team_data(home_team, None)
            away_data, _ = self.team_resolver.get_team_data(away_team, None)
            
            home_elo = home_data['elo']
            away_elo = away_data['elo']
            
            # Simple Elo-based 1X2 calculation
            elo_diff = home_elo - away_elo
            
            # Convert Elo difference to probabilities (simplified)
            # This is a basic implementation - could be enhanced
            home_advantage = 100  # Home advantage in Elo points
            adjusted_diff = elo_diff + home_advantage
            
            # Sigmoid-like transformation
            p_home = 1 / (1 + 10 ** (-adjusted_diff / 400))
            p_away = 1 / (1 + 10 ** (adjusted_diff / 400))
            p_draw = max(0.1, 1 - p_home - p_away)  # Ensure minimum draw probability
            
            # Normalize
            total = p_home + p_draw + p_away
            return {
                '1': p_home / total,
                'X': p_draw / total,
                '2': p_away / total
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error getting Elo 1X2 probabilities: {e}")
            return None
    
    def predict_with_hybrid_1x2(self, 
                               home_team: str, 
                               away_team: str, 
                               league: Optional[str] = None, 
                               date: Optional[str] = None) -> Dict:
        """
        Enhanced predictions with Hybrid 1X2 model
        
        Args:
            home_team: Home team name
            away_team: Away team name
            league: League name
            date: Date
            
        Returns:
            Predictions with hybrid 1X2 enhancement
        """
        self.logger.info(f"Hybrid 1X2 prediction за: {home_team} vs {away_team}")
        
        # Get base predictions with scoreline enhancement
        base_predictions = self.predict_with_scoreline_enhancement(home_team, away_team, league, date)
        
        try:
            # Create match features for hybrid prediction
            home_data, _ = self.team_resolver.get_team_data(home_team, league)
            away_data, _ = self.team_resolver.get_team_data(away_team, league)
            
            match_features = self._create_improved_features(
                home_team, away_team, home_data, away_data, league, date
            )
            
            # Generate hybrid 1X2 prediction
            hybrid_1x2_result = self._predict_1x2_hybrid(match_features, home_team, away_team, league)
            
            # Add hybrid 1X2 to existing predictions
            if 'prediction_1x2' not in base_predictions:
                base_predictions['prediction_1x2'] = {}
            
            # Add hybrid results to 1X2 prediction
            base_predictions['prediction_1x2']['hybrid'] = {
                '1': hybrid_1x2_result['hybrid_probs']['1'],
                'X': hybrid_1x2_result['hybrid_probs']['X'],
                '2': hybrid_1x2_result['hybrid_probs']['2'],
                'predicted_outcome': max(hybrid_1x2_result['hybrid_probs'], 
                                       key=hybrid_1x2_result['hybrid_probs'].get),
                'confidence': hybrid_1x2_result['uncertainty']['confidence'],
                'uncertainty': hybrid_1x2_result['uncertainty'],
                'components': hybrid_1x2_result['components'],
                'weights_used': hybrid_1x2_result['weights_used']
            }
            
            # Add hybrid metadata
            base_predictions['hybrid_1x2_info'] = {
                'enabled': True,
                'sources_used': len([k for k, v in hybrid_1x2_result['components'].items() 
                                   if v is not None]),
                'combination_method': hybrid_1x2_result.get('combination_method', 'weighted_average'),
                'league': league,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Hybrid 1X2 prediction completed for {home_team} vs {away_team}")
            return base_predictions
            
        except Exception as e:
            self.logger.error(f"❌ Error in hybrid 1X2 prediction: {e}")
            # Add error info but return base predictions
            base_predictions['hybrid_1x2_info'] = {
                'enabled': False,
                'error': str(e),
                'fallback_reason': 'hybrid_1x2_failed'
            }
            return base_predictions
    
    def _create_improved_features(
        self,
        home_team: str,
        away_team: str,
        home_data: Dict,
        away_data: Dict,
        league: Optional[str] = None,
        date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Създава features с подобрените данни от TeamResolver
        """
        features = {
            'home_team': home_team,
            'away_team': away_team,
            'league': league or 'Unknown',
            'date': date or datetime.now().strftime('%Y-%m-%d'),
            
            # Elo
            'home_elo_before': home_data['elo'],
            'away_elo_before': away_data['elo'],
            'elo_diff': home_data['elo'] - away_data['elo'],
            
            # Form
            'home_form_5': home_data['form'],
            'away_form_5': away_data['form'],
            
            # Goals
            'home_goals_scored_avg_5': home_data['goals_avg'],
            'away_goals_scored_avg_5': away_data['goals_avg'],
            'home_goals_conceded_avg_5': 1.5,  # TODO: Добави в TeamResolver
            'away_goals_conceded_avg_5': 1.5,
            
            # xG
            'home_xg_proxy': home_data['xg_proxy'],
            'away_xg_proxy': away_data['xg_proxy'],
            
            # Efficiency
            'home_shooting_efficiency': home_data['shooting_efficiency'],
            'away_shooting_efficiency': away_data['shooting_efficiency'],
            
            # Home advantage
            'is_home': 1,
            'home_rest_days': 7,
            'away_rest_days': 7
        }
        
        # Попълваме останалите features с 0
        for col in self.feature_columns:
            if col not in features:
                features[col] = 0.0
        
        return pd.DataFrame([features])
    
    def _make_all_predictions(self, match_df: pd.DataFrame, home_team_id: int, away_team_id: int, league: Optional[str] = None) -> Dict:
        """Прави всички predictions"""
        predictions = {}
        
        try:
            # Poisson predictions
            poisson_pred = self.models['poisson'].predict_match_probabilities(
                home_team_id, away_team_id
            )
            
            # 1X2 prediction с калибрация
            X_1x2, meta_1x2 = align_features(match_df, self.feature_lists['1x2'], league)
            pred_1x2_raw = self.models['1x2'].predict_proba(X_1x2)[0:1]  # Keep as 2D array
            pred_1x2 = self._apply_1x2_calibration(pred_1x2_raw)[0]  # Apply calibration and get first row
            
            predictions['prediction_1x2'] = {
                'prob_home_win': float(pred_1x2[0]),
                'prob_draw': float(pred_1x2[1]),
                'prob_away_win': float(pred_1x2[2]),
                'predicted_outcome': ['1', 'X', '2'][np.argmax(pred_1x2)],
                'confidence': float(np.max(pred_1x2))
            }
            
            # Over/Under 2.5 с калибрация
            X_ou25, meta_ou25 = align_features(match_df, self.feature_lists['ou25'], league)
            pred_ou25_raw = self.models['ou25'].predict_proba(X_ou25)[0]
            
            # Прилага калибрация на prob_over
            prob_over_calibrated = self._apply_ou25_calibration(pred_ou25_raw[1], league)
            prob_under_calibrated = 1.0 - prob_over_calibrated  # Нормализация
            
            predictions['prediction_ou25'] = {
                'prob_over': float(prob_over_calibrated),
                'prob_under': float(prob_under_calibrated),
                'predicted_outcome': 'Over' if prob_over_calibrated > 0.5 else 'Under',
                'confidence': float(max(prob_over_calibrated, prob_under_calibrated))
            }
            
            # BTTS (using improved model with 0.6 threshold)
            X_btts, meta_btts = align_features(match_df, self.feature_lists['btts'], league)
            pred_btts = self.models['btts'].predict_proba(X_btts)[0]
            
            # Use 0.6 threshold for improved BTTS model
            btts_prob_yes = float(pred_btts[1])
            predicted_outcome = 'Yes' if btts_prob_yes >= 0.6 else 'No'
            
            predictions['prediction_btts'] = {
                'prob_yes': btts_prob_yes,
                'prob_no': float(pred_btts[0]),
                'predicted_outcome': predicted_outcome,
                'confidence': float(max(pred_btts)),
                'model_source': 'improved_btts' if os.path.exists('models/model_btts_improved/btts_model_improved.pkl') else 'legacy_btts',
                'threshold_used': 0.6
            }
            
            # FII Score (упростен)
            elo_diff = match_df['elo_diff'].iloc[0]
            fii_score = 5.0 + (elo_diff / 200.0)  # Базов FII
            fii_score = max(0, min(10, fii_score))  # Ограничаваме между 0-10
            
            predictions['fii'] = {
                'score': float(fii_score),
                'confidence_level': self._get_confidence_level(0.7),  # Default
                'components': {
                    'elo_diff': float(elo_diff),
                    'form_diff': float(match_df['home_form_5'].iloc[0] - match_df['away_form_5'].iloc[0]),
                    'xg_efficiency_diff': float(match_df['home_xg_proxy'].iloc[0] - match_df['away_xg_proxy'].iloc[0]),
                    'finishing_efficiency_diff': float(match_df['home_shooting_efficiency'].iloc[0] - match_df['away_shooting_efficiency'].iloc[0])
                }
            }
            
            # Model versions
            predictions['model_versions'] = {
                'poisson': 'v1',
                '1x2': 'v1',
                'ou25': 'v1',
                'btts': 'v1',
                'ensemble': 'v1'
            }
            
            predictions['timestamp'] = datetime.now().isoformat()
            
            # Добавяме feature quality информация
            predictions['feature_quality'] = {
                '1x2_model': {
                    'data_quality_score': meta_1x2.get('data_quality_score', 1.0),
                    'missing_features': meta_1x2.get('missing_features', []),
                    'imputed_count': len(meta_1x2.get('imputed_features', {}))
                },
                'ou25_model': {
                    'data_quality_score': meta_ou25.get('data_quality_score', 1.0),
                    'missing_features': meta_ou25.get('missing_features', []),
                    'imputed_count': len(meta_ou25.get('imputed_features', {}))
                },
                'btts_model': {
                    'data_quality_score': meta_btts.get('data_quality_score', 1.0),
                    'missing_features': meta_btts.get('missing_features', []),
                    'imputed_count': len(meta_btts.get('imputed_features', {}))
                }
            }
            
        except Exception as e:
            self.logger.error(f"Грешка при predictions: {e}")
            raise
        
        return predictions
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Конвертира confidence score в текстово ниво"""
        if confidence >= 0.9:
            return "High"
        elif confidence >= 0.7:
            return "Medium"
        elif confidence >= 0.5:
            return "Low"
        else:
            return "Very Low"
    
    def _load_1x2_calibrators(self, model_dir: str):
        """Зарежда IsotonicRegression калибратори за 1X2 модела"""
        try:
            self.calibrators_1x2 = {}
            class_names = ['1', 'X', '2']
            
            for class_name in class_names:
                calibrator_file = f"{model_dir}/calibrator_{class_name}.pkl"
                if os.path.exists(calibrator_file):
                    self.calibrators_1x2[class_name] = joblib.load(calibrator_file)
                    self.logger.info(f"✓ Калибратор {class_name} зареден")
                else:
                    self.logger.warning(f"⚠ Калибратор {class_name} не е намерен: {calibrator_file}")
                    self.calibrators_1x2[class_name] = None
            
            if all(cal is not None for cal in self.calibrators_1x2.values()):
                self.logger.info("✅ Всички 1X2 калибратори заредени успешно")
            else:
                self.logger.warning("⚠ Някои 1X2 калибратори липсват")
                
        except Exception as e:
            self.logger.error(f"❌ Грешка при зареждане на 1X2 калибратори: {e}")
            self.calibrators_1x2 = {}
    
    def _apply_1x2_calibration(self, raw_probs: np.ndarray) -> np.ndarray:
        """Прилага независими калибратори за всеки клас и нормализира"""
        if not hasattr(self, 'calibrators_1x2') or not self.calibrators_1x2:
            self.logger.warning("Няма заредени калибратори, връщам raw probabilities")
            return raw_probs
        
        try:
            calibrated_probs = np.zeros_like(raw_probs)
            class_names = ['1', 'X', '2']
            
            # Прилагаме калибрация за всеки клас
            for i, class_name in enumerate(class_names):
                if self.calibrators_1x2.get(class_name) is not None:
                    calibrated_probs[:, i] = self.calibrators_1x2[class_name].predict(raw_probs[:, i])
                else:
                    # Fallback към raw probability ако калибраторът липсва
                    calibrated_probs[:, i] = raw_probs[:, i]
            
            # Нормализация до сума = 1
            row_sums = calibrated_probs.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)  # Избягване на деление на 0
            calibrated_probs = calibrated_probs / row_sums
            
            return calibrated_probs
            
        except Exception as e:
            self.logger.error(f"Грешка при калибрация: {e}, връщам raw probabilities")
            return raw_probs
    
    def _load_global_ou25_calibrator(self):
        """Зарежда глобален OU2.5 калибратор ако съществува"""
        try:
            global_calibrator_path = 'models/model_ou25_v1/calibrator.pkl'
            if os.path.exists(global_calibrator_path):
                self.global_ou25_calibrator = joblib.load(global_calibrator_path)
                self.logger.info("✓ Глобален OU2.5 калибратор зареден")
            else:
                self.global_ou25_calibrator = None
                self.logger.info("ℹ Глобален OU2.5 калибратор не е намерен")
        except Exception as e:
            self.logger.warning(f"⚠ Грешка при зареждане на глобален OU2.5 калибратор: {e}")
            self.global_ou25_calibrator = None
    
    def _apply_ou25_calibration(self, raw_prob: float, league: Optional[str] = None) -> float:
        """Прилага OU2.5 калибрация с fallback логика"""
        try:
            # Опитва се да намери league-specific калибратор (ако има per-league система)
            league_calibrator = None
            if league and hasattr(self, 'ou25_calibrators_by_league'):
                from core.league_utils import get_league_slug
                league_slug = get_league_slug(league)
                if league_slug:
                    league_calibrator = self.ou25_calibrators_by_league.get(league_slug)
            
            # Прилага калибрация - първо league-specific, после глобален, накрая raw
            if league_calibrator is not None:
                # League-specific калибратор
                calibrated_prob = league_calibrator.predict([raw_prob])[0]
                self.logger.debug(f"Използван {league} калибратор за OU2.5")
                return calibrated_prob
            elif hasattr(self, 'global_ou25_calibrator') and self.global_ou25_calibrator is not None:
                # Fallback към глобален калибратор
                calibrated_prob = self.global_ou25_calibrator.predict([raw_prob])[0]
                self.logger.debug("Използван глобален OU2.5 калибратор")
                return calibrated_prob
            else:
                # Няма калибратор - използва raw probability
                self.logger.warning(f"Няма калибратор за OU2.5 ({league or 'unknown'}), използвам raw probability")
                return raw_prob
                
        except Exception as e:
            self.logger.error(f"Грешка при OU2.5 калибрация: {e}, връщам raw probability")
            return raw_prob
    
    def _get_recommendation(self, confidence: float) -> str:
        """Дава препоръка според confidence нивото"""
        if confidence >= 0.9:
            return "Прогнозата е базирана на пълни исторически данни"
        elif confidence >= 0.7:
            return "Прогнозата е базирана на частични данни - използвайте с внимание"
        elif confidence >= 0.5:
            return "Прогнозата е базирана на ограничени данни - ниска надеждност"
        else:
            return "Прогнозата е базирана на default стойности - много ниска надеждност"
