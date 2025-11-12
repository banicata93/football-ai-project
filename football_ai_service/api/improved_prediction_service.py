"""
Improved Prediction Service - Подобрена версия с по-добра обработка на непознати отбори
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
import json
from typing import Dict, Tuple, Optional
from datetime import datetime

from core.utils import setup_logging
from core.ml_utils import get_feature_columns, prepare_features, align_features
from core.ensemble import EnsembleModel, FootballIntelligenceIndex
from core.team_resolver import TeamResolver


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
        
        self.logger.info("ImprovedPredictionService инициализиран успешно")
    
    def _load_models(self):
        """Зареждане на всички модели (същото като оригинала)"""
        self.logger.info("Зареждане на модели...")
        
        try:
            # Poisson
            self.models['poisson'] = joblib.load('models/model_poisson_v1/poisson_model.pkl')
            
            # ML Models with feature lists
            ml_models = {
                '1x2': 'models/model_1x2_v1',
                'ou25': 'models/model_ou25_v1',
                'btts': 'models/model_btts_v1'
            }
            
            for model_name, model_dir in ml_models.items():
                model_file = f"{model_dir}/{model_name}_model.pkl"
                self.models[model_name] = joblib.load(model_file)
                
                feature_list_file = f"{model_dir}/feature_list.json"
                try:
                    with open(feature_list_file, 'r') as f:
                        self.feature_lists[model_name] = json.load(f)
                    self.logger.info(f"✓ {model_name} model: {len(self.feature_lists[model_name])} features")
                except FileNotFoundError:
                    self.logger.warning(f"⚠ Feature list not found for {model_name}, using all features")
                    self.feature_lists[model_name] = get_feature_columns()
            
            # Ensemble
            self.models['ensemble'] = joblib.load('models/ensemble_v1/ensemble_model.pkl')
            self.models['fii'] = joblib.load('models/ensemble_v1/fii_model.pkl')
            
            self.feature_columns = get_feature_columns()
            self.logger.info(f"Всички модели заредени успешно ({len(self.models)} модела)")
            
        except Exception as e:
            self.logger.error(f"Грешка при зареждане на модели: {e}")
            raise
    
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
        predictions = self._make_all_predictions(match_df, home_team_id, away_team_id)
        
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
    
    def _make_all_predictions(self, match_df: pd.DataFrame, home_team_id: int, away_team_id: int) -> Dict:
        """Прави всички predictions"""
        predictions = {}
        
        try:
            # Poisson predictions
            poisson_pred = self.models['poisson'].predict_match_probabilities(
                home_team_id, away_team_id
            )
            
            # 1X2 prediction
            X_1x2 = align_features(match_df, self.feature_lists['1x2'])
            pred_1x2 = self.models['1x2'].predict_proba(X_1x2)[0]
            
            predictions['prediction_1x2'] = {
                'prob_home_win': float(pred_1x2[0]),
                'prob_draw': float(pred_1x2[1]),
                'prob_away_win': float(pred_1x2[2]),
                'predicted_outcome': ['1', 'X', '2'][np.argmax(pred_1x2)],
                'confidence': float(np.max(pred_1x2))
            }
            
            # Over/Under 2.5
            X_ou25 = align_features(match_df, self.feature_lists['ou25'])
            pred_ou25 = self.models['ou25'].predict_proba(X_ou25)[0]
            
            predictions['prediction_ou25'] = {
                'prob_over': float(pred_ou25[1]),
                'prob_under': float(pred_ou25[0]),
                'predicted_outcome': 'Over' if pred_ou25[1] > 0.5 else 'Under',
                'confidence': float(max(pred_ou25))
            }
            
            # BTTS
            X_btts = align_features(match_df, self.feature_lists['btts'])
            pred_btts = self.models['btts'].predict_proba(X_btts)[0]
            
            predictions['prediction_btts'] = {
                'prob_yes': float(pred_btts[1]),
                'prob_no': float(pred_btts[0]),
                'predicted_outcome': 'Yes' if pred_btts[1] > 0.5 else 'No',
                'confidence': float(max(pred_btts))
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
