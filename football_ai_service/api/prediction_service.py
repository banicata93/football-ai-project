"""
Prediction Service - Централизирана логика за predictions
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


class PredictionService:
    """
    Сървис за predictions с всички модели
    """
    
    def __init__(self):
        """Инициализация на сървиса"""
        self.logger = setup_logging()
        self.models = {}
        self.feature_columns = []
        self.feature_lists = {}  # Feature lists for each model
        self.elo_ratings = {}
        self.team_stats = {}
        self.team_names = {}  # Real team names mapping
        
        self._load_models()
        self._load_team_data()
        self._load_team_names()
        
        self.logger.info("PredictionService инициализиран успешно")
    
    def _load_models(self):
        """Зареждане на всички модели"""
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
                # Load model
                model_file = f"{model_dir}/{model_name}_model.pkl"
                self.models[model_name] = joblib.load(model_file)
                
                # Load feature list
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
            
            # Feature columns (all features for feature engineering)
            self.feature_columns = get_feature_columns()
            
            self.logger.info(f"Всички модели заредени успешно ({len(self.models)} модела)")
            
        except Exception as e:
            self.logger.error(f"Грешка при зареждане на модели: {e}")
            raise
    
    def _load_team_data(self):
        """Зареждане на team data (Elo, stats)"""
        self.logger.info("Зареждане на team data...")
        
        try:
            # Зареждане на последните данни
            df = pd.read_parquet("data/processed/test_final_predictions.parquet")
            
            # Вземаме последните Elo ratings за всеки отбор
            home_teams = df.groupby('home_team_id').last()
            away_teams = df.groupby('away_team_id').last()
            
            # Обработка на home teams
            for team_id, row in home_teams.iterrows():
                team_name = row.get('home_team', f'Team_{team_id}')
                self.elo_ratings[team_name] = {
                    'elo': row.get('home_elo_before', 1500),
                    'form': row.get('home_form_5', 0),
                    'goals_avg': row.get('home_goals_scored_avg_5', 1.5),
                    'xg_proxy': row.get('home_xg_proxy', 1.5),
                    'shooting_efficiency': row.get('home_shooting_efficiency', 0.3)
                }
            
            # Обработка на away teams
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
        """Зареждане на реални имена на отборите"""
        try:
            with open('models/team_names_mapping.json', 'r') as f:
                team_mapping = json.load(f)
            
            # Convert keys to int and create lookup by Team_ID format
            for team_id_str, info in team_mapping.items():
                team_id = int(team_id_str)
                team_key = f"Team_{team_id}"
                self.team_names[team_key] = info
            
            self.logger.info(f"✓ Реални имена заредени за {len(self.team_names)} отбора")
            
        except Exception as e:
            self.logger.warning(f"Не мога да заредя team names: {e}")
            self.team_names = {}
    
    def get_team_display_name(self, team_key: str) -> str:
        """
        Получаване на display име за отбор
        
        Args:
            team_key: Team key (напр. "Team_363")
        
        Returns:
            Display име или оригиналния key ако няма mapping
        """
        if team_key in self.team_names:
            info = self.team_names[team_key]
            return info['display_name']
        return team_key
    
    def _create_match_features(
        self,
        home_team: str,
        away_team: str,
        league: Optional[str] = None,
        date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Създаване на features за мач
        
        Args:
            home_team: Домакин
            away_team: Гост
            league: Лига
            date: Дата
        
        Returns:
            DataFrame с features
        """
        # Вземаме данни за отборите
        home_data = self.elo_ratings.get(home_team, {
            'elo': 1500, 'form': 0, 'goals_avg': 1.5,
            'xg_proxy': 1.5, 'shooting_efficiency': 0.3
        })
        
        away_data = self.elo_ratings.get(away_team, {
            'elo': 1500, 'form': 0, 'goals_avg': 1.5,
            'xg_proxy': 1.5, 'shooting_efficiency': 0.3
        })
        
        # Създаваме базови features
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
            'home_goals_conceded_avg_5': 1.5,
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
        
        # Попълваме останалите features с default стойности
        for col in self.feature_columns:
            if col not in features:
                features[col] = 0.0
        
        df = pd.DataFrame([features])
        
        return df
    
    def predict(
        self,
        home_team: str,
        away_team: str,
        league: Optional[str] = None,
        date: Optional[str] = None
    ) -> Dict:
        """
        Пълна прогноза за мач
        
        Args:
            home_team: Домакин
            away_team: Гост
            league: Лига
            date: Дата
        
        Returns:
            Dictionary с predictions
        """
        self.logger.info(f"Prediction за: {home_team} vs {away_team}")
        
        # Създаване на features
        match_df = self._create_match_features(home_team, away_team, league, date)
        
        # Poisson predictions - използваме dummy team IDs
        # В реална среда би трябвало да имаме mapping на team names към IDs
        home_team_id = hash(home_team) % 10000
        away_team_id = hash(away_team) % 10000
        
        try:
            poisson_pred = self.models['poisson'].predict_match_probabilities(
                home_team_id, away_team_id
            )
            # Конвертиране на формата
            poisson_pred_formatted = {
                'probs_1x2': np.array([
                    poisson_pred['prob_home_win'],
                    poisson_pred['prob_draw'],
                    poisson_pred['prob_away_win']
                ]),
                'prob_over25': poisson_pred['prob_over_25'],
                'prob_btts': poisson_pred['prob_btts_yes'],
                'lambda_home': poisson_pred['lambda_home'],
                'lambda_away': poisson_pred['lambda_away'],
                'expected_goals': poisson_pred['expected_total_goals']
            }
            poisson_pred = poisson_pred_formatted
        except Exception as e:
            self.logger.warning(f"Poisson prediction failed: {e}, using fallback")
            # Fallback ако отборите не са в Poisson модела
            poisson_pred = {
                'probs_1x2': np.array([0.33, 0.33, 0.34]),
                'prob_over25': 0.5,
                'prob_btts': 0.5,
                'lambda_home': 1.5,
                'lambda_away': 1.2,
                'expected_goals': 2.7
            }
        
        # Добавяме Poisson predictions към features
        match_df['poisson_prob_1'] = poisson_pred['probs_1x2'][0]
        match_df['poisson_prob_x'] = poisson_pred['probs_1x2'][1]
        match_df['poisson_prob_2'] = poisson_pred['probs_1x2'][2]
        match_df['poisson_prob_over25'] = poisson_pred['prob_over25']
        match_df['poisson_prob_btts'] = poisson_pred['prob_btts']
        match_df['poisson_expected_goals'] = poisson_pred['expected_goals']
        match_df['poisson_lambda_home'] = poisson_pred['lambda_home']
        match_df['poisson_lambda_away'] = poisson_pred['lambda_away']
        
        # ML predictions
        # Prepare features (използваме legacy метод за съвместимост)
        X_all, _ = prepare_features(match_df, self.feature_columns, use_intelligent_imputation=False, legacy_fill_na=True)
        
        # Align features for each model (използваме legacy метод)
        X_1x2, _ = align_features(X_all, self.feature_lists['1x2'], use_intelligent_imputation=False)
        X_ou25, _ = align_features(X_all, self.feature_lists['ou25'], use_intelligent_imputation=False)
        X_btts, _ = align_features(X_all, self.feature_lists['btts'], use_intelligent_imputation=False)
        
        ml_1x2 = self.models['1x2'].predict_proba(X_1x2)[0]
        ml_ou25 = self.models['ou25'].predict_proba(X_ou25)[0, 1]
        ml_btts_raw = self.models['btts'].predict_proba(X_btts)[0, 1]
        
        # BTTS Calibration layer (reduce overconfidence)
        ml_btts_calibrated = 0.5 + (ml_btts_raw - 0.5) * 0.85
        ml_btts_calibrated = np.clip(ml_btts_calibrated, 0.05, 0.95)
        
        # Blend with Poisson for BTTS
        ml_btts = 0.8 * ml_btts_calibrated + 0.2 * poisson_pred['prob_btts']
        
        # Ensemble predictions
        ensemble_1x2 = self.models['ensemble'].predict(
            poisson_pred['probs_1x2'].reshape(1, -1),
            ml_1x2.reshape(1, -1)
        )[0]
        
        ensemble_ou25 = self.models['ensemble'].predict(
            np.array([[poisson_pred['prob_over25']]]),
            np.array([[ml_ou25]])
        )[0, 0]
        
        ensemble_btts = self.models['ensemble'].predict(
            np.array([[poisson_pred['prob_btts']]]),
            np.array([[ml_btts]])
        )[0, 0]
        
        # FII
        fii_score, fii_conf = self.models['fii'].calculate_fii(
            elo_diff=match_df['elo_diff'].iloc[0],
            form_diff=match_df['home_form_5'].iloc[0] - match_df['away_form_5'].iloc[0],
            xg_efficiency_diff=match_df['home_xg_proxy'].iloc[0] - match_df['away_xg_proxy'].iloc[0],
            finishing_efficiency_diff=match_df['home_shooting_efficiency'].iloc[0] - match_df['away_shooting_efficiency'].iloc[0],
            is_home=1
        )
        
        # Форматиране на резултата
        result = {
            'match_info': {
                'home_team': home_team,
                'away_team': away_team,
                'league': league or 'Unknown',
                'date': date or datetime.now().strftime('%Y-%m-%d')
            },
            'prediction_1x2': {
                'prob_home_win': float(ensemble_1x2[0]),
                'prob_draw': float(ensemble_1x2[1]),
                'prob_away_win': float(ensemble_1x2[2]),
                'predicted_outcome': ['1', 'X', '2'][np.argmax(ensemble_1x2)],
                'confidence': float(np.max(ensemble_1x2))
            },
            'prediction_ou25': {
                'prob_over': float(ensemble_ou25),
                'prob_under': float(1 - ensemble_ou25),
                'predicted_outcome': 'Over' if ensemble_ou25 > 0.5 else 'Under',
                'confidence': float(max(ensemble_ou25, 1 - ensemble_ou25))
            },
            'prediction_btts': {
                'prob_yes': float(ensemble_btts),
                'prob_no': float(1 - ensemble_btts),
                'predicted_outcome': self._get_btts_outcome(
                    ensemble_btts, 
                    match_df['elo_diff'].iloc[0]
                ),
                'confidence': float(max(ensemble_btts, 1 - ensemble_btts))
            },
            'fii': {
                'score': float(fii_score),
                'confidence_level': fii_conf,
                'components': {
                    'elo_diff': float(match_df['elo_diff'].iloc[0]),
                    'form_diff': float(match_df['home_form_5'].iloc[0] - match_df['away_form_5'].iloc[0]),
                    'xg_efficiency_diff': float(match_df['home_xg_proxy'].iloc[0] - match_df['away_xg_proxy'].iloc[0]),
                    'finishing_efficiency_diff': float(match_df['home_shooting_efficiency'].iloc[0] - match_df['away_shooting_efficiency'].iloc[0])
                }
            },
            'model_versions': {
                'poisson': 'v1',
                '1x2': 'v1',
                'ou25': 'v1',
                'btts': 'v1',
                'ensemble': 'v1'
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def _get_btts_outcome(self, prob_btts: float, elo_diff: float) -> str:
        """
        Dynamic threshold за BTTS базиран на Elo difference
        
        Args:
            prob_btts: BTTS вероятност
            elo_diff: Elo разлика
        
        Returns:
            'Yes' или 'No'
        """
        # Dynamic threshold based on match context
        if abs(elo_diff) < 200:
            threshold = 0.50  # Равностойни отбори
        else:
            threshold = 0.53  # Голяма разлика (по-малко вероятно и двата да отбележат)
        
        return 'Yes' if prob_btts > threshold else 'No'
    
    def get_model_info(self) -> Dict:
        """Информация за моделите"""
        
        models_list = []
        
        # Зареждане на метрики
        model_configs = [
            ('Poisson', 'v1', 'models/model_poisson_v1/metrics.json'),
            ('1X2', 'v1', 'models/model_1x2_v1/metrics.json'),
            ('OU2.5', 'v1', 'models/model_ou25_v1/metrics.json'),
            ('BTTS', 'v1', 'models/model_btts_v1/metrics.json'),
            ('Ensemble', 'v1', 'models/ensemble_v1/metrics.json')
        ]
        
        for name, version, metrics_path in model_configs:
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                
                model_info = {
                    'model_name': name,
                    'version': version,
                    'trained_date': 'N/A',
                    'accuracy': None,
                    'metrics': metrics.get('validation', {}) if name != 'Ensemble' else metrics.get('test', {})
                }
                
                models_list.append(model_info)
            except Exception as e:
                self.logger.warning(f"Не мога да заредя метрики за {name}: {e}")
        
        return {
            'models': models_list,
            'total_models': len(models_list)
        }
    
    def health_check(self) -> Dict:
        """Health check"""
        
        return {
            'status': 'healthy' if len(self.models) > 0 else 'unhealthy',
            'models_loaded': len(self.models) > 0,
            'num_models': len(self.models),
            'num_teams': len(self.elo_ratings)
        }


if __name__ == "__main__":
    # Test
    service = PredictionService()
    
    result = service.predict(
        home_team="Manchester United",
        away_team="Liverpool",
        league="Premier League"
    )
    
    print(json.dumps(result, indent=2))
