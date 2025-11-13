"""
Prediction Service - Централизирана логика за predictions
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
import json
from typing import Dict, Tuple, Optional
from datetime import datetime

from core.utils import setup_logging
from core.feature_validator import FeatureValidator
from core.ml_utils import align_features, prepare_features
from monitoring.prediction_logger import PredictionLogger
from core.league_utils import get_league_slug, get_per_league_model_path
from core.ensemble import EnsembleModel, FootballIntelligenceIndex
from core.team_name_resolver import TeamNameResolver
from core.btts_features import BTTSFeatureEngineer
from core.btts_ensemble import BTTSEnsemble


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
        
        # Инициализиране на prediction logger
        self.prediction_logger = PredictionLogger()
        
        # Per-league модели
        self.ou25_models_by_league = {}
        self.ou25_calibrators_by_league = {}
        
        # Team name resolver
        self.team_resolver = TeamNameResolver()
        
        # BTTS подобрения
        self.btts_feature_engineer = BTTSFeatureEngineer()
        self.btts_ensemble = BTTSEnsemble()
        self.improved_btts_model = None
        
        # Зареждане на всички компоненти
        self._load_models()
        self._load_team_data()
        self._load_per_league_models()
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
                    self.logger.warning(f"⚠ Feature list not found for {model_name}, using empty list")
                    self.feature_lists[model_name] = []
            
            # Ensemble
            self.models['ensemble'] = joblib.load('models/ensemble_v1/ensemble_model.pkl')
            self.models['fii'] = joblib.load('models/ensemble_v1/fii_model.pkl')
            
            # Improved BTTS model (ако е налично)
            try:
                improved_btts_path = 'models/model_btts_improved/btts_model_improved.pkl'
                if os.path.exists(improved_btts_path):
                    self.improved_btts_model = joblib.load(improved_btts_path)
                    
                    # Зарежда feature list за improved BTTS
                    improved_features_path = 'models/model_btts_improved/feature_columns.json'
                    if os.path.exists(improved_features_path):
                        with open(improved_features_path, 'r') as f:
                            feature_data = json.load(f)
                            if isinstance(feature_data, dict) and 'features' in feature_data:
                                self.feature_lists['btts_improved'] = feature_data['features']
                            else:
                                self.feature_lists['btts_improved'] = feature_data
                    
                    self.logger.info(f"✓ Improved BTTS model зареден с {len(self.feature_lists.get('btts_improved', []))} features")
                else:
                    self.logger.info("⚠ Improved BTTS model не е намерен, използвам стандартния")
            except Exception as e:
                self.logger.warning(f"⚠ Грешка при зареждане на improved BTTS: {e}")
            
            # Feature columns (all features for feature engineering)
            try:
                from core.ml_utils import get_feature_columns
                self.feature_columns = get_feature_columns()
            except ImportError:
                # Fallback ако функцията не съществува
                self.feature_columns = []
                self.logger.warning("get_feature_columns not found, using empty feature list")
            
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
    
    def _load_per_league_models(self):
        """Зареждане на per-league OU2.5 модели"""
        try:
            from core.utils import load_config
            config = load_config("config/model_config.yaml")
            
            # Проверява дали per-league е включен
            per_league_config = config.get('model_ou25', {}).get('per_league', {})
            if not per_league_config.get('enabled', False):
                self.logger.info("Per-league модели са изключени в конфигурацията")
                return
            
            target_leagues = per_league_config.get('target_leagues', [])
            lazy_loading = per_league_config.get('lazy_loading', True)
            
            if lazy_loading:
                self.logger.info(f"Per-league модели ще се зареждат lazy за {len(target_leagues)} лиги")
                # Запазваме само списъка с поддържани лиги
                self._supported_per_league = set(target_leagues)
            else:
                # Зарежда всички модели веднага
                self.logger.info(f"Зареждане на per-league модели за {len(target_leagues)} лиги...")
                loaded_count = 0
                
                for league_slug in target_leagues:
                    if self._load_league_model(league_slug):
                        loaded_count += 1
                
                self.logger.info(f"✓ Заредени {loaded_count}/{len(target_leagues)} per-league OU2.5 модела")
                
        except Exception as e:
            self.logger.warning(f"Грешка при зареждане на per-league модели: {e}")
            # Fallback към празни структури
            self.ou25_models_by_league = {}
            self.ou25_calibrators_by_league = {}
    
    def _load_league_model(self, league_slug: str) -> bool:
        """
        Зарежда модел за конкретна лига
        
        Args:
            league_slug: League slug (premier_league, la_liga, etc.)
        
        Returns:
            True ако модела е зареден успешно
        """
        try:
            model_dir = get_per_league_model_path(league_slug, 'ou25', 'v1')
            model_file = f"{model_dir}/ou25_model.pkl"
            calibrator_file = f"{model_dir}/calibrator.pkl"
            feature_file = f"{model_dir}/feature_columns.json"
            
            # Проверява дали файловете съществуват
            if not os.path.exists(model_file):
                return False
            
            # Зарежда модела
            model = joblib.load(model_file)
            self.ou25_models_by_league[league_slug] = model
            
            # Зарежда калибратора (ако съществува)
            if os.path.exists(calibrator_file):
                calibrator = joblib.load(calibrator_file)
                self.ou25_calibrators_by_league[league_slug] = calibrator
            
            # Проверява feature consistency
            if os.path.exists(feature_file):
                with open(feature_file, 'r') as f:
                    league_features = json.load(f)
                
                # Сравнява с глобалните features
                global_features = self.feature_lists.get('ou25', [])
                if global_features and league_features != global_features:
                    self.logger.warning(f"Feature mismatch за {league_slug}: {len(league_features)} vs {len(global_features)}")
            
            self.logger.info(f"✓ {league_slug} OU2.5 модел зареден")
            return True
            
        except Exception as e:
            self.logger.warning(f"Грешка при зареждане на {league_slug} модел: {e}")
            return False
    
    def _get_ou25_model_for_league(self, league: Optional[str] = None) -> Tuple[object, object, str]:
        """
        Получава OU2.5 модел за дадена лига с fallback към глобален
        
        Args:
            league: League име
        
        Returns:
            Tuple (model, calibrator, source) където source е "league_ou25" или "global_ou25"
        """
        # Опитва се да намери league-specific модел
        if league:
            league_slug = get_league_slug(league)
            
            if league_slug:
                # Lazy loading ако е необходимо
                if (hasattr(self, '_supported_per_league') and 
                    league_slug in self._supported_per_league and 
                    league_slug not in self.ou25_models_by_league):
                    
                    self.logger.info(f"Lazy loading на {league_slug} OU2.5 модел...")
                    self._load_league_model(league_slug)
                
                # Проверява дали модела е зареден
                if league_slug in self.ou25_models_by_league:
                    model = self.ou25_models_by_league[league_slug]
                    calibrator = self.ou25_calibrators_by_league.get(league_slug)
                    return model, calibrator, "league_ou25"
        
        # Fallback към глобален модел
        global_model = self.models.get('ou25')
        global_calibrator = None  # Глобалният модел може да няма калибратор
        
        return global_model, global_calibrator, "global_ou25"
    
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
        # Намираме team keys за lookup на данните
        home_key = self.team_resolver.find_team_key(home_team)
        away_key = self.team_resolver.find_team_key(away_team)
        
        # Вземаме данни за отборите използвайки team keys
        home_data = self.elo_ratings.get(home_key or home_team, {
            'elo': 1500, 'form': 0, 'goals_avg': 1.5,
            'xg_proxy': 1.5, 'shooting_efficiency': 0.3
        })
        
        away_data = self.elo_ratings.get(away_key or away_team, {
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
        # Резолва имената на отборите
        resolved_home = self.resolve_team_name(home_team)
        resolved_away = self.resolve_team_name(away_team)
        
        self.logger.info(f"Prediction за: {resolved_home} vs {resolved_away}")
        
        # Валидира отборите (използва оригиналните имена)
        home_validation = self.validate_team(home_team)
        away_validation = self.validate_team(away_team)
        
        # Добавя информация за резолването
        home_validation['resolved_to'] = resolved_home
        away_validation['resolved_to'] = resolved_away
        
        # Създаване на features с резолваните имена
        match_df = self._create_match_features(resolved_home, resolved_away, league, date)
        
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
            self.logger.warning(f"Poisson prediction failed: {e}, using league-specific fallback")
            # League-specific fallback стойности
            fallback_values = self._get_league_fallback(league)
            poisson_pred = {
                'probs_1x2': np.array(fallback_values['probs_1x2']),
                'prob_over25': fallback_values['prob_over25'],
                'prob_btts': fallback_values['prob_btts'],
                'lambda_home': fallback_values['lambda_home'],
                'lambda_away': fallback_values['lambda_away'],
                'expected_goals': fallback_values['lambda_home'] + fallback_values['lambda_away']
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
        
        # OU2.5 prediction с per-league модел или fallback
        ou25_model, ou25_calibrator, ou25_source = self._get_ou25_model_for_league(league)
        
        if ou25_model is not None:
            # За per-league модели използваме всички features (72)
            if ou25_source == "league_ou25":
                ml_ou25_raw = ou25_model.predict(X_all.iloc[:1])[0] if hasattr(ou25_model, 'predict') else ou25_model.predict_proba(X_all.iloc[:1])[0, 1]
            else:
                ml_ou25_raw = ou25_model.predict(X_ou25)[0] if hasattr(ou25_model, 'predict') else ou25_model.predict_proba(X_ou25)[0, 1]
            
            # Прилага калибрация ако е налична
            if ou25_calibrator is not None:
                # IsotonicRegression калибрация
                ml_ou25 = ou25_calibrator.predict([ml_ou25_raw])[0]
            else:
                ml_ou25 = ml_ou25_raw
        else:
            # Fallback към глобален модел
            ml_ou25 = self.models['ou25'].predict_proba(X_ou25)[0, 1]
            ou25_source = "global_ou25"
        ml_btts_raw = self.models['btts'].predict_proba(X_btts)[0, 1]
        
        # BTTS Calibration layer (reduce overconfidence)
        ml_btts_calibrated = 0.5 + (ml_btts_raw - 0.5) * 0.85
        ml_btts_calibrated = np.clip(ml_btts_calibrated, 0.05, 0.95)
        
        # Blend with Poisson for BTTS
        ml_btts = 0.8 * ml_btts_calibrated + 0.2 * poisson_pred['prob_btts']
        
        # Ensemble predictions with dynamic weighting
        # Map league name to ID for ensemble (простo mapping за демонстрация)
        league_id_map = {
            'Premier League': 1,
            'La Liga': 2,
            'Serie A': 3,
            'Bundesliga': 4,
            'Ligue 1': 5
        }
        league_id = league_id_map.get(league, 0)
        
        ensemble_1x2 = self.models['ensemble'].predict(
            poisson_pred['probs_1x2'].reshape(1, -1),
            ml_1x2.reshape(1, -1),
            league_id=league_id
        )[0]
        
        ensemble_ou25 = self.models['ensemble'].predict(
            np.array([[poisson_pred['prob_over25']]]),
            np.array([[ml_ou25]]),
            league_id=league_id
        )[0, 0]
        
        # Подобрена BTTS прогноза
        btts_improved = self.predict_btts_improved(match_df, poisson_pred['prob_btts'])
        ensemble_btts = btts_improved['prob_yes']
        
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
                'home_team': resolved_home,
                'away_team': resolved_away,
                'original_home_team': home_team,
                'original_away_team': away_team,
                'league': league or 'Unknown',
                'date': date or datetime.now().strftime('%Y-%m-%d')
            },
            'team_validation': {
                'home_team': home_validation,
                'away_team': away_validation
            },
            'prediction_1x2': {
                'prob_home_win': float(ensemble_1x2[0]),
                'prob_draw': float(ensemble_1x2[1]),
                'prob_away_win': float(ensemble_1x2[2]),
                'predicted_outcome': ['1', 'X', '2'][np.argmax(ensemble_1x2)],
                'confidence': float(np.max(ensemble_1x2) - np.mean(ensemble_1x2))
            },
            'prediction_ou25': {
                'prob_over': float(ensemble_ou25),
                'prob_under': float(1 - ensemble_ou25),
                'predicted_outcome': 'Over' if ensemble_ou25 > 0.5 else 'Under',
                'confidence': float(abs(ensemble_ou25 - 0.5) * 2)
            },
            'prediction_btts': {
                'prob_yes': float(btts_improved['prob_yes']),
                'prob_no': float(btts_improved['prob_no']),
                'predicted_outcome': btts_improved['predicted_outcome'],
                'confidence': float(btts_improved['confidence']),
                'confidence_level': btts_improved['confidence_level'],
                'model_source': btts_improved['model_source'],
                'threshold_recommendation': btts_improved['threshold_recommendation'],
                'enhanced_features': btts_improved.get('features_used', 0)
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
            'model_sources': {
                'ou25': ou25_source  # "league_ou25" или "global_ou25"
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Логва прогнозата за калибрационен мониторинг
        try:
            self.prediction_logger.log_prediction(
                home_team=home_team,
                away_team=away_team,
                league=league or 'Unknown',
                prediction_data=result,
                match_date=date
            )
        except Exception as e:
            self.logger.warning(f"Failed to log prediction: {e}")
        
        return result
    
    def _get_league_fallback(self, league: Optional[str] = None) -> Dict[str, any]:
        """
        Получава league-specific fallback стойности
        
        Args:
            league: Име на лигата
        
        Returns:
            Dictionary с fallback стойности
        """
        # League-specific статистики (базирани на исторически данни)
        league_stats = {
            'Premier League': {
                'probs_1x2': [0.46, 0.27, 0.27],  # Home bias
                'prob_over25': 0.58,
                'prob_btts': 0.52,
                'lambda_home': 1.7,
                'lambda_away': 1.3
            },
            'La Liga': {
                'probs_1x2': [0.44, 0.28, 0.28],
                'prob_over25': 0.54,
                'prob_btts': 0.48,
                'lambda_home': 1.6,
                'lambda_away': 1.2
            },
            'Serie A': {
                'probs_1x2': [0.42, 0.30, 0.28],
                'prob_over25': 0.51,
                'prob_btts': 0.46,
                'lambda_home': 1.5,
                'lambda_away': 1.1
            },
            'Bundesliga': {
                'probs_1x2': [0.48, 0.26, 0.26],
                'prob_over25': 0.62,
                'prob_btts': 0.55,
                'lambda_home': 1.8,
                'lambda_away': 1.4
            },
            'Ligue 1': {
                'probs_1x2': [0.43, 0.29, 0.28],
                'prob_over25': 0.49,
                'prob_btts': 0.44,
                'lambda_home': 1.4,
                'lambda_away': 1.0
            }
        }
        
        # Default fallback
        default_stats = {
            'probs_1x2': [0.44, 0.28, 0.28],
            'prob_over25': 0.54,
            'prob_btts': 0.49,
            'lambda_home': 1.5,
            'lambda_away': 1.2
        }
        
        return league_stats.get(league, default_stats)

    def _confidence_binary(self, p_ml: float, p_poi: float) -> float:
        """
        Изчислява confidence за binary prediction базиран на ентропия и agreement
        
        Args:
            p_ml: ML model вероятност
            p_poi: Poisson model вероятност
        
        Returns:
            Confidence score (0-1)
        """
        import math
        
        # Soft clip ML probability
        p = np.clip(0.5 + (p_ml - 0.5) * 0.9, 0.02, 0.98)
        
        # Entropy-based confidence
        entropy = -(p * math.log(p) + (1-p) * math.log(1-p)) / math.log(2)
        ent_conf = 1 - entropy
        
        # Agreement-based confidence
        agree = 1 - abs(p_ml - p_poi)
        
        # Combined confidence
        return float(0.6 * ent_conf + 0.4 * agree)
    
    def _confidence_1x2(self, probs_ml: np.ndarray, probs_poi: np.ndarray) -> float:
        """
        Изчислява confidence за 1X2 prediction базиран на ентропия и agreement
        
        Args:
            probs_ml: ML model вероятности [prob_1, prob_X, prob_2]
            probs_poi: Poisson model вероятности [prob_1, prob_X, prob_2]
        
        Returns:
            Confidence score (0-1)
        """
        import math
        
        # Soft clip probabilities
        probs_ml = np.clip(probs_ml, 0.02, 0.98)
        probs_ml = probs_ml / probs_ml.sum()  # Renormalize
        
        # Entropy-based confidence
        entropy = -np.sum(probs_ml * np.log(probs_ml)) / math.log(3)
        ent_conf = 1 - entropy
        
        # Agreement-based confidence (mean L1 distance)
        agree = 1 - np.mean(np.abs(probs_ml - probs_poi))
        
        # Combined confidence
        return float(0.6 * ent_conf + 0.4 * agree)

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
    
    def resolve_team_name(self, team_name: str) -> str:
        """Резолва името на отбор използвайки TeamNameResolver"""
        return self.team_resolver.get_team_display_name(team_name)
    
    def find_similar_teams(self, team_name: str, limit: int = 5) -> list:
        """Намира подобни отбори"""
        return self.team_resolver.get_similar_teams(team_name, limit)
    
    def validate_team(self, team_name: str) -> Dict:
        """Валидира отбор и връща информация"""
        info = self.team_resolver.get_team_info(team_name)
        
        result = {
            'original_name': team_name,
            'resolved_name': info['resolved_name'],
            'is_valid': info['is_valid'],
            'team_key': info.get('team_key')
        }
        
        # Добавя предупреждения за проблемни отбори
        warnings = []
        if info.get('is_women'):
            warnings.append('Женски отбор')
        if info.get('is_youth'):
            warnings.append('Младежки отбор')
        if info.get('is_reserve'):
            warnings.append('Резервен отбор')
        if info.get('is_duplicate'):
            warnings.append('Дубликат на друг отбор')
        if info.get('is_unknown'):
            warnings.append('Неразпознат отбор')
            
        if warnings:
            result['warnings'] = warnings
            
        # Ако не е валиден, предлага алтернативи
        if not info['is_valid']:
            similar = self.find_similar_teams(team_name, 3)
            if similar:
                result['suggestions'] = [name for name, score in similar]
        
        return result
    
    def predict_btts_improved(self, match_df: pd.DataFrame, poisson_btts_prob: float) -> Dict:
        """
        Подобрена BTTS прогноза с enhanced features и ensemble logic
        
        Args:
            match_df: Match features DataFrame
            poisson_btts_prob: Poisson BTTS вероятност
            
        Returns:
            Enhanced BTTS prediction
        """
        try:
            if self.improved_btts_model is None:
                # Fallback към стандартния модел
                return self._predict_btts_standard(match_df, poisson_btts_prob)
            
            # Прилага BTTS feature engineering
            enhanced_df = self.btts_feature_engineer.create_btts_features(match_df)
            
            # Подготвя features за improved модел
            improved_features = self.feature_lists.get('btts_improved', [])
            available_features = [f for f in improved_features if f in enhanced_df.columns]
            
            if len(available_features) < len(improved_features) * 0.8:  # Минимум 80% features
                self.logger.warning(f"Недостатъчно features за improved BTTS: {len(available_features)}/{len(improved_features)}")
                return self._predict_btts_standard(match_df, poisson_btts_prob)
            
            # ML prediction с improved модел
            X_improved = enhanced_df[available_features].fillna(0)
            ml_btts_prob = self.improved_btts_model.predict_proba(X_improved)[:, 1][0]
            
            # Enhanced ensemble logic
            ensemble_result = self.btts_ensemble.enhanced_btts_ensemble(
                ml_prob=ml_btts_prob,
                poisson_prob=poisson_btts_prob,
                ml_weight=0.85  # По-висока тежест за improved модел
            )
            
            # Threshold препоръки
            threshold_rec = self.btts_ensemble.get_threshold_recommendation(
                ensemble_result['probability'], 
                ensemble_result['confidence']
            )
            
            return {
                'prob_yes': ensemble_result['probability'],
                'prob_no': 1 - ensemble_result['probability'],
                'predicted_outcome': ensemble_result['predicted_outcome'],
                'confidence': ensemble_result['confidence'],
                'confidence_level': ensemble_result['confidence_level'],
                'model_source': 'improved_btts',
                'components': ensemble_result['components'],
                'threshold_recommendation': threshold_rec['recommended_threshold'],
                'features_used': len(available_features)
            }
            
        except Exception as e:
            self.logger.warning(f"Грешка в improved BTTS: {e}, fallback към стандартен")
            return self._predict_btts_standard(match_df, poisson_btts_prob)
    
    def _predict_btts_standard(self, match_df: pd.DataFrame, poisson_btts_prob: float) -> Dict:
        """Стандартна BTTS прогноза (fallback)"""
        try:
            # Използва стандартния BTTS модел
            btts_features = self.feature_lists.get('btts', [])
            if btts_features:
                X_btts = match_df[btts_features].fillna(0)
                ml_btts_prob = self.models['btts'].predict_proba(X_btts)[:, 1][0]
            else:
                ml_btts_prob = 0.5  # Default
            
            # Стандартна ensemble логика
            ensemble_prob = 0.8 * ml_btts_prob + 0.2 * poisson_btts_prob
            confidence = abs(ensemble_prob - 0.5) * 2
            
            return {
                'prob_yes': ensemble_prob,
                'prob_no': 1 - ensemble_prob,
                'predicted_outcome': 'Yes' if ensemble_prob > 0.5 else 'No',
                'confidence': confidence,
                'confidence_level': 'Medium' if confidence > 0.3 else 'Low',
                'model_source': 'standard_btts',
                'threshold_recommendation': 0.5,
                'features_used': len(btts_features)
            }
            
        except Exception as e:
            self.logger.error(f"Грешка и в стандартния BTTS: {e}")
            return {
                'prob_yes': 0.5,
                'prob_no': 0.5,
                'predicted_outcome': 'Unknown',
                'confidence': 0.0,
                'confidence_level': 'Low',
                'model_source': 'fallback',
                'threshold_recommendation': 0.5,
                'features_used': 0
            }
    
    def health_check(self) -> Dict:
        """Health check"""
        
        return {
            'status': 'healthy' if len(self.models) > 0 else 'unhealthy',
            'models_loaded': len(self.models) > 0,
            'num_models': len(self.models),
            'num_teams': len(self.elo_ratings),
            'team_resolver_loaded': self.team_resolver is not None,
            'improved_btts_loaded': self.improved_btts_model is not None,
            'btts_features_available': len(self.feature_lists.get('btts_improved', []))
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
