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
import pickle
from typing import Dict, Tuple, Optional, List
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
from core.poisson_v2 import PoissonV2Model
from core.calibration_multiclass import MulticlassCalibrator
from core.features_1x2 import Features1X2
from core.hybrid_1x2_predictor import Hybrid1X2Predictor


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
        
        # Initialize Hybrid 1X2 Predictor
        try:
            self.hybrid_predictor = Hybrid1X2Predictor()
            self.hybrid_enabled = self.hybrid_predictor.is_available()
            self.logger.info(f"🎯 Hybrid 1X2 Predictor: {'✅ Enabled' if self.hybrid_enabled else '❌ Disabled'}")
        except Exception as e:
            self.logger.warning(f"⚠️ Hybrid 1X2 Predictor not available: {e}")
            self.hybrid_predictor = None
            self.hybrid_enabled = False
        
        # Per-league модели
        self.ou25_models_by_league = {}
        self.ou25_calibrators_by_league = {}
        
        # Team name resolver
        self.team_resolver = TeamNameResolver()
        
        # BTTS подобрения
        self.btts_feature_engineer = BTTSFeatureEngineer()
        self.btts_ensemble = BTTSEnsemble()
        self.improved_btts_model = None
        
        # 1X2 v2 components
        self.x1x2_v2_models = {}  # Per-league binary models
        self.x1x2_v2_calibrators = {}  # Per-league calibrators
        self.poisson_v2_models = {}  # Per-league Poisson v2 models
        self.features_1x2 = Features1X2()
        self.x1x2_v2_enabled = True
        
        # Зареждане на всички компоненти
        self._load_models()
        self._load_team_data()
        self._load_per_league_models()
        self._load_1x2_v2_models()
        self._load_team_names()
        
        self.logger.info("PredictionService инициализиран успешно")
    
    def _load_models(self):
        """Зареждане на всички модели"""
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
                # Load model
                model_file = f"{model_dir}/{model_name}_model.pkl"
                self.models[model_name] = joblib.load(model_file)
                
                # Специално зареждане на калибратори за 1X2 модел
                if model_name == '1x2':
                    self._load_1x2_calibrators(model_dir)
                
                # Load feature list
                feature_list_file = f"{model_dir}/feature_list.json"
                try:
                    with open(feature_list_file, 'r') as f:
                        self.feature_lists[model_name] = json.load(f)
                    self.logger.info(f"✓ {model_name} model: {len(self.feature_lists[model_name])} features")
                except FileNotFoundError:
                    self.logger.warning(f"⚠ Feature list not found for {model_name}, using empty list")
                    self.feature_lists[model_name] = []
            
            # Load BTTS model (prioritize improved version)
            self._load_btts_models()
            
            # Зареждане на глобален OU2.5 калибратор
            self._load_global_ou25_calibrator()
            
            # Ensemble
            self.models['ensemble'] = joblib.load('models/ensemble_v1/ensemble_model.pkl')
            self.models['fii'] = joblib.load('models/ensemble_v1/fii_model.pkl')
            
            
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
    
    def _load_btts_models(self):
        """Зареждане на BTTS модели с приоритет на improved версията"""
        btts_loaded = False
        
        # Опит за зареждане на improved BTTS модел
        try:
            improved_btts_path = 'models/model_btts_improved/btts_model_improved.pkl'
            improved_features_path = 'models/model_btts_improved/feature_columns.json'
            
            if os.path.exists(improved_btts_path):
                self.improved_btts_model = joblib.load(improved_btts_path)
                
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
                    # Няма improved модел за fallback
                    self.improved_btts_model = None
                else:
                    self.logger.error("❌ Нито improved, нито legacy BTTS модел не могат да се заредят")
                    self.models['btts'] = None
                    self.improved_btts_model = None
                    self.feature_lists['btts'] = []
                    
            except Exception as e:
                self.logger.error(f"❌ Грешка при зареждане на legacy BTTS: {e}")
                self.models['btts'] = None
                self.improved_btts_model = None
                self.feature_lists['btts'] = []
    
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
    
    def _load_1x2_v2_models(self):
        """Зареждане на 1X2 v2 модели и компоненти"""
        if not self.x1x2_v2_enabled:
            self.logger.info("1X2 v2 е изключен")
            return
            
        self.logger.info("🔄 Зареждане на 1X2 v2 модели...")
        
        try:
            # Major leagues за per-league modeling
            major_leagues = [
                'premier_league', 'la_liga', 'serie_a', 'bundesliga',
                'ligue_1', 'eredivisie', 'primeira_liga', 'championship'
            ]
            
            loaded_leagues = 0
            
            # Зареждане на модели за всяка лига
            for league in major_leagues:
                if self._load_1x2_v2_league_models(league):
                    loaded_leagues += 1
            
            # Зареждане на глобален fallback модел
            if self._load_1x2_v2_league_models('global'):
                loaded_leagues += 1
            
            # Зареждане на Poisson v2 модели
            self._load_poisson_v2_models()
            
            self.logger.info(f"✅ Заредени 1X2 v2 модели за {loaded_leagues} лиги/глобален")
            
        except Exception as e:
            self.logger.error(f"❌ Грешка при зареждане на 1X2 v2 модели: {e}")
            self.x1x2_v2_enabled = False
    
    def _load_1x2_v2_league_models(self, league: str) -> bool:
        """
        Зареждане на 1X2 v2 модели за конкретна лига
        
        Args:
            league: League slug или 'global'
            
        Returns:
            True ако моделите са заредени успешно
        """
        try:
            model_dir = Path(f"models/leagues/{league}/1x2_v2")
            
            if not model_dir.exists():
                self.logger.warning(f"⚠️ Директория за {league} 1X2 v2 не съществува: {model_dir}")
                return False
            
            # Зареждане на 3-те binary модела
            binary_models = {}
            model_files = {
                'homewin': model_dir / 'homewin_model.pkl',
                'draw': model_dir / 'draw_model.pkl', 
                'awaywin': model_dir / 'awaywin_model.pkl'
            }
            
            for model_name, model_file in model_files.items():
                if model_file.exists():
                    with open(model_file, 'rb') as f:
                        binary_models[model_name] = pickle.load(f)
                else:
                    self.logger.warning(f"⚠️ Липсва {model_name} модел за {league}")
                    return False
            
            # Зареждане на калибратор
            calibrator_file = model_dir / 'calibrator.pkl'
            calibrator = None
            if calibrator_file.exists():
                calibrator = MulticlassCalibrator.load_calibrator(str(calibrator_file))
            
            # Зареждане на feature list
            feature_file = model_dir / 'feature_list.json'
            feature_list = []
            if feature_file.exists():
                with open(feature_file, 'r') as f:
                    feature_list = json.load(f)
            
            # Съхраняване на моделите
            self.x1x2_v2_models[league] = {
                'models': binary_models,
                'feature_list': feature_list
            }
            
            if calibrator:
                self.x1x2_v2_calibrators[league] = calibrator
            
            self.logger.info(f"✅ Заредени 1X2 v2 модели за {league}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Грешка при зареждане на 1X2 v2 модели за {league}: {e}")
            return False
    
    def _load_poisson_v2_models(self):
        """Зареждане на Poisson v2 модели"""
        try:
            poisson_dir = Path("models/leagues/poisson_v2")
            
            if not poisson_dir.exists():
                self.logger.warning("⚠️ Poisson v2 директория не съществува")
                return
            
            # Зареждане на всички Poisson v2 модели
            for poisson_file in poisson_dir.glob("*_poisson_v2.pkl"):
                league = poisson_file.stem.replace('_poisson_v2', '')
                
                try:
                    poisson_model = PoissonV2Model.load_model(str(poisson_file))
                    self.poisson_v2_models[league] = poisson_model
                    self.logger.info(f"✅ Зареден Poisson v2 за {league}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Грешка при зареждане на Poisson v2 за {league}: {e}")
            
            self.logger.info(f"✅ Заредени {len(self.poisson_v2_models)} Poisson v2 модела")
            
        except Exception as e:
            self.logger.error(f"❌ Грешка при зареждане на Poisson v2 модели: {e}")

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
            
            self.logger.info(f"✅ Имена на отборите заредени: {len(self.team_names)} отбора")
        except Exception as e:
            self.logger.warning(f"⚠ Грешка при зареждане на имена на отборите: {e}")
            self.team_names = {}
    
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
        # Опитва се да зареди глобален калибратор
        global_calibrator = getattr(self, 'global_ou25_calibrator', None)
        
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
        
        # 1X2 prediction с калибрация
        ml_1x2_raw = self.models['1x2'].predict_proba(X_1x2)[0:1]  # Keep as 2D array
        ml_1x2 = self._apply_1x2_calibration(ml_1x2_raw)[0]  # Apply calibration and get first row
        
        # OU2.5 prediction с per-league модел или fallback
        ou25_model, ou25_calibrator, ou25_source = self._get_ou25_model_for_league(league)
        
        if ou25_model is not None:
            # За per-league модели използваме всички features (72)
            if ou25_source == "league_ou25":
                ml_ou25_raw = ou25_model.predict(X_all.iloc[:1])[0] if hasattr(ou25_model, 'predict') else ou25_model.predict_proba(X_all.iloc[:1])[0, 1]
            else:
                ml_ou25_raw = ou25_model.predict(X_ou25)[0] if hasattr(ou25_model, 'predict') else ou25_model.predict_proba(X_ou25)[0, 1]
            
            # Прилага калибрация - първо league-specific, после глобален, накрая raw
            if ou25_calibrator is not None:
                # League-specific калибратор
                ml_ou25 = ou25_calibrator.predict([ml_ou25_raw])[0]
                self.logger.debug(f"Използван {ou25_source} калибратор за OU2.5")
            elif hasattr(self, 'global_ou25_calibrator') and self.global_ou25_calibrator is not None:
                # Fallback към глобален калибратор
                ml_ou25 = self.global_ou25_calibrator.predict([ml_ou25_raw])[0]
                self.logger.debug("Използван глобален OU2.5 калибратор")
            else:
                # Няма калибратор - използва raw probability
                ml_ou25 = ml_ou25_raw
                self.logger.warning(f"Няма калибратор за OU2.5 ({ou25_source}), използвам raw probability")
        else:
            # Fallback към глобален модел
            ml_ou25_raw = self.models['ou25'].predict_proba(X_ou25)[0, 1]
            ou25_source = "global_ou25"
            
            # Прилага глобален калибратор ако е наличен
            if hasattr(self, 'global_ou25_calibrator') and self.global_ou25_calibrator is not None:
                ml_ou25 = self.global_ou25_calibrator.predict([ml_ou25_raw])[0]
                self.logger.debug("Използван глобален OU2.5 калибратор за fallback модел")
            else:
                ml_ou25 = ml_ou25_raw
                self.logger.warning("Няма калибратор за глобален OU2.5 модел, използвам raw probability")
        
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
        
        # Enhanced OU2.5 prediction with overconfidence fixes
        ensemble_ou25 = self.models['ensemble'].predict_ou25(
            np.array([[poisson_pred['prob_over25']]]),
            np.array([[ml_ou25]]),
            league=league,
            league_id=league_id
        )[0, 0]
        
        # Подобрена BTTS прогноза with league context
        btts_improved = self.predict_btts_improved(match_df, poisson_pred['prob_btts'], league=league)
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
        """Информация за моделите с пълни метрики и статус"""
        
        models_list = []
        
        # 1X2 v1 Model
        models_list.append(self._get_single_model_info(
            name='1X2',
            version='v1',
            model_key='1x2',
            metrics_path='models/model_1x2_v1/metrics.json',
            use_val=True
        ))
        
        # 1X2 v2 Per-League Models (aggregated)
        models_list.append(self._get_1x2_v2_aggregated_info())
        
        # 1X2 Hybrid Model
        models_list.append(self._get_hybrid_1x2_info())
        
        # Poisson v1
        models_list.append(self._get_single_model_info(
            name='Poisson',
            version='v1',
            model_key='poisson',
            metrics_path='models/model_poisson_v1/metrics.json',
            use_val=True,
            metric_prefix='1x2'
        ))
        
        # Poisson v2 Per-League (aggregated)
        models_list.append(self._get_poisson_v2_aggregated_info())
        
        # OU2.5 v1 Global
        models_list.append(self._get_single_model_info(
            name='OU2.5',
            version='v1',
            model_key='ou25',
            metrics_path='models/model_ou25_v1/metrics.json',
            use_val=True
        ))
        
        # OU2.5 Per-League (aggregated)
        models_list.append(self._get_ou25_per_league_info())
        
        # BTTS v1
        models_list.append(self._get_single_model_info(
            name='BTTS',
            version='v1',
            model_key='btts',
            metrics_path='models/model_btts_v1/metrics.json',
            use_val=True
        ))
        
        # BTTS v2
        models_list.append(self._get_single_model_info(
            name='BTTS',
            version='v2',
            model_key='btts_improved',
            metrics_path='models/model_btts_v2/metrics.json',
            use_val=True
        ))
        
        # Draw Specialist
        models_list.append(self._get_draw_specialist_info())
        
        # Scoreline v1
        models_list.append(self._get_scoreline_info())
        
        # Ensemble
        models_list.append(self._get_ensemble_info())
        
        return {
            'models': models_list,
            'total_models': len(models_list)
        }
    
    def _get_single_model_info(self, name: str, version: str, model_key: str, 
                                metrics_path: str, use_val: bool = True, 
                                use_test: bool = False, metric_prefix: str = None) -> Dict:
        """Зарежда информация за единичен модел"""
        
        errors = []
        metrics = {}
        accuracy = None
        trained_date = 'N/A'
        loaded = model_key in self.models
        
        try:
            with open(metrics_path, 'r') as f:
                metrics_data = json.load(f)
            
            # Избери правилния dataset
            if use_test and 'test' in metrics_data:
                dataset = metrics_data['test']
            elif use_val and 'val' in metrics_data:
                dataset = metrics_data['val']
            elif 'validation' in metrics_data:
                dataset = metrics_data['validation']
            else:
                dataset = metrics_data.get('train', {})
            
            # Извлечи метрики
            if metric_prefix:
                accuracy = dataset.get(f'accuracy_{metric_prefix}')
                metrics = {
                    'accuracy': dataset.get(f'accuracy_{metric_prefix}'),
                    'log_loss': dataset.get(f'log_loss_{metric_prefix}')
                }
            else:
                accuracy = dataset.get('accuracy')
                metrics = {
                    'accuracy': dataset.get('accuracy'),
                    'log_loss': dataset.get('log_loss'),
                    'brier_score': dataset.get('brier_score'),
                    'roc_auc': dataset.get('roc_auc')
                }
            
            # Премахни None стойности
            metrics = {k: v for k, v in metrics.items() if v is not None}
            
            # Опит за извличане на дата
            import os
            if os.path.exists(metrics_path):
                import datetime
                mtime = os.path.getmtime(metrics_path)
                trained_date = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                
        except FileNotFoundError:
            errors.append('metrics_file_missing')
            loaded = False
        except Exception as e:
            errors.append(f'error_loading_metrics: {str(e)}')
            self.logger.warning(f"Грешка при зареждане на метрики за {name} {version}: {e}")
        
        return {
            'model_name': name,
            'version': version,
            'trained_date': trained_date,
            'accuracy': accuracy,
            'metrics': metrics,
            'loaded': loaded,
            'errors': errors
        }
    
    def _get_1x2_v2_aggregated_info(self) -> Dict:
        """Агрегирана информация за 1X2 v2 per-league модели"""
        
        errors = []
        leagues_trained = len(self.x1x2_v2_models)
        loaded = leagues_trained > 0
        
        if leagues_trained == 0:
            errors.append('no_leagues_trained')
            return {
                'model_name': '1X2',
                'version': 'v2',
                'trained_date': 'N/A',
                'accuracy': None,
                'metrics': {},
                'loaded': False,
                'errors': errors,
                'leagues_trained': 0
            }
        
        # Агрегирай метрики от всички лиги
        accuracies = []
        log_losses = []
        
        for league in ['premier_league', 'la_liga', 'bundesliga', 'serie_a', 
                       'ligue_1', 'eredivisie', 'primeira_liga', 'championship']:
            metrics_path = f'models/leagues/{league}/1x2_v2/metrics.json'
            try:
                with open(metrics_path, 'r') as f:
                    metrics_data = json.load(f)
                    val_data = metrics_data.get('val', {})
                    if 'accuracy' in val_data:
                        accuracies.append(val_data['accuracy'])
                    if 'log_loss' in val_data:
                        log_losses.append(val_data['log_loss'])
            except:
                pass
        
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else None
        avg_log_loss = sum(log_losses) / len(log_losses) if log_losses else None
        
        return {
            'model_name': '1X2',
            'version': 'v2',
            'trained_date': 'N/A',
            'accuracy': avg_accuracy,
            'metrics': {
                'accuracy': avg_accuracy,
                'log_loss': avg_log_loss,
                'leagues_count': len(accuracies)
            },
            'loaded': loaded,
            'errors': errors,
            'leagues_trained': leagues_trained
        }
    
    def _get_hybrid_1x2_info(self) -> Dict:
        """Информация за Hybrid 1X2 модел"""
        
        loaded = self.hybrid_enabled and self.hybrid_predictor is not None
        errors = [] if loaded else ['hybrid_not_available']
        
        # Опит за зареждане на метрики
        metrics = {}
        accuracy = None
        
        try:
            metrics_path = 'models/1x2_hybrid_v1/metrics.json'
            with open(metrics_path, 'r') as f:
                metrics_data = json.load(f)
                val_data = metrics_data.get('val', metrics_data.get('validation', {}))
                accuracy = val_data.get('accuracy')
                metrics = {
                    'accuracy': val_data.get('accuracy'),
                    'log_loss': val_data.get('log_loss')
                }
                metrics = {k: v for k, v in metrics.items() if v is not None}
        except:
            errors.append('metrics_file_missing')
        
        return {
            'model_name': '1X2 Hybrid',
            'version': 'hybrid_v1',
            'trained_date': 'N/A',
            'accuracy': accuracy,
            'metrics': metrics,
            'loaded': loaded,
            'errors': errors
        }
    
    def _get_poisson_v2_aggregated_info(self) -> Dict:
        """Агрегирана информация за Poisson v2 per-league модели"""
        
        leagues_trained = len(self.poisson_v2_models)
        loaded = leagues_trained > 0
        errors = [] if loaded else ['no_leagues_trained']
        
        return {
            'model_name': 'Poisson',
            'version': 'v2',
            'trained_date': 'N/A',
            'accuracy': None,
            'metrics': {},
            'loaded': loaded,
            'errors': errors,
            'leagues_trained': leagues_trained
        }
    
    def _get_ou25_per_league_info(self) -> Dict:
        """Агрегирана информация за OU2.5 per-league модели"""
        
        # Провери колко лиги имат тренирани модели на диска (не в паметта)
        leagues_on_disk = []
        target_leagues = ['premier_league', 'la_liga', 'serie_a', 'bundesliga', 
                         'ligue_1', 'eredivisie', 'primeira_liga', 'championship']
        
        for league in target_leagues:
            model_path = f'models/leagues/{league}/ou25_v1/ou25_model.pkl'
            if os.path.exists(model_path):
                leagues_on_disk.append(league)
        
        leagues_trained = len(leagues_on_disk)
        loaded = leagues_trained > 0
        
        if leagues_trained == 0:
            return {
                'model_name': 'OU2.5 Per-League',
                'version': 'v1',
                'trained_date': 'N/A',
                'accuracy': None,
                'metrics': {},
                'loaded': False,
                'errors': ['no_leagues_trained'],
                'leagues_trained': 0
            }
        
        # Агрегирай метрики от всички тренирани лиги
        accuracies = []
        log_losses = []
        
        for league in leagues_on_disk:
            metrics_path = f'models/leagues/{league}/ou25_v1/metrics.json'
            try:
                with open(metrics_path, 'r') as f:
                    metrics_data = json.load(f)
                    val_data = metrics_data.get('val', {})
                    if 'accuracy' in val_data:
                        accuracies.append(val_data['accuracy'])
                    if 'log_loss' in val_data:
                        log_losses.append(val_data['log_loss'])
            except:
                pass
        
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else None
        avg_log_loss = sum(log_losses) / len(log_losses) if log_losses else None
        
        return {
            'model_name': 'OU2.5 Per-League',
            'version': 'v1',
            'trained_date': 'N/A',
            'accuracy': avg_accuracy,
            'metrics': {
                'accuracy': avg_accuracy,
                'log_loss': avg_log_loss,
                'leagues_count': float(len(accuracies))
            },
            'loaded': loaded,
            'errors': [],
            'leagues_trained': leagues_trained
        }
    
    def _get_draw_specialist_info(self) -> Dict:
        """Информация за Draw Specialist модел"""
        
        # Провери дали е зареден
        loaded = hasattr(self, 'draw_predictor') and self.draw_predictor is not None
        errors = []
        
        if not loaded:
            errors.append('optional_feature_not_trained')
        
        # Draw Specialist е optional feature - не е критичен за системата
        return {
            'model_name': 'Draw Specialist',
            'version': 'v1',
            'trained_date': 'N/A',
            'accuracy': None,
            'metrics': {},
            'loaded': loaded,
            'errors': errors
        }
    
    def _get_scoreline_info(self) -> Dict:
        """Информация за Scoreline модел"""
        
        # Провери дали е зареден
        loaded = 'poisson' in self.models
        errors = [] if loaded else ['model_not_loaded']
        
        # Scoreline използва Poisson, така че вземи метриките от Poisson
        accuracy = None
        metrics = {}
        trained_date = 'N/A'
        
        try:
            metrics_path = 'models/model_poisson_v1/metrics.json'
            with open(metrics_path, 'r') as f:
                metrics_data = json.load(f)
                val_data = metrics_data.get('validation', {})
                
                # Scoreline е базиран на Poisson, така че използваме неговите метрики
                accuracy = val_data.get('accuracy_1x2')
                metrics = {
                    'accuracy_1x2': val_data.get('accuracy_1x2'),
                    'log_loss_1x2': val_data.get('log_loss_1x2')
                }
                metrics = {k: v for k, v in metrics.items() if v is not None}
                
                import os
                if os.path.exists(metrics_path):
                    import datetime
                    mtime = os.path.getmtime(metrics_path)
                    trained_date = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        except:
            pass
        
        return {
            'model_name': 'Scoreline',
            'version': 'v1',
            'trained_date': trained_date,
            'accuracy': accuracy,
            'metrics': metrics,
            'loaded': loaded,
            'errors': errors
        }
    
    def _get_ensemble_info(self) -> Dict:
        """Информация за Ensemble модел"""
        
        loaded = 'ensemble' in self.models
        errors = [] if loaded else ['model_not_loaded']
        
        accuracy = None
        metrics = {}
        trained_date = 'N/A'
        
        try:
            metrics_path = 'models/ensemble_v1/metrics.json'
            with open(metrics_path, 'r') as f:
                metrics_data = json.load(f)
                test_data = metrics_data.get('test', {})
                
                # Изчисли средна accuracy от всички задачи
                accuracies = [
                    test_data.get('1x2_accuracy'),
                    test_data.get('ou25_accuracy'),
                    test_data.get('btts_accuracy')
                ]
                accuracies = [a for a in accuracies if a is not None]
                
                if accuracies:
                    accuracy = sum(accuracies) / len(accuracies)
                
                # Върни всички метрики
                metrics = {
                    'avg_accuracy': accuracy,
                    '1x2_accuracy': test_data.get('1x2_accuracy'),
                    '1x2_log_loss': test_data.get('1x2_log_loss'),
                    'ou25_accuracy': test_data.get('ou25_accuracy'),
                    'ou25_log_loss': test_data.get('ou25_log_loss'),
                    'btts_accuracy': test_data.get('btts_accuracy'),
                    'btts_log_loss': test_data.get('btts_log_loss')
                }
                metrics = {k: v for k, v in metrics.items() if v is not None}
                
                import os
                if os.path.exists(metrics_path):
                    import datetime
                    mtime = os.path.getmtime(metrics_path)
                    trained_date = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                    
        except Exception as e:
            errors.append(f'error_loading_metrics: {str(e)}')
            self.logger.warning(f"Грешка при зареждане на Ensemble метрики: {e}")
        
        return {
            'model_name': 'Ensemble',
            'version': 'v1',
            'trained_date': trained_date,
            'accuracy': accuracy,
            'metrics': metrics,
            'loaded': loaded,
            'errors': errors
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
    
    def predict_btts_improved(self, match_df: pd.DataFrame, poisson_btts_prob: float, league: str = None) -> Dict:
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
                self.logger.warning("Improved BTTS model не е наличен, използвам fallback")
                return self._predict_btts_standard(match_df, poisson_btts_prob)
            
            # Прилага BTTS feature engineering
            enhanced_df = self.btts_feature_engineer.create_btts_features(match_df)
            
            # Подготвя features за improved модел
            improved_features = self.feature_lists.get('btts', [])
            available_features = [f for f in improved_features if f in enhanced_df.columns]
            
            if len(available_features) < len(improved_features) * 0.8:  # Минимум 80% features
                self.logger.warning(f"Недостатъчно features за improved BTTS: {len(available_features)}/{len(improved_features)}, fallback към стандартен")
                return self._predict_btts_standard(match_df, poisson_btts_prob)
            
            # ML prediction с improved модел (вече калибриран)
            X_improved = enhanced_df[available_features].fillna(0)
            ml_btts_prob = self.improved_btts_model.predict_proba(X_improved)[:, 1][0]
            
            # Enhanced ensemble logic with league-aware regularization
            ensemble_result = self.btts_ensemble.enhanced_btts_ensemble(
                ml_prob=ml_btts_prob,
                poisson_prob=poisson_btts_prob,
                ml_weight=0.85,  # По-висока тежест за improved модел
                league=league    # League for base rate regularization
            )
            
            # Използваме 0.6 threshold за predicted outcome
            final_prob = ensemble_result['probability']
            predicted_outcome = 'Yes' if final_prob >= 0.6 else 'No'
            
            # Threshold препоръки
            threshold_rec = self.btts_ensemble.get_threshold_recommendation(
                final_prob, 
                ensemble_result['confidence']
            )
            
            return {
                'prob_yes': final_prob,
                'prob_no': 1 - final_prob,
                'predicted_outcome': predicted_outcome,
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
            self.logger.info("Използвам стандартен BTTS модел като fallback")
            
            # Използва стандартния BTTS модел
            btts_features = self.feature_lists.get('btts', [])
            if btts_features and self.models.get('btts') is not None:
                X_btts = match_df[btts_features].fillna(0)
                ml_btts_prob = self.models['btts'].predict_proba(X_btts)[:, 1][0]
                model_source = 'legacy_btts'
            else:
                self.logger.warning("Няма наличен BTTS модел, използвам default стойности")
                ml_btts_prob = 0.5  # Default
                model_source = 'fallback_default'
            
            # Стандартна ensemble логика
            ensemble_prob = 0.8 * ml_btts_prob + 0.2 * poisson_btts_prob
            confidence = abs(ensemble_prob - 0.5) * 2
            
            # Използваме 0.6 threshold за consistency
            predicted_outcome = 'Yes' if ensemble_prob >= 0.6 else 'No'
            
            return {
                'prob_yes': ensemble_prob,
                'prob_no': 1 - ensemble_prob,
                'predicted_outcome': predicted_outcome,
                'confidence': confidence,
                'confidence_level': 'Medium' if confidence > 0.3 else 'Low',
                'model_source': model_source,
                'threshold_recommendation': 0.6,
                'features_used': len(btts_features) if btts_features else 0
            }
            
        except Exception as e:
            self.logger.error(f"Грешка и в стандартния BTTS: {e}, използвам default стойности")
            return {
                'prob_yes': 0.5,
                'prob_no': 0.5,
                'predicted_outcome': 'No',  # Conservative default with 0.6 threshold
                'confidence': 0.0,
                'confidence_level': 'Very Low',
                'model_source': 'error_fallback',
                'threshold_recommendation': 0.6,
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
            'btts_features_available': len(self.feature_lists.get('btts', []))
        }


    def predict_league_round(self, league_slug: str) -> Dict:
        """
        Predict all matches in the next round for a specific league
        
        Args:
            league_slug: League identifier (e.g., '2025-26-english-premier-league')
            
        Returns:
            Dict: Complete round predictions with structure:
                {
                    "league": league_slug,
                    "round": detected_round,
                    "round_date": "2025-11-22",
                    "total_matches": 10,
                    "matches": [
                        {
                            "home_team": "...",
                            "away_team": "...", 
                            "date": "2025-11-22T15:00:00Z",
                            "predictions": {
                                "1x2": {...},
                                "ou25": {...},
                                "btts": {...}
                            }
                        }
                    ]
                }
        """
        try:
            self.logger.info(f"🎯 Predicting next round for league: {league_slug}")
            
            # Import fixtures loader
            from core.fixtures_loader import FixturesLoader
            
            # Load next round fixtures
            fixtures_loader = FixturesLoader()
            fixtures_df = fixtures_loader.get_next_round(league_slug)
            
            if fixtures_df.empty:
                self.logger.warning(f"⚠️  No fixtures found for league: {league_slug}")
                return {
                    "league": league_slug,
                    "round": None,
                    "round_date": None,
                    "total_matches": 0,
                    "matches": [],
                    "error": "No upcoming fixtures found for this league"
                }
            
            # Extract round information
            round_date = fixtures_df.iloc[0]['round_date']
            round_date_str = round_date.strftime('%Y-%m-%d') if round_date else None
            
            # Predict each match
            match_predictions = []
            successful_predictions = 0
            
            for _, fixture in fixtures_df.iterrows():
                try:
                    # Map ESPN league slug to our system's league name
                    our_league_name = self._map_espn_league_to_our_system(league_slug)
                    
                    # Make prediction using existing predict method
                    prediction = self.predict(
                        home_team=fixture['home_team'],
                        away_team=fixture['away_team'],
                        league=our_league_name
                    )
                    
                    # Structure the match prediction
                    match_prediction = {
                        "home_team": fixture['home_team'],
                        "away_team": fixture['away_team'],
                        "date": fixture['date'].isoformat(),
                        "event_id": fixture.get('event_id'),
                        "predictions": {
                            "1x2": prediction['prediction_1x2'],
                            "ou25": prediction['prediction_ou25'],
                            "btts": prediction['prediction_btts']
                        },
                        "confidence": {
                            "overall": prediction.get('confidence', 0.5),
                            "fii_score": prediction.get('fii_score', 0.5)
                        }
                    }
                    
                    match_predictions.append(match_prediction)
                    successful_predictions += 1
                    
                    self.logger.debug(f"✅ Predicted: {fixture['home_team']} vs {fixture['away_team']}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to predict {fixture['home_team']} vs {fixture['away_team']}: {e}")
                    
                    # Add failed prediction with error info
                    match_predictions.append({
                        "home_team": fixture['home_team'],
                        "away_team": fixture['away_team'],
                        "date": fixture['date'].isoformat(),
                        "event_id": fixture.get('event_id'),
                        "error": str(e),
                        "predictions": None
                    })
            
            # Build final result
            result = {
                "league": league_slug,
                "round": f"Round {round_date_str}" if round_date_str else "Next Round",
                "round_date": round_date_str,
                "total_matches": len(fixtures_df),
                "successful_predictions": successful_predictions,
                "failed_predictions": len(fixtures_df) - successful_predictions,
                "matches": match_predictions,
                "generated_at": pd.Timestamp.now(tz='UTC').isoformat()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error in next round prediction: {e}")
            raise
    
    def _map_espn_league_to_our_system(self, espn_league_slug: str) -> str:
        """
        Map ESPN league slug to our system's league names
        
        Args:
            espn_league_slug: ESPN league identifier
            
        Returns:
            str: Our system's league name
        """
        mapping = {
            '2025-26-english-premier-league': 'Premier League',
            '2025-26-laliga': 'La Liga',
            '2025-26-italian-serie-a': 'Serie A', 
            '2025-26-german-bundesliga': 'Bundesliga',
            '2025-26-ligue-1': 'Ligue 1',
            '2025-26-portuguese-primeira-liga': 'Primeira Liga',
            '2025-26-dutch-eredivisie': 'Eredivisie',
            '2025-26-english-championship': 'Championship'
        }
        
        return mapping.get(espn_league_slug, 'Premier League')  # Default fallback
    
    def _predict_1x2_v2(self, home_team: str, away_team: str, league: str = None) -> Dict:
        """
        1X2 v2 prediction using per-league binary models + Poisson v2 + calibration
        
        Args:
            home_team: Home team name
            away_team: Away team name
            league: League name
            
        Returns:
            Dictionary with 1X2 v2 predictions
        """
        if not self.x1x2_v2_enabled:
            self.logger.warning("1X2 v2 е изключен, използвам fallback")
            return self._predict_1x2_fallback(home_team, away_team, league)
        
        try:
            self.logger.info(f"🎯 1X2 v2 prediction: {home_team} vs {away_team}")
            
            # Determine league slug
            from core.league_utils import get_league_slug
            league_slug = get_league_slug(league) if league else None
            
            # Get appropriate models (per-league or global fallback)
            models_info = self._get_1x2_v2_models_for_league(league_slug)
            if not models_info:
                self.logger.warning(f"⚠️ Няма 1X2 v2 модели за {league_slug}, използвам fallback")
                return self._predict_1x2_fallback(home_team, away_team, league)
            
            binary_models = models_info['models']
            feature_list = models_info['feature_list']
            calibrator = models_info.get('calibrator')
            
            # Create features
            features = self._create_1x2_v2_features(home_team, away_team, league)
            
            # Align features with model expectations
            feature_vector = self._align_1x2_v2_features(features, feature_list)
            
            # Get predictions from 3 binary models
            pred_homewin = binary_models['homewin'].predict_proba(feature_vector.reshape(1, -1))[0, 1]
            pred_draw = binary_models['draw'].predict_proba(feature_vector.reshape(1, -1))[0, 1]
            pred_awaywin = binary_models['awaywin'].predict_proba(feature_vector.reshape(1, -1))[0, 1]
            
            # Combine and normalize ML predictions
            ml_predictions = np.array([pred_homewin, pred_draw, pred_awaywin])
            ml_predictions = ml_predictions / np.sum(ml_predictions)
            
            # Get Poisson v2 predictions
            poisson_predictions = self._get_poisson_v2_predictions(home_team, away_team, league_slug)
            
            # Combine ML and Poisson predictions
            ml_weight = 0.7  # Can be made configurable
            poisson_weight = 0.3
            
            combined_predictions = (ml_weight * ml_predictions + 
                                  poisson_weight * poisson_predictions)
            combined_predictions = combined_predictions / np.sum(combined_predictions)
            
            # Check if hybrid prediction should be used
            if self.hybrid_enabled and self.hybrid_predictor:
                try:
                    # Create features DataFrame for hybrid predictor
                    features_df = pd.DataFrame([features])
                    
                    context = {
                        'home_team': home_team,
                        'away_team': away_team,
                        'league': league,
                        'home_team_id': features.get('home_team_id'),
                        'away_team_id': features.get('away_team_id')
                    }
                    
                    # Get hybrid prediction
                    hybrid_result = self.hybrid_predictor.predict_hybrid_1x2(features_df, context)
                    
                    # Return hybrid result with additional metadata
                    return {
                        'prob_home_win': hybrid_result['prob_home_win'],
                        'prob_draw': hybrid_result['prob_draw'],
                        'prob_away_win': hybrid_result['prob_away_win'],
                        'predicted_outcome': hybrid_result['predicted_outcome'],
                        'confidence': hybrid_result['confidence'],
                        'model_version': '1x2_hybrid_v1',
                        'league_used': league or 'default',
                        'using_hybrid': True,
                        'hybrid_sources': hybrid_result.get('sources_used', {}),
                        'calibrated': hybrid_result.get('calibrated', False),
                        'components': hybrid_result.get('components', {}),
                        'weights_used': hybrid_result.get('weights_used', {}),
                        'timestamp': hybrid_result.get('timestamp')
                    }
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Hybrid prediction failed, using ML v2: {e}")
                    # Continue with ML v2 prediction below
            
            # Apply calibration if available
            if calibrator:
                # Convert to logits for calibration
                logits = np.log(np.clip(combined_predictions, 1e-15, 1.0))
                calibrated_predictions = calibrator.predict_proba(
                    combined_predictions.reshape(1, -1), 
                    logits.reshape(1, -1)
                )[0]
            else:
                calibrated_predictions = combined_predictions
            
            # Ensure probabilities sum to 1
            calibrated_predictions = calibrated_predictions / np.sum(calibrated_predictions)
            
            # Determine predicted outcome
            predicted_class = np.argmax(calibrated_predictions)
            outcome_map = {0: '1', 1: 'X', 2: '2'}
            predicted_outcome = outcome_map[predicted_class]
            
            # Calculate confidence
            max_prob = np.max(calibrated_predictions)
            confidence = max_prob
            
            result = {
                'prob_home_win': float(calibrated_predictions[0]),
                'prob_draw': float(calibrated_predictions[1]),
                'prob_away_win': float(calibrated_predictions[2]),
                'predicted_outcome': predicted_outcome,
                'confidence': float(confidence),
                'model_version': '1x2_v2',
                'league_used': league_slug or 'global',
                'using_hybrid': False,
                'ml_predictions': {
                    'home': float(ml_predictions[0]),
                    'draw': float(ml_predictions[1]),
                    'away': float(ml_predictions[2])
                },
                'poisson_predictions': {
                    'home': float(poisson_predictions[0]),
                    'draw': float(poisson_predictions[1]),
                    'away': float(poisson_predictions[2])
                },
                'calibrated': calibrator is not None
            }
            
            self.logger.info(f"✅ 1X2 v2: {predicted_outcome} ({confidence:.3f} confidence)")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Грешка в 1X2 v2 prediction: {e}")
            return self._predict_1x2_fallback(home_team, away_team, league)
    
    def _get_1x2_v2_models_for_league(self, league_slug: str = None) -> Dict:
        """Get 1X2 v2 models for league with fallback to global"""
        if league_slug and league_slug in self.x1x2_v2_models:
            models_info = self.x1x2_v2_models[league_slug].copy()
            models_info['calibrator'] = self.x1x2_v2_calibrators.get(league_slug)
            return models_info
        elif 'global' in self.x1x2_v2_models:
            models_info = self.x1x2_v2_models['global'].copy()
            models_info['calibrator'] = self.x1x2_v2_calibrators.get('global')
            return models_info
        else:
            return None
    
    def _create_1x2_v2_features(self, home_team: str, away_team: str, league: str) -> Dict:
        """Create features for 1X2 v2 prediction"""
        try:
            # Load historical data for feature creation
            from core.data_loader import ESPNDataLoader
            data_loader = ESPNDataLoader()
            df = data_loader.load_fixtures()
            
            if df is None or df.empty:
                self.logger.warning("⚠️ Няма исторически данни за features")
                return {}
            
            # Create 1X2-specific features
            features = self.features_1x2.create_1x2_features(
                home_team, away_team, league, df, datetime.now()
            )
            
            # Add standard features (simplified)
            standard_features = {
                'home_team_basic': hash(home_team) % 10000,
                'away_team_basic': hash(away_team) % 10000,
                'league_basic': hash(league) % 100 if league else 0
            }
            
            # Combine features
            combined_features = {**standard_features, **features}
            
            return combined_features
            
        except Exception as e:
            self.logger.error(f"❌ Грешка при създаване на 1X2 v2 features: {e}")
            return {}
    
    def _align_1x2_v2_features(self, features: Dict, feature_list: List[str]) -> np.ndarray:
        """Align features with model expectations"""
        try:
            feature_vector = []
            
            for feature_name in feature_list:
                if feature_name in features:
                    value = features[feature_name]
                    # Handle non-numeric values
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        feature_vector.append(float(value))
                    else:
                        feature_vector.append(0.0)
                else:
                    feature_vector.append(0.0)  # Default value for missing features
            
            return np.array(feature_vector)
            
        except Exception as e:
            self.logger.error(f"❌ Грешка при align на features: {e}")
            return np.zeros(len(feature_list))
    
    def _get_poisson_v2_predictions(self, home_team: str, away_team: str, 
                                   league_slug: str = None) -> np.ndarray:
        """Get Poisson v2 predictions"""
        try:
            # Get appropriate Poisson v2 model
            if league_slug and league_slug in self.poisson_v2_models:
                poisson_model = self.poisson_v2_models[league_slug]
            elif 'global' in self.poisson_v2_models:
                poisson_model = self.poisson_v2_models['global']
            else:
                # Fallback to default probabilities
                self.logger.warning("⚠️ Няма Poisson v2 модел, използвам default")
                return np.array([0.45, 0.25, 0.30])  # Default home/draw/away
            
            # Get Poisson prediction
            prediction = poisson_model.predict_match(home_team, away_team, league_slug)
            
            return np.array([
                prediction['poisson_p_home'],
                prediction['poisson_p_draw'],
                prediction['poisson_p_away']
            ])
            
        except Exception as e:
            self.logger.error(f"❌ Грешка в Poisson v2 prediction: {e}")
            return np.array([0.45, 0.25, 0.30])  # Default fallback
    
    def _predict_1x2_fallback(self, home_team: str, away_team: str, league: str = None) -> Dict:
        """Fallback 1X2 prediction using existing models"""
        try:
            # Use existing 1X2 prediction logic as fallback
            # This would call the original predict method's 1X2 logic
            self.logger.info("🔄 Using 1X2 fallback prediction")
            
            # Simplified fallback - in practice this would use existing models
            return {
                'prob_home_win': 0.45,
                'prob_draw': 0.25,
                'prob_away_win': 0.30,
                'predicted_outcome': '1',
                'confidence': 0.45,
                'model_version': '1x2_fallback',
                'league_used': 'fallback'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Грешка в 1X2 fallback: {e}")
            return {
                'prob_home_win': 0.33,
                'prob_draw': 0.34,
                'prob_away_win': 0.33,
                'predicted_outcome': 'X',
                'confidence': 0.34,
                'model_version': '1x2_default',
                'league_used': 'default'
            }
    
    def predict_draw_specialist(self, home_team: str, away_team: str, 
                              league: str = None) -> Dict[str, any]:
        """
        Predict draw probability using specialized draw model
        
        ADDITIVE method - does not modify existing 1X2 prediction logic.
        Provides enhanced draw probability estimation.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            league: League name (optional)
            
        Returns:
            Dictionary with draw specialist prediction
        """
        try:
            self.logger.info(f"🎯 Draw specialist prediction: {home_team} vs {away_team}")
            
            # Load draw predictor (lazy loading)
            if not hasattr(self, 'draw_predictor'):
                try:
                    from core.draw_predictor import DrawPredictor
                    self.draw_predictor = DrawPredictor()
                    self.logger.info("✅ Draw predictor loaded")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not load draw predictor: {e}")
                    self.draw_predictor = None
            
            # Get historical data for features
            from core.data_loader import ESPNDataLoader
            data_loader = ESPNDataLoader()
            df = data_loader.load_fixtures()
            
            if df is None or df.empty:
                self.logger.warning("⚠️ No historical data for draw prediction")
                return self._draw_specialist_fallback(home_team, away_team, league)
            
            # Add required columns
            df['league'] = df['league_id'].astype(str) if 'league_id' in df.columns else 'unknown'
            df['home_team'] = df['home_team_id'].astype(str) if 'home_team_id' in df.columns else 'unknown'
            df['away_team'] = df['away_team_id'].astype(str) if 'away_team_id' in df.columns else 'unknown'
            
            # Get existing 1X2 prediction for ML draw probability
            existing_prediction = None
            p_ml_draw = None
            p_poisson_draw = None
            
            try:
                existing_prediction = self.predict(home_team, away_team, league)
                if existing_prediction and 'prediction_1x2' in existing_prediction:
                    p_ml_draw = existing_prediction['prediction_1x2'].get('prob_draw', 0.25)
                    # Try to get Poisson draw probability if available
                    p_poisson_draw = existing_prediction['prediction_1x2'].get('poisson_p_draw', 0.25)
            except Exception as e:
                self.logger.warning(f"⚠️ Could not get existing 1X2 prediction: {e}")
            
            # Use draw predictor if available
            if self.draw_predictor:
                draw_result = self.draw_predictor.predict_draw_probability(
                    home_team=home_team,
                    away_team=away_team,
                    league=league or 'unknown',
                    df=df,
                    reference_date=datetime.now(),
                    p_ml_draw=p_ml_draw,
                    p_poisson_draw=p_poisson_draw
                )
                
                # Enhance with existing prediction context
                if existing_prediction:
                    draw_result['enhanced_1x2_prediction'] = {
                        'original_draw_prob': p_ml_draw,
                        'enhanced_draw_prob': draw_result['draw_probability'],
                        'improvement': draw_result['draw_probability'] - (p_ml_draw or 0.25),
                        'other_probs': {
                            'prob_home_win': existing_prediction['prediction_1x2'].get('prob_home_win', 0.33),
                            'prob_away_win': existing_prediction['prediction_1x2'].get('prob_away_win', 0.33)
                        }
                    }
                
                return draw_result
            else:
                return self._draw_specialist_fallback(home_team, away_team, league, p_ml_draw)
                
        except Exception as e:
            self.logger.error(f"❌ Error in draw specialist prediction: {e}")
            return self._draw_specialist_fallback(home_team, away_team, league)
    
    def _draw_specialist_fallback(self, home_team: str, away_team: str, 
                                league: str = None, p_ml_draw: float = None) -> Dict[str, any]:
        """
        Fallback prediction when draw specialist fails
        
        Args:
            home_team: Home team name
            away_team: Away team name
            league: League name
            p_ml_draw: ML draw probability if available
            
        Returns:
            Fallback draw prediction
        """
        fallback_prob = p_ml_draw if p_ml_draw is not None else 0.25
        
        return {
            'draw_probability': fallback_prob,
            'confidence': 0.3,  # Low confidence for fallback
            'components': {
                'draw_model': 0.25,
                'ml_1x2': fallback_prob,
                'poisson': 0.25,
                'league_prior': 0.25
            },
            'weights_used': {
                'draw_model': 0.0,
                'ml_1x2': 1.0,
                'poisson': 0.0,
                'league_prior': 0.0
            },
            'model_version': 'draw_specialist_fallback',
            'is_model_loaded': False,
            'fallback_reason': 'Draw specialist model not available',
            'match_info': {
                'home_team': home_team,
                'away_team': away_team,
                'league': league
            }
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
