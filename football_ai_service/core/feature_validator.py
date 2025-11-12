"""
Feature Validator - Интелигентно валидиране и попълване на features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
import logging

from .utils import setup_logging


class FeatureGroup(Enum):
    """Групи от features за различно третиране"""
    CRITICAL = "critical"  # Elo, основни статистики
    FORM = "form"  # Форма, momentum
    GOALS = "goals"  # Голове, xG, ефективност
    CONTEXT = "context"  # Дни почивка, лига, дата
    ADVANCED = "advanced"  # Сложни статистики
    POISSON = "poisson"  # Poisson predictions


class ImputationMethod(Enum):
    """Методи за попълване на липсващи стойности"""
    ZERO = "zero"
    MEAN = "mean"
    MEDIAN = "median"
    LEAGUE_MEAN = "league_mean"
    HISTORICAL = "historical"
    INTERPOLATE = "interpolate"
    REQUIRED = "required"  # Задължителни - хвърля грешка ако липсват


class FeatureValidator:
    """
    Интелигентен валидатор за features с групово попълване
    """
    
    def __init__(self):
        """Инициализация на валидатора"""
        self.logger = setup_logging()
        
        # Дефиниране на feature групите
        self._define_feature_groups()
        
        # Зареждане на исторически статистики
        self._load_historical_stats()
    
    def _define_feature_groups(self):
        """Дефинира групите от features и техните методи за попълване"""
        
        self.feature_groups = {
            FeatureGroup.CRITICAL: {
                'features': [
                    'home_elo_before', 'away_elo_before', 'elo_diff',
                    'home_team', 'away_team', 'is_home'
                ],
                'method': ImputationMethod.REQUIRED,
                'description': 'Критични features - задължителни за точни прогнози'
            },
            
            FeatureGroup.FORM: {
                'features': [
                    'home_form_5', 'away_form_5', 'home_form_10', 'away_form_10',
                    'home_win_rate_5', 'away_win_rate_5', 'home_points_per_game_5', 'away_points_per_game_5'
                ],
                'method': ImputationMethod.LEAGUE_MEAN,
                'fallback': ImputationMethod.ZERO,
                'description': 'Форма и momentum - използва лигови средни'
            },
            
            FeatureGroup.GOALS: {
                'features': [
                    'home_goals_scored_avg_5', 'away_goals_scored_avg_5',
                    'home_goals_conceded_avg_5', 'away_goals_conceded_avg_5',
                    'home_goals_scored_avg_10', 'away_goals_scored_avg_10',
                    'home_xg_proxy', 'away_xg_proxy',
                    'home_shooting_efficiency', 'away_shooting_efficiency'
                ],
                'method': ImputationMethod.LEAGUE_MEAN,
                'fallback': ImputationMethod.HISTORICAL,
                'description': 'Голове и ефективност - лигови средни или исторически данни'
            },
            
            FeatureGroup.CONTEXT: {
                'features': [
                    'home_rest_days', 'away_rest_days', 'rest_advantage',
                    'league', 'season', 'month', 'day_of_week'
                ],
                'method': ImputationMethod.MEDIAN,
                'fallback': ImputationMethod.ZERO,
                'description': 'Контекстуални features - медиана или нули'
            },
            
            FeatureGroup.ADVANCED: {
                'features': [
                    'home_shots_avg_5', 'away_shots_avg_5',
                    'home_shots_on_target_avg_5', 'away_shots_on_target_avg_5',
                    'home_possession_avg_5', 'away_possession_avg_5',
                    'home_pass_accuracy_avg_5', 'away_pass_accuracy_avg_5',
                    'home_corners_avg_5', 'away_corners_avg_5',
                    'home_fouls_avg_5', 'away_fouls_avg_5'
                ],
                'method': ImputationMethod.MEAN,
                'fallback': ImputationMethod.ZERO,
                'description': 'Напреднали статистики - средни стойности'
            },
            
            FeatureGroup.POISSON: {
                'features': [
                    'poisson_prob_1', 'poisson_prob_x', 'poisson_prob_2',
                    'poisson_prob_over25', 'poisson_prob_btts',
                    'poisson_expected_goals', 'poisson_lambda_home', 'poisson_lambda_away'
                ],
                'method': ImputationMethod.ZERO,
                'description': 'Poisson predictions - нули ако липсват'
            }
        }
    
    def _load_historical_stats(self):
        """Зарежда исторически статистики за попълване"""
        
        # Default исторически стойности по лиги
        self.historical_stats = {
            'Premier League': {
                'goals_scored_avg': 1.8,
                'goals_conceded_avg': 1.2,
                'xg_proxy': 1.7,
                'shooting_efficiency': 0.35,
                'form': 0.1,
                'shots_avg': 12.5,
                'shots_on_target_avg': 4.2,
                'possession_avg': 50.0,
                'pass_accuracy_avg': 82.5,
                'corners_avg': 5.5,
                'fouls_avg': 11.2
            },
            'La Liga': {
                'goals_scored_avg': 1.6,
                'goals_conceded_avg': 1.1,
                'xg_proxy': 1.6,
                'shooting_efficiency': 0.33,
                'form': 0.05,
                'shots_avg': 11.8,
                'shots_on_target_avg': 3.9,
                'possession_avg': 52.0,
                'pass_accuracy_avg': 85.2,
                'corners_avg': 5.2,
                'fouls_avg': 12.1
            },
            'Serie A': {
                'goals_scored_avg': 1.5,
                'goals_conceded_avg': 1.1,
                'xg_proxy': 1.5,
                'shooting_efficiency': 0.32,
                'form': 0.0,
                'shots_avg': 11.2,
                'shots_on_target_avg': 3.7,
                'possession_avg': 49.5,
                'pass_accuracy_avg': 83.8,
                'corners_avg': 5.0,
                'fouls_avg': 13.5
            },
            'Bundesliga': {
                'goals_scored_avg': 1.9,
                'goals_conceded_avg': 1.3,
                'xg_proxy': 1.8,
                'shooting_efficiency': 0.34,
                'form': 0.08,
                'shots_avg': 13.1,
                'shots_on_target_avg': 4.5,
                'possession_avg': 51.2,
                'pass_accuracy_avg': 81.9,
                'corners_avg': 6.1,
                'fouls_avg': 10.8
            },
            'Ligue 1': {
                'goals_scored_avg': 1.4,
                'goals_conceded_avg': 1.0,
                'xg_proxy': 1.4,
                'shooting_efficiency': 0.30,
                'form': 0.02,
                'shots_avg': 10.9,
                'shots_on_target_avg': 3.5,
                'possession_avg': 50.8,
                'pass_accuracy_avg': 84.1,
                'corners_avg': 4.8,
                'fouls_avg': 12.8
            },
            # Global default
            'Default': {
                'goals_scored_avg': 1.5,
                'goals_conceded_avg': 1.2,
                'xg_proxy': 1.5,
                'shooting_efficiency': 0.30,
                'form': 0.0,
                'shots_avg': 11.5,
                'shots_on_target_avg': 4.0,
                'possession_avg': 50.0,
                'pass_accuracy_avg': 83.0,
                'corners_avg': 5.5,
                'fouls_avg': 12.0
            }
        }
    
    def validate_and_impute(
        self,
        df: pd.DataFrame,
        required_features: List[str],
        league: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Валидира и попълва липсващи features
        
        Args:
            df: DataFrame с features
            required_features: Списък с нужни features
            league: Лига за лигово-специфично попълване
        
        Returns:
            Tuple (обработен DataFrame, metadata за попълването)
        """
        self.logger.info(f"Валидиране на {len(required_features)} features...")
        
        # Metadata за проследяване на промените
        metadata = {
            'original_features': len(df.columns),
            'required_features': len(required_features),
            'missing_features': [],
            'imputed_features': {},
            'critical_missing': [],
            'warnings': [],
            'data_quality_score': 1.0
        }
        
        df_processed = df.copy()
        
        # Проверка за критични features
        critical_features = self.feature_groups[FeatureGroup.CRITICAL]['features']
        missing_critical = [f for f in critical_features if f in required_features and f not in df.columns]
        
        if missing_critical:
            metadata['critical_missing'] = missing_critical
            metadata['data_quality_score'] = 0.0
            error_msg = f"Критични features липсват: {missing_critical}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Групово попълване на features
        for group, config in self.feature_groups.items():
            group_features = [f for f in config['features'] if f in required_features]
            missing_in_group = [f for f in group_features if f not in df.columns]
            
            if missing_in_group:
                self.logger.warning(f"Липсващи {group.value} features: {missing_in_group}")
                metadata['missing_features'].extend(missing_in_group)
                
                # Попълване според групата
                imputed_values = self._impute_group_features(
                    missing_in_group, group, config, league
                )
                
                # Добавяне на попълнените features
                for feature, value in imputed_values.items():
                    df_processed[feature] = value
                    metadata['imputed_features'][feature] = {
                        'method': config['method'].value,
                        'value': value,
                        'group': group.value
                    }
        
        # Изчисляване на data quality score
        metadata['data_quality_score'] = self._calculate_quality_score(
            len(df.columns), len(metadata['missing_features']), len(required_features)
        )
        
        # Добавяне на предупреждения
        if metadata['missing_features']:
            metadata['warnings'].append(
                f"Попълнени {len(metadata['missing_features'])} липсващи features"
            )
        
        if metadata['data_quality_score'] < 0.8:
            metadata['warnings'].append(
                f"Ниско качество на данните: {metadata['data_quality_score']:.2f}"
            )
        
        # Финално подреждане на колоните
        df_final = df_processed.reindex(columns=required_features, fill_value=0.0)
        
        self.logger.info(
            f"Валидиране завършено: {len(df.columns)} → {len(df_final.columns)} features, "
            f"качество: {metadata['data_quality_score']:.2f}"
        )
        
        return df_final, metadata
    
    def _impute_group_features(
        self,
        missing_features: List[str],
        group: FeatureGroup,
        config: Dict,
        league: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Попълва липсващи features за конкретна група
        
        Args:
            missing_features: Липсващи features
            group: Група от features
            config: Конфигурация на групата
            league: Лига
        
        Returns:
            Dictionary с попълнени стойности
        """
        imputed = {}
        method = config['method']
        
        league_key = league if league in self.historical_stats else 'Default'
        league_stats = self.historical_stats[league_key]
        
        for feature in missing_features:
            if method == ImputationMethod.ZERO:
                imputed[feature] = 0.0
                
            elif method == ImputationMethod.LEAGUE_MEAN:
                # Мапване на feature към статистика
                stat_key = self._map_feature_to_stat(feature)
                if stat_key and stat_key in league_stats:
                    imputed[feature] = league_stats[stat_key]
                else:
                    # Fallback
                    fallback_method = config.get('fallback', ImputationMethod.ZERO)
                    imputed[feature] = self._get_fallback_value(feature, fallback_method, league_stats)
                    
            elif method == ImputationMethod.HISTORICAL:
                stat_key = self._map_feature_to_stat(feature)
                if stat_key and stat_key in league_stats:
                    imputed[feature] = league_stats[stat_key]
                else:
                    imputed[feature] = 0.0
                    
            elif method == ImputationMethod.MEDIAN:
                # За контекстуални features
                if 'rest_days' in feature:
                    imputed[feature] = 7.0  # Типична седмица между мачове
                elif 'month' in feature:
                    imputed[feature] = 6.0  # Средата на сезона
                elif 'day_of_week' in feature:
                    imputed[feature] = 6.0  # Събота (типичен ден за мачове)
                else:
                    imputed[feature] = 0.0
                    
            elif method == ImputationMethod.MEAN:
                # За напреднали статистики
                stat_key = self._map_feature_to_stat(feature)
                if stat_key and stat_key in league_stats:
                    imputed[feature] = league_stats[stat_key]
                else:
                    imputed[feature] = 0.0
            
            else:
                imputed[feature] = 0.0
        
        return imputed
    
    def _map_feature_to_stat(self, feature: str) -> Optional[str]:
        """
        Мапва feature име към статистически ключ
        
        Args:
            feature: Име на feature
        
        Returns:
            Ключ в historical_stats или None
        """
        mapping = {
            # Голове
            'goals_scored_avg': 'goals_scored_avg',
            'goals_conceded_avg': 'goals_conceded_avg',
            'xg_proxy': 'xg_proxy',
            'shooting_efficiency': 'shooting_efficiency',
            
            # Форма
            'form': 'form',
            
            # Статистики
            'shots_avg': 'shots_avg',
            'shots_on_target_avg': 'shots_on_target_avg',
            'possession_avg': 'possession_avg',
            'pass_accuracy_avg': 'pass_accuracy_avg',
            'corners_avg': 'corners_avg',
            'fouls_avg': 'fouls_avg'
        }
        
        # Търси частично съвпадение
        for pattern, stat_key in mapping.items():
            if pattern in feature:
                return stat_key
        
        return None
    
    def _get_fallback_value(
        self,
        feature: str,
        fallback_method: ImputationMethod,
        league_stats: Dict[str, float]
    ) -> float:
        """Получава fallback стойност"""
        
        if fallback_method == ImputationMethod.ZERO:
            return 0.0
        elif fallback_method == ImputationMethod.HISTORICAL:
            # Опитва се да намери подходяща статистика
            stat_key = self._map_feature_to_stat(feature)
            return league_stats.get(stat_key, 0.0)
        else:
            return 0.0
    
    def _calculate_quality_score(
        self,
        original_count: int,
        missing_count: int,
        required_count: int
    ) -> float:
        """
        Изчислява quality score за данните
        
        Args:
            original_count: Брой оригинални features
            missing_count: Брой липсващи features
            required_count: Брой нужни features
        
        Returns:
            Quality score между 0.0 и 1.0
        """
        if required_count == 0:
            return 1.0
        
        # Базов score според процента налични features
        available_ratio = (required_count - missing_count) / required_count
        
        # Намаляване за много липсващи features
        if missing_count > required_count * 0.5:
            penalty = 0.3
        elif missing_count > required_count * 0.2:
            penalty = 0.1
        else:
            penalty = 0.0
        
        return max(0.0, available_ratio - penalty)
    
    def get_feature_groups_info(self) -> Dict:
        """Връща информация за feature групите"""
        
        info = {}
        for group, config in self.feature_groups.items():
            info[group.value] = {
                'features': config['features'],
                'method': config['method'].value,
                'description': config['description'],
                'count': len(config['features'])
            }
        
        return info
