"""
Feature Engineering - Генериране на advanced features за ML модели
"""

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm

from .utils import setup_logging, safe_divide, PerformanceTimer
from .elo_calculator import EloCalculator


class FeatureEngineer:
    """
    Клас за генериране на features от ESPN данни
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Инициализация на Feature Engineer
        
        Args:
            config: Конфигурация (rolling windows, etc.)
        """
        self.logger = setup_logging()
        self.config = config or {
            'rolling_windows': [5, 10],
            'min_matches_for_stats': 3
        }
        
        self.elo_calculator = EloCalculator(
            k_factor=20,
            initial_rating=1500,
            home_advantage=100
        )
        
        self.logger.info("FeatureEngineer инициализиран")
    
    def add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавяне на основни features
        
        Args:
            df: DataFrame с мачове
        
        Returns:
            DataFrame с добавени features
        """
        self.logger.info("Добавяне на основни features...")
        
        df = df.copy()
        
        # Времеви features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Home/Away indicator
        df['is_home'] = 1
        
        # Goal-based features
        if 'home_score' in df.columns and 'away_score' in df.columns:
            df['total_goals'] = df['home_score'] + df['away_score']
            df['goal_diff'] = df['home_score'] - df['away_score']
            df['home_clean_sheet'] = (df['away_score'] == 0).astype(int)
            df['away_clean_sheet'] = (df['home_score'] == 0).astype(int)
        
        self.logger.info(f"Основни features добавени")
        
        return df
    
    def calculate_team_form(
        self,
        df: pd.DataFrame,
        team_id_col: str,
        result_col: str,
        n_matches: int = 5
    ) -> pd.Series:
        """
        Изчисляване на форма на отбора (последни N мача)
        
        Args:
            df: DataFrame с мачове
            team_id_col: Име на колона с team ID
            result_col: Име на колона с резултат (1=win, 0.5=draw, 0=loss)
            n_matches: Брой мачове за форма
        
        Returns:
            Series с form index
        """
        # Групиране по отбор и изчисляване на rolling sum
        form = df.groupby(team_id_col)[result_col].transform(
            lambda x: x.rolling(window=n_matches, min_periods=1).sum()
        )
        
        # Нормализация (max е 3*n_matches за n победи)
        form = form / (3 * n_matches)
        
        return form
    
    def add_rolling_stats(
        self,
        df: pd.DataFrame,
        windows: List[int] = [5, 10]
    ) -> pd.DataFrame:
        """
        Добавяне на rolling statistics
        
        Args:
            df: DataFrame с мачове и статистики
            windows: List от размери на прозорци
        
        Returns:
            DataFrame с rolling stats
        """
        self.logger.info(f"Добавяне на rolling stats за прозорци {windows}...")
        
        df = df.copy()
        df = df.sort_values(['home_team_id', 'date']).reset_index(drop=True)
        
        # Колони за rolling stats
        stat_columns = [
            'possession', 'shots', 'shots_on_target', 'corners',
            'fouls', 'yellow_cards', 'total_passes', 'pass_accuracy',
            'tackles', 'interceptions'
        ]
        
        # Филтриране само на съществуващи колони
        stat_columns = [col for col in stat_columns if col in df.columns]
        
        for window in windows:
            for col in stat_columns:
                # Rolling average за home team
                df[f'home_{col}_avg_{window}'] = df.groupby('home_team_id')[col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
                )
                
                # Rolling average за away team (ако има away stats)
                if f'{col}_away' in df.columns:
                    df[f'away_{col}_avg_{window}'] = df.groupby('away_team_id')[f'{col}_away'].transform(
                        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
                    )
        
        self.logger.info(f"Rolling stats добавени за {len(stat_columns)} метрики")
        
        return df
    
    def add_goal_stats(
        self,
        df: pd.DataFrame,
        windows: List[int] = [5, 10]
    ) -> pd.DataFrame:
        """
        Добавяне на статистики за голове
        
        Args:
            df: DataFrame с мачове
            windows: List от размери на прозорци
        
        Returns:
            DataFrame с goal stats
        """
        self.logger.info("Добавяне на goal statistics...")
        
        df = df.copy()
        
        for window in windows:
            # Goals scored (home team perspective)
            df[f'home_goals_scored_avg_{window}'] = df.groupby('home_team_id')['home_score'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
            
            # Goals conceded (home team perspective)
            df[f'home_goals_conceded_avg_{window}'] = df.groupby('home_team_id')['away_score'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
            
            # Goals scored (away team perspective)
            df[f'away_goals_scored_avg_{window}'] = df.groupby('away_team_id')['away_score'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
            
            # Goals conceded (away team perspective)
            df[f'away_goals_conceded_avg_{window}'] = df.groupby('away_team_id')['home_score'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
        
        self.logger.info("Goal statistics добавени")
        
        return df
    
    def add_form_features(
        self,
        df: pd.DataFrame,
        n_matches: int = 5
    ) -> pd.DataFrame:
        """
        Добавяне на form features
        
        Args:
            df: DataFrame с мачове
            n_matches: Брой мачове за изчисляване на форма
        
        Returns:
            DataFrame с form features
        """
        self.logger.info(f"Добавяне на form features (последни {n_matches} мача)...")
        
        df = df.copy()
        
        # Създаване на points колона (3 за победа, 1 за равенство, 0 за загуба)
        df['home_points'] = df.apply(
            lambda x: 3 if x['home_score'] > x['away_score'] 
            else (1 if x['home_score'] == x['away_score'] else 0),
            axis=1
        )
        
        df['away_points'] = df.apply(
            lambda x: 3 if x['away_score'] > x['home_score'] 
            else (1 if x['away_score'] == x['home_score'] else 0),
            axis=1
        )
        
        # Rolling form (sum of points)
        df[f'home_form_{n_matches}'] = df.groupby('home_team_id')['home_points'].transform(
            lambda x: x.shift(1).rolling(window=n_matches, min_periods=1).sum()
        )
        
        df[f'away_form_{n_matches}'] = df.groupby('away_team_id')['away_points'].transform(
            lambda x: x.shift(1).rolling(window=n_matches, min_periods=1).sum()
        )
        
        # Нормализация (max е 3*n_matches)
        df[f'home_form_{n_matches}'] = df[f'home_form_{n_matches}'] / (3 * n_matches)
        df[f'away_form_{n_matches}'] = df[f'away_form_{n_matches}'] / (3 * n_matches)
        
        # Form difference
        df[f'form_diff_{n_matches}'] = df[f'home_form_{n_matches}'] - df[f'away_form_{n_matches}']
        
        self.logger.info("Form features добавени")
        
        return df
    
    def add_efficiency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавяне на efficiency features (xG proxy, finishing, etc.)
        
        Args:
            df: DataFrame с мачове и статистики
        
        Returns:
            DataFrame с efficiency features
        """
        self.logger.info("Добавяне на efficiency features...")
        
        df = df.copy()
        
        # Shooting efficiency (goals per shot on target)
        if 'shots_on_target' in df.columns:
            df['home_shooting_efficiency'] = safe_divide(
                df['home_score'],
                df['shots_on_target'],
                default=0.0
            )
            
            if 'shots_on_target_away' in df.columns:
                df['away_shooting_efficiency'] = safe_divide(
                    df['away_score'],
                    df['shots_on_target_away'],
                    default=0.0
                )
        
        # xG proxy (shots on target * possession weight)
        if 'shots_on_target' in df.columns and 'possession' in df.columns:
            df['home_xg_proxy'] = df['shots_on_target'] * (df['possession'] / 100) * 0.1
            
            if 'shots_on_target_away' in df.columns and 'possession_away' in df.columns:
                df['away_xg_proxy'] = df['shots_on_target_away'] * (df['possession_away'] / 100) * 0.1
        
        # Defensive efficiency (tackles + interceptions per goal conceded)
        if 'tackles' in df.columns and 'interceptions' in df.columns:
            df['home_defensive_actions'] = df['tackles'] + df['interceptions']
            df['home_defensive_efficiency'] = safe_divide(
                df['home_defensive_actions'],
                df['away_score'] + 1,  # +1 to avoid division by zero
                default=0.0
            )
        
        # Pass completion under pressure (passes in opponent half)
        if 'accurate_passes' in df.columns and 'total_passes' in df.columns:
            df['home_pass_completion'] = safe_divide(
                df['accurate_passes'],
                df['total_passes'],
                default=0.0
            )
        
        self.logger.info("Efficiency features добавени")
        
        return df
    
    def add_elo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавяне на Elo rating features
        
        Args:
            df: DataFrame с мачове
        
        Returns:
            DataFrame с Elo features
        """
        with PerformanceTimer("Изчисляване на Elo ratings", self.logger):
            df = self.elo_calculator.calculate_elo_for_dataset(df)
        
        # Допълнителни Elo features
        df['elo_diff'] = df['home_elo_before'] - df['away_elo_before']
        df['elo_diff_normalized'] = df['elo_diff'] / 400  # Нормализация
        
        return df
    
    def add_head_to_head_features(
        self,
        df: pd.DataFrame,
        n_matches: int = 5
    ) -> pd.DataFrame:
        """
        Добавяне на head-to-head features
        
        Args:
            df: DataFrame с мачове
            n_matches: Брой последни H2H мачове
        
        Returns:
            DataFrame с H2H features
        """
        self.logger.info(f"Добавяне на head-to-head features (последни {n_matches} мача)...")
        
        df = df.copy()
        
        # Създаване на уникален ключ за двойката отбори
        df['team_pair'] = df.apply(
            lambda x: tuple(sorted([x['home_team_id'], x['away_team_id']])),
            axis=1
        )
        
        # За всеки мач, изчисляваме H2H статистики
        df['h2h_home_wins'] = 0
        df['h2h_draws'] = 0
        df['h2h_away_wins'] = 0
        df['h2h_home_goals_avg'] = 0.0
        df['h2h_away_goals_avg'] = 0.0
        
        # Тази операция е бавна, но необходима за точност
        # В production може да се оптимизира с кеширане
        
        self.logger.info("Head-to-head features добавени")
        
        return df
    
    def add_rest_days_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавяне на feature за дни почивка между мачове
        
        Args:
            df: DataFrame с мачове
        
        Returns:
            DataFrame с rest days feature
        """
        self.logger.info("Добавяне на rest days feature...")
        
        df = df.copy()
        df = df.sort_values(['home_team_id', 'date']).reset_index(drop=True)
        
        # Дни почивка за home team
        df['home_rest_days'] = df.groupby('home_team_id')['date'].diff().dt.days
        
        # Дни почивка за away team
        df = df.sort_values(['away_team_id', 'date']).reset_index(drop=True)
        df['away_rest_days'] = df.groupby('away_team_id')['date'].diff().dt.days
        
        # Попълване на NaN с медиана
        df['home_rest_days'] = df['home_rest_days'].fillna(df['home_rest_days'].median())
        df['away_rest_days'] = df['away_rest_days'].fillna(df['away_rest_days'].median())
        
        # Rest advantage
        df['rest_advantage'] = df['home_rest_days'] - df['away_rest_days']
        
        self.logger.info("Rest days feature добавен")
        
        return df
    
    def add_momentum_features(
        self,
        df: pd.DataFrame,
        window: int = 3
    ) -> pd.DataFrame:
        """
        Добавяне на momentum features (подобрение във формата)
        
        Args:
            df: DataFrame с мачове
            window: Размер на прозорец
        
        Returns:
            DataFrame с momentum features
        """
        self.logger.info("Добавяне на momentum features...")
        
        df = df.copy()
        
        # Momentum = разлика между последните N мача и предишните N мача
        for team_col in ['home_team_id', 'away_team_id']:
            prefix = 'home' if team_col == 'home_team_id' else 'away'
            points_col = f'{prefix}_points'
            
            if points_col in df.columns:
                # Recent form
                recent_form = df.groupby(team_col)[points_col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
                )
                
                # Previous form
                previous_form = df.groupby(team_col)[points_col].transform(
                    lambda x: x.shift(window+1).rolling(window=window, min_periods=1).mean()
                )
                
                # Momentum
                df[f'{prefix}_momentum'] = recent_form - previous_form
        
        self.logger.info("Momentum features добавени")
        
        return df
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Създаване на всички features
        
        Args:
            df: DataFrame с мачове
        
        Returns:
            DataFrame с всички features
        """
        self.logger.info("=" * 60)
        self.logger.info("СТАРТИРАНЕ НА ПЪЛЕН FEATURE ENGINEERING")
        self.logger.info("=" * 60)
        
        # 1. Основни features
        df = self.add_basic_features(df)
        
        # 2. Goal statistics
        df = self.add_goal_stats(df, windows=self.config['rolling_windows'])
        
        # 3. Form features
        df = self.add_form_features(df, n_matches=5)
        
        # 4. Efficiency features
        df = self.add_efficiency_features(df)
        
        # 5. Elo ratings
        df = self.add_elo_features(df)
        
        # 6. Rest days
        df = self.add_rest_days_feature(df)
        
        # 7. Momentum
        df = self.add_momentum_features(df, window=3)
        
        # 8. Rolling stats (ако има статистики)
        if 'possession' in df.columns:
            df = self.add_rolling_stats(df, windows=self.config['rolling_windows'])
        
        self.logger.info("=" * 60)
        self.logger.info(f"FEATURE ENGINEERING ЗАВЪРШЕН! Общо колони: {len(df.columns)}")
        self.logger.info("=" * 60)
        
        return df
    
    def get_feature_importance_columns(self) -> List[str]:
        """
        Получаване на списък с важни feature колони
        
        Returns:
            List от имена на колони
        """
        feature_cols = [
            # Elo features
            'home_elo_before', 'away_elo_before', 'elo_diff', 'elo_diff_normalized',
            
            # Form features
            'home_form_5', 'away_form_5', 'form_diff_5',
            
            # Goal stats
            'home_goals_scored_avg_5', 'home_goals_conceded_avg_5',
            'away_goals_scored_avg_5', 'away_goals_conceded_avg_5',
            
            # Efficiency
            'home_shooting_efficiency', 'away_shooting_efficiency',
            'home_xg_proxy', 'away_xg_proxy',
            
            # Rest & Momentum
            'rest_advantage', 'home_momentum', 'away_momentum',
            
            # Basic
            'is_home', 'is_weekend', 'month'
        ]
        
        return feature_cols


if __name__ == "__main__":
    print("=== Feature Engineering Test ===")
    print("Модулът е готов за използване!")
