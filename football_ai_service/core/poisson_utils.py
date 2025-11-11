"""
Poisson Model Utilities - Статистически модел за прогнозиране на голове
"""

import logging
from typing import Dict, Tuple, Optional, List

import pandas as pd
import numpy as np
from scipy.stats import poisson

from .utils import setup_logging, safe_divide


class PoissonModel:
    """
    Poisson модел за прогнозиране на футболни резултати
    
    Базиран на:
    - Attack strength (способност за вкарване на голове)
    - Defense strength (способност за предотвратяване на голове)
    - Home advantage
    - League average goals
    """
    
    def __init__(
        self,
        home_advantage: float = 1.15,
        league_normalization: bool = True,
        min_matches: int = 5
    ):
        """
        Инициализация на Poisson модел
        
        Args:
            home_advantage: Множител за предимство на домакина
            league_normalization: Дали да нормализираме по лига
            min_matches: Минимален брой мачове за изчисляване на strength
        """
        self.home_advantage = home_advantage
        self.league_normalization = league_normalization
        self.min_matches = min_matches
        self.logger = setup_logging()
        
        # Съхранение на team strengths
        self.attack_strength: Dict[int, float] = {}
        self.defense_strength: Dict[int, float] = {}
        
        # League averages
        self.league_avg_goals_home: Dict[int, float] = {}
        self.league_avg_goals_away: Dict[int, float] = {}
        
        self.logger.info("PoissonModel инициализиран")
    
    def calculate_team_strengths(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Изчисляване на attack и defense strength за всеки отбор
        
        Args:
            df: DataFrame с мачове (трябва да има: home_team_id, away_team_id,
                home_score, away_score, league_id)
        
        Returns:
            DataFrame с добавени strength колони
        """
        self.logger.info("Изчисляване на team strengths...")
        
        df = df.copy()
        
        # Изчисляване на league averages
        if self.league_normalization and 'league_id' in df.columns:
            league_stats = df.groupby('league_id').agg({
                'home_score': 'mean',
                'away_score': 'mean'
            }).reset_index()
            
            self.league_avg_goals_home = dict(zip(
                league_stats['league_id'],
                league_stats['home_score']
            ))
            self.league_avg_goals_away = dict(zip(
                league_stats['league_id'],
                league_stats['away_score']
            ))
        else:
            # Global average
            global_avg_home = df['home_score'].mean()
            global_avg_away = df['away_score'].mean()
            self.league_avg_goals_home = {0: global_avg_home}
            self.league_avg_goals_away = {0: global_avg_away}
        
        # Изчисляване на attack strength за всеки отбор
        # Attack strength = (голове вкарани) / (средно голове в лигата)
        
        # Home attack strength
        home_attack = df.groupby('home_team_id').agg({
            'home_score': 'mean',
            'league_id': 'first'
        }).reset_index()
        
        home_attack['league_avg'] = home_attack['league_id'].map(
            lambda x: self.league_avg_goals_home.get(x, self.league_avg_goals_home.get(0, 1.5))
        )
        home_attack['attack_strength'] = home_attack['home_score'] / home_attack['league_avg']
        
        # Away attack strength
        away_attack = df.groupby('away_team_id').agg({
            'away_score': 'mean',
            'league_id': 'first'
        }).reset_index()
        
        away_attack['league_avg'] = away_attack['league_id'].map(
            lambda x: self.league_avg_goals_away.get(x, self.league_avg_goals_away.get(0, 1.2))
        )
        away_attack['attack_strength'] = away_attack['away_score'] / away_attack['league_avg']
        
        # Комбиниране на home и away attack
        all_teams = set(df['home_team_id'].unique()) | set(df['away_team_id'].unique())
        
        for team_id in all_teams:
            home_str = home_attack[home_attack['home_team_id'] == team_id]['attack_strength'].values
            away_str = away_attack[away_attack['away_team_id'] == team_id]['attack_strength'].values
            
            # Средно от home и away attack
            strengths = []
            if len(home_str) > 0:
                strengths.append(home_str[0])
            if len(away_str) > 0:
                strengths.append(away_str[0])
            
            if strengths:
                self.attack_strength[team_id] = np.mean(strengths)
            else:
                self.attack_strength[team_id] = 1.0
        
        # Изчисляване на defense strength
        # Defense strength = (голове получени) / (средно голове в лигата)
        
        # Home defense (голове получени като домакин)
        home_defense = df.groupby('home_team_id').agg({
            'away_score': 'mean',
            'league_id': 'first'
        }).reset_index()
        
        home_defense['league_avg'] = home_defense['league_id'].map(
            lambda x: self.league_avg_goals_away.get(x, self.league_avg_goals_away.get(0, 1.2))
        )
        home_defense['defense_strength'] = home_defense['away_score'] / home_defense['league_avg']
        
        # Away defense (голове получени като гост)
        away_defense = df.groupby('away_team_id').agg({
            'home_score': 'mean',
            'league_id': 'first'
        }).reset_index()
        
        away_defense['league_avg'] = away_defense['league_id'].map(
            lambda x: self.league_avg_goals_home.get(x, self.league_avg_goals_home.get(0, 1.5))
        )
        away_defense['defense_strength'] = away_defense['home_score'] / away_defense['league_avg']
        
        # Комбиниране на home и away defense
        for team_id in all_teams:
            home_def = home_defense[home_defense['home_team_id'] == team_id]['defense_strength'].values
            away_def = away_defense[away_defense['away_team_id'] == team_id]['defense_strength'].values
            
            strengths = []
            if len(home_def) > 0:
                strengths.append(home_def[0])
            if len(away_def) > 0:
                strengths.append(away_def[0])
            
            if strengths:
                self.defense_strength[team_id] = np.mean(strengths)
            else:
                self.defense_strength[team_id] = 1.0
        
        self.logger.info(f"Team strengths изчислени за {len(all_teams)} отбора")
        
        return df
    
    def calculate_lambda(
        self,
        home_team_id: int,
        away_team_id: int,
        league_id: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Изчисляване на λ (очаквани голове) за home и away отбор
        
        Args:
            home_team_id: ID на домакина
            away_team_id: ID на гостите
            league_id: ID на лигата (за нормализация)
        
        Returns:
            Tuple (λ_home, λ_away)
        """
        # League average
        if league_id and league_id in self.league_avg_goals_home:
            avg_home = self.league_avg_goals_home[league_id]
            avg_away = self.league_avg_goals_away[league_id]
        else:
            avg_home = self.league_avg_goals_home.get(0, 1.5)
            avg_away = self.league_avg_goals_away.get(0, 1.2)
        
        # Team strengths
        home_attack = self.attack_strength.get(home_team_id, 1.0)
        home_defense = self.defense_strength.get(home_team_id, 1.0)
        away_attack = self.attack_strength.get(away_team_id, 1.0)
        away_defense = self.defense_strength.get(away_team_id, 1.0)
        
        # λ_home = league_avg_home * home_attack * away_defense * home_advantage
        lambda_home = avg_home * home_attack * away_defense * self.home_advantage
        
        # λ_away = league_avg_away * away_attack * home_defense
        lambda_away = avg_away * away_attack * home_defense
        
        return lambda_home, lambda_away
    
    def predict_match_probabilities(
        self,
        home_team_id: int,
        away_team_id: int,
        league_id: Optional[int] = None,
        max_goals: int = 10
    ) -> Dict[str, float]:
        """
        Прогнозиране на вероятности за мач
        
        Args:
            home_team_id: ID на домакина
            away_team_id: ID на гостите
            league_id: ID на лигата
            max_goals: Максимален брой голове за изчисления
        
        Returns:
            Dictionary с вероятности
        """
        # Изчисляване на lambda
        lambda_home, lambda_away = self.calculate_lambda(
            home_team_id, away_team_id, league_id
        )
        
        # Матрица с вероятности за всеки резултат
        prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
        
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                prob_matrix[i, j] = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
        
        # 1X2 вероятности
        prob_home_win = np.sum(np.tril(prob_matrix, -1))  # Под диагонала
        prob_draw = np.sum(np.diag(prob_matrix))  # Диагонал
        prob_away_win = np.sum(np.triu(prob_matrix, 1))  # Над диагонала
        
        # Over/Under 2.5
        prob_over_25 = 0
        prob_under_25 = 0
        
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                if i + j > 2.5:
                    prob_over_25 += prob_matrix[i, j]
                else:
                    prob_under_25 += prob_matrix[i, j]
        
        # Both Teams To Score
        prob_btts_yes = 0
        prob_btts_no = 0
        
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                if i > 0 and j > 0:
                    prob_btts_yes += prob_matrix[i, j]
                else:
                    prob_btts_no += prob_matrix[i, j]
        
        # Expected goals
        expected_home_goals = lambda_home
        expected_away_goals = lambda_away
        expected_total_goals = lambda_home + lambda_away
        
        return {
            'lambda_home': lambda_home,
            'lambda_away': lambda_away,
            'prob_home_win': prob_home_win,
            'prob_draw': prob_draw,
            'prob_away_win': prob_away_win,
            'prob_over_25': prob_over_25,
            'prob_under_25': prob_under_25,
            'prob_btts_yes': prob_btts_yes,
            'prob_btts_no': prob_btts_no,
            'expected_home_goals': expected_home_goals,
            'expected_away_goals': expected_away_goals,
            'expected_total_goals': expected_total_goals
        }
    
    def predict_dataset(
        self,
        df: pd.DataFrame,
        max_goals: int = 10
    ) -> pd.DataFrame:
        """
        Прогнозиране за целия dataset
        
        Args:
            df: DataFrame с мачове
            max_goals: Максимален брой голове
        
        Returns:
            DataFrame с добавени прогнози
        """
        self.logger.info(f"Прогнозиране за {len(df)} мача...")
        
        df = df.copy()
        
        # Инициализация на колони
        df['poisson_lambda_home'] = 0.0
        df['poisson_lambda_away'] = 0.0
        df['poisson_prob_1'] = 0.0
        df['poisson_prob_x'] = 0.0
        df['poisson_prob_2'] = 0.0
        df['poisson_prob_over25'] = 0.0
        df['poisson_prob_btts'] = 0.0
        df['poisson_expected_goals'] = 0.0
        
        # Прогнозиране за всеки мач
        skipped = 0
        for idx, row in df.iterrows():
            home_id = row['home_team_id']
            away_id = row['away_team_id']
            league_id = row.get('league_id', None)
            
            # Проверка дали отборите са в модела
            if home_id not in self.attack_strength or away_id not in self.attack_strength:
                skipped += 1
                # Използваме default вероятности
                df.at[idx, 'poisson_lambda_home'] = 1.5
                df.at[idx, 'poisson_lambda_away'] = 1.2
                df.at[idx, 'poisson_prob_1'] = 0.40
                df.at[idx, 'poisson_prob_x'] = 0.30
                df.at[idx, 'poisson_prob_2'] = 0.30
                df.at[idx, 'poisson_prob_over25'] = 0.50
                df.at[idx, 'poisson_prob_btts'] = 0.50
                df.at[idx, 'poisson_expected_goals'] = 2.7
                continue
            
            try:
                pred = self.predict_match_probabilities(
                    home_id, away_id, league_id, max_goals
                )
                
                df.at[idx, 'poisson_lambda_home'] = pred['lambda_home']
                df.at[idx, 'poisson_lambda_away'] = pred['lambda_away']
                df.at[idx, 'poisson_prob_1'] = pred['prob_home_win']
                df.at[idx, 'poisson_prob_x'] = pred['prob_draw']
                df.at[idx, 'poisson_prob_2'] = pred['prob_away_win']
                df.at[idx, 'poisson_prob_over25'] = pred['prob_over_25']
                df.at[idx, 'poisson_prob_btts'] = pred['prob_btts_yes']
                df.at[idx, 'poisson_expected_goals'] = pred['expected_total_goals']
            except Exception as e:
                skipped += 1
                # Default values при грешка
                df.at[idx, 'poisson_lambda_home'] = 1.5
                df.at[idx, 'poisson_lambda_away'] = 1.2
                df.at[idx, 'poisson_prob_1'] = 0.40
                df.at[idx, 'poisson_prob_x'] = 0.30
                df.at[idx, 'poisson_prob_2'] = 0.30
                df.at[idx, 'poisson_prob_over25'] = 0.50
                df.at[idx, 'poisson_prob_btts'] = 0.50
                df.at[idx, 'poisson_expected_goals'] = 2.7
        
        if skipped > 0:
            self.logger.warning(f"Пропуснати {skipped} мача (липсващи отбори или грешки)")
        
        self.logger.info("Прогнозиране завършено")
        
        return df
    
    def evaluate_predictions(
        self,
        df: pd.DataFrame,
        actual_result_col: str = 'result',
        actual_over25_col: str = 'over_25',
        actual_btts_col: str = 'btts'
    ) -> Dict[str, float]:
        """
        Оценка на точността на прогнозите
        
        Args:
            df: DataFrame с прогнози и действителни резултати
            actual_result_col: Колона с действителен резултат (1, X, 2)
            actual_over25_col: Колона с over 2.5 (0 или 1)
            actual_btts_col: Колона с BTTS (0 или 1)
        
        Returns:
            Dictionary с метрики
        """
        self.logger.info("Оценка на Poisson predictions...")
        
        # 1X2 accuracy
        df['poisson_pred_1x2'] = df[['poisson_prob_1', 'poisson_prob_x', 'poisson_prob_2']].idxmax(axis=1)
        df['poisson_pred_1x2'] = df['poisson_pred_1x2'].map({
            'poisson_prob_1': '1',
            'poisson_prob_x': 'X',
            'poisson_prob_2': '2'
        })
        
        accuracy_1x2 = (df['poisson_pred_1x2'] == df[actual_result_col]).mean()
        
        # Over/Under 2.5 accuracy
        df['poisson_pred_over25'] = (df['poisson_prob_over25'] > 0.5).astype(int)
        accuracy_over25 = (df['poisson_pred_over25'] == df[actual_over25_col]).mean()
        
        # BTTS accuracy
        df['poisson_pred_btts'] = (df['poisson_prob_btts'] > 0.5).astype(int)
        accuracy_btts = (df['poisson_pred_btts'] == df[actual_btts_col]).mean()
        
        # Log Loss (за вероятности)
        from sklearn.metrics import log_loss
        
        # 1X2 log loss
        y_true_1x2 = pd.get_dummies(df[actual_result_col])[['1', 'X', '2']].values
        y_pred_1x2 = df[['poisson_prob_1', 'poisson_prob_x', 'poisson_prob_2']].values
        
        # Премахване на NaN и clip probabilities
        y_pred_1x2 = np.nan_to_num(y_pred_1x2, nan=0.33)  # Default равномерно разпределение
        y_pred_1x2 = np.clip(y_pred_1x2, 1e-15, 1 - 1e-15)
        
        # Нормализация на вероятностите (да сумират до 1)
        row_sums = y_pred_1x2.sum(axis=1, keepdims=True)
        y_pred_1x2 = y_pred_1x2 / row_sums
        
        logloss_1x2 = log_loss(y_true_1x2, y_pred_1x2)
        
        # Over/Under log loss
        y_pred_over25 = df['poisson_prob_over25'].fillna(0.5).values
        y_pred_over25 = np.clip(y_pred_over25, 1e-15, 1 - 1e-15)
        logloss_over25 = log_loss(df[actual_over25_col], y_pred_over25)
        
        # BTTS log loss
        y_pred_btts = df['poisson_prob_btts'].fillna(0.5).values
        y_pred_btts = np.clip(y_pred_btts, 1e-15, 1 - 1e-15)
        logloss_btts = log_loss(df[actual_btts_col], y_pred_btts)
        
        metrics = {
            'accuracy_1x2': accuracy_1x2,
            'accuracy_over25': accuracy_over25,
            'accuracy_btts': accuracy_btts,
            'log_loss_1x2': logloss_1x2,
            'log_loss_over25': logloss_over25,
            'log_loss_btts': logloss_btts,
            'mean_lambda_home': df['poisson_lambda_home'].mean(),
            'mean_lambda_away': df['poisson_lambda_away'].mean(),
            'mean_expected_goals': df['poisson_expected_goals'].mean()
        }
        
        self.logger.info(f"Accuracy 1X2: {accuracy_1x2:.3f}")
        self.logger.info(f"Accuracy Over/Under 2.5: {accuracy_over25:.3f}")
        self.logger.info(f"Accuracy BTTS: {accuracy_btts:.3f}")
        self.logger.info(f"Log Loss 1X2: {logloss_1x2:.3f}")
        
        return metrics
    
    def get_most_likely_score(
        self,
        home_team_id: int,
        away_team_id: int,
        league_id: Optional[int] = None,
        top_n: int = 5
    ) -> List[Tuple[int, int, float]]:
        """
        Най-вероятни резултати
        
        Args:
            home_team_id: ID на домакина
            away_team_id: ID на гостите
            league_id: ID на лигата
            top_n: Брой резултати
        
        Returns:
            List от tuples (home_goals, away_goals, probability)
        """
        lambda_home, lambda_away = self.calculate_lambda(
            home_team_id, away_team_id, league_id
        )
        
        # Изчисляване на вероятности за всички резултати до 10-10
        scores = []
        for i in range(11):
            for j in range(11):
                prob = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
                scores.append((i, j, prob))
        
        # Сортиране по вероятност
        scores.sort(key=lambda x: x[2], reverse=True)
        
        return scores[:top_n]


if __name__ == "__main__":
    print("=== Poisson Model Test ===")
    print("Модулът е готов за използване!")
