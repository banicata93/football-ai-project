"""
Elo Rating Calculator - Изчисляване на Elo рейтинг за футболни отбори
"""

import logging
from typing import Dict, Tuple, Optional

import pandas as pd
import numpy as np

from .utils import setup_logging


class EloCalculator:
    """
    Elo Rating система за футболни отбори
    
    Базирана на шахматната Elo система, адаптирана за футбол:
    - K-factor: Колко бързо се променя рейтинга
    - Home advantage: Бонус за домакина
    - Goal difference: Множител базиран на разликата в головете
    """
    
    def __init__(
        self,
        k_factor: float = 20,
        initial_rating: float = 1500,
        home_advantage: float = 100
    ):
        """
        Инициализация на Elo calculator
        
        Args:
            k_factor: K-фактор (колко бързо се променя рейтинга)
            initial_rating: Начален рейтинг за нови отбори
            home_advantage: Предимство на домакина (в Elo точки)
        """
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.home_advantage = home_advantage
        self.logger = setup_logging()
        
        # Текущи рейтинги на отборите
        self.ratings: Dict[int, float] = {}
        
        # История на рейтингите
        self.history: list = []
    
    def get_rating(self, team_id: int) -> float:
        """
        Получаване на текущ рейтинг на отбор
        
        Args:
            team_id: ID на отбора
        
        Returns:
            Elo рейтинг
        """
        if team_id not in self.ratings:
            self.ratings[team_id] = self.initial_rating
        return self.ratings[team_id]
    
    def expected_score(
        self,
        rating_a: float,
        rating_b: float,
        is_home: bool = False
    ) -> float:
        """
        Изчисляване на очакван резултат
        
        Args:
            rating_a: Elo рейтинг на отбор A
            rating_b: Elo рейтинг на отбор B
            is_home: Дали отбор A е домакин
        
        Returns:
            Очакван резултат (0-1)
        """
        # Добавяне на home advantage
        if is_home:
            rating_a += self.home_advantage
        
        # Формула на Elo
        expected = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        return expected
    
    def goal_difference_multiplier(self, goal_diff: int) -> float:
        """
        Множител базиран на разликата в головете
        
        Args:
            goal_diff: Разлика в головете
        
        Returns:
            Множител
        """
        abs_diff = abs(goal_diff)
        
        if abs_diff <= 1:
            return 1.0
        elif abs_diff == 2:
            return 1.5
        else:
            return (11 + abs_diff) / 8
    
    def actual_score(
        self,
        home_score: int,
        away_score: int,
        for_home: bool = True
    ) -> float:
        """
        Определяне на действителен резултат
        
        Args:
            home_score: Голове на домакина
            away_score: Голове на гостите
            for_home: Дали изчисляваме за домакина
        
        Returns:
            Действителен резултат (1 за победа, 0.5 за равенство, 0 за загуба)
        """
        if for_home:
            if home_score > away_score:
                return 1.0
            elif home_score == away_score:
                return 0.5
            else:
                return 0.0
        else:
            if away_score > home_score:
                return 1.0
            elif away_score == home_score:
                return 0.5
            else:
                return 0.0
    
    def update_ratings(
        self,
        home_team_id: int,
        away_team_id: int,
        home_score: int,
        away_score: int,
        match_date: Optional[pd.Timestamp] = None,
        match_id: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Обновяване на рейтингите след мач
        
        Args:
            home_team_id: ID на домакина
            away_team_id: ID на гостите
            home_score: Голове на домакина
            away_score: Голове на гостите
            match_date: Дата на мача
            match_id: ID на мача
        
        Returns:
            Tuple (нов рейтинг домакин, нов рейтинг гост)
        """
        # Текущи рейтинги
        home_rating_before = self.get_rating(home_team_id)
        away_rating_before = self.get_rating(away_team_id)
        
        # Очаквани резултати
        home_expected = self.expected_score(home_rating_before, away_rating_before, is_home=True)
        away_expected = 1 - home_expected
        
        # Действителни резултати
        home_actual = self.actual_score(home_score, away_score, for_home=True)
        away_actual = self.actual_score(home_score, away_score, for_home=False)
        
        # Множител за разлика в головете
        goal_diff = abs(home_score - away_score)
        multiplier = self.goal_difference_multiplier(goal_diff)
        
        # Обновяване на рейтингите
        home_rating_after = home_rating_before + self.k_factor * multiplier * (home_actual - home_expected)
        away_rating_after = away_rating_before + self.k_factor * multiplier * (away_actual - away_expected)
        
        # Запазване на новите рейтинги
        self.ratings[home_team_id] = home_rating_after
        self.ratings[away_team_id] = away_rating_after
        
        # Запазване в историята
        self.history.append({
            'match_id': match_id,
            'date': match_date,
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'home_score': home_score,
            'away_score': away_score,
            'home_rating_before': home_rating_before,
            'away_rating_before': away_rating_before,
            'home_rating_after': home_rating_after,
            'away_rating_after': away_rating_after,
            'home_expected': home_expected,
            'away_expected': away_expected
        })
        
        return home_rating_after, away_rating_after
    
    def calculate_elo_for_dataset(self, fixtures_df: pd.DataFrame) -> pd.DataFrame:
        """
        Изчисляване на Elo рейтинги за целия dataset
        
        Args:
            fixtures_df: DataFrame с мачове (трябва да има колони:
                        home_team_id, away_team_id, home_score, away_score, date)
        
        Returns:
            DataFrame с добавени Elo колони
        """
        self.logger.info("Изчисляване на Elo рейтинги за dataset...")
        
        # Копиране на DataFrame
        df = fixtures_df.copy()
        
        # Сортиране по дата
        df = df.sort_values('date').reset_index(drop=True)
        
        # Инициализация на колони
        df['home_elo_before'] = 0.0
        df['away_elo_before'] = 0.0
        df['home_elo_after'] = 0.0
        df['away_elo_after'] = 0.0
        df['elo_diff_before'] = 0.0
        df['home_win_prob'] = 0.0
        
        # Изчисляване на Elo за всеки мач
        for idx, row in df.iterrows():
            home_id = row['home_team_id']
            away_id = row['away_team_id']
            home_score = row['home_score']
            away_score = row['away_score']
            
            # Текущи рейтинги (преди мача)
            home_elo_before = self.get_rating(home_id)
            away_elo_before = self.get_rating(away_id)
            
            # Очаквана вероятност за победа на домакина
            home_win_prob = self.expected_score(home_elo_before, away_elo_before, is_home=True)
            
            # Обновяване на рейтингите
            home_elo_after, away_elo_after = self.update_ratings(
                home_id, away_id, home_score, away_score,
                match_date=row.get('date'),
                match_id=row.get('match_id')
            )
            
            # Запазване в DataFrame
            df.at[idx, 'home_elo_before'] = home_elo_before
            df.at[idx, 'away_elo_before'] = away_elo_before
            df.at[idx, 'home_elo_after'] = home_elo_after
            df.at[idx, 'away_elo_after'] = away_elo_after
            df.at[idx, 'elo_diff_before'] = home_elo_before - away_elo_before
            df.at[idx, 'home_win_prob'] = home_win_prob
        
        self.logger.info(f"Elo рейтинги изчислени за {len(df)} мача")
        self.logger.info(f"Общо отбори с рейтинг: {len(self.ratings)}")
        
        return df
    
    def get_top_teams(self, n: int = 10) -> list:
        """
        Получаване на топ N отбори по Elo рейтинг
        
        Args:
            n: Брой отбори
        
        Returns:
            List от tuples (team_id, rating)
        """
        sorted_teams = sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
        return sorted_teams[:n]
    
    def get_history_df(self) -> pd.DataFrame:
        """
        Получаване на историята като DataFrame
        
        Returns:
            DataFrame с история на рейтингите
        """
        return pd.DataFrame(self.history)
    
    def reset(self):
        """Нулиране на всички рейтинги и история"""
        self.ratings = {}
        self.history = []
        self.logger.info("Elo calculator нулиран")
    
    def save_ratings(self, filepath: str):
        """
        Запазване на текущите рейтинги
        
        Args:
            filepath: Път до файл
        """
        ratings_df = pd.DataFrame([
            {'team_id': team_id, 'elo_rating': rating}
            for team_id, rating in self.ratings.items()
        ])
        ratings_df.to_csv(filepath, index=False)
        self.logger.info(f"Elo рейтинги запазени в {filepath}")
    
    def load_ratings(self, filepath: str):
        """
        Зареждане на рейтинги от файл
        
        Args:
            filepath: Път до файл
        """
        ratings_df = pd.read_csv(filepath)
        self.ratings = dict(zip(ratings_df['team_id'], ratings_df['elo_rating']))
        self.logger.info(f"Elo рейтинги заредени от {filepath}")


if __name__ == "__main__":
    # Тестване на Elo calculator
    print("=== Elo Calculator Test ===\n")
    
    elo = EloCalculator(k_factor=20, initial_rating=1500, home_advantage=100)
    
    # Симулация на мачове
    matches = [
        (1, 2, 3, 1),  # Отбор 1 vs Отбор 2: 3-1
        (2, 3, 2, 2),  # Отбор 2 vs Отбор 3: 2-2
        (3, 1, 0, 2),  # Отбор 3 vs Отбор 1: 0-2
        (1, 2, 1, 0),  # Отбор 1 vs Отбор 2: 1-0
    ]
    
    for home_id, away_id, home_score, away_score in matches:
        home_before = elo.get_rating(home_id)
        away_before = elo.get_rating(away_id)
        
        home_after, away_after = elo.update_ratings(home_id, away_id, home_score, away_score)
        
        print(f"Мач: Отбор {home_id} {home_score}-{away_score} Отбор {away_id}")
        print(f"  Отбор {home_id}: {home_before:.1f} → {home_after:.1f} ({home_after-home_before:+.1f})")
        print(f"  Отбор {away_id}: {away_before:.1f} → {away_after:.1f} ({away_after-away_before:+.1f})")
        print()
    
    print("Топ отбори:")
    for team_id, rating in elo.get_top_teams(3):
        print(f"  Отбор {team_id}: {rating:.1f}")
