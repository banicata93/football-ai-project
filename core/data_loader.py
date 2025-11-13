"""
ESPN Data Loader - Зареждане и обработка на ESPN футболни данни
"""

import os
import logging
from typing import Dict, Optional, List, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

from .utils import setup_logging, load_config, PerformanceTimer


class ESPNDataLoader:
    """
    Клас за зареждане и обработка на ESPN данни
    """
    
    def __init__(self, config_path: str = "config/data_config.yaml"):
        """
        Инициализация на data loader
        
        Args:
            config_path: Път до конфигурационен файл
        """
        self.logger = setup_logging()
        self.config = load_config(config_path)
        self.data_paths = self.config['data_paths']
        self.base_files = self.config['base_files']
        
        # Кеш за заредени данни
        self._cache = {}
        
        self.logger.info("ESPNDataLoader инициализиран успешно")
    
    def load_fixtures(self) -> pd.DataFrame:
        """
        Зареждане на fixtures (мачове)
        
        Returns:
            DataFrame с мачове
        """
        if 'fixtures' in self._cache:
            self.logger.info("Зареждане на fixtures от кеш")
            return self._cache['fixtures']
        
        with PerformanceTimer("Зареждане на fixtures", self.logger):
            filepath = os.path.join(self.data_paths['base_data'], self.base_files['fixtures'])
            df = pd.read_csv(filepath)
            
            # Обработка на колони
            df['date'] = pd.to_datetime(df['date'])
            
            # Преименуване на колони за яснота
            column_mapping = {
                'eventId': 'match_id',
                'leagueId': 'league_id',
                'homeTeamId': 'home_team_id',
                'awayTeamId': 'away_team_id',
                'homeTeamScore': 'home_score',
                'awayTeamScore': 'away_score',
                'venueId': 'venue_id',
                'statusId': 'status_id',
                'homeTeamWinner': 'home_win',
                'awayTeamWinner': 'away_win'
            }
            df = df.rename(columns=column_mapping)
            
            # Добавяне на изчислени колони
            df['total_goals'] = df['home_score'] + df['away_score']
            df['goal_diff'] = df['home_score'] - df['away_score']
            
            # Определяне на резултат (1X2)
            df['result'] = df.apply(self._determine_result, axis=1)
            
            # Over/Under 2.5
            df['over_25'] = (df['total_goals'] > 2.5).astype(int)
            
            # Both Teams To Score
            df['btts'] = ((df['home_score'] > 0) & (df['away_score'] > 0)).astype(int)
            
            # Сортиране по дата
            df = df.sort_values('date').reset_index(drop=True)
            
            self._cache['fixtures'] = df
            self.logger.info(f"Заредени {len(df)} мача")
            
            return df
    
    def load_teams(self) -> pd.DataFrame:
        """
        Зареждане на отбори
        
        Returns:
            DataFrame с отбори
        """
        if 'teams' in self._cache:
            return self._cache['teams']
        
        with PerformanceTimer("Зареждане на teams", self.logger):
            filepath = os.path.join(self.data_paths['base_data'], self.base_files['teams'])
            df = pd.read_csv(filepath)
            
            # Преименуване
            df = df.rename(columns={
                'teamId': 'team_id',
                'displayName': 'team_name',
                'venueId': 'venue_id'
            })
            
            self._cache['teams'] = df
            self.logger.info(f"Заредени {len(df)} отбора")
            
            return df
    
    def load_team_stats(self) -> pd.DataFrame:
        """
        Зареждане на статистики на отборите
        
        Returns:
            DataFrame със статистики
        """
        if 'team_stats' in self._cache:
            return self._cache['team_stats']
        
        with PerformanceTimer("Зареждане на team stats", self.logger):
            filepath = os.path.join(self.data_paths['base_data'], self.base_files['team_stats'])
            df = pd.read_csv(filepath)
            
            # Преименуване
            column_mapping = {
                'eventId': 'match_id',
                'teamId': 'team_id',
                'teamOrder': 'team_order',
                'possessionPct': 'possession',
                'foulsCommitted': 'fouls',
                'yellowCards': 'yellow_cards',
                'redCards': 'red_cards',
                'wonCorners': 'corners',
                'totalShots': 'shots',
                'shotsOnTarget': 'shots_on_target',
                'accuratePasses': 'accurate_passes',
                'totalPasses': 'total_passes',
                'passPct': 'pass_accuracy',
                'totalCrosses': 'crosses',
                'accurateCrosses': 'accurate_crosses',
                'totalTackles': 'tackles',
                'effectiveTackles': 'effective_tackles',
                'interceptions': 'interceptions',
                'totalClearance': 'clearances',
                'blockedShots': 'blocked_shots'
            }
            df = df.rename(columns=column_mapping)
            
            # Изчисляване на допълнителни метрики
            df['shot_accuracy'] = df['shots_on_target'] / df['shots'].replace(0, np.nan)
            df['tackle_success'] = df['effective_tackles'] / df['tackles'].replace(0, np.nan)
            df['cross_accuracy'] = df['accurate_crosses'] / df['crosses'].replace(0, np.nan)
            
            # Попълване на NaN с 0
            df = df.fillna(0)
            
            self._cache['team_stats'] = df
            self.logger.info(f"Заредени статистики за {len(df)} мача")
            
            return df
    
    def load_players(self) -> pd.DataFrame:
        """
        Зареждане на играчи
        
        Returns:
            DataFrame с играчи
        """
        if 'players' in self._cache:
            return self._cache['players']
        
        with PerformanceTimer("Зареждане на players", self.logger):
            filepath = os.path.join(self.data_paths['base_data'], self.base_files['players'])
            df = pd.read_csv(filepath)
            
            # Преименуване
            df = df.rename(columns={
                'athleteId': 'player_id',
                'displayName': 'player_name',
                'positionId': 'position_id',
                'positionName': 'position',
                'dateOfBirth': 'birth_date'
            })
            
            # Обработка на дата на раждане
            df['birth_date'] = pd.to_datetime(df['birth_date'], errors='coerce')
            
            self._cache['players'] = df
            self.logger.info(f"Заредени {len(df)} играчи")
            
            return df
    
    def load_leagues(self) -> pd.DataFrame:
        """
        Зареждане на лиги
        
        Returns:
            DataFrame с лиги
        """
        if 'leagues' in self._cache:
            return self._cache['leagues']
        
        with PerformanceTimer("Зареждане на leagues", self.logger):
            filepath = os.path.join(self.data_paths['base_data'], self.base_files['leagues'])
            df = pd.read_csv(filepath)
            
            # Преименуване
            df = df.rename(columns={
                'leagueId': 'league_id',
                'leagueName': 'league_name',
                'seasonName': 'season_name'
            })
            
            self._cache['leagues'] = df
            self.logger.info(f"Заредени {len(df)} лиги/сезони")
            
            return df
    
    def load_standings(self) -> pd.DataFrame:
        """
        Зареждане на класирания
        
        Returns:
            DataFrame с класирания
        """
        if 'standings' in self._cache:
            return self._cache['standings']
        
        with PerformanceTimer("Зареждане на standings", self.logger):
            filepath = os.path.join(self.data_paths['base_data'], self.base_files['standings'])
            df = pd.read_csv(filepath)
            
            # Преименуване
            column_mapping = {
                'teamId': 'team_id',
                'leagueId': 'league_id'
            }
            df = df.rename(columns=column_mapping)
            
            self._cache['standings'] = df
            self.logger.info(f"Заредени {len(df)} записа в класирания")
            
            return df
    
    def load_venues(self) -> pd.DataFrame:
        """
        Зареждане на стадиони
        
        Returns:
            DataFrame със стадиони
        """
        if 'venues' in self._cache:
            return self._cache['venues']
        
        with PerformanceTimer("Зареждане на venues", self.logger):
            filepath = os.path.join(self.data_paths['base_data'], self.base_files['venues'])
            df = pd.read_csv(filepath)
            
            # Преименуване
            df = df.rename(columns={
                'venueId': 'venue_id',
                'venueName': 'venue_name'
            })
            
            self._cache['venues'] = df
            self.logger.info(f"Заредени {len(df)} стадиона")
            
            return df
    
    def load_team_roster(self) -> pd.DataFrame:
        """
        Зареждане на състави на отборите
        
        Returns:
            DataFrame със състави
        """
        if 'team_roster' in self._cache:
            return self._cache['team_roster']
        
        with PerformanceTimer("Зареждане на team roster", self.logger):
            filepath = os.path.join(self.data_paths['base_data'], self.base_files['team_roster'])
            df = pd.read_csv(filepath)
            
            # Преименуване
            df = df.rename(columns={
                'athleteId': 'player_id',
                'teamId': 'team_id',
                'leagueId': 'league_id'
            })
            
            self._cache['team_roster'] = df
            self.logger.info(f"Заредени {len(df)} записа в състави")
            
            return df
    
    def load_key_events(self, league_pattern: Optional[str] = None) -> pd.DataFrame:
        """
        Зареждане на ключови събития (голове, картони и т.н.)
        
        Args:
            league_pattern: Филтър за лиги (напр. "ENG", "ESP")
        
        Returns:
            DataFrame с ключови събития
        """
        key = f'key_events_{league_pattern}' if league_pattern else 'key_events_all'
        
        if key in self._cache:
            return self._cache[key]
        
        with PerformanceTimer(f"Зареждане на key events", self.logger):
            key_events_dir = self.data_paths['key_events']
            
            # Намиране на всички CSV файлове
            all_files = [f for f in os.listdir(key_events_dir) if f.endswith('.csv')]
            
            if league_pattern:
                all_files = [f for f in all_files if league_pattern in f]
            
            dfs = []
            for file in tqdm(all_files, desc="Зареждане на key events"):
                filepath = os.path.join(key_events_dir, file)
                try:
                    df = pd.read_csv(filepath)
                    dfs.append(df)
                except Exception as e:
                    self.logger.warning(f"Грешка при зареждане на {file}: {e}")
            
            if not dfs:
                self.logger.warning("Няма заредени key events файлове")
                return pd.DataFrame()
            
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Преименуване
            if 'eventId' in combined_df.columns:
                combined_df = combined_df.rename(columns={'eventId': 'match_id'})
            
            self._cache[key] = combined_df
            self.logger.info(f"Заредени {len(combined_df)} ключови събития")
            
            return combined_df
    
    def load_all_base_data(self) -> Dict[str, pd.DataFrame]:
        """
        Зареждане на всички основни данни
        
        Returns:
            Dictionary с всички DataFrames
        """
        self.logger.info("Зареждане на всички основни данни...")
        
        data = {
            'fixtures': self.load_fixtures(),
            'teams': self.load_teams(),
            'team_stats': self.load_team_stats(),
            'players': self.load_players(),
            'leagues': self.load_leagues(),
            'standings': self.load_standings(),
            'venues': self.load_venues(),
            'team_roster': self.load_team_roster()
        }
        
        self.logger.info("Всички основни данни заредени успешно")
        return data
    
    def merge_fixtures_with_stats(self) -> pd.DataFrame:
        """
        Обединяване на fixtures с team stats
        
        Returns:
            DataFrame с мачове и статистики
        """
        self.logger.info("Обединяване на fixtures и team stats...")
        
        fixtures = self.load_fixtures()
        team_stats = self.load_team_stats()
        
        # Merge за home team
        merged = fixtures.merge(
            team_stats,
            left_on='match_id',
            right_on='match_id',
            how='left',
            suffixes=('', '_home')
        )
        
        # Филтриране само на home team stats (team_order == 0)
        merged = merged[merged['team_order'] == 0].copy()
        
        # Merge за away team
        away_stats = team_stats[team_stats['team_order'] == 1].copy()
        away_stats = away_stats.add_suffix('_away')
        away_stats = away_stats.rename(columns={'match_id_away': 'match_id'})
        
        merged = merged.merge(
            away_stats,
            on='match_id',
            how='left'
        )
        
        self.logger.info(f"Обединени {len(merged)} мача със статистики")
        
        return merged
    
    def get_team_match_history(
        self,
        team_id: int,
        end_date: Optional[pd.Timestamp] = None,
        n_matches: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Извличане на историята на мачове за даден отбор
        
        Args:
            team_id: ID на отбора
            end_date: Крайна дата (за избягване на data leakage)
            n_matches: Брой последни мачове
        
        Returns:
            DataFrame с мачове на отбора
        """
        fixtures = self.load_fixtures()
        
        # Филтриране на мачове с този отбор
        team_matches = fixtures[
            (fixtures['home_team_id'] == team_id) | 
            (fixtures['away_team_id'] == team_id)
        ].copy()
        
        # Филтриране по дата
        if end_date:
            team_matches = team_matches[team_matches['date'] < end_date]
        
        # Сортиране по дата
        team_matches = team_matches.sort_values('date', ascending=False)
        
        # Ограничаване на брой мачове
        if n_matches:
            team_matches = team_matches.head(n_matches)
        
        return team_matches
    
    @staticmethod
    def _determine_result(row) -> str:
        """
        Определяне на резултат (1X2)
        
        Args:
            row: Ред от DataFrame
        
        Returns:
            '1' (home win), 'X' (draw), '2' (away win)
        """
        if row['home_win']:
            return '1'
        elif row['away_win']:
            return '2'
        else:
            return 'X'
    
    def clear_cache(self):
        """Изчистване на кеша"""
        self._cache = {}
        self.logger.info("Кешът е изчистен")
    
    def get_cache_info(self) -> Dict[str, int]:
        """
        Информация за кеша
        
        Returns:
            Dictionary с размери на кешираните данни
        """
        return {key: len(df) for key, df in self._cache.items()}


if __name__ == "__main__":
    # Тестване на data loader
    loader = ESPNDataLoader()
    
    # Зареждане на всички данни
    data = loader.load_all_base_data()
    
    print("\n=== ESPN Data Loader Test ===")
    for name, df in data.items():
        print(f"{name}: {len(df)} записа, {len(df.columns)} колони")
    
    # Тест на merge
    merged = loader.merge_fixtures_with_stats()
    print(f"\nMerged fixtures with stats: {len(merged)} записа")
    
    # Кеш информация
    print(f"\nCache info: {loader.get_cache_info()}")
