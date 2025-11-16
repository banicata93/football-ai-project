"""
Team Resolver - Интелигентно търсене и обработка на отбори
"""

import re
import json
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher
import logging

from .utils import setup_logging


class TeamResolver:
    """
    Клас за интелигентно търсене на отбори и обработка на непознати имена
    """
    
    def __init__(self, team_database: Dict, team_names_mapping: Dict):
        """
        Инициализация на TeamResolver
        
        Args:
            team_database: База данни с отбори и техните статистики
            team_names_mapping: Mapping на Team_ID към реални имена
        """
        self.logger = setup_logging()
        self.team_database = team_database
        self.team_names_mapping = team_names_mapping
        
        # Създаваме индекс за бързо търсене
        self._build_search_index()
        
        # Лигови default стойности
        self._load_league_defaults()
    
    def _build_search_index(self):
        """Създава индекс за бързо търсене на отбори"""
        self.search_index = {}
        
        # Добавяме всички варианти на имена
        for team_key, data in self.team_database.items():
            # Оригинално име
            self.search_index[team_key.lower()] = team_key
            
            # Реално име ако има
            if team_key in self.team_names_mapping:
                real_name = self.team_names_mapping[team_key]['display_name']
                self.search_index[real_name.lower()] = team_key
                
                # Алтернативни имена
                if 'alternative_names' in self.team_names_mapping[team_key]:
                    for alt_name in self.team_names_mapping[team_key]['alternative_names']:
                        self.search_index[alt_name.lower()] = team_key
    
    def _load_league_defaults(self):
        """Зарежда default стойности по лиги"""
        self.league_defaults = {
            'Premier League': {
                'elo': 1650, 'form': 0.1, 'goals_avg': 1.8,
                'xg_proxy': 1.7, 'shooting_efficiency': 0.35
            },
            'La Liga': {
                'elo': 1620, 'form': 0.05, 'goals_avg': 1.6,
                'xg_proxy': 1.6, 'shooting_efficiency': 0.33
            },
            'Serie A': {
                'elo': 1600, 'form': 0.0, 'goals_avg': 1.5,
                'xg_proxy': 1.5, 'shooting_efficiency': 0.32
            },
            'Bundesliga': {
                'elo': 1630, 'form': 0.08, 'goals_avg': 1.9,
                'xg_proxy': 1.8, 'shooting_efficiency': 0.34
            },
            'Ligue 1': {
                'elo': 1580, 'form': 0.02, 'goals_avg': 1.4,
                'xg_proxy': 1.4, 'shooting_efficiency': 0.30
            },
            # Default за непознати лиги
            'Unknown': {
                'elo': 1500, 'form': 0.0, 'goals_avg': 1.5,
                'xg_proxy': 1.5, 'shooting_efficiency': 0.30
            }
        }
    
    def find_team(self, team_name: str, league: Optional[str] = None) -> Tuple[Optional[str], float, Dict]:
        """
        Търси отбор в базата данни
        
        Args:
            team_name: Име на отбора
            league: Лига (за по-добро търсене)
        
        Returns:
            Tuple (team_key, confidence, metadata)
        """
        # Точно съвпадение
        team_key_lower = team_name.lower()
        if team_key_lower in self.search_index:
            team_key = self.search_index[team_key_lower]
            return team_key, 1.0, {'method': 'exact_match'}
        
        # Fuzzy matching
        best_match = None
        best_score = 0.0
        
        for indexed_name, team_key in self.search_index.items():
            score = SequenceMatcher(None, team_key_lower, indexed_name).ratio()
            if score > best_score and score >= 0.8:  # Минимум 80% съвпадение
                best_score = score
                best_match = team_key
        
        if best_match:
            return best_match, best_score, {'method': 'fuzzy_match'}
        
        # Частично съвпадение (съдържа думи)
        team_words = set(team_name.lower().split())
        for indexed_name, team_key in self.search_index.items():
            indexed_words = set(indexed_name.split())
            common_words = team_words.intersection(indexed_words)
            
            if len(common_words) >= 2 or (len(common_words) == 1 and len(team_words) == 1):
                score = len(common_words) / max(len(team_words), len(indexed_words))
                if score > best_score and score >= 0.6:
                    best_score = score
                    best_match = team_key
        
        if best_match:
            return best_match, best_score, {'method': 'partial_match'}
        
        # Не е намерен
        return None, 0.0, {'method': 'not_found'}
    
    def get_team_data(self, team_name: str, league: Optional[str] = None) -> Tuple[Dict, Dict]:
        """
        Получава данни за отбор с fallback стратегии
        
        Args:
            team_name: Име на отбора
            league: Лига
        
        Returns:
            Tuple (team_data, metadata)
        """
        # Опитваме се да намерим отбора
        team_key, confidence, search_meta = self.find_team(team_name, league)
        
        metadata = {
            'original_name': team_name,
            'found_team': team_key,
            'confidence': confidence,
            'search_method': search_meta['method'],
            'data_source': 'unknown'
        }
        
        if team_key and confidence >= 0.8:
            # Намерен отбор с висока увереност
            team_data = self.team_database[team_key].copy()
            metadata['data_source'] = 'database'
            metadata['real_name'] = self.get_display_name(team_key)
            
        elif team_key and confidence >= 0.6:
            # Намерен отбор с ниска увереност - използваме данните но с предупреждение
            team_data = self.team_database[team_key].copy()
            metadata['data_source'] = 'database_uncertain'
            metadata['real_name'] = self.get_display_name(team_key)
            metadata['warning'] = f"Неточно съвпадение за '{team_name}' -> '{self.get_display_name(team_key)}'"
            
        else:
            # Не е намерен - използваме лигови defaults
            league_key = league if league in self.league_defaults else 'Unknown'
            team_data = self.league_defaults[league_key].copy()
            metadata['data_source'] = 'league_default'
            metadata['league_used'] = league_key
            metadata['warning'] = f"Отборът '{team_name}' не е намерен в базата данни"
        
        return team_data, metadata
    
    def get_display_name(self, team_key: str) -> str:
        """Получава display име за отбор"""
        if team_key in self.team_names_mapping:
            return self.team_names_mapping[team_key]['display_name']
        return team_key
    
    def get_team_id_for_poisson(self, team_name: str, league: Optional[str] = None) -> Tuple[int, Dict]:
        """
        Получава team ID за Poisson модела
        
        Args:
            team_name: Име на отбора
            league: Лига
        
        Returns:
            Tuple (team_id, metadata)
        """
        team_key, confidence, search_meta = self.find_team(team_name, league)
        
        metadata = {
            'method': 'unknown',
            'confidence': confidence,
            'warning': None
        }
        
        if team_key and confidence >= 0.8:
            # Извличаме ID от team_key (напр. "Team_363" -> 363)
            if '_' in team_key:
                team_id = int(team_key.split('_')[1])
                metadata['method'] = 'database_id'
                return team_id, metadata
        
        # Fallback - използваме hash но с предупреждение
        team_id = abs(hash(team_name)) % 10000
        metadata['method'] = 'hash_fallback'
        metadata['warning'] = f"Използва се hash ID за непознат отбор '{team_name}'"
        
        return team_id, metadata


class ImprovedPredictionService:
    """
    Подобрена версия на PredictionService с по-добра обработка на отбори
    """
    
    def __init__(self):
        """Инициализация с подобрена обработка на отбори"""
        self.logger = setup_logging()
        # ... (същата инициализация като оригинала)
        
        # Създаваме TeamResolver
        self.team_resolver = TeamResolver(
            self.elo_ratings, 
            self.team_names
        )
    
    def predict_with_confidence(
        self,
        home_team: str,
        away_team: str,
        league: Optional[str] = None,
        date: Optional[str] = None
    ) -> Dict:
        """
        Прогноза с confidence scoring и предупреждения
        
        Returns:
            Dictionary с predictions и metadata за качеството на данните
        """
        # Получаваме данни за отборите
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
        
        # Правим стандартната прогноза
        prediction = self._make_prediction(home_data, away_data, league, date)
        
        # Добавяме metadata
        prediction['data_quality'] = {
            'overall_confidence': overall_confidence,
            'confidence_level': self._get_confidence_level(overall_confidence),
            'home_team_meta': home_meta,
            'away_team_meta': away_meta,
            'warnings': warnings,
            'recommendation': self._get_recommendation(overall_confidence)
        }
        
        return prediction
    
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
