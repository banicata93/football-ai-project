#!/usr/bin/env python3
"""
Team Name Resolver - Резолва и нормализира имената на отборите за API
"""

import json
import re
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher
import logging

class TeamNameResolver:
    """Резолва имената на отборите използвайки почистения mapping"""
    
    def __init__(self, clean_mapping_path: str = 'models/team_mapping_clean.json'):
        self.logger = logging.getLogger(__name__)
        self.clean_mapping = self._load_clean_mapping(clean_mapping_path)
        self.name_to_key = self._build_name_index()
        
    def _load_clean_mapping(self, path: str) -> Dict:
        """Зарежда почистения team mapping"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Грешка при зареждане на clean mapping: {e}")
            return {}
    
    def _build_name_index(self) -> Dict[str, str]:
        """Създава индекс от име към team key с приоритизиране на основните отбори"""
        name_index = {}
        
        # Първо добавя основните отбори (не младежки, не резервни, не женски)
        for team_key, team_data in self.clean_mapping.items():
            display_name = team_data.get('display_name', '')
            
            # Пропуска дублираните и проблемните отбори
            if (team_data.get('is_duplicate') or 
                team_data.get('is_women') or 
                display_name.startswith('[DUP]')):
                continue
            
            # Пропуска младежки и резервни отбори в първия проход
            if (team_data.get('is_youth') or 
                team_data.get('is_reserve')):
                continue
                
            # Добавя основното име
            name_index[display_name.lower()] = team_key
            
            # Добавя алтернативни имена
            original_name = team_data.get('original_name', '')
            if original_name and original_name.lower() != display_name.lower():
                name_index[original_name.lower()] = team_key
            
            # Добавя кратко име
            short_name = team_data.get('short_name', '')
            if short_name and short_name.strip():
                name_index[short_name.lower().strip()] = team_key
            
            # Добавя абревиатура
            abbreviation = team_data.get('abbreviation', '')
            if abbreviation and len(abbreviation) >= 2:
                name_index[abbreviation.lower()] = team_key
        
        # Втори проход - добавя младежки и резервни отбори само ако няма конфликт
        for team_key, team_data in self.clean_mapping.items():
            display_name = team_data.get('display_name', '')
            
            # Пропуска дублираните и проблемните отбори
            if (team_data.get('is_duplicate') or 
                team_data.get('is_women') or 
                display_name.startswith('[DUP]')):
                continue
            
            # Само младежки и резервни отбори в този проход
            if not (team_data.get('is_youth') or team_data.get('is_reserve')):
                continue
                
            # Добавя само ако няма конфликт с основен отбор
            if display_name.lower() not in name_index:
                name_index[display_name.lower()] = team_key
            
            original_name = team_data.get('original_name', '')
            if original_name and original_name.lower() not in name_index:
                name_index[original_name.lower()] = team_key
        
        return name_index
    
    def _normalize_name(self, name: str) -> str:
        """Нормализира името за търсене"""
        if not name:
            return ""
            
        # Премахва излишни символи и нормализира
        normalized = re.sub(r'[^\w\s]', ' ', name.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Премахва общи суфикси
        suffixes = ['fc', 'cf', 'sc', 'ac', 'as', 'rc', 'cd', 'sd', 'ud', 'ad', 'club', 'united', 'city']
        words = normalized.split()
        if words and words[-1] in suffixes:
            words = words[:-1]
            normalized = ' '.join(words)
        
        return normalized
    
    def _calculate_similarity(self, name1: str, name2: str) -> float:
        """Изчислява сходството между две имена"""
        return SequenceMatcher(None, name1.lower(), name2.lower()).ratio()
    
    def find_team_key(self, team_name: str, threshold: float = 0.85) -> Optional[str]:
        """Намира team key за дадено име"""
        if not team_name:
            return None
        
        # Проверява дали е Team_XXXX формат
        if team_name.startswith('Team_'):
            return team_name if team_name in self.clean_mapping else None
        
        normalized_input = self._normalize_name(team_name)
        
        # 1. Точно съвпадение (case insensitive)
        for indexed_name, team_key in self.name_to_key.items():
            if normalized_input == indexed_name:
                return team_key
        
        # 2. Точно съвпадение с оригиналното име
        original_lower = team_name.lower().strip()
        if original_lower in self.name_to_key:
            return self.name_to_key[original_lower]
        
        # 3. Частично съвпадение - само ако е достатъчно специфично
        if len(normalized_input) >= 4:  # Минимум 4 символа за частично търсене
            for indexed_name, team_key in self.name_to_key.items():
                # Проверява дали цялото търсено име е в индексираното
                if normalized_input in indexed_name and len(normalized_input) >= len(indexed_name) * 0.6:
                    return team_key
                # Или обратно - индексираното име е в търсеното
                if indexed_name in normalized_input and len(indexed_name) >= len(normalized_input) * 0.6:
                    return team_key
        
        # 4. Fuzzy matching само за високо сходство
        best_match = None
        best_score = 0
        
        for indexed_name, team_key in self.name_to_key.items():
            score = self._calculate_similarity(normalized_input, indexed_name)
            if score > best_score and score >= threshold:
                best_score = score
                best_match = team_key
        
        if best_match:
            team_data = self.clean_mapping.get(best_match, {})
            self.logger.info(f"Fuzzy match: '{team_name}' -> '{team_data.get('display_name')}' (score: {best_score:.2f})")
        
        return best_match
    
    def get_team_display_name(self, team_name: str) -> str:
        """Връща почистеното display име за отбор"""
        team_key = self.find_team_key(team_name)
        
        if team_key and team_key in self.clean_mapping:
            team_data = self.clean_mapping[team_key]
            
            # Проверява за проблемни отбори
            if team_data.get('is_duplicate'):
                preferred_id = team_data.get('preferred_id')
                if preferred_id:
                    # Търси предпочитания отбор
                    for key, data in self.clean_mapping.items():
                        if str(data.get('id')) == preferred_id:
                            return data.get('display_name', team_name)
                
                # Ако няма предпочитан, връща оригиналното име без [DUP]
                original = team_data.get('original_name', team_name)
                return original.replace('[DUP] ', '')
            
            return team_data.get('display_name', team_name)
        
        # Ако не е намерен, връща оригиналното име
        return team_name
    
    def get_similar_teams(self, team_name: str, limit: int = 5) -> List[Tuple[str, float]]:
        """Връща списък с подобни отбори"""
        if not team_name:
            return []
        
        normalized_input = self._normalize_name(team_name)
        similarities = []
        
        for indexed_name, team_key in self.name_to_key.items():
            team_data = self.clean_mapping.get(team_key, {})
            
            # Пропуска проблемните отбори
            if (team_data.get('is_duplicate') or 
                team_data.get('is_women') or 
                team_data.get('display_name', '').startswith('[DUP]')):
                continue
            
            score = self._calculate_similarity(normalized_input, indexed_name)
            if score > 0.3:  # Минимален threshold
                display_name = team_data.get('display_name', indexed_name)
                similarities.append((display_name, score))
        
        # Сортира по score и връща топ резултатите
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]
    
    def is_valid_team(self, team_name: str) -> bool:
        """Проверява дали отборът е валиден (не е женски, младежки, резервен)"""
        team_key = self.find_team_key(team_name)
        
        if team_key and team_key in self.clean_mapping:
            team_data = self.clean_mapping[team_key]
            
            # Проверява за проблемни категории
            if (team_data.get('is_women') or 
                team_data.get('is_youth') or 
                team_data.get('is_reserve') or
                team_data.get('is_duplicate')):
                return False
            
            return True
        
        return False
    
    def get_team_info(self, team_name: str) -> Dict:
        """Връща пълна информация за отбор"""
        team_key = self.find_team_key(team_name)
        
        if team_key and team_key in self.clean_mapping:
            team_data = self.clean_mapping[team_key].copy()
            team_data['team_key'] = team_key
            team_data['resolved_name'] = self.get_team_display_name(team_name)
            team_data['is_valid'] = self.is_valid_team(team_name)
            return team_data
        
        return {
            'team_key': None,
            'display_name': team_name,
            'resolved_name': team_name,
            'is_valid': False,
            'is_unknown': True
        }

def main():
    """Тестова функция"""
    resolver = TeamNameResolver()
    
    print("🔍 ТЕСТВАНЕ НА TEAM NAME RESOLVER")
    print("=" * 50)
    
    test_names = [
        "Manchester United",
        "Man Utd", 
        "Barcelona",
        "Barca",
        "Real Madrid",
        "Bayern Munich",
        "Liverpool",
        "Chelsea FC",
        "Arsenal",
        "Juventus",
        "AC Milan",
        "Inter Milan",
        "Team_3841",  # Проблемно име
        "Team_11420"  # Проблемно име
    ]
    
    for name in test_names:
        resolved = resolver.get_team_display_name(name)
        info = resolver.get_team_info(name)
        
        print(f"\n'{name}':")
        print(f"  -> Resolved: {resolved}")
        print(f"  -> Valid: {info['is_valid']}")
        
        if not info['is_valid'] and not info.get('is_unknown'):
            flags = []
            if info.get('is_women'): flags.append('женски')
            if info.get('is_youth'): flags.append('младежки') 
            if info.get('is_reserve'): flags.append('резервен')
            if info.get('is_duplicate'): flags.append('дубликат')
            
            if flags:
                print(f"  -> Причина: {', '.join(flags)}")
        
        if info.get('is_unknown'):
            similar = resolver.get_similar_teams(name, 3)
            if similar:
                print(f"  -> Подобни: {[f'{n} ({s:.2f})' for n, s in similar]}")

if __name__ == "__main__":
    main()
