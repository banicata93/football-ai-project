#!/usr/bin/env python3
"""
Team Name Cleaner - Почиства и нормализира имената на отборите
"""

import json
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

class TeamNameCleaner:
    """Почиства и нормализира имената на отборите"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Ключови думи за женски отбори
        self.women_keywords = [
            'women', 'ladies', 'femmes', 'dames', 'femenino', 'feminino', 
            'w.f.c', 'wfc', 'female', 'womens', 'féminin', 'donne', 'frauen'
        ]
        
        # Ключови думи за младежки отбори
        self.youth_keywords = [
            'u21', 'u23', 'u19', 'u18', 'u17', 'u16', 'under 21', 'under 23',
            'youth', 'junior', 'juvenil', 'jeunes', 'giovanili', 'jugend'
        ]
        
        # Ключови думи за резервни отбори
        self.reserve_keywords = [
            ' ii', ' iii', ' b', ' c', ' reserve', ' reserves', ' segunda', 
            ' segundo', ' filial', ' amateur', ' 2', ' 3'
        ]
        
        # Известни дублирани отбори с предпочитани ID-та
        self.preferred_teams = {
            'Barcelona': '83',  # Основният Барселона
            'Real Madrid': '86',  # Основният Реал Мадрид
            'River Plate': '16',  # Аржентинският Ривър Плейт
            'Athletic Club': '93',  # Атлетик Билбао
            'Valencia': '94',  # Валенсия
            'Real Sociedad': '89',  # Реал Сосиедад
        }
        
    def load_team_mappings(self) -> Tuple[Dict, Dict]:
        """Зарежда team mapping файловете"""
        try:
            with open('models/team_mapping.json', 'r', encoding='utf-8') as f:
                team_mapping = json.load(f)
            
            with open('models/team_names_mapping.json', 'r', encoding='utf-8') as f:
                names_mapping = json.load(f)
                
            return team_mapping, names_mapping
        except Exception as e:
            self.logger.error(f"Грешка при зареждане на mappings: {e}")
            return {}, {}
    
    def is_women_team(self, name: str) -> bool:
        """Проверява дали отборът е женски"""
        name_lower = name.lower()
        return any(keyword in name_lower for keyword in self.women_keywords)
    
    def is_youth_team(self, name: str) -> bool:
        """Проверява дали отборът е младежки"""
        name_lower = name.lower()
        return any(keyword in name_lower for keyword in self.youth_keywords)
    
    def is_reserve_team(self, name: str) -> bool:
        """Проверява дали отборът е резервен"""
        name_lower = name.lower()
        return any(keyword in name_lower for keyword in self.reserve_keywords)
    
    def clean_team_name(self, name: str) -> str:
        """Почиства името на отбора"""
        # Премахва излишни скоби и символи
        cleaned = re.sub(r'\s+', ' ', name.strip())
        
        # Премахва излишни скоби в края
        cleaned = re.sub(r'\s*\([^)]*\)\s*$', '', cleaned)
        
        # Премахва FC, CF в края ако не е част от основното име
        cleaned = re.sub(r'\s+(FC|CF|SC|AC|AS|RC|CD|SD|UD|AD)$', '', cleaned, flags=re.IGNORECASE)
        
        return cleaned.strip()
    
    def get_team_priority_score(self, team_data: Dict) -> int:
        """Изчислява приоритетен score за отбор (по-високо = по-важен)"""
        name = team_data.get('display_name', '')
        
        score = 0
        
        # Наказва женски отбори
        if self.is_women_team(name):
            score -= 100
            
        # Наказва младежки отбори
        if self.is_youth_team(name):
            score -= 50
            
        # Наказва резервни отбори
        if self.is_reserve_team(name):
            score -= 30
            
        # Бонус за по-кратки имена (обикновено са основните)
        score += max(0, 50 - len(name))
        
        # Бонус за отбори без скоби
        if '(' not in name:
            score += 20
            
        return score
    
    def resolve_duplicate_teams(self, names_mapping: Dict) -> Dict[str, str]:
        """Решава дублираните отбори и връща mapping от име към предпочитано ID"""
        name_to_ids = {}
        
        # Групира по имена
        for team_id, data in names_mapping.items():
            clean_name = self.clean_team_name(data['display_name'])
            if clean_name not in name_to_ids:
                name_to_ids[clean_name] = []
            name_to_ids[clean_name].append((team_id, data))
        
        # Решава дубликатите
        resolved_mapping = {}
        
        for clean_name, teams in name_to_ids.items():
            if len(teams) == 1:
                # Няма дубликати
                team_id, data = teams[0]
                resolved_mapping[clean_name] = team_id
            else:
                # Има дубликати - избира най-добрия
                if clean_name in self.preferred_teams:
                    # Използва предварително дефинираното предпочитание
                    preferred_id = self.preferred_teams[clean_name]
                    if any(team_id == preferred_id for team_id, _ in teams):
                        resolved_mapping[clean_name] = preferred_id
                        self.logger.info(f"Използвано предпочитание за {clean_name}: ID {preferred_id}")
                        continue
                
                # Избира базирано на score
                best_team = max(teams, key=lambda x: self.get_team_priority_score(x[1]))
                team_id, data = best_team
                resolved_mapping[clean_name] = team_id
                
                self.logger.info(f"Решен дубликат за '{clean_name}': избрано ID {team_id} ({data['display_name']})")
                
                # Показва отхвърлените
                for other_id, other_data in teams:
                    if other_id != team_id:
                        reason = []
                        if self.is_women_team(other_data['display_name']):
                            reason.append("женски")
                        if self.is_youth_team(other_data['display_name']):
                            reason.append("младежки")
                        if self.is_reserve_team(other_data['display_name']):
                            reason.append("резервен")
                        
                        reason_str = ", ".join(reason) if reason else "по-нисък приоритет"
                        self.logger.info(f"  Отхвърлено ID {other_id} ({other_data['display_name']}) - {reason_str}")
        
        return resolved_mapping
    
    def create_clean_team_mapping(self) -> Dict[str, Dict]:
        """Създава почистен team mapping"""
        team_mapping, names_mapping = self.load_team_mappings()
        
        if not team_mapping or not names_mapping:
            self.logger.error("Не могат да се заредят team mappings")
            return {}
        
        # Решава дубликатите
        resolved_mapping = self.resolve_duplicate_teams(names_mapping)
        
        # Създава новия mapping
        clean_mapping = {}
        
        for team_key, team_data in team_mapping.items():
            team_id = str(team_data['id'])
            
            if team_id in names_mapping:
                real_data = names_mapping[team_id]
                clean_name = self.clean_team_name(real_data['display_name'])
                
                # Проверява дали това ID е предпочитаното за това име
                if resolved_mapping.get(clean_name) == team_id:
                    clean_mapping[team_key] = {
                        'id': team_data['id'],
                        'display_name': clean_name,
                        'short_name': real_data.get('short_name', clean_name[:10]),
                        'abbreviation': real_data.get('abbreviation', clean_name[:3].upper()),
                        'original_name': real_data['display_name'],
                        'is_women': self.is_women_team(real_data['display_name']),
                        'is_youth': self.is_youth_team(real_data['display_name']),
                        'is_reserve': self.is_reserve_team(real_data['display_name'])
                    }
                else:
                    # Това ID е дубликат - маркира го
                    clean_mapping[team_key] = {
                        'id': team_data['id'],
                        'display_name': f"[DUP] {real_data['display_name']}",
                        'short_name': team_data.get('short_name', f"T{team_data['id']}"),
                        'abbreviation': f"D{team_data['id']}",
                        'original_name': real_data['display_name'],
                        'is_duplicate': True,
                        'preferred_id': resolved_mapping.get(clean_name),
                        'is_women': self.is_women_team(real_data['display_name']),
                        'is_youth': self.is_youth_team(real_data['display_name']),
                        'is_reserve': self.is_reserve_team(real_data['display_name'])
                    }
            else:
                # Няма реално име - запазва generic
                clean_mapping[team_key] = team_data.copy()
                clean_mapping[team_key]['is_generic'] = True
        
        return clean_mapping
    
    def save_clean_mapping(self, clean_mapping: Dict, output_path: str = 'models/team_mapping_clean.json'):
        """Запазва почистения mapping"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(clean_mapping, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Почистен team mapping запазен в {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Грешка при запазване: {e}")
            return False
    
    def generate_report(self, clean_mapping: Dict) -> Dict:
        """Генерира отчет за почистването"""
        report = {
            'total_teams': len(clean_mapping),
            'generic_teams': 0,
            'duplicate_teams': 0,
            'women_teams': 0,
            'youth_teams': 0,
            'reserve_teams': 0,
            'clean_teams': 0
        }
        
        for team_data in clean_mapping.values():
            if team_data.get('is_generic'):
                report['generic_teams'] += 1
            elif team_data.get('is_duplicate'):
                report['duplicate_teams'] += 1
            elif team_data.get('is_women'):
                report['women_teams'] += 1
            elif team_data.get('is_youth'):
                report['youth_teams'] += 1
            elif team_data.get('is_reserve'):
                report['reserve_teams'] += 1
            else:
                report['clean_teams'] += 1
        
        return report

def main():
    """Основна функция за почистване на team names"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    cleaner = TeamNameCleaner()
    
    print("🧹 СТАРТИРАНЕ НА TEAM NAME CLEANER")
    print("=" * 50)
    
    # Създава почистен mapping
    clean_mapping = cleaner.create_clean_team_mapping()
    
    if not clean_mapping:
        print("❌ Грешка при създаване на clean mapping")
        return
    
    # Запазва резултата
    success = cleaner.save_clean_mapping(clean_mapping)
    
    if success:
        # Генерира отчет
        report = cleaner.generate_report(clean_mapping)
        
        print("\n📊 ОТЧЕТ ЗА ПОЧИСТВАНЕТО:")
        print(f"  Общо отбори: {report['total_teams']}")
        print(f"  Почистени отбори: {report['clean_teams']}")
        print(f"  Generic отбори: {report['generic_teams']}")
        print(f"  Дублирани отбори: {report['duplicate_teams']}")
        print(f"  Женски отбори: {report['women_teams']}")
        print(f"  Младежки отбори: {report['youth_teams']}")
        print(f"  Резервни отбори: {report['reserve_teams']}")
        
        print(f"\n✅ Team mapping почистен успешно!")
        print("📁 Резултат запазен в: models/team_mapping_clean.json")
    else:
        print("❌ Грешка при запазване на резултата")

if __name__ == "__main__":
    main()
