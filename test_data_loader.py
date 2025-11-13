"""
Тестов скрипт за проверка на ESPN Data Loader
"""

import sys
import os

# Добавяне на проектната директория към path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_loader import ESPNDataLoader
from core.utils import setup_logging


def main():
    """Главна функция за тестване"""
    
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("ТЕСТВАНЕ НА ESPN DATA LOADER")
    logger.info("=" * 60)
    
    # Инициализация на loader
    loader = ESPNDataLoader()
    
    # Тест 1: Зареждане на fixtures
    print("\n[1] Тестване на fixtures...")
    fixtures = loader.load_fixtures()
    print(f"   ✓ Заредени {len(fixtures)} мача")
    print(f"   ✓ Колони: {list(fixtures.columns[:10])}...")
    print(f"   ✓ Период: {fixtures['date'].min()} до {fixtures['date'].max()}")
    print(f"   ✓ Резултати - 1: {(fixtures['result']=='1').sum()}, "
          f"X: {(fixtures['result']=='X').sum()}, "
          f"2: {(fixtures['result']=='2').sum()}")
    
    # Тест 2: Зареждане на teams
    print("\n[2] Тестване на teams...")
    teams = loader.load_teams()
    print(f"   ✓ Заредени {len(teams)} отбора")
    print(f"   ✓ Примерни отбори: {teams['team_name'].head(5).tolist()}")
    
    # Тест 3: Зареждане на team stats
    print("\n[3] Тестване на team stats...")
    team_stats = loader.load_team_stats()
    print(f"   ✓ Заредени {len(team_stats)} статистики")
    print(f"   ✓ Средно владеене: {team_stats['possession'].mean():.1f}%")
    print(f"   ✓ Средни удари: {team_stats['shots'].mean():.1f}")
    print(f"   ✓ Средни ъглови: {team_stats['corners'].mean():.1f}")
    
    # Тест 4: Зареждане на players
    print("\n[4] Тестване на players...")
    players = loader.load_players()
    print(f"   ✓ Заредени {len(players)} играчи")
    print(f"   ✓ Средна възраст: {players['age'].mean():.1f} години")
    
    # Тест 5: Зареждане на leagues
    print("\n[5] Тестване на leagues...")
    leagues = loader.load_leagues()
    print(f"   ✓ Заредени {len(leagues)} лиги/сезони")
    unique_leagues = leagues['league_name'].nunique()
    print(f"   ✓ Уникални лиги: {unique_leagues}")
    
    # Тест 6: Зареждане на standings
    print("\n[6] Тестване на standings...")
    standings = loader.load_standings()
    print(f"   ✓ Заредени {len(standings)} записа в класирания")
    
    # Тест 7: Зареждане на venues
    print("\n[7] Тестване на venues...")
    venues = loader.load_venues()
    print(f"   ✓ Заредени {len(venues)} стадиона")
    
    # Тест 8: Merge fixtures with stats
    print("\n[8] Тестване на merge fixtures + stats...")
    merged = loader.merge_fixtures_with_stats()
    print(f"   ✓ Обединени {len(merged)} мача")
    print(f"   ✓ Колони: {len(merged.columns)}")
    
    # Тест 9: Team match history
    print("\n[9] Тестване на team match history...")
    # Вземаме първия отбор
    first_team_id = teams['team_id'].iloc[0]
    first_team_name = teams['team_name'].iloc[0]
    history = loader.get_team_match_history(first_team_id, n_matches=10)
    print(f"   ✓ История на {first_team_name}: {len(history)} мача")
    
    # Тест 10: Cache info
    print("\n[10] Кеш информация...")
    cache_info = loader.get_cache_info()
    for key, size in cache_info.items():
        print(f"   ✓ {key}: {size} записа")
    
    # Обобщение
    print("\n" + "=" * 60)
    print("РЕЗУЛТАТ: Всички тестове преминаха успешно! ✓")
    print("=" * 60)
    
    # Примерни данни
    print("\n[ПРИМЕРНИ ДАННИ]")
    print("\nПоследни 5 мача:")
    print(fixtures[['date', 'home_team_id', 'away_team_id', 'home_score', 
                    'away_score', 'result', 'over_25', 'btts']].tail())
    
    print("\nСтатистики на последен мач:")
    last_match_id = fixtures['match_id'].iloc[-1]
    last_match_stats = team_stats[team_stats['match_id'] == last_match_id]
    if not last_match_stats.empty:
        print(last_match_stats[['team_id', 'possession', 'shots', 
                                'shots_on_target', 'corners']].to_string())


if __name__ == "__main__":
    main()
