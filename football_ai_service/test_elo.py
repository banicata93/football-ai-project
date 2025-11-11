"""
Тестов скрипт за Elo Calculator
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.elo_calculator import EloCalculator
from core.data_loader import ESPNDataLoader
from core.utils import setup_logging


def test_basic_elo():
    """Тест на основна Elo функционалност"""
    print("\n" + "=" * 60)
    print("ТЕСТ 1: Основна Elo функционалност")
    print("=" * 60)
    
    elo = EloCalculator(k_factor=20, initial_rating=1500, home_advantage=100)
    
    # Симулация на мачове
    matches = [
        (1, 2, 3, 1, "Отбор A vs Отбор B: 3-1"),
        (2, 3, 2, 2, "Отбор B vs Отбор C: 2-2"),
        (3, 1, 0, 2, "Отбор C vs Отбор A: 0-2"),
        (1, 2, 1, 0, "Отбор A vs Отбор B: 1-0"),
    ]
    
    for home_id, away_id, home_score, away_score, desc in matches:
        home_before = elo.get_rating(home_id)
        away_before = elo.get_rating(away_id)
        
        home_after, away_after = elo.update_ratings(home_id, away_id, home_score, away_score)
        
        print(f"\n{desc}")
        print(f"  Отбор {home_id}: {home_before:.1f} → {home_after:.1f} ({home_after-home_before:+.1f})")
        print(f"  Отбор {away_id}: {away_before:.1f} → {away_after:.1f} ({away_after-away_before:+.1f})")
    
    print("\n✓ Тест 1 завършен успешно!")


def test_elo_with_real_data():
    """Тест с реални ESPN данни"""
    print("\n" + "=" * 60)
    print("ТЕСТ 2: Elo с реални ESPN данни")
    print("=" * 60)
    
    logger = setup_logging()
    
    # Зареждане на данни
    loader = ESPNDataLoader()
    fixtures = loader.load_fixtures()
    
    # Вземаме само първите 1000 мача за бърз тест
    fixtures_sample = fixtures.head(1000).copy()
    
    print(f"\nЗаредени {len(fixtures_sample)} мача за тестване")
    
    # Изчисляване на Elo
    elo = EloCalculator(k_factor=20, initial_rating=1500, home_advantage=100)
    fixtures_with_elo = elo.calculate_elo_for_dataset(fixtures_sample)
    
    print(f"\n✓ Elo изчислен за {len(fixtures_with_elo)} мача")
    print(f"✓ Отбори с рейтинг: {len(elo.ratings)}")
    
    # Статистики
    print(f"\nElo статистики:")
    print(f"  Min: {fixtures_with_elo['home_elo_before'].min():.1f}")
    print(f"  Max: {fixtures_with_elo['home_elo_before'].max():.1f}")
    print(f"  Mean: {fixtures_with_elo['home_elo_before'].mean():.1f}")
    print(f"  Std: {fixtures_with_elo['home_elo_before'].std():.1f}")
    
    # Топ 5 отбори
    print(f"\nТоп 5 отбори по Elo:")
    teams = loader.load_teams()
    
    for i, (team_id, rating) in enumerate(elo.get_top_teams(5), 1):
        team_name = teams[teams['team_id'] == team_id]['team_name'].values
        team_name = team_name[0] if len(team_name) > 0 else f"Team {team_id}"
        print(f"  {i}. {team_name:30s} - {rating:.1f}")
    
    # Примерни мачове с Elo
    print(f"\nПримерни мачове с Elo predictions:")
    sample = fixtures_with_elo[['date', 'home_team_id', 'away_team_id', 
                                  'home_score', 'away_score', 
                                  'home_elo_before', 'away_elo_before', 
                                  'elo_diff_before', 'home_win_prob']].tail(5)
    print(sample.to_string())
    
    print("\n✓ Тест 2 завършен успешно!")


def test_elo_prediction_accuracy():
    """Тест на точността на Elo predictions"""
    print("\n" + "=" * 60)
    print("ТЕСТ 3: Точност на Elo predictions")
    print("=" * 60)
    
    loader = ESPNDataLoader()
    fixtures = loader.load_fixtures().head(1000)
    
    elo = EloCalculator(k_factor=20, initial_rating=1500, home_advantage=100)
    fixtures_with_elo = elo.calculate_elo_for_dataset(fixtures)
    
    # Изчисляване на точност
    # Prediction: home win ако home_win_prob > 0.5
    fixtures_with_elo['predicted_home_win'] = (fixtures_with_elo['home_win_prob'] > 0.5).astype(int)
    fixtures_with_elo['actual_home_win'] = (fixtures_with_elo['result'] == '1').astype(int)
    
    correct = (fixtures_with_elo['predicted_home_win'] == fixtures_with_elo['actual_home_win']).sum()
    total = len(fixtures_with_elo)
    accuracy = correct / total * 100
    
    print(f"\nТочност на Elo predictions:")
    print(f"  Правилни: {correct}/{total}")
    print(f"  Accuracy: {accuracy:.2f}%")
    
    # Разпределение на резултати
    print(f"\nРазпределение на резултати:")
    print(f"  Home wins: {(fixtures_with_elo['result'] == '1').sum()}")
    print(f"  Draws: {(fixtures_with_elo['result'] == 'X').sum()}")
    print(f"  Away wins: {(fixtures_with_elo['result'] == '2').sum()}")
    
    print("\n✓ Тест 3 завършен успешно!")


def main():
    """Главна функция"""
    print("\n" + "=" * 60)
    print("ELO CALCULATOR - ТЕСТВАНЕ")
    print("=" * 60)
    
    # Тест 1: Основна функционалност
    test_basic_elo()
    
    # Тест 2: Реални данни
    test_elo_with_real_data()
    
    # Тест 3: Точност
    test_elo_prediction_accuracy()
    
    print("\n" + "=" * 60)
    print("ВСИЧКИ ТЕСТОВЕ ЗАВЪРШЕНИ УСПЕШНО! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
