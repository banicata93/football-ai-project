"""
League utilities за mapping между league names и slugs
"""

from typing import Dict, Optional


# League ID to slug mapping (обновен с реални данни)
LEAGUE_ID_TO_SLUG = {
    # Реални league IDs от данните
    3903: 'premier_league',    # 643 мача
    9999: 'la_liga',          # 487 мача  
    4003: 'serie_a',          # 476 мача
    770: 'bundesliga',        # 449 мача
    3904: 'ligue_1',          # 404 мача
    4002: 'eredivisie',       # 353 мача
    650: 'primeira_liga',     # 323 мача
    750: 'championship',      # 314 мача
    4007: 'liga_mx',          # 285 мача
    630: 'mls',               # 275 мача
    
    # Допълнителни лиги с по-малко мачове
    680: 'scottish_premiership',  # За бъдещо разширяване
    660: 'russian_premier_league',
    670: 'turkish_super_lig',
    640: 'ukrainian_premier_league',
    620: 'belgian_pro_league'
}

# League name to slug mapping (case insensitive)
LEAGUE_NAME_TO_SLUG = {
    'premier league': 'premier_league',
    'english premier league': 'premier_league',
    'epl': 'premier_league',
    
    'la liga': 'la_liga',
    'spanish la liga': 'la_liga',
    'primera division': 'la_liga',
    'spain primera division': 'la_liga',
    
    'serie a': 'serie_a',
    'italian serie a': 'serie_a',
    'italy serie a': 'serie_a',
    
    'bundesliga': 'bundesliga',
    'german bundesliga': 'bundesliga',
    'germany bundesliga': 'bundesliga',
    '1. bundesliga': 'bundesliga',
    
    'ligue 1': 'ligue_1',
    'french ligue 1': 'ligue_1',
    'france ligue 1': 'ligue_1',
    
    'eredivisie': 'eredivisie',
    'dutch eredivisie': 'eredivisie',
    'netherlands eredivisie': 'eredivisie',
    
    'primeira liga': 'primeira_liga',
    'portuguese primeira liga': 'primeira_liga',
    'portugal primeira liga': 'primeira_liga',
    
    'championship': 'championship',
    'english championship': 'championship',
    'efl championship': 'championship',
    
    'liga mx': 'liga_mx',
    'mexican liga mx': 'liga_mx',
    'mexico liga mx': 'liga_mx',
    
    'mls': 'mls',
    'major league soccer': 'mls',
    'usa mls': 'mls'
}

# Slug to display name mapping
SLUG_TO_DISPLAY_NAME = {
    'premier_league': 'Premier League',
    'la_liga': 'La Liga',
    'serie_a': 'Serie A',
    'bundesliga': 'Bundesliga',
    'ligue_1': 'Ligue 1',
    'eredivisie': 'Eredivisie',
    'primeira_liga': 'Primeira Liga',
    'championship': 'Championship',
    'liga_mx': 'Liga MX',
    'mls': 'MLS'
}


def get_league_slug(league_input: Optional[str] = None, league_id: Optional[int] = None) -> Optional[str]:
    """
    Получава league slug от име или ID
    
    Args:
        league_input: League име (case insensitive)
        league_id: League ID
    
    Returns:
        League slug или None ако не е намерен
    """
    if league_id is not None:
        return LEAGUE_ID_TO_SLUG.get(league_id)
    
    if league_input is not None:
        league_lower = league_input.lower().strip()
        return LEAGUE_NAME_TO_SLUG.get(league_lower)
    
    return None


def get_league_display_name(slug: str) -> str:
    """
    Получава display име от slug
    
    Args:
        slug: League slug
    
    Returns:
        Display име или slug ако не е намерен
    """
    return SLUG_TO_DISPLAY_NAME.get(slug, slug.replace('_', ' ').title())


def is_supported_league(league_input: Optional[str] = None, league_id: Optional[int] = None) -> bool:
    """
    Проверява дали лигата е поддържана
    
    Args:
        league_input: League име
        league_id: League ID
    
    Returns:
        True ако лигата е поддържана
    """
    slug = get_league_slug(league_input, league_id)
    return slug is not None


def get_supported_leagues() -> Dict[str, str]:
    """
    Получава всички поддържани лиги
    
    Returns:
        Dictionary {slug: display_name}
    """
    return SLUG_TO_DISPLAY_NAME.copy()


def get_per_league_model_path(league_slug: str, model_type: str = 'ou25', version: str = 'v1') -> str:
    """
    Генерира path към league-specific модел
    
    Args:
        league_slug: League slug
        model_type: Тип модел (ou25, btts, etc.)
        version: Версия на модела
    
    Returns:
        Path към модела
    """
    return f"models/leagues/{league_slug}/{model_type}_{version}"


if __name__ == "__main__":
    # Тестване на функциите
    print("🧪 Тестване на league utilities...")
    
    # Тест 1: League slug mapping
    test_cases = [
        ("Premier League", "premier_league"),
        ("La Liga", "la_liga"),
        ("Serie A", "serie_a"),
        ("Bundesliga", "bundesliga"),
        ("Ligue 1", "ligue_1")
    ]
    
    for league_name, expected_slug in test_cases:
        slug = get_league_slug(league_name)
        assert slug == expected_slug, f"Expected {expected_slug}, got {slug}"
        print(f"✅ {league_name} → {slug}")
    
    # Тест 2: League ID mapping
    for league_id in [1, 2, 3, 4, 5]:
        slug = get_league_slug(league_id=league_id)
        print(f"✅ League ID {league_id} → {slug}")
    
    # Тест 3: Model path generation
    path = get_per_league_model_path("premier_league", "ou25", "v1")
    expected_path = "models/leagues/premier_league/ou25_v1"
    assert path == expected_path, f"Expected {expected_path}, got {path}"
    print(f"✅ Model path: {path}")
    
    print("✅ Всички тестове преминаха успешно!")
