"""
League Name Formatter and Organizer

Cleans up and organizes league names for better UI display
"""

from typing import List, Dict, Tuple


# League icons mapping
LEAGUE_ICONS = {
    # Top European Leagues
    "premier_league": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "la_liga": "🇪🇸",
    "serie_a": "🇮🇹",
    "bundesliga": "🇩🇪",
    "ligue_1": "🇫🇷",
    "eredivisie": "🇳🇱",
    "primeira_liga": "🇵🇹",
    "championship": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    
    # International
    "world_cup": "🌍",
    "uefa_nations_league": "🇪🇺",
    "uefa_champions_league": "⭐",
    "uefa_europa_league": "🌟",
    "copa_libertadores": "🏆",
    "copa_america": "🏆",
    "euro": "🇪🇺",
    
    # Other leagues
    "mls": "🇺🇸",
    "liga_mx": "🇲🇽",
    "j_league": "🇯🇵",
    "k_league": "🇰🇷",
    "a_league": "🇦🇺",
    "saudi_pro_league": "🇸🇦",
}


# Priority order for leagues (higher = more important)
LEAGUE_PRIORITY = {
    # Top 5 European leagues
    "English Premier League": 100,
    "Spanish La Liga": 99,
    "Italian Serie A": 98,
    "German Bundesliga": 97,
    "French Ligue 1": 96,
    
    # Other major European leagues
    "Dutch Eredivisie": 90,
    "Portuguese Primeira Liga": 89,
    "English Championship": 88,
    "Scottish Premiership": 85,
    "Belgian Pro League": 84,
    "Turkish Süper Lig": 83,
    "Russian Premier League": 82,
    
    # International tournaments
    "FIFA World Cup": 200,
    "UEFA Champions League": 195,
    "UEFA Europa League": 190,
    "UEFA Nations League": 185,
    "Copa Libertadores": 180,
    "Copa America": 175,
    "UEFA European Championship": 170,
    
    # Americas
    "MLS": 80,
    "Liga MX": 79,
    "Argentine Liga": 78,
    "Brazilian Serie A": 77,
    
    # Asia
    "Japanese J.League": 70,
    "Korean K League": 69,
    "Saudi Pro League": 68,
    
    # Default
    "default": 50
}


# Clean name mappings
LEAGUE_NAME_MAPPINGS = {
    # Premier League variations
    "English Premier League": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League",
    "Premier League": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League",
    
    # La Liga
    "Spanish La Liga": "🇪🇸 La Liga",
    "LaLiga": "🇪🇸 La Liga",
    "La Liga": "🇪🇸 La Liga",
    
    # Serie A
    "Italian Serie A": "🇮🇹 Serie A",
    "Serie A": "🇮🇹 Serie A",
    
    # Bundesliga
    "German Bundesliga": "🇩🇪 Bundesliga",
    "Bundesliga": "🇩🇪 Bundesliga",
    
    # Ligue 1
    "French Ligue 1": "🇫🇷 Ligue 1",
    "Ligue 1": "🇫🇷 Ligue 1",
    
    # Other European
    "Dutch Eredivisie": "🇳🇱 Eredivisie",
    "Eredivisie": "🇳🇱 Eredivisie",
    "Portuguese Primeira Liga": "🇵🇹 Primeira Liga",
    "Primeira Liga": "🇵🇹 Primeira Liga",
    "English Championship": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Championship",
    "Championship": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Championship",
    
    # International
    "FIFA World Cup": "🌍 World Cup",
    "FIFA World Cup Qualifying - UEFA": "🇪🇺 WC Qualifying (UEFA)",
    "FIFA World Cup Qualifying - Concacaf": "🌎 WC Qualifying (CONCACAF)",
    "FIFA World Cup Qualifying - Conmebol": "🌎 WC Qualifying (CONMEBOL)",
    "FIFA World Cup Qualifying - CAF": "🌍 WC Qualifying (Africa)",
    "FIFA World Cup Qualifying - AFC": "🌏 WC Qualifying (Asia)",
    
    "UEFA Champions League": "⭐ Champions League",
    "UEFA Europa League": "🌟 Europa League",
    "UEFA Nations League": "🇪🇺 Nations League",
    "UEFA European Championship": "🇪🇺 Euro",
    
    "CONMEBOL Libertadores": "🏆 Copa Libertadores",
    "CONMEBOL Sudamericana": "🏆 Copa Sudamericana",
    "Copa America": "🏆 Copa America",
    
    # Americas
    "MLS": "🇺🇸 MLS",
    "Liga MX": "🇲🇽 Liga MX",
    "Argentine Liga Profesional de Fútbol": "🇦🇷 Argentine Liga",
    "Brazilian Serie A": "🇧🇷 Brasileirão",
    
    # Asia
    "Japanese J.League": "🇯🇵 J.League",
    "Korean K League": "🇰🇷 K League",
    "Saudi Pro League": "🇸🇦 Saudi Pro League",
    
    # Friendlies
    "International Friendly": "🤝 International Friendly",
    "Women's International Friendly": "🤝 Women's Friendly",
}


def clean_league_name(original_name: str, slug: str = "") -> str:
    """
    Clean and format league name for display
    
    Args:
        original_name: Original league name from API
        slug: League slug
    
    Returns:
        Cleaned and formatted name with icon
    """
    # Check direct mapping first
    if original_name in LEAGUE_NAME_MAPPINGS:
        return LEAGUE_NAME_MAPPINGS[original_name]
    
    # Try to find partial match
    for key, value in LEAGUE_NAME_MAPPINGS.items():
        if key.lower() in original_name.lower():
            return value
    
    # Clean up common patterns
    name = original_name
    
    # Remove year patterns
    import re
    name = re.sub(r'\b20\d{2}\b', '', name)
    name = re.sub(r'\b\d{4}-\d{2}\b', '', name)
    
    # Clean up extra spaces
    name = ' '.join(name.split())
    
    # Add icon based on keywords
    if "premier league" in name.lower():
        return f"🏴󠁧󠁢󠁥󠁮󠁧󠁿 {name}"
    elif "la liga" in name.lower():
        return f"🇪🇸 {name}"
    elif "serie a" in name.lower():
        return f"🇮🇹 {name}"
    elif "bundesliga" in name.lower():
        return f"🇩🇪 {name}"
    elif "ligue 1" in name.lower():
        return f"🇫🇷 {name}"
    elif "champions league" in name.lower():
        return f"⭐ {name}"
    elif "europa league" in name.lower():
        return f"🌟 {name}"
    elif "world cup" in name.lower():
        return f"🌍 {name}"
    elif "nations league" in name.lower():
        return f"🇪🇺 {name}"
    elif "libertadores" in name.lower():
        return f"🏆 {name}"
    elif "mls" in name.lower():
        return f"🇺🇸 {name}"
    
    # Default: just return cleaned name
    return f"⚽ {name}"


def get_league_priority(league_name: str) -> int:
    """
    Get priority score for league (higher = more important)
    
    Args:
        league_name: League name
    
    Returns:
        Priority score
    """
    # Check exact match
    if league_name in LEAGUE_PRIORITY:
        return LEAGUE_PRIORITY[league_name]
    
    # Check partial match
    for key, value in LEAGUE_PRIORITY.items():
        if key.lower() in league_name.lower():
            return value
    
    return LEAGUE_PRIORITY["default"]


def group_leagues_by_category(leagues: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group leagues by category for better organization
    
    Args:
        leagues: List of league dictionaries
    
    Returns:
        Dictionary with categories as keys
    """
    categories = {
        "🌟 Top European Leagues": [],
        "🌍 International Tournaments": [],
        "🇪🇺 European Competitions": [],
        "🌎 Americas": [],
        "🌏 Asia & Others": [],
        "⚽ Other Leagues": []
    }
    
    for league in leagues:
        original_name = league.get("original_name", league.get("name", ""))
        
        # Top European leagues
        if any(x in original_name for x in ["Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1", "Eredivisie", "Primeira Liga", "Championship"]):
            categories["🌟 Top European Leagues"].append(league)
        
        # International tournaments
        elif any(x in original_name for x in ["World Cup", "Copa America", "Euro"]):
            categories["🌍 International Tournaments"].append(league)
        
        # European competitions
        elif any(x in original_name for x in ["Champions League", "Europa League", "Nations League", "Conference League"]):
            categories["🇪🇺 European Competitions"].append(league)
        
        # Americas
        elif any(x in original_name for x in ["MLS", "Liga MX", "Argentine", "Brazilian", "Libertadores", "Sudamericana"]):
            categories["🌎 Americas"].append(league)
        
        # Asia & Others
        elif any(x in original_name for x in ["J.League", "K League", "Saudi", "Australian", "Chinese"]):
            categories["🌏 Asia & Others"].append(league)
        
        # Other
        else:
            categories["⚽ Other Leagues"].append(league)
    
    # Remove empty categories
    return {k: v for k, v in categories.items() if v}


def format_leagues_for_display(leagues: List[Dict], group_by_category: bool = True) -> List[Tuple[str, str, str]]:
    """
    Format leagues for display in UI
    
    Args:
        leagues: List of league dictionaries from API
        group_by_category: Whether to group by category
    
    Returns:
        List of tuples: (slug, display_name, category)
    """
    formatted = []
    
    for league in leagues:
        slug = league.get("slug", "")
        original_name = league.get("original_name", league.get("name", ""))
        
        # Clean name
        display_name = clean_league_name(original_name, slug)
        
        # Get priority
        priority = get_league_priority(original_name)
        
        formatted.append({
            "slug": slug,
            "display_name": display_name,
            "original_name": original_name,
            "priority": priority
        })
    
    # Sort by priority (descending)
    formatted.sort(key=lambda x: x["priority"], reverse=True)
    
    if group_by_category:
        # Group by category
        grouped = group_leagues_by_category(formatted)
        
        # Flatten with category headers
        result = []
        for category, leagues_in_cat in grouped.items():
            for league in leagues_in_cat:
                result.append((league["slug"], league["display_name"], category))
        
        return result
    else:
        # Return flat list
        return [(lg["slug"], lg["display_name"], "") for lg in formatted]


def get_league_display_name(league_dict: Dict) -> str:
    """
    Get display name for a single league
    
    Args:
        league_dict: League dictionary from API
    
    Returns:
        Formatted display name
    """
    original_name = league_dict.get("original_name", league_dict.get("name", ""))
    slug = league_dict.get("slug", "")
    
    return clean_league_name(original_name, slug)
