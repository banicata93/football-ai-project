"""
Mapping subsystem for canonical team and league names.

This module provides name normalization and ID resolution without breaking
existing prediction logic. All functions use safe fallback behavior.
"""

from .teams_mapping import (
    load_team_mapping,
    normalize_team_name,
    resolve_team_id_to_name,
    get_team_canonical_name
)

from .leagues_mapping import (
    load_league_mapping,
    normalize_league_name,
    resolve_league_id_to_name,
    get_league_canonical_name
)

__all__ = [
    'load_team_mapping',
    'normalize_team_name',
    'resolve_team_id_to_name',
    'get_team_canonical_name',
    'load_league_mapping',
    'normalize_league_name',
    'resolve_league_id_to_name',
    'get_league_canonical_name',
]
