"""
Team Mapping System - Canonical Names and Aliases

This module provides safe team name normalization without breaking existing logic.
All functions use fallback behavior to ensure backward compatibility.

SAFE MODE GUARANTEES:
- Never throws errors
- Never stops predictions
- Always returns valid team name (fallback to input)
- No breaking changes to existing code
"""

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Optional fuzzy matching (safe fallback if not available)
try:
    from rapidfuzz import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False

logger = logging.getLogger(__name__)

# Global mapping cache
_TEAM_MAPPING: Optional[Dict] = None
_TEAM_ID_MAP: Optional[Dict[int, str]] = None
_TEAM_ALIAS_MAP: Optional[Dict[str, str]] = None


def load_team_mapping(csv_path: str = None) -> Dict:
    """
    Load team mapping from CSV file.
    
    Args:
        csv_path: Path to teams.csv (optional, uses default if None)
        
    Returns:
        Dictionary with team mapping data
        
    Structure:
        {
            'by_id': {team_id: canonical_name},
            'by_canonical': {canonical_name: team_data},
            'by_alias': {alias_clean: canonical_name}
        }
    """
    global _TEAM_MAPPING, _TEAM_ID_MAP, _TEAM_ALIAS_MAP
    
    # Return cached if already loaded
    if _TEAM_MAPPING is not None:
        return _TEAM_MAPPING
    
    # Default path
    if csv_path is None:
        csv_path = Path(__file__).parent.parent.parent / 'data' / 'mappings' / 'teams.csv'
    else:
        csv_path = Path(csv_path)
    
    # Initialize empty mapping (safe fallback)
    mapping = {
        'by_id': {},
        'by_canonical': {},
        'by_alias': {}
    }
    
    # Try to load CSV (safe - no errors if file missing)
    try:
        if not csv_path.exists():
            logger.warning(f"Team mapping file not found: {csv_path}. Using empty mapping (safe fallback).")
            _TEAM_MAPPING = mapping
            _TEAM_ID_MAP = {}
            _TEAM_ALIAS_MAP = {}
            return mapping
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                try:
                    team_id = int(row['team_id']) if row.get('team_id') else None
                    canonical_name = row['canonical_name'].strip()
                    display_name = row.get('display_name', canonical_name).strip()
                    aliases_str = row.get('aliases', '')
                    
                    # Parse aliases
                    aliases = []
                    if aliases_str:
                        aliases = [a.strip() for a in aliases_str.split(';') if a.strip()]
                    
                    # Store team data
                    team_data = {
                        'team_id': team_id,
                        'canonical_name': canonical_name,
                        'display_name': display_name,
                        'aliases': aliases
                    }
                    
                    # Index by ID
                    if team_id:
                        mapping['by_id'][team_id] = canonical_name
                    
                    # Index by canonical name (cleaned)
                    canonical_clean = _clean_name(canonical_name)
                    mapping['by_canonical'][canonical_clean] = team_data
                    
                    # Index by aliases (cleaned)
                    for alias in aliases:
                        alias_clean = _clean_name(alias)
                        if alias_clean:
                            mapping['by_alias'][alias_clean] = canonical_name
                    
                    # Also index canonical name as alias
                    mapping['by_alias'][canonical_clean] = canonical_name
                    
                except Exception as e:
                    logger.warning(f"Error parsing team row: {row}. Error: {e}. Skipping.")
                    continue
        
        logger.info(f"Loaded {len(mapping['by_canonical'])} teams from {csv_path}")
        
    except Exception as e:
        logger.warning(f"Error loading team mapping: {e}. Using empty mapping (safe fallback).")
    
    # Cache the mapping
    _TEAM_MAPPING = mapping
    _TEAM_ID_MAP = mapping['by_id']
    _TEAM_ALIAS_MAP = mapping['by_alias']
    
    return mapping


def _clean_name(name: str) -> str:
    """
    Clean and normalize a team/league name for matching.
    
    Args:
        name: Raw name string
        
    Returns:
        Cleaned lowercase name
    """
    if not name:
        return ""
    
    # Lowercase
    cleaned = name.lower().strip()
    
    # Remove extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Remove common punctuation (but keep hyphens)
    cleaned = re.sub(r'[^\w\s\-]', '', cleaned)
    
    return cleaned


def normalize_team_name(raw_name: str, fuzzy_threshold: int = 85) -> str:
    """
    Normalize team name to canonical form.
    
    SAFE MODE: Always returns a valid team name (fallback to input if no match).
    
    Args:
        raw_name: Raw team name from user input or data
        fuzzy_threshold: Minimum fuzzy match score (0-100)
        
    Returns:
        Canonical team name (or original if no match found)
        
    Matching strategy:
        1. Exact canonical name match
        2. Exact alias match
        3. Fuzzy match (if enabled and threshold met)
        4. Fallback to original input (SAFE)
    """
    # Load mapping if not loaded
    if _TEAM_MAPPING is None:
        load_team_mapping()
    
    # Safety check
    if not raw_name or not isinstance(raw_name, str):
        return raw_name if raw_name else ""
    
    # Clean input
    cleaned_input = _clean_name(raw_name)
    
    if not cleaned_input:
        return raw_name
    
    # 1. Check exact canonical match
    if cleaned_input in _TEAM_MAPPING['by_canonical']:
        return _TEAM_MAPPING['by_canonical'][cleaned_input]['canonical_name']
    
    # 2. Check exact alias match
    if cleaned_input in _TEAM_ALIAS_MAP:
        return _TEAM_ALIAS_MAP[cleaned_input]
    
    # 3. Fuzzy matching (optional, safe)
    if FUZZY_AVAILABLE and fuzzy_threshold > 0:
        best_match = None
        best_score = 0
        
        # Check against all aliases
        for alias_clean, canonical in _TEAM_ALIAS_MAP.items():
            score = fuzz.ratio(cleaned_input, alias_clean)
            if score >= fuzzy_threshold and score > best_score:
                best_score = score
                best_match = canonical
        
        if best_match:
            logger.debug(f"Fuzzy matched '{raw_name}' -> '{best_match}' (score: {best_score})")
            return best_match
    
    # 4. Fallback to original input (SAFE - no breaking changes)
    return raw_name


def resolve_team_id_to_name(team_id: int) -> str:
    """
    Resolve team ID to canonical name.
    
    SAFE MODE: Returns string representation of ID if no match found.
    
    Args:
        team_id: ESPN team ID
        
    Returns:
        Canonical team name (or str(team_id) as fallback)
    """
    # Load mapping if not loaded
    if _TEAM_MAPPING is None:
        load_team_mapping()
    
    # Safety check
    if team_id is None:
        return ""
    
    try:
        team_id = int(team_id)
    except (ValueError, TypeError):
        return str(team_id)
    
    # Lookup by ID
    if team_id in _TEAM_ID_MAP:
        return _TEAM_ID_MAP[team_id]
    
    # Fallback to ID string (SAFE)
    return str(team_id)


def get_team_canonical_name(team_name: str) -> str:
    """
    Get canonical name for a team (alias for normalize_team_name).
    
    Args:
        team_name: Team name to normalize
        
    Returns:
        Canonical team name
    """
    return normalize_team_name(team_name)


def get_team_display_name(team_name: str) -> str:
    """
    Get display name for a team.
    
    Args:
        team_name: Team name (canonical or alias)
        
    Returns:
        Display name (or canonical if not found)
    """
    # Load mapping if not loaded
    if _TEAM_MAPPING is None:
        load_team_mapping()
    
    # Normalize first
    canonical = normalize_team_name(team_name)
    
    # Get display name
    canonical_clean = _clean_name(canonical)
    if canonical_clean in _TEAM_MAPPING['by_canonical']:
        return _TEAM_MAPPING['by_canonical'][canonical_clean]['display_name']
    
    return canonical


def get_all_team_aliases(team_name: str) -> List[str]:
    """
    Get all known aliases for a team.
    
    Args:
        team_name: Team name (canonical or alias)
        
    Returns:
        List of aliases (empty list if not found)
    """
    # Load mapping if not loaded
    if _TEAM_MAPPING is None:
        load_team_mapping()
    
    # Normalize first
    canonical = normalize_team_name(team_name)
    
    # Get aliases
    canonical_clean = _clean_name(canonical)
    if canonical_clean in _TEAM_MAPPING['by_canonical']:
        return _TEAM_MAPPING['by_canonical'][canonical_clean]['aliases']
    
    return []


# Auto-load on import (safe - no errors if file missing)
try:
    load_team_mapping()
except Exception as e:
    logger.warning(f"Could not auto-load team mapping: {e}")
