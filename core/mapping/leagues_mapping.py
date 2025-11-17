"""
League Mapping System - Canonical Names and Slugs

This module provides safe league name normalization without breaking existing logic.
All functions use fallback behavior to ensure backward compatibility.

SAFE MODE GUARANTEES:
- Never throws errors
- Never stops predictions
- Always returns valid league name (fallback to input)
- No breaking changes to existing code
"""

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Optional fuzzy matching (safe fallback if not available)
try:
    from rapidfuzz import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False

logger = logging.getLogger(__name__)

# Global mapping cache
_LEAGUE_MAPPING: Optional[Dict] = None
_LEAGUE_ID_MAP: Optional[Dict[int, str]] = None
_LEAGUE_ALIAS_MAP: Optional[Dict[str, str]] = None
_LEAGUE_SLUG_MAP: Optional[Dict[str, str]] = None


def load_league_mapping(csv_path: str = None) -> Dict:
    """
    Load league mapping from CSV file.
    
    Args:
        csv_path: Path to leagues.csv (optional, uses default if None)
        
    Returns:
        Dictionary with league mapping data
        
    Structure:
        {
            'by_id': {league_id: canonical_name},
            'by_canonical': {canonical_name: league_data},
            'by_alias': {alias_clean: canonical_name},
            'by_slug': {slug: canonical_name}
        }
    """
    global _LEAGUE_MAPPING, _LEAGUE_ID_MAP, _LEAGUE_ALIAS_MAP, _LEAGUE_SLUG_MAP
    
    # Return cached if already loaded
    if _LEAGUE_MAPPING is not None:
        return _LEAGUE_MAPPING
    
    # Default path
    if csv_path is None:
        csv_path = Path(__file__).parent.parent.parent / 'data' / 'mappings' / 'leagues.csv'
    else:
        csv_path = Path(csv_path)
    
    # Initialize empty mapping (safe fallback)
    mapping = {
        'by_id': {},
        'by_canonical': {},
        'by_alias': {},
        'by_slug': {}
    }
    
    # Try to load CSV (safe - no errors if file missing)
    try:
        if not csv_path.exists():
            logger.warning(f"League mapping file not found: {csv_path}. Using empty mapping (safe fallback).")
            _LEAGUE_MAPPING = mapping
            _LEAGUE_ID_MAP = {}
            _LEAGUE_ALIAS_MAP = {}
            _LEAGUE_SLUG_MAP = {}
            return mapping
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                try:
                    league_id = int(row['league_id']) if row.get('league_id') else None
                    canonical_name = row['canonical_name'].strip()
                    slug = row.get('slug', '').strip()
                    aliases_str = row.get('aliases', '')
                    
                    # Parse aliases
                    aliases = []
                    if aliases_str:
                        aliases = [a.strip() for a in aliases_str.split(';') if a.strip()]
                    
                    # Store league data
                    league_data = {
                        'league_id': league_id,
                        'canonical_name': canonical_name,
                        'slug': slug,
                        'aliases': aliases
                    }
                    
                    # Index by ID
                    if league_id:
                        mapping['by_id'][league_id] = canonical_name
                    
                    # Index by canonical name (cleaned)
                    canonical_clean = _clean_name(canonical_name)
                    mapping['by_canonical'][canonical_clean] = league_data
                    
                    # Index by slug
                    if slug:
                        mapping['by_slug'][slug] = canonical_name
                        # Also clean slug version
                        slug_clean = _clean_name(slug)
                        mapping['by_slug'][slug_clean] = canonical_name
                    
                    # Index by aliases (cleaned)
                    for alias in aliases:
                        alias_clean = _clean_name(alias)
                        if alias_clean:
                            mapping['by_alias'][alias_clean] = canonical_name
                    
                    # Also index canonical name as alias
                    mapping['by_alias'][canonical_clean] = canonical_name
                    
                except Exception as e:
                    logger.warning(f"Error parsing league row: {row}. Error: {e}. Skipping.")
                    continue
        
        logger.info(f"Loaded {len(mapping['by_canonical'])} leagues from {csv_path}")
        
    except Exception as e:
        logger.warning(f"Error loading league mapping: {e}. Using empty mapping (safe fallback).")
    
    # Cache the mapping
    _LEAGUE_MAPPING = mapping
    _LEAGUE_ID_MAP = mapping['by_id']
    _LEAGUE_ALIAS_MAP = mapping['by_alias']
    _LEAGUE_SLUG_MAP = mapping['by_slug']
    
    return mapping


def _clean_name(name: str) -> str:
    """
    Clean and normalize a league name for matching.
    
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


def normalize_league_name(raw_name: str, fuzzy_threshold: int = 85) -> str:
    """
    Normalize league name to canonical form.
    
    SAFE MODE: Always returns a valid league name (fallback to input if no match).
    
    Args:
        raw_name: Raw league name from user input or data
        fuzzy_threshold: Minimum fuzzy match score (0-100)
        
    Returns:
        Canonical league name (or original if no match found)
        
    Matching strategy:
        1. Exact canonical name match
        2. Exact slug match
        3. Exact alias match
        4. Fuzzy match (if enabled and threshold met)
        5. Fallback to original input (SAFE)
    """
    # Load mapping if not loaded
    if _LEAGUE_MAPPING is None:
        load_league_mapping()
    
    # Safety check
    if not raw_name or not isinstance(raw_name, str):
        return raw_name if raw_name else ""
    
    # Clean input
    cleaned_input = _clean_name(raw_name)
    
    if not cleaned_input:
        return raw_name
    
    # 1. Check exact canonical match
    if cleaned_input in _LEAGUE_MAPPING['by_canonical']:
        return _LEAGUE_MAPPING['by_canonical'][cleaned_input]['canonical_name']
    
    # 2. Check exact slug match
    if cleaned_input in _LEAGUE_SLUG_MAP:
        return _LEAGUE_SLUG_MAP[cleaned_input]
    
    # 3. Check exact alias match
    if cleaned_input in _LEAGUE_ALIAS_MAP:
        return _LEAGUE_ALIAS_MAP[cleaned_input]
    
    # 4. Fuzzy matching (optional, safe)
    if FUZZY_AVAILABLE and fuzzy_threshold > 0:
        best_match = None
        best_score = 0
        
        # Check against all aliases
        for alias_clean, canonical in _LEAGUE_ALIAS_MAP.items():
            score = fuzz.ratio(cleaned_input, alias_clean)
            if score >= fuzzy_threshold and score > best_score:
                best_score = score
                best_match = canonical
        
        if best_match:
            logger.debug(f"Fuzzy matched '{raw_name}' -> '{best_match}' (score: {best_score})")
            return best_match
    
    # 5. Fallback to original input (SAFE - no breaking changes)
    return raw_name


def resolve_league_id_to_name(league_id: int) -> str:
    """
    Resolve league ID to canonical name.
    
    SAFE MODE: Returns string representation of ID if no match found.
    
    Args:
        league_id: ESPN league ID
        
    Returns:
        Canonical league name (or str(league_id) as fallback)
    """
    # Load mapping if not loaded
    if _LEAGUE_MAPPING is None:
        load_league_mapping()
    
    # Safety check
    if league_id is None:
        return ""
    
    try:
        league_id = int(league_id)
    except (ValueError, TypeError):
        return str(league_id)
    
    # Lookup by ID
    if league_id in _LEAGUE_ID_MAP:
        return _LEAGUE_ID_MAP[league_id]
    
    # Fallback to ID string (SAFE)
    return str(league_id)


def get_league_canonical_name(league_name: str) -> str:
    """
    Get canonical name for a league (alias for normalize_league_name).
    
    Args:
        league_name: League name to normalize
        
    Returns:
        Canonical league name
    """
    return normalize_league_name(league_name)


def get_league_slug(league_name: str) -> str:
    """
    Get slug for a league.
    
    Args:
        league_name: League name (canonical or alias)
        
    Returns:
        League slug (or canonical name if not found)
    """
    # Load mapping if not loaded
    if _LEAGUE_MAPPING is None:
        load_league_mapping()
    
    # Normalize first
    canonical = normalize_league_name(league_name)
    
    # Get slug
    canonical_clean = _clean_name(canonical)
    if canonical_clean in _LEAGUE_MAPPING['by_canonical']:
        slug = _LEAGUE_MAPPING['by_canonical'][canonical_clean]['slug']
        return slug if slug else canonical
    
    return canonical


def get_all_league_aliases(league_name: str) -> List[str]:
    """
    Get all known aliases for a league.
    
    Args:
        league_name: League name (canonical or alias)
        
    Returns:
        List of aliases (empty list if not found)
    """
    # Load mapping if not loaded
    if _LEAGUE_MAPPING is None:
        load_league_mapping()
    
    # Normalize first
    canonical = normalize_league_name(league_name)
    
    # Get aliases
    canonical_clean = _clean_name(canonical)
    if canonical_clean in _LEAGUE_MAPPING['by_canonical']:
        return _LEAGUE_MAPPING['by_canonical'][canonical_clean]['aliases']
    
    return []


# Auto-load on import (safe - no errors if file missing)
try:
    load_league_mapping()
except Exception as e:
    logger.warning(f"Could not auto-load league mapping: {e}")
