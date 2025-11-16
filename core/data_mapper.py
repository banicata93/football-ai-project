"""
Data Mapper - Maps IDs to names for training

This module provides functions to map team_id and league_id to their names
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class DataMapper:
    """Maps IDs to names for teams and leagues"""
    
    def __init__(self, base_data_path: str = "data_raw/espn/base_data"):
        """
        Initialize DataMapper
        
        Args:
            base_data_path: Path to base data directory
        """
        self.base_data_path = Path(base_data_path)
        self.teams_df = None
        self.leagues_df = None
        self._load_mappings()
    
    def _load_mappings(self):
        """Load teams and leagues CSV files"""
        try:
            # Load teams
            teams_path = self.base_data_path / "teams.csv"
            if teams_path.exists():
                self.teams_df = pd.read_csv(teams_path)
                logger.info(f"✅ Loaded {len(self.teams_df)} teams")
            else:
                logger.warning(f"⚠️ Teams file not found: {teams_path}")
            
            # Load leagues
            leagues_path = self.base_data_path / "leagues.csv"
            if leagues_path.exists():
                self.leagues_df = pd.read_csv(leagues_path)
                logger.info(f"✅ Loaded {len(self.leagues_df)} leagues")
            else:
                logger.warning(f"⚠️ Leagues file not found: {leagues_path}")
                
        except Exception as e:
            logger.error(f"❌ Error loading mappings: {e}")
    
    def get_team_name(self, team_id: int) -> str:
        """
        Get team name from team ID
        
        Args:
            team_id: Team ID
        
        Returns:
            Team display name or f"Team_{team_id}" if not found
        """
        if self.teams_df is None:
            return f"Team_{team_id}"
        
        team = self.teams_df[self.teams_df['teamId'] == team_id]
        if not team.empty:
            return team.iloc[0]['displayName']
        
        return f"Team_{team_id}"
    
    def get_league_name(self, league_id: int) -> str:
        """
        Get league name from league ID
        
        Args:
            league_id: League ID
        
        Returns:
            League name or f"League_{league_id}" if not found
        """
        if self.leagues_df is None:
            return f"League_{league_id}"
        
        league = self.leagues_df[self.leagues_df['leagueId'] == league_id]
        if not league.empty:
            return league.iloc[0]['leagueName']
        
        return f"League_{league_id}"
    
    def get_league_slug(self, league_id: int) -> str:
        """
        Get league slug from league ID
        
        Args:
            league_id: League ID
        
        Returns:
            League slug or f"league_{league_id}" if not found
        """
        if self.leagues_df is None:
            return f"league_{league_id}"
        
        league = self.leagues_df[self.leagues_df['leagueId'] == league_id]
        if not league.empty:
            # Use leagueName and convert to slug
            name = league.iloc[0]['leagueName']
            slug = name.lower().replace(' ', '_').replace('-', '_')
            return slug
        
        return f"league_{league_id}"
    
    def enrich_fixtures(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich fixtures DataFrame with team and league names
        
        Args:
            df: Fixtures DataFrame with IDs
        
        Returns:
            Enriched DataFrame with names
        """
        if df is None or df.empty:
            return df
        
        df = df.copy()
        
        # Add team names
        if 'home_team_id' in df.columns:
            df['home_team'] = df['home_team_id'].apply(self.get_team_name)
        
        if 'away_team_id' in df.columns:
            df['away_team'] = df['away_team_id'].apply(self.get_team_name)
        
        # Add league name
        if 'league_id' in df.columns:
            df['league'] = df['league_id'].apply(self.get_league_name)
            df['league_slug'] = df['league_id'].apply(self.get_league_slug)
        
        logger.info(f"✅ Enriched {len(df)} fixtures with names")
        
        return df
    
    def get_all_leagues_with_data(self, df: pd.DataFrame, min_matches: int = 100) -> Dict[int, Dict]:
        """
        Get all leagues with sufficient data
        
        Args:
            df: Fixtures DataFrame
            min_matches: Minimum matches required
        
        Returns:
            Dictionary of {league_id: {name, slug, count}}
        """
        if df is None or df.empty or 'league_id' not in df.columns:
            return {}
        
        league_counts = df['league_id'].value_counts()
        leagues_with_data = {}
        
        for league_id, count in league_counts.items():
            if count >= min_matches:
                leagues_with_data[league_id] = {
                    'id': league_id,
                    'name': self.get_league_name(league_id),
                    'slug': self.get_league_slug(league_id),
                    'matches': count
                }
        
        logger.info(f"✅ Found {len(leagues_with_data)} leagues with ≥{min_matches} matches")
        
        return leagues_with_data
