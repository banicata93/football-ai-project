#!/usr/bin/env python3
"""
Fixtures Loader Module

Loads upcoming fixtures from ESPN Kaggle dataset and provides
next round prediction functionality.

Author: Football AI System
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

try:
    from .utils import setup_logging
except ImportError:
    # For standalone testing
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import setup_logging


class FixturesLoader:
    """
    Loads and processes upcoming fixtures from ESPN dataset
    """
    
    def __init__(self):
        self.logger = setup_logging()
        self.project_root = Path(__file__).parent.parent
        self.espn_data_dir = self.project_root / "data_raw" / "espn" / "base_data"
        
        # Cache for loaded data
        self._fixtures_df = None
        self._teams_df = None
        self._leagues_df = None
        
        # League slug mappings (ESPN to our system)
        self.league_mappings = {
            'eng.1': 'Premier League',
            'esp.1': 'La Liga', 
            'ita.1': 'Serie A',
            'ger.1': 'Bundesliga',
            'fra.1': 'Ligue 1',
            'ned.1': 'Eredivisie',
            'por.1': 'Primeira Liga',
            'eng.2': 'Championship'
        }
        
    def _load_fixtures(self) -> pd.DataFrame:
        """Load fixtures data from ESPN dataset"""
        if self._fixtures_df is not None:
            return self._fixtures_df
            
        try:
            fixtures_path = self.espn_data_dir / "fixtures.csv"
            if not fixtures_path.exists():
                self.logger.error(f"❌ Fixtures file not found: {fixtures_path}")
                return pd.DataFrame()
            
            self.logger.info(f"📁 Loading fixtures from: {fixtures_path}")
            df = pd.read_csv(fixtures_path)
            
            # Parse date column and make timezone-aware
            df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
            
            # Filter out invalid dates
            df = df.dropna(subset=['date'])
            
            self.logger.info(f"✅ Loaded {len(df)} fixtures")
            self._fixtures_df = df
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error loading fixtures: {e}")
            return pd.DataFrame()
    
    def _load_teams(self) -> pd.DataFrame:
        """Load teams data from ESPN dataset"""
        if self._teams_df is not None:
            return self._teams_df
            
        try:
            teams_path = self.espn_data_dir / "teams.csv"
            if not teams_path.exists():
                self.logger.error(f"❌ Teams file not found: {teams_path}")
                return pd.DataFrame()
            
            self.logger.info(f"📁 Loading teams from: {teams_path}")
            df = pd.read_csv(teams_path)
            
            self.logger.info(f"✅ Loaded {len(df)} teams")
            self._teams_df = df
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error loading teams: {e}")
            return pd.DataFrame()
    
    def _load_leagues(self) -> pd.DataFrame:
        """Load leagues data from ESPN dataset"""
        if self._leagues_df is not None:
            return self._leagues_df
            
        try:
            leagues_path = self.espn_data_dir / "leagues.csv"
            if not leagues_path.exists():
                self.logger.error(f"❌ Leagues file not found: {leagues_path}")
                return pd.DataFrame()
            
            self.logger.info(f"📁 Loading leagues from: {leagues_path}")
            df = pd.read_csv(leagues_path)
            
            self.logger.info(f"✅ Loaded {len(df)} leagues")
            self._leagues_df = df
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error loading leagues: {e}")
            return pd.DataFrame()
    
    def _get_team_name(self, team_id: int) -> str:
        """Get team display name from team ID"""
        teams_df = self._load_teams()
        if teams_df.empty:
            return f"Team_{team_id}"
            
        team_row = teams_df[teams_df['teamId'] == team_id]
        if team_row.empty:
            return f"Team_{team_id}"
            
        # Prefer displayName, fallback to name
        display_name = team_row.iloc[0].get('displayName', '')
        name = team_row.iloc[0].get('name', '')
        
        return display_name if display_name else name if name else f"Team_{team_id}"
    
    def _get_league_info(self, league_id: int) -> Dict[str, str]:
        """Get league information from league ID"""
        leagues_df = self._load_leagues()
        if leagues_df.empty:
            return {"name": f"League_{league_id}", "slug": f"league_{league_id}"}
            
        # Find the most recent season for this league
        league_rows = leagues_df[leagues_df['leagueId'] == league_id]
        if league_rows.empty:
            return {"name": f"League_{league_id}", "slug": f"league_{league_id}"}
        
        # Get the most recent entry
        latest_row = league_rows.iloc[-1]
        
        league_name = latest_row.get('leagueName', f'League_{league_id}')
        league_slug = latest_row.get('seasonSlug', f'league_{league_id}')
        
        return {
            "name": league_name,
            "slug": league_slug
        }
    
    def _normalize_league_name(self, league_slug: str) -> str:
        """Convert ESPN league slug to our system's league name"""
        return self.league_mappings.get(league_slug, league_slug)
    
    def _detect_next_round_date(self, fixtures_df: pd.DataFrame) -> Optional[datetime]:
        """
        Detect the next round date from upcoming fixtures
        
        Args:
            fixtures_df: DataFrame with upcoming fixtures
            
        Returns:
            datetime: Date of the next round, or None if no fixtures
        """
        if fixtures_df.empty:
            return None
            
        # Get the earliest upcoming date
        next_date = fixtures_df['date'].min()
        
        # Find all matches within 3 days of this date (same matchweek)
        date_threshold = pd.Timedelta(days=3)
        same_round_mask = (fixtures_df['date'] - next_date).abs() <= date_threshold
        
        return next_date
    
    def get_available_leagues(self) -> List[Dict[str, str]]:
        """
        Get list of available leagues with upcoming fixtures
        
        Returns:
            List[Dict]: List of leagues with name and slug
        """
        try:
            fixtures_df = self._load_fixtures()
            if fixtures_df.empty:
                return []
            
            # Filter upcoming fixtures
            now = pd.Timestamp.now(tz='UTC')
            upcoming_fixtures = fixtures_df[fixtures_df['date'] >= now]
            
            if upcoming_fixtures.empty:
                self.logger.warning("⚠️  No upcoming fixtures found")
                return []
            
            # Get unique league IDs
            unique_leagues = upcoming_fixtures['leagueId'].unique()
            
            leagues_info = []
            for league_id in unique_leagues:
                league_info = self._get_league_info(league_id)
                normalized_name = self._normalize_league_name(league_info['slug'])
                
                leagues_info.append({
                    'id': int(league_id),
                    'name': normalized_name,
                    'original_name': league_info['name'],
                    'slug': league_info['slug']
                })
            
            self.logger.info(f"✅ Found {len(leagues_info)} leagues with upcoming fixtures")
            return leagues_info
            
        except Exception as e:
            self.logger.error(f"❌ Error getting available leagues: {e}")
            return []
    
    def get_next_round(self, league_slug: str) -> pd.DataFrame:
        """
        Get next round fixtures for a specific league
        
        Args:
            league_slug: League identifier (e.g., 'Premier League', 'eng.1')
            
        Returns:
            pd.DataFrame: Next round fixtures with columns:
                - date, home_team, away_team, league, round_date
        """
        try:
            self.logger.info(f"🔍 Getting next round for league: {league_slug}")
            
            fixtures_df = self._load_fixtures()
            if fixtures_df.empty:
                self.logger.error("❌ No fixtures data available")
                return pd.DataFrame()
            
            # Filter for upcoming fixtures only
            now = pd.Timestamp.now(tz='UTC')
            upcoming_fixtures = fixtures_df[fixtures_df['date'] >= now].copy()
            
            if upcoming_fixtures.empty:
                self.logger.warning("⚠️  No upcoming fixtures found")
                return pd.DataFrame()
            
            # Find matching league
            target_league_id = None
            
            # Try to match by normalized name first
            for league_id in upcoming_fixtures['leagueId'].unique():
                league_info = self._get_league_info(league_id)
                normalized_name = self._normalize_league_name(league_info['slug'])
                
                if (normalized_name.lower() == league_slug.lower() or 
                    league_info['slug'].lower() == league_slug.lower() or
                    league_info['name'].lower() == league_slug.lower()):
                    target_league_id = league_id
                    break
            
            if target_league_id is None:
                self.logger.error(f"❌ League not found: {league_slug}")
                available_leagues = self.get_available_leagues()
                self.logger.info(f"Available leagues: {[l['name'] for l in available_leagues]}")
                return pd.DataFrame()
            
            # Filter fixtures for target league
            league_fixtures = upcoming_fixtures[upcoming_fixtures['leagueId'] == target_league_id].copy()
            
            if league_fixtures.empty:
                self.logger.warning(f"⚠️  No upcoming fixtures for league: {league_slug}")
                return pd.DataFrame()
            
            # Detect next round date
            next_round_date = self._detect_next_round_date(league_fixtures)
            if next_round_date is None:
                return pd.DataFrame()
            
            # Get fixtures for the next round (within 3 days)
            date_threshold = pd.Timedelta(days=3)
            next_round_mask = (league_fixtures['date'] - next_round_date).abs() <= date_threshold
            next_round_fixtures = league_fixtures[next_round_mask].copy()
            
            # Build result DataFrame
            result_fixtures = []
            
            for _, fixture in next_round_fixtures.iterrows():
                home_team = self._get_team_name(fixture['homeTeamId'])
                away_team = self._get_team_name(fixture['awayTeamId'])
                
                result_fixtures.append({
                    'date': fixture['date'],
                    'home_team': home_team,
                    'away_team': away_team,
                    'league': league_slug,
                    'round_date': next_round_date,
                    'event_id': fixture['eventId'],
                    'home_team_id': fixture['homeTeamId'],
                    'away_team_id': fixture['awayTeamId']
                })
            
            result_df = pd.DataFrame(result_fixtures)
            
            # Sort by date
            result_df = result_df.sort_values('date')
            
            self.logger.info(f"✅ Found {len(result_df)} fixtures for next round of {league_slug}")
            self.logger.info(f"📅 Next round date: {next_round_date.strftime('%Y-%m-%d')}")
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"❌ Error getting next round: {e}")
            return pd.DataFrame()
    
    def get_fixture_by_teams(self, home_team: str, away_team: str, league: str = None) -> Optional[Dict]:
        """
        Find a specific fixture by team names
        
        Args:
            home_team: Home team name
            away_team: Away team name  
            league: Optional league filter
            
        Returns:
            Dict: Fixture info or None if not found
        """
        try:
            fixtures_df = self._load_fixtures()
            if fixtures_df.empty:
                return None
            
            # Filter upcoming fixtures
            now = pd.Timestamp.now(tz='UTC')
            upcoming_fixtures = fixtures_df[fixtures_df['date'] >= now].copy()
            
            # Search for matching fixture
            for _, fixture in upcoming_fixtures.iterrows():
                fixture_home = self._get_team_name(fixture['homeTeamId'])
                fixture_away = self._get_team_name(fixture['awayTeamId'])
                
                if (fixture_home.lower() == home_team.lower() and 
                    fixture_away.lower() == away_team.lower()):
                    
                    # Check league if specified
                    if league:
                        league_info = self._get_league_info(fixture['leagueId'])
                        normalized_name = self._normalize_league_name(league_info['slug'])
                        if normalized_name.lower() != league.lower():
                            continue
                    
                    return {
                        'date': fixture['date'],
                        'home_team': fixture_home,
                        'away_team': fixture_away,
                        'league': self._normalize_league_name(self._get_league_info(fixture['leagueId'])['slug']),
                        'event_id': fixture['eventId']
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error finding fixture: {e}")
            return None


def main():
    """Test the fixtures loader"""
    loader = FixturesLoader()
    
    print("🧪 Testing Fixtures Loader")
    print("=" * 40)
    
    # Test 1: Get available leagues
    leagues = loader.get_available_leagues()
    print(f"📊 Available leagues: {len(leagues)}")
    for league in leagues[:5]:
        print(f"   • {league['name']} ({league['slug']})")
    
    # Test 2: Get next round for Premier League
    if leagues:
        test_league = leagues[0]['name']
        print(f"\n🔍 Testing next round for: {test_league}")
        fixtures = loader.get_next_round(test_league)
        print(f"📅 Found {len(fixtures)} fixtures")
        
        if not fixtures.empty:
            print("Sample fixtures:")
            for _, fixture in fixtures.head(3).iterrows():
                print(f"   • {fixture['home_team']} vs {fixture['away_team']} ({fixture['date'].strftime('%Y-%m-%d %H:%M')})")


if __name__ == "__main__":
    main()
