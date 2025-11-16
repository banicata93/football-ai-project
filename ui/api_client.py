import os
import requests
from typing import Dict, Optional


class FootballAPIClient:
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (
            base_url or 
            os.getenv('FOOTBALL_API_URL', 'http://localhost:8000')
        ).rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        self.timeout = 30

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make HTTP request with error handling"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            kwargs.setdefault('timeout', self.timeout)
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            
            if not response.content:
                return {"ok": True, "data": {}}
                
            return {"ok": True, "data": response.json()}
            
        except requests.exceptions.ConnectionError:
            return {"ok": False, "error": f"Cannot connect to API at {self.base_url}"}
        except requests.exceptions.Timeout:
            return {"ok": False, "error": f"Request timeout after {self.timeout}s"}
        except requests.exceptions.HTTPError as e:
            return {"ok": False, "error": f"HTTP {response.status_code}: {str(e)}"}
        except Exception as e:
            return {"ok": False, "error": f"Unexpected error: {str(e)}"}

    def get_health(self) -> Dict:
        """Get API health status"""
        return self._make_request('GET', '/health')

    def get_stats(self) -> Dict:
        """Get system statistics"""
        return self._make_request('GET', '/stats')

    def get_models(self) -> Dict:
        """Get model information"""
        return self._make_request('GET', '/models')

    def get_teams(self, limit: int = 100) -> Dict:
        """Get teams list"""
        return self._make_request('GET', f'/teams?limit={limit}')

    def get_leagues(self) -> Dict:
        """Get available leagues"""
        return self._make_request('GET', '/predict/leagues')
    
    def get_teams_by_league(self, league_slug: str = None, limit: int = 100) -> Dict:
        """Get teams for a specific league"""
        params = {'limit': limit}
        if league_slug:
            params['league_slug'] = league_slug
        return self._make_request('GET', '/teams/by-league', params=params)

    def predict_improved(self, home_team: str, away_team: str,
                        league: Optional[str] = None,
                        date: Optional[str] = None) -> Dict:
        """Predict match using improved service"""
        payload = {
            'home_team': home_team,
            'away_team': away_team
        }
        
        if league:
            payload['league'] = league
        if date:
            payload['date'] = date
            
        return self._make_request('POST', '/predict/improved', json=payload)

    def predict_next_round(self, league_slug: str) -> Dict:
        """Get next round predictions for league"""
        return self._make_request('GET', f'/predict/next-round?league={league_slug}')

    def call_post(self, endpoint: str, payload: Dict) -> Dict:
        """Generic POST call for API explorer"""
        return self._make_request('POST', endpoint, json=payload)
