"""
Test script за API
"""

import requests
import json
import time


def test_api(base_url: str = "http://localhost:8000"):
    """
    Тестване на всички API endpoints
    
    Args:
        base_url: Base URL на API
    """
    print("=" * 70)
    print("ТЕСТВАНЕ НА FOOTBALL AI PREDICTION API")
    print("=" * 70)
    
    # Test 1: Root endpoint
    print("\n[1/7] Test Root Endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        print(f"✓ Status: {response.status_code}")
        print(f"  Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 2: Health check
    print("\n[2/7] Test Health Check...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✓ Status: {response.status_code}")
        print(f"  Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 3: Models list
    print("\n[3/7] Test Models List...")
    try:
        response = requests.get(f"{base_url}/models")
        print(f"✓ Status: {response.status_code}")
        data = response.json()
        print(f"  Total models: {data['total_models']}")
        for model in data['models']:
            print(f"  - {model['model_name']} ({model['version']})")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 4: Stats
    print("\n[4/7] Test Stats...")
    try:
        response = requests.get(f"{base_url}/stats")
        print(f"✓ Status: {response.status_code}")
        data = response.json()
        print(f"  Version: {data['version']}")
        print(f"  Uptime: {data['uptime_hours']:.2f} hours")
        print(f"  Models: {data['models_loaded']}")
        print(f"  Teams: {data['teams_in_database']}")
        print(f"  Features: {data['features_used']}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 5: Teams list
    print("\n[5/7] Test Teams List...")
    try:
        response = requests.get(f"{base_url}/teams")
        print(f"✓ Status: {response.status_code}")
        data = response.json()
        print(f"  Total teams: {data['total_teams']}")
        print(f"  Top 5 teams by Elo:")
        for i, team in enumerate(data['teams'][:5], 1):
            print(f"    {i}. {team['name']}: {team['elo']:.0f}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 6: Prediction (POST)
    print("\n[6/7] Test Prediction (POST)...")
    try:
        payload = {
            "home_team": "Manchester United",
            "away_team": "Liverpool",
            "league": "Premier League",
            "date": "2024-03-15"
        }
        
        response = requests.post(f"{base_url}/predict", json=payload)
        print(f"✓ Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"\n  Match: {data['match_info']['home_team']} vs {data['match_info']['away_team']}")
            print(f"  League: {data['match_info']['league']}")
            
            print(f"\n  1X2 Prediction:")
            print(f"    Home Win: {data['prediction_1x2']['prob_home_win']:.3f}")
            print(f"    Draw:     {data['prediction_1x2']['prob_draw']:.3f}")
            print(f"    Away Win: {data['prediction_1x2']['prob_away_win']:.3f}")
            print(f"    → Predicted: {data['prediction_1x2']['predicted_outcome']} (confidence: {data['prediction_1x2']['confidence']:.3f})")
            
            print(f"\n  Over/Under 2.5:")
            print(f"    Over:  {data['prediction_ou25']['prob_over']:.3f}")
            print(f"    Under: {data['prediction_ou25']['prob_under']:.3f}")
            print(f"    → Predicted: {data['prediction_ou25']['predicted_outcome']} (confidence: {data['prediction_ou25']['confidence']:.3f})")
            
            print(f"\n  BTTS:")
            print(f"    Yes: {data['prediction_btts']['prob_yes']:.3f}")
            print(f"    No:  {data['prediction_btts']['prob_no']:.3f}")
            print(f"    → Predicted: {data['prediction_btts']['predicted_outcome']} (confidence: {data['prediction_btts']['confidence']:.3f})")
            
            print(f"\n  FII Score: {data['fii']['score']:.2f}/10 ({data['fii']['confidence_level']})")
        else:
            print(f"  Error: {response.json()}")
            
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 7: Prediction (GET)
    print("\n[7/7] Test Prediction (GET)...")
    try:
        response = requests.get(
            f"{base_url}/predict/Barcelona/vs/Real Madrid",
            params={"league": "La Liga"}
        )
        print(f"✓ Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"  Match: {data['match_info']['home_team']} vs {data['match_info']['away_team']}")
            print(f"  Predicted 1X2: {data['prediction_1x2']['predicted_outcome']}")
            print(f"  Predicted OU2.5: {data['prediction_ou25']['predicted_outcome']}")
            print(f"  Predicted BTTS: {data['prediction_btts']['predicted_outcome']}")
            print(f"  FII: {data['fii']['score']:.2f}/10")
        else:
            print(f"  Error: {response.json()}")
            
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "=" * 70)
    print("ТЕСТВАНЕ ЗАВЪРШЕНО")
    print("=" * 70)


if __name__ == "__main__":
    # Wait for server to start
    print("Изчакване на сървъра да стартира...")
    time.sleep(2)
    
    # Run tests
    test_api()
