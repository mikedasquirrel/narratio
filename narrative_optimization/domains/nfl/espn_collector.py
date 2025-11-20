
import requests
import json
import time
from datetime import datetime, timedelta

def collect_nfl_week(season, week):
    """Collect one week of NFL games from ESPN"""
    
    # ESPN scoreboard API
    url = "http://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    
    # Calculate approximate date for this week
    # Week 1 typically starts first Thursday of September
    season_start = datetime(season, 9, 7)  # Approximate
    week_date = season_start + timedelta(days=(week-1)*7)
    
    params = {
        'dates': week_date.strftime('%Y%m%d'),
        'limit': 100
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        games = []
        for event in data.get('events', []):
            competitions = event.get('competitions', [])
            if competitions:
                comp = competitions[0]
                
                competitors = comp.get('competitors', [])
                home_team = next((c for c in competitors if c.get('homeAway') == 'home'), None)
                away_team = next((c for c in competitors if c.get('homeAway') == 'away'), None)
                
                if home_team and away_team:
                    game = {
                        'season': season,
                        'week': week,
                        'gameday': event.get('date', '')[:10],
                        'home_team': home_team.get('team', {}).get('abbreviation'),
                        'away_team': away_team.get('team', {}).get('abbreviation'),
                        'home_score': int(home_team.get('score', 0)),
                        'away_score': int(away_team.get('score', 0)),
                        'home_won': int(home_team.get('score', 0)) > int(away_team.get('score', 0)),
                        'status': comp.get('status', {}).get('type', {}).get('name', 'unknown')
                    }
                    games.append(game)
        
        return games
        
    except Exception as e:
        print(f"Error collecting week {week}: {e}")
        return []

# Collect current season
print("Collecting 2025 season (Weeks 1-10)...")
all_games = []

for week in range(1, 11):
    print(f"  Week {week}...", end=" ", flush=True)
    games = collect_nfl_week(2025, week)
    print(f"✓ {len(games)} games")
    all_games.extend(games)
    time.sleep(1)  # Be respectful

print(f"\nCollected {len(all_games)} games")

# Save
with open('nfl_2025_current.json', 'w') as f:
    json.dump(all_games, f, indent=2)

print("✓ Saved to nfl_2025_current.json")
