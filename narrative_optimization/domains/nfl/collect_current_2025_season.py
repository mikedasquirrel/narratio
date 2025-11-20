"""
Collect Current 2025 NFL Season Data
Real-time collection for ongoing season

Current Date: November 14, 2025
Season Status: Week 10 (approximately)

SOURCES FOR LIVE DATA:
1. ESPN API (has current season)
2. NFL.com RSS feeds
3. Sports data APIs (The Odds API, etc.)
4. Manual entry from public box scores
"""

import json
from pathlib import Path
from datetime import datetime

print("="*80)
print("2025 NFL SEASON - CURRENT DATA COLLECTION")
print("="*80)

print(f"\nCurrent date: {datetime.now().strftime('%Y-%m-%d')}")
print(f"Expected: 2025 season Week 10 (started Sept 2025)")

# ============================================================================
# CHECK WHAT WE HAVE
# ============================================================================

existing_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'nfl_games_with_odds.json'
with open(existing_path) as f:
    existing_games = json.load(f)

season_2025 = [g for g in existing_games if g.get('season') == 2025]

print(f"\nCurrent 2025 data: {len(season_2025)} games")

if not season_2025:
    print(f"  ⚠ No 2025 season data in database")
    print(f"\n  NEED TO COLLECT:")
    print(f"    - Weeks 1-10 (approximately 160 games)")
    print(f"    - QB starters")
    print(f"    - Coach names")
    print(f"    - Game outcomes")
    print(f"    - Betting odds (for validation)")

# ============================================================================
# COLLECTION METHODS
# ============================================================================

print("\n" + "="*80)
print("DATA COLLECTION OPTIONS")
print("="*80)

print("""
OPTION 1: ESPN API (Most current)
```python
import requests

# ESPN has public API for scores
url = f'http://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard'
params = {'dates': '20250914'}  # Week 1 2025
response = requests.get(url, params=params)
games = response.json()
```

OPTION 2: The Odds API (Has betting data)
- Requires API key (free tier available)
- Has current odds + historical results
- Best for betting validation

OPTION 3: Manual from ESPN.com
- Week-by-week box scores
- Copy outcomes + key players
- Tedious but reliable

OPTION 4: Wait for sportsipy update
- May add current season later
- Not reliable for real-time

RECOMMENDED: ESPN API for scores, The Odds API for betting data
""")

# ============================================================================
# GENERATE COLLECTION SCRIPT
# ============================================================================

print("\n[CREATING] ESPN API collector...")

espn_collector = '''
import requests
import json
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

print(f"\\nCollected {len(all_games)} games")

# Save
with open('nfl_2025_current.json', 'w') as f:
    json.dump(all_games, f, indent=2)

print("✓ Saved to nfl_2025_current.json")
'''

collector_path = Path(__file__).parent / 'espn_collector.py'
with open(collector_path, 'w') as f:
    f.write(espn_collector)

print(f"✓ Created ESPN collector: {collector_path.name}")

# ============================================================================
# INSTRUCTIONS
# ============================================================================

print("\n" + "="*80)
print("COLLECTION INSTRUCTIONS")
print("="*80)

print(f"""
TO COLLECT 2025 SEASON:

1. Run ESPN collector:
   ```bash
   cd {Path(__file__).parent}
   python3 espn_collector.py
   ```

2. This will create: nfl_2025_current.json
   With: ~160 games (Weeks 1-10)

3. Add QB/Coach names (manual or from rosters)

4. Merge with existing database

5. Regenerate features

6. Test model on Weeks 1-10 (most recent performance)

ALTERNATIVE - Manual Quick Entry:
Just enter this week's games (Week 10, ~16 games):
- Outcomes from ESPN.com
- QB starters from depth charts
- Coaches (same as 2024 mostly)
- Test model predictions vs actual

This tests "does it work THIS week" which is most important.
""")

print(f"\n✓ Collection setup complete")
print("="*80)

