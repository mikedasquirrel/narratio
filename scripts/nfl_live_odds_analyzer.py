#!/usr/bin/env python3
"""
NFL LIVE Odds Analyzer
Gets ACTUAL upcoming games with REAL current odds
"""

import sys
import json
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# The Odds API (free tier: 500 requests/month)
# Get your free API key at: https://the-odds-api.com/
ODDS_API_KEY = "YOUR_API_KEY_HERE"  # Replace with actual key
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Alternative: Use odds from ESPN/other free sources
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"

def fetch_upcoming_nfl_games_espn():
    """Fetch upcoming NFL games from ESPN (FREE)"""
    print("📥 Fetching upcoming NFL games from ESPN...")
    
    try:
        response = requests.get(ESPN_SCOREBOARD, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        events = data.get('events', [])
        print(f"✓ Found {len(events)} games")
        
        games = []
        for event in events:
            try:
                comp = event['competitions'][0]
                
                # Get teams
                home_team = next(t for t in comp['competitors'] if t['homeAway'] == 'home')
                away_team = next(t for t in comp['competitors'] if t['homeAway'] == 'away')
                
                # Get odds if available
                odds = comp.get('odds', [{}])[0] if comp.get('odds') else {}
                spread = odds.get('spread', 0)
                over_under = odds.get('overUnder', 0)
                
                # Game status
                status = event['status']['type']['name']
                game_date = event['date']
                
                game = {
                    'game_id': event['id'],
                    'name': event['name'],
                    'date': game_date,
                    'status': status,
                    'home_team': home_team['team']['abbreviation'],
                    'away_team': away_team['team']['abbreviation'],
                    'home_team_full': home_team['team']['displayName'],
                    'away_team_full': away_team['team']['displayName'],
                    'spread_line': float(spread) if spread else None,
                    'total_line': float(over_under) if over_under else None,
                    'home_score': int(home_team.get('score', 0)),
                    'away_score': int(away_team.get('score', 0)),
                    'is_upcoming': status in ['STATUS_SCHEDULED', 'STATUS_PREGAME'],
                    'is_live': status in ['STATUS_IN_PROGRESS', 'STATUS_HALFTIME'],
                    'is_final': status in ['STATUS_FINAL', 'STATUS_FINAL_OVERTIME'],
                }
                
                games.append(game)
                
            except Exception as e:
                print(f"  ⚠ Error parsing game: {e}")
                continue
        
        return games
        
    except Exception as e:
        print(f"✗ Failed to fetch from ESPN: {e}")
        return []

def fetch_odds_from_api(api_key=None):
    """Fetch from The Odds API (requires key)"""
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        print("⚠ No Odds API key - using ESPN data instead")
        return None
    
    print("📥 Fetching from The Odds API...")
    
    url = f"{ODDS_API_BASE}/sports/americanfootball_nfl/odds/"
    
    try:
        response = requests.get(url, params={
            'apiKey': api_key,
            'regions': 'us',
            'markets': 'spreads,h2h,totals',
            'oddsFormat': 'american',
        }, timeout=10)
        
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Found {len(data)} games with odds")
        return data
        
    except Exception as e:
        print(f"✗ API fetch failed: {e}")
        return None

def load_historical_context():
    """Load historical data for team context"""
    context_file = Path(__file__).parent.parent / "data" / "domains" / "nfl_enriched_with_rosters.json"
    
    with open(context_file) as f:
        data = json.load(f)
    
    # Build team records and matchup history
    games = data['games']
    current_season = 2025
    
    # Get latest records for each team
    team_records = {}
    for game in sorted(games, key=lambda g: (g['season'], g.get('week', 0))):
        if game['season'] == current_season:
            team_records[game['home_team']] = game.get('home_record_before', '0-0')
            team_records[game['away_team']] = game.get('away_record_before', '0-0')
    
    # Get matchup history
    matchup_history = {}
    for game in games:
        key = tuple(sorted([game['home_team'], game['away_team']]))
        if key not in matchup_history:
            matchup_history[key] = []
        matchup_history[key].append(game)
    
    return team_records, matchup_history

def analyze_live_game(game, team_records, matchup_history, patterns):
    """Analyze a single live/upcoming game"""
    
    home = game['home_team']
    away = game['away_team']
    spread = game.get('spread_line', 0)
    
    if spread is None or spread == 0:
        return None
    
    # Calculate features
    features = {
        'is_home_dog': spread > 0,
        'is_big_dog': spread >= 7,
        'is_huge_dog': spread >= 3.5,
        'spread_line': spread,
    }
    
    # Team records
    home_rec = team_records.get(home, '0-0')
    away_rec = team_records.get(away, '0-0')
    
    def parse_record(rec):
        try:
            w, l = rec.split('-')
            return int(w), int(l)
        except:
            return 0, 0
    
    home_w, home_l = parse_record(home_rec)
    away_w, away_l = parse_record(away_rec)
    
    features['record_diff'] = home_w - away_w
    features['high_momentum'] = features['record_diff'] > 2
    
    # Matchup history
    matchup_key = tuple(sorted([home, away]))
    history = matchup_history.get(matchup_key, [])
    features['rivalry_games'] = len(history)
    features['is_rivalry'] = len(history) > 15
    
    # Match against patterns
    matches = []
    
    if features['is_huge_dog']:
        matches.append(('Huge Home Underdog (+7+)', 80.3, 94.4) if features['is_big_dog'] 
                      else ('Big Home Underdog (+3.5+)', 65.5, 86.7))
    
    if features['high_momentum']:
        matches.append(('Strong Record Home', 72.7, 90.5))
    
    if features['is_home_dog']:
        matches.append(('Home Underdog', 47.4, 77.2))
    
    if not matches:
        return None
    
    # Best match
    best = max(matches, key=lambda x: x[1])
    
    return {
        'matchup': f"{away} @ {home}",
        'home_team': home,
        'away_team': away,
        'spread': spread,
        'total': game.get('total_line'),
        'records': f"{away_rec} @ {home_rec}",
        'status': game['status'],
        'date': game['date'],
        'recommendations': [
            {
                'bet_type': 'SPREAD',
                'bet': f"HOME {home} {spread:+.1f}",
                'expected_roi': f"{best[1]:.1f}%",
                'win_rate': f"{best[2]:.1f}%",
                'pattern': best[0],
                'confidence': 'HIGH' if best[1] > 60 else 'MEDIUM',
                'units': 2 if best[1] > 70 else 1,
            }
        ],
        'matching_patterns': [{'pattern': p[0], 'roi': p[1], 'win_rate': p[2]} for p in matches],
    }

def main():
    print("="*70)
    print(f"🏈 NFL LIVE BETTING ANALYZER - {datetime.now().strftime('%A, %B %d, %Y %I:%M %p')}")
    print("="*70)
    
    # Fetch live games
    games = fetch_upcoming_nfl_games_espn()
    
    if not games:
        print("\n✗ No games found")
        return 1
    
    # Filter to upcoming/live only
    active_games = [g for g in games if g['is_upcoming'] or g['is_live']]
    
    print(f"\n📊 Today's Schedule:")
    print(f"  Total games: {len(games)}")
    print(f"  Upcoming/Live: {len(active_games)}")
    print(f"  Final: {len([g for g in games if g['is_final']])}")
    
    if not active_games:
        print("\n⚠ No upcoming games today")
        print("   (All games may have finished or none scheduled)")
        print("\n   Showing recent completed games for analysis...")
        active_games = games[:6]  # Show recent for demo
    
    # Load context
    print("\n📚 Loading historical context...")
    team_records, matchup_history = load_historical_context()
    print(f"  ✓ Team records loaded")
    
    # Load patterns
    patterns_file = Path(__file__).parent.parent / "data" / "domains" / "nfl_betting_patterns_FIXED.json"
    with open(patterns_file) as f:
        patterns = json.load(f)
    
    # Analyze each game
    print(f"\n🔄 Analyzing {len(active_games)} games...")
    
    opportunities = []
    for game in active_games:
        analysis = analyze_live_game(game, team_records, matchup_history, patterns)
        if analysis:
            opportunities.append(analysis)
    
    print(f"✓ Found {len(opportunities)} betting opportunities")
    
    # Display results
    if not opportunities:
        print("\n⚠ No betting opportunities match our patterns today")
        return 0
    
    print("\n" + "="*70)
    print(f"💰 BETTING OPPORTUNITIES - {len(opportunities)} FLAGGED")
    print("="*70)
    
    for i, opp in enumerate(opportunities, 1):
        print(f"\n{i}. {opp['matchup']}")
        print(f"   Spread: {opp['spread']:+.1f} | Records: {opp['records']}")
        print(f"   Status: {opp['status']}")
        
        for rec in opp['recommendations']:
            print(f"\n   💰 {rec['bet_type']}: {rec['bet']}")
            print(f"      ├─ Confidence: {rec['confidence']}")
            print(f"      ├─ Win Rate: {rec['win_rate']} (historical)")
            print(f"      ├─ Expected ROI: {rec['expected_roi']}")
            print(f"      ├─ Pattern: {rec['pattern']}")
            print(f"      └─ Recommended: {rec['units']} units")
    
    # Parlays
    high_conf = [opp for opp in opportunities 
                 if any(r['confidence'] == 'HIGH' for r in opp['recommendations'])]
    
    if len(high_conf) >= 2:
        print("\n" + "="*70)
        print("🎰 SUGGESTED PARLAYS")
        print("="*70)
        
        print(f"\n2-LEG PARLAY (High Confidence):")
        for opp in high_conf[:2]:
            rec = next(r for r in opp['recommendations'] if r['confidence'] == 'HIGH')
            print(f"  ✓ {rec['bet']}")
        print(f"  Payout: +264 (2.64:1) | Bet: 1u")
        
        if len(high_conf) >= 3:
            print(f"\n3-LEG PARLAY (High Confidence):")
            for opp in high_conf[:3]:
                rec = next(r for r in opp['recommendations'] if r['confidence'] == 'HIGH')
                print(f"  ✓ {rec['bet']}")
            print(f"  Payout: +596 (5.96:1) | Bet: 0.5u")
    
    print("\n" + "="*70)
    
    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'games_analyzed': len(active_games),
        'opportunities': opportunities,
    }
    
    output_file = Path(__file__).parent.parent / "data" / "domains" / "nfl_today_bets.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✓ Saved to: {output_file.name}")
    print("="*70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

