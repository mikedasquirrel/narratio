"""
The Odds API Configuration

Official paid API key for comprehensive odds coverage.

API Documentation: https://the-odds-api.com/liveapi/guides/v4/

Coverage:
- Pre-game odds (moneyline, spread, totals)
- Live in-game odds
- Player props (points, rebounds, assists, etc.)
- Multiple sportsbooks (best line shopping)

Sports Available:
- NHL (americanfootball_nhl)
- NBA (basketball_nba)
- NFL (americanfootball_nfl)
- MLB (baseball_mlb)
- Soccer, Tennis, Golf, UFC, Boxing, etc.

Author: Odds Integration System
Date: November 19, 2025
"""

import os

# Official API Key
ODDS_API_KEY = "2e330948334c9505ed5542a82fcfa3b9"

# API Endpoints
BASE_URL = "https://api.the-odds-api.com/v4"

# Sports keys
SPORTS = {
    'nhl': 'icehockey_nhl',
    'nba': 'basketball_nba',
    'nfl': 'americanfootball_nfl',
    'mlb': 'baseball_mlb',
    'ncaab': 'basketball_ncaab',
    'ncaaf': 'americanfootball_ncaaf',
    'soccer_epl': 'soccer_epl',
    'soccer_uefa': 'soccer_uefa_champs_league',
    'tennis': 'tennis_atp',
    'ufc': 'mma_mixed_martial_arts',
    'boxing': 'boxing_boxing',
    'golf': 'golf_pga',
}

# Markets available
MARKETS = {
    'h2h': 'Moneyline (head to head)',
    'spreads': 'Point spreads',
    'totals': 'Over/under totals',
    'player_points': 'Player points props',
    'player_rebounds': 'Player rebounds props',
    'player_assists': 'Player assists props',
    'player_threes': 'Player 3-pointers props',
}

# Regions (for best odds)
REGIONS = ['us', 'us2', 'uk', 'eu', 'au']

# Sportsbooks to track
BOOKMAKERS = [
    'draftkings',
    'fanduel',
    'betmgm',
    'caesars',
    'pointsbet',
    'barstool',
    'unibet_us',
]

# Rate limits (paid tier)
RATE_LIMIT = {
    'requests_per_month': 50000,  # Check your plan
    'requests_per_second': 10,
}

# Cache settings
CACHE_DURATION = {
    'pre_game': 300,  # 5 minutes
    'live': 30,       # 30 seconds
    'props': 180,     # 3 minutes
}

