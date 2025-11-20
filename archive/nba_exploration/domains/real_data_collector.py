"""
REAL NBA Data Collector - Integrates with actual NBA data sources

Collects genuine NBA game data, team information, and narratives
from official sources and sports media.
"""

from typing import List, Dict, Any, Optional
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path
import time

try:
    from nba_api.stats.endpoints import leaguegamefinder, teamgamelog, scoreboardv2
    from nba_api.stats.static import teams as nba_teams
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False
    print("Warning: nba_api not installed. Install with: pip install nba_api")

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("Warning: beautifulsoup4 not installed. Install with: pip install beautifulsoup4")


class RealNBADataCollector:
    """
    Collects REAL NBA data from official sources.
    
    Data sources:
    - NBA Stats API (official stats, games, scores)
    - Basketball Reference (team info, narratives)
    - ESPN (game previews, team descriptions)
    - The Odds API (betting lines) - optional, requires API key
    """
    
    def __init__(self, odds_api_key: Optional[str] = None):
        """
        Initialize real NBA data collector.
        
        Parameters
        ----------
        odds_api_key : str, optional
            API key for The Odds API (for betting lines)
            Get free key at: https://the-odds-api.com/
        """
        if not NBA_API_AVAILABLE:
            raise ImportError("nba_api required. Install with: pip install nba_api")
        
        self.odds_api_key = odds_api_key
        self.teams = self._load_nba_teams()
        self.team_narratives_cache = {}
    
    def _load_nba_teams(self) -> Dict[str, Dict]:
        """Load real NBA teams from official API."""
        all_teams = nba_teams.get_teams()
        
        teams_dict = {}
        for team in all_teams:
            teams_dict[team['abbreviation']] = {
                'id': team['id'],
                'name': team['full_name'],
                'abbreviation': team['abbreviation'],
                'city': team['city'],
                'nickname': team['nickname'],
                'year_founded': team['year_founded']
            }
        
        print(f"✅ Loaded {len(teams_dict)} NBA teams")
        return teams_dict
    
    def fetch_season_games(self, season_year: str) -> List[Dict]:
        """
        Fetch real NBA games for a season.
        
        Parameters
        ----------
        season_year : str
            Season in format '2023-24'
        
        Returns
        -------
        games : list of dict
            Real game data with scores, dates, teams
        """
        print(f"Fetching games for {season_year} season...")
        
        try:
            # Use NBA API to get games
            gamefinder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season_year,
                league_id_nullable='00'  # NBA
            )
            
            games_df = gamefinder.get_data_frames()[0]
            
            # Process games
            games = []
            processed_game_ids = set()
            
            for _, row in games_df.iterrows():
                game_id = row['GAME_ID']
                
                # Each game appears twice (once per team), only process once
                if game_id in processed_game_ids:
                    continue
                processed_game_ids.add(game_id)
                
                game_data = {
                    'game_id': game_id,
                    'season': season_year,
                    'date': row['GAME_DATE'],
                    'team_id': row['TEAM_ID'],
                    'team_abbreviation': row['TEAM_ABBREVIATION'],
                    'team_name': row['TEAM_NAME'],
                    'matchup': row['MATCHUP'],
                    'wl': row['WL'],  # W or L
                    'points': int(row['PTS']),
                    'plus_minus': int(row['PLUS_MINUS']) if row['PLUS_MINUS'] else 0
                }
                
                games.append(game_data)
                
                # Rate limit
                time.sleep(0.6)  # NBA API rate limit
            
            print(f"✅ Fetched {len(games)} games for {season_year}")
            return games
        
        except Exception as e:
            print(f"Error fetching games: {e}")
            return []
    
    def fetch_team_description(self, team_abbr: str) -> str:
        """
        Fetch real team description/narrative from Basketball Reference.
        
        Parameters
        ----------
        team_abbr : str
            Team abbreviation (e.g., 'LAL')
        
        Returns
        -------
        narrative : str
            Team description and narrative
        """
        if not BS4_AVAILABLE:
            return self._get_fallback_narrative(team_abbr)
        
        # Check cache
        if team_abbr in self.team_narratives_cache:
            return self.team_narratives_cache[team_abbr]
        
        try:
            # Basketball Reference URL
            team_code = team_abbr.lower()
            url = f"https://www.basketball-reference.com/teams/{team_code}/2024.html"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Try to find team description
                # Basketball Reference has meta descriptions
                meta_desc = soup.find('meta', {'name': 'description'})
                if meta_desc and meta_desc.get('content'):
                    narrative = meta_desc['content']
                else:
                    # Fallback: construct from team info
                    narrative = self._get_fallback_narrative(team_abbr)
                
                self.team_narratives_cache[team_abbr] = narrative
                time.sleep(3)  # Be respectful to Basketball Reference
                return narrative
            else:
                return self._get_fallback_narrative(team_abbr)
        
        except Exception as e:
            print(f"Warning: Could not fetch narrative for {team_abbr}: {e}")
            return self._get_fallback_narrative(team_abbr)
    
    def _get_fallback_narrative(self, team_abbr: str) -> str:
        """Generate fallback narrative from known team info."""
        if team_abbr not in self.teams:
            return f"{team_abbr} is an NBA team competing at the highest level of professional basketball."
        
        team = self.teams[team_abbr]
        return f"The {team['name']} are a professional basketball team based in {team['city']}, competing in the NBA since {team['year_founded']}. Known for their competitive spirit and dedication to excellence."
    
    def fetch_betting_lines(self, date: str = None) -> Dict[str, Dict]:
        """
        Fetch real betting lines from The Odds API.
        
        Parameters
        ----------
        date : str, optional
            Date in ISO format (YYYY-MM-DD)
            If None, fetches today's games
        
        Returns
        -------
        odds : dict
            Betting lines by game
        """
        if not self.odds_api_key:
            print("Warning: No Odds API key provided. Betting lines unavailable.")
            return {}
        
        try:
            url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"
            params = {
                'apiKey': self.odds_api_key,
                'regions': 'us',
                'markets': 'spreads,totals',
                'oddsFormat': 'american'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                odds_data = response.json()
                print(f"✅ Fetched betting lines for {len(odds_data)} games")
                return self._process_odds_data(odds_data)
            else:
                print(f"Warning: Odds API returned status {response.status_code}")
                return {}
        
        except Exception as e:
            print(f"Warning: Could not fetch betting lines: {e}")
            return {}
    
    def _process_odds_data(self, odds_data: List[Dict]) -> Dict[str, Dict]:
        """Process raw odds data into usable format."""
        processed = {}
        
        for game in odds_data:
            game_key = f"{game['home_team']}_{game['away_team']}"
            
            # Extract spread (point spread)
            spread = None
            if game.get('bookmakers'):
                for bookmaker in game['bookmakers']:
                    if bookmaker['markets']:
                        for market in bookmaker['markets']:
                            if market['key'] == 'spreads':
                                for outcome in market['outcomes']:
                                    if outcome['name'] == game['home_team']:
                                        spread = outcome['point']
                                        break
            
            processed[game_key] = {
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'spread': spread,
                'commence_time': game.get('commence_time')
            }
        
        return processed
    
    def fetch_espn_game_preview(self, game_id: str) -> Optional[str]:
        """
        Fetch game preview narrative from ESPN.
        
        Parameters
        ----------
        game_id : str
            ESPN game ID
        
        Returns
        -------
        preview : str or None
            Game preview text with narrative framing
        """
        if not BS4_AVAILABLE:
            return None
        
        try:
            url = f"https://www.espn.com/nba/game/_/gameId/{game_id}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for game preview/recap text
                preview_section = soup.find('div', class_='game-story')
                if preview_section:
                    preview_text = preview_section.get_text(strip=True, separator=' ')
                    return preview_text[:1000]  # First 1000 chars
            
            return None
        
        except Exception as e:
            print(f"Warning: Could not fetch ESPN preview: {e}")
            return None
    
    def create_complete_dataset(self, seasons: List[str], save_path: Optional[str] = None) -> List[Dict]:
        """
        Create complete dataset with real data.
        
        Parameters
        ----------
        seasons : list of str
            Seasons to collect (e.g., ['2022-23', '2023-24'])
        save_path : str, optional
            Path to save collected data
        
        Returns
        -------
        dataset : list of dict
            Complete games with narratives
        """
        print(f"\n{'='*70}")
        print("COLLECTING REAL NBA DATA")
        print(f"{'='*70}\n")
        
        complete_dataset = []
        
        for season in seasons:
            print(f"\nProcessing {season} season...")
            
            # Get games
            games = self.fetch_season_games(season)
            
            # Add narratives for each team
            teams_in_season = set()
            for game in games:
                teams_in_season.add(game['team_abbreviation'])
            
            print(f"Fetching narratives for {len(teams_in_season)} teams...")
            for team_abbr in teams_in_season:
                narrative = self.fetch_team_description(team_abbr)
                # Cache for reuse
                self.team_narratives_cache[team_abbr] = narrative
            
            # Enrich games with narratives
            for game in games:
                team_abbr = game['team_abbreviation']
                game['team_narrative'] = self.team_narratives_cache.get(
                    team_abbr,
                    self._get_fallback_narrative(team_abbr)
                )
                complete_dataset.append(game)
            
            print(f"✅ Completed {season}: {len(games)} games")
        
        print(f"\n{'='*70}")
        print(f"TOTAL COLLECTED: {len(complete_dataset)} games")
        print(f"{'='*70}\n")
        
        # Save if requested
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(complete_dataset, f, indent=2)
            print(f"✅ Data saved to {save_path}")
        
        return complete_dataset
    
    def get_todays_games(self) -> List[Dict]:
        """
        Get today's NBA games with betting lines.
        
        Returns
        -------
        games : list of dict
            Today's games with teams, time, and betting info
        """
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            scoreboard = scoreboardv2.ScoreboardV2(game_date=today)
            games_df = scoreboard.get_data_frames()[0]
            
            todays_games = []
            for _, row in games_df.iterrows():
                game_info = {
                    'game_id': row['GAME_ID'],
                    'game_date': row['GAME_DATE_EST'],
                    'home_team': row['HOME_TEAM_ID'],
                    'away_team': row['VISITOR_TEAM_ID'],
                    'game_status_text': row['GAME_STATUS_TEXT']
                }
                todays_games.append(game_info)
            
            print(f"✅ Found {len(todays_games)} games today")
            return todays_games
        
        except Exception as e:
            print(f"Error fetching today's games: {e}")
            return []


class HybridNBACollector:
    """
    Hybrid collector: Real NBA game data + enhanced narrative generation.
    
    Uses real NBA API for games/scores but generates rich narratives
    based on actual team performance and statistics.
    """
    
    def __init__(self):
        if not NBA_API_AVAILABLE:
            raise ImportError("nba_api required for hybrid collector")
        
        self.real_collector = RealNBADataCollector()
        self.teams = self.real_collector.teams
    
    def fetch_games_with_narratives(self, season: str) -> List[Dict]:
        """
        Fetch real games and generate narrative-rich descriptions.
        
        Parameters
        ----------
        season : str
            Season (e.g., '2023-24')
        
        Returns
        -------
        games : list of dict
            Real games with generated narratives
        """
        print(f"\nFetching REAL NBA games for {season}...")
        
        # Get real games
        real_games = self.real_collector.fetch_season_games(season)
        
        # Enrich with narratives
        enriched_games = []
        for game in real_games:
            # Generate narrative based on real outcome
            team_abbr = game['team_abbreviation']
            won = (game['wl'] == 'W')
            
            # Create narrative incorporating real performance
            narrative = self._create_performance_narrative(
                team_abbr,
                won,
                game['points'],
                game['plus_minus']
            )
            
            game['narrative'] = narrative
            enriched_games.append(game)
        
        return enriched_games
    
    def _create_performance_narrative(self, team_abbr: str, won: bool, points: int, plus_minus: int) -> str:
        """
        Generate narrative based on real performance.
        
        Incorporates actual game outcome to create realistic narratives.
        """
        team_info = self.teams.get(team_abbr, {})
        team_name = team_info.get('name', team_abbr)
        
        # Performance-based narrative
        if won:
            if plus_minus > 15:
                performance = "dominated with a commanding victory"
                energy = "championship-level intensity"
            elif plus_minus > 5:
                performance = "secured a solid win"
                energy = "competitive excellence"
            else:
                performance = "fought hard for a close victory"
                energy = "resilient determination"
        else:
            if plus_minus < -15:
                performance = "struggled significantly in a tough loss"
                energy = "rebuilding momentum"
            elif plus_minus < -5:
                performance = "fell short in a competitive game"
                energy = "learning and adapting"
            else:
                performance = "narrowly lost a hard-fought battle"
                energy = "maintaining competitive spirit"
        
        # Build narrative
        narrative = f"""The {team_name} {performance}, scoring {points} points. 
        The team demonstrated {energy} and continues to compete at the highest level. 
        {'Their winning mentality and championship aspirations drive every game.' if won else 'They look to bounce back stronger and return to their winning ways.'}
        """
        
        return narrative.strip()
    
    def split_train_test_temporal(self, games: List[Dict], test_every_nth: int = 10) -> Tuple[List, List]:
        """
        Split games with proper temporal validation.
        
        Parameters
        ----------
        games : list
            All games
        test_every_nth : int
            Test on every Nth season
        
        Returns
        -------
        train_games, test_games : tuple
            Training and testing splits
        """
        # Group by season
        seasons = {}
        for game in games:
            season = game['season']
            if season not in seasons:
                seasons[season] = []
            seasons[season].append(game)
        
        # Sort seasons chronologically
        sorted_seasons = sorted(seasons.keys())
        
        train_games = []
        test_games = []
        
        for idx, season in enumerate(sorted_seasons):
            if (idx + 1) % test_every_nth == 0:
                # Test season
                test_games.extend(seasons[season])
                print(f"  Test season: {season} ({len(seasons[season])} games)")
            else:
                # Train season
                train_games.extend(seasons[season])
        
        print(f"\n✅ Split complete:")
        print(f"   Training: {len(train_games)} games ({len(train_games)//len(sorted_seasons)} seasons)")
        print(f"   Testing: {len(test_games)} games ({len(test_games)//len(sorted_seasons)} seasons)")
        
        return train_games, test_games

