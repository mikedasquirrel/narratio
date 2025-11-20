"""
MLB Game Data Collector - Archetype Betting System
Fetches current season games with odds, rosters, and context for narrative analysis

Author: Narrative Optimization Framework
Date: November 2024
"""

import requests
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import time

class MLBGameCollector:
    """Collect MLB games with betting odds and roster data"""
    
    def __init__(self):
        self.base_url = "https://statsapi.mlb.com/api/v1"
        self.odds_api_key = None  # Set via environment or config
        self.data_dir = Path(__file__).parent / 'data'
        self.data_dir.mkdir(exist_ok=True)
        
        # MLB team mappings
        self.teams = {
            'NYY': {'name': 'New York Yankees', 'id': 147, 'division': 'AL East'},
            'BOS': {'name': 'Boston Red Sox', 'id': 111, 'division': 'AL East'},
            'TB': {'name': 'Tampa Bay Rays', 'id': 139, 'division': 'AL East'},
            'TOR': {'name': 'Toronto Blue Jays', 'id': 141, 'division': 'AL East'},
            'BAL': {'name': 'Baltimore Orioles', 'id': 110, 'division': 'AL East'},
            'CLE': {'name': 'Cleveland Guardians', 'id': 114, 'division': 'AL Central'},
            'CWS': {'name': 'Chicago White Sox', 'id': 145, 'division': 'AL Central'},
            'DET': {'name': 'Detroit Tigers', 'id': 116, 'division': 'AL Central'},
            'KC': {'name': 'Kansas City Royals', 'id': 118, 'division': 'AL Central'},
            'MIN': {'name': 'Minnesota Twins', 'id': 142, 'division': 'AL Central'},
            'HOU': {'name': 'Houston Astros', 'id': 117, 'division': 'AL West'},
            'LAA': {'name': 'Los Angeles Angels', 'id': 108, 'division': 'AL West'},
            'OAK': {'name': 'Oakland Athletics', 'id': 133, 'division': 'AL West'},
            'SEA': {'name': 'Seattle Mariners', 'id': 136, 'division': 'AL West'},
            'TEX': {'name': 'Texas Rangers', 'id': 140, 'division': 'AL West'},
            'ATL': {'name': 'Atlanta Braves', 'id': 144, 'division': 'NL East'},
            'MIA': {'name': 'Miami Marlins', 'id': 146, 'division': 'NL East'},
            'NYM': {'name': 'New York Mets', 'id': 121, 'division': 'NL East'},
            'PHI': {'name': 'Philadelphia Phillies', 'id': 143, 'division': 'NL East'},
            'WSH': {'name': 'Washington Nationals', 'id': 120, 'division': 'NL East'},
            'CHC': {'name': 'Chicago Cubs', 'id': 112, 'division': 'NL Central'},
            'CIN': {'name': 'Cincinnati Reds', 'id': 113, 'division': 'NL Central'},
            'MIL': {'name': 'Milwaukee Brewers', 'id': 158, 'division': 'NL Central'},
            'PIT': {'name': 'Pittsburgh Pirates', 'id': 134, 'division': 'NL Central'},
            'STL': {'name': 'St. Louis Cardinals', 'id': 138, 'division': 'NL Central'},
            'ARI': {'name': 'Arizona Diamondbacks', 'id': 109, 'division': 'NL West'},
            'COL': {'name': 'Colorado Rockies', 'id': 115, 'division': 'NL West'},
            'LAD': {'name': 'Los Angeles Dodgers', 'id': 119, 'division': 'NL West'},
            'SD': {'name': 'San Diego Padres', 'id': 135, 'division': 'NL West'},
            'SF': {'name': 'San Francisco Giants', 'id': 137, 'division': 'NL West'}
        }
        
        # Historic stadiums
        self.historic_stadiums = {
            'Wrigley Field': 'CHC',
            'Fenway Park': 'BOS',
            'Dodger Stadium': 'LAD',
            'Yankee Stadium': 'NYY',
            'Oracle Park': 'SF'
        }
        
        # Major rivalries
        self.rivalries = [
            ('NYY', 'BOS'),  # Yankees-Red Sox
            ('LAD', 'SF'),   # Dodgers-Giants
            ('CHC', 'STL'),  # Cubs-Cardinals
            ('HOU', 'TEX'),  # Astros-Rangers (Lone Star Series)
            ('NYM', 'PHI'),  # Mets-Phillies
            ('BAL', 'WSH'),  # Beltway Series
            ('OAK', 'SF'),   # Bay Bridge Series
            ('CWS', 'CHC')   # Chicago Crosstown
        ]
    
    def fetch_games(self, start_date: str, end_date: str, season: int = 2024) -> List[Dict]:
        """
        Fetch games for date range
        
        Args:
            start_date: YYYY-MM-DD format
            end_date: YYYY-MM-DD format
            season: Season year
            
        Returns:
            List of game dictionaries with full context
        """
        games = []
        
        try:
            # Get schedule
            url = f"{self.base_url}/schedule"
            params = {
                'sportId': 1,
                'startDate': start_date,
                'endDate': end_date,
                'hydrate': 'team,linescore,probablePitcher'
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            for date_data in data.get('dates', []):
                for game in date_data.get('games', []):
                    game_info = self._extract_game_info(game, season)
                    if game_info:
                        games.append(game_info)
                        
            print(f"Fetched {len(games)} games from {start_date} to {end_date}")
            
        except Exception as e:
            print(f"Error fetching games: {e}")
            
        return games
    
    def _extract_game_info(self, game: Dict, season: int) -> Optional[Dict]:
        """Extract relevant game information"""
        try:
            teams = game.get('teams', {})
            away_team = teams.get('away', {}).get('team', {})
            home_team = teams.get('home', {}).get('team', {})
            
            # Get team codes
            away_code = self._get_team_code(away_team.get('id'))
            home_code = self._get_team_code(home_team.get('id'))
            
            if not away_code or not home_code:
                return None
            
            # Extract basic info
            game_info = {
                'game_id': game.get('gamePk'),
                'game_date': game.get('gameDate'),
                'season': season,
                'away_team': away_code,
                'home_team': home_code,
                'away_team_name': away_team.get('name'),
                'home_team_name': home_team.get('name'),
                'venue': game.get('venue', {}).get('name'),
                'status': game.get('status', {}).get('detailedState')
            }
            
            # Extract scores if final
            if game.get('status', {}).get('codedGameState') in ['F', 'O']:
                game_info['away_score'] = teams.get('away', {}).get('score')
                game_info['home_score'] = teams.get('home', {}).get('score')
                game_info['home_wins'] = game_info['home_score'] > game_info['away_score']
            
            # Add context
            game_info['is_rivalry'] = self._is_rivalry(away_code, home_code)
            game_info['is_historic_stadium'] = game_info['venue'] in self.historic_stadiums
            game_info['month'] = datetime.fromisoformat(game.get('gameDate').replace('Z', '+00:00')).month
            
            # Get probable pitchers
            away_pitcher = teams.get('away', {}).get('probablePitcher', {})
            home_pitcher = teams.get('home', {}).get('probablePitcher', {})
            
            if away_pitcher:
                game_info['away_pitcher'] = away_pitcher.get('fullName')
                game_info['away_pitcher_id'] = away_pitcher.get('id')
            
            if home_pitcher:
                game_info['home_pitcher'] = home_pitcher.get('fullName')
                game_info['home_pitcher_id'] = home_pitcher.get('id')
            
            return game_info
            
        except Exception as e:
            print(f"Error extracting game info: {e}")
            return None
    
    def _get_team_code(self, team_id: int) -> Optional[str]:
        """Get team code from team ID"""
        for code, info in self.teams.items():
            if info['id'] == team_id:
                return code
        return None
    
    def _is_rivalry(self, team1: str, team2: str) -> bool:
        """Check if game is a rivalry"""
        return (team1, team2) in self.rivalries or (team2, team1) in self.rivalries
    
    def fetch_team_roster(self, team_code: str, season: int = 2024) -> List[Dict]:
        """
        Fetch team roster
        
        Args:
            team_code: Team abbreviation
            season: Season year
            
        Returns:
            List of player dictionaries
        """
        roster = []
        
        try:
            team_id = self.teams.get(team_code, {}).get('id')
            if not team_id:
                return roster
            
            url = f"{self.base_url}/teams/{team_id}/roster"
            params = {'season': season}
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            for player in data.get('roster', []):
                person = player.get('person', {})
                position = player.get('position', {})
                
                roster.append({
                    'player_id': person.get('id'),
                    'full_name': person.get('fullName'),
                    'position': position.get('name'),
                    'position_code': position.get('code'),
                    'jersey_number': player.get('jerseyNumber')
                })
                
        except Exception as e:
            print(f"Error fetching roster for {team_code}: {e}")
            
        return roster
    
    def fetch_team_stats(self, team_code: str, season: int = 2024) -> Dict:
        """
        Fetch team season statistics
        
        Args:
            team_code: Team abbreviation
            season: Season year
            
        Returns:
            Dictionary of team stats
        """
        stats = {}
        
        try:
            team_id = self.teams.get(team_code, {}).get('id')
            if not team_id:
                return stats
            
            url = f"{self.base_url}/teams/{team_id}"
            params = {
                'season': season,
                'hydrate': 'standings'
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            team_data = data.get('teams', [{}])[0]
            record = team_data.get('record', {}).get('leagueRecord', {})
            
            stats = {
                'wins': record.get('wins', 0),
                'losses': record.get('losses', 0),
                'win_pct': record.get('pct', '0.500'),
                'division': self.teams[team_code]['division']
            }
            
        except Exception as e:
            print(f"Error fetching stats for {team_code}: {e}")
            
        return stats
    
    def collect_season_games(self, season: int = 2024) -> List[Dict]:
        """
        Collect all games for a season
        
        Args:
            season: Season year
            
        Returns:
            List of all games with full context
        """
        all_games = []
        
        # Season runs April-September typically
        start_date = f"{season}-03-28"
        end_date = f"{season}-10-01"
        
        # Fetch in monthly chunks to avoid overwhelming API
        current = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        while current < end:
            chunk_start = current.strftime('%Y-%m-%d')
            chunk_end = (current + timedelta(days=30)).strftime('%Y-%m-%d')
            
            games = self.fetch_games(chunk_start, chunk_end, season)
            all_games.extend(games)
            
            current += timedelta(days=30)
            time.sleep(0.5)  # Be nice to the API
        
        # Save to file
        output_file = self.data_dir / f'mlb_games_{season}.json'
        with open(output_file, 'w') as f:
            json.dump(all_games, f, indent=2)
        
        print(f"Collected {len(all_games)} games for {season} season")
        print(f"Saved to {output_file}")
        
        return all_games
    
    def get_todays_games(self) -> List[Dict]:
        """Get today's games with context"""
        today = datetime.now().strftime('%Y-%m-%d')
        return self.fetch_games(today, today)


if __name__ == '__main__':
    # Example usage
    collector = MLBGameCollector()
    
    # Collect 2024 season
    games_2024 = collector.collect_season_games(2024)
    
    print(f"\nTotal games collected: {len(games_2024)}")
    print(f"Rivalry games: {sum(1 for g in games_2024 if g.get('is_rivalry'))}")
    print(f"Historic stadium games: {sum(1 for g in games_2024 if g.get('is_historic_stadium'))}")

