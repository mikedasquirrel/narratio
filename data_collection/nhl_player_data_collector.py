"""
NHL Player Data Collector

Collects player-level data for prop betting:
- Player game logs (goals, assists, shots, saves)
- Season statistics and recent form
- Lineup status and projected ice time
- Historical matchup performance
- Injury reports and availability

Data Sources:
- NHL API: Player stats and game logs
- Daily lineups: Starting goalies and forward lines
- Historical performance vs opponents

Author: Prop Betting System
Date: November 20, 2024
"""

import json
import time
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np


class NHLPlayerDataCollector:
    """
    Collect player-level data for prop betting.
    
    Philosophy:
    - Player narratives drive prop outcomes
    - Recent form matters more than season averages
    - Matchups create exploitable edges
    - Ice time and lineups are critical
    """
    
    def __init__(self, cache_dir: Path = None):
        """Initialize collector with caching"""
        self.cache_dir = cache_dir or Path("data/nhl_player_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # NHL API endpoints
        self.base_url = "https://api-web.nhle.com/v1"
        self.stats_api = "https://statsapi.web.nhl.com/api/v1"
        
        # Rate limiting
        self.last_request = 0
        self.min_delay = 0.5  # seconds between requests
        
    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
        self.last_request = time.time()
        
    def _get_json(self, url: str, params: Dict = None) -> Optional[Dict]:
        """Fetch JSON with rate limiting and error handling"""
        self._rate_limit()
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None
            
    def get_player_season_stats(self, player_id: int, season: str = "20242025") -> Dict:
        """
        Get player season statistics.
        
        Parameters
        ----------
        player_id : int
            NHL player ID
        season : str
            Season in YYYYYYY format (e.g., "20242025")
            
        Returns
        -------
        stats : dict
            Season statistics including goals, assists, shots, etc.
        """
        # Try new API first
        url = f"{self.base_url}/player/{player_id}/landing"
        data = self._get_json(url)
        
        if data and 'featuredStats' in data:
            stats = data['featuredStats']['regularSeason']['subSeason']
            
            first = data.get('firstName', '')
            last = data.get('lastName', '')
            if isinstance(first, dict):
                first = first.get('default', '')
            if isinstance(last, dict):
                last = last.get('default', '')
            
            # Extract key stats for props
            return {
                'player_id': player_id,
                'player_name': f"{first} {last}".strip(),
                'position': data.get('position', ''),
                'games_played': stats.get('gamesPlayed', 0),
                'goals': stats.get('goals', 0),
                'assists': stats.get('assists', 0),
                'points': stats.get('points', 0),
                'shots': stats.get('shots', 0),
                'shooting_pct': stats.get('shootingPctg', 0.0),
                'plus_minus': stats.get('plusMinus', 0),
                'pim': stats.get('penaltyMinutes', 0),
                'toi_per_game': stats.get('avgToi', "0:00"),
                'goals_per_game': stats.get('goals', 0) / max(stats.get('gamesPlayed', 1), 1),
                'assists_per_game': stats.get('assists', 0) / max(stats.get('gamesPlayed', 1), 1),
                'shots_per_game': stats.get('shots', 0) / max(stats.get('gamesPlayed', 1), 1),
            }
            
        # Fallback to old API
        url = f"{self.stats_api}/people/{player_id}/stats"
        params = {'stats': 'statsSingleSeason', 'season': season}
        data = self._get_json(url, params)
        
        if data and 'stats' in data:
            stats = data['stats'][0]['splits'][0]['stat'] if data['stats'] else {}
            return self._parse_player_stats(stats, player_id)
            
        return {}
        
    def get_player_game_logs(self, player_id: int, last_n_games: int = 10) -> List[Dict]:
        """
        Get player's recent game logs.
        
        Parameters
        ----------
        player_id : int
            NHL player ID
        last_n_games : int
            Number of recent games to fetch
            
        Returns
        -------
        games : list of dict
            Recent game performances
        """
        # Try new API
        url = f"{self.base_url}/player/{player_id}/game-log/now"
        data = self._get_json(url)
        
        if data and 'gameLog' in data:
            games = []
            for game in data['gameLog'][:last_n_games]:
                games.append({
                    'date': game.get('gameDate'),
                    'opponent': game.get('opponentAbbrev'),
                    'home_away': 'home' if game.get('homeRoadFlag') == 'H' else 'away',
                    'goals': game.get('goals', 0),
                    'assists': game.get('assists', 0),
                    'points': game.get('points', 0),
                    'shots': game.get('shots', 0),
                    'plus_minus': game.get('plusMinus', 0),
                    'pim': game.get('penaltyMinutes', 0),
                    'toi': game.get('toi', "0:00"),
                    'powerplay_goals': game.get('powerPlayGoals', 0),
                    'powerplay_points': game.get('powerPlayPoints', 0),
                    'game_winning_goals': game.get('gameWinningGoals', 0),
                })
            return games
            
        # Fallback to old API with date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)  # Get last 2 months
        
        url = f"{self.stats_api}/people/{player_id}/stats"
        params = {
            'stats': 'gameLog',
            'season': '20242025',
            'startDate': start_date.strftime('%Y-%m-%d'),
            'endDate': end_date.strftime('%Y-%m-%d')
        }
        
        data = self._get_json(url, params)
        if data and 'stats' in data and data['stats']:
            games = []
            splits = data['stats'][0].get('splits', [])[:last_n_games]
            
            for game in splits:
                stat = game['stat']
                games.append({
                    'date': game.get('date'),
                    'opponent': game['opponent']['abbreviation'],
                    'home_away': 'home' if game['isHome'] else 'away',
                    'goals': stat.get('goals', 0),
                    'assists': stat.get('assists', 0),
                    'points': stat.get('points', 0),
                    'shots': stat.get('shots', 0),
                    'plus_minus': stat.get('plusMinus', 0),
                    'pim': stat.get('penaltyMinutes', 0),
                    'toi': stat.get('timeOnIce', "0:00"),
                    'powerplay_goals': stat.get('powerPlayGoals', 0),
                    'powerplay_points': stat.get('powerPlayPoints', 0),
                })
            
            return games
            
        return []
        
    def get_goalie_game_logs(self, player_id: int, last_n_games: int = 10) -> List[Dict]:
        """
        Get goalie's recent game logs.
        
        Returns saves, goals against, save percentage per game.
        """
        url = f"{self.base_url}/player/{player_id}/game-log/now"
        data = self._get_json(url)
        
        if data and 'gameLog' in data:
            games = []
            for game in data['gameLog'][:last_n_games]:
                games.append({
                    'date': game.get('gameDate'),
                    'opponent': game.get('opponentAbbrev'),
                    'home_away': 'home' if game.get('homeRoadFlag') == 'H' else 'away',
                    'saves': game.get('saves', 0),
                    'shots_against': game.get('shotsAgainst', 0),
                    'goals_against': game.get('goalsAgainst', 0),
                    'save_pct': game.get('savePctg', 0.0),
                    'toi': game.get('toi', "0:00"),
                    'decision': game.get('decision', ''),  # W, L, OT, SO
                    'shutout': game.get('shutouts', 0) > 0,
                })
            return games
            
        return []
        
    def get_player_vs_team_stats(self, player_id: int, team_abbrev: str, 
                                 last_n_seasons: int = 3) -> Dict:
        """
        Get player's historical performance vs specific team.
        
        Returns
        -------
        stats : dict
            Career and recent stats vs team
        """
        # This would require game-by-game parsing
        # For now, return aggregated estimate
        game_logs = self.get_player_game_logs(player_id, last_n_games=50)
        
        vs_team_games = [g for g in game_logs if g['opponent'] == team_abbrev]
        
        if not vs_team_games:
            return {
                'games_played': 0,
                'avg_goals': 0.0,
                'avg_assists': 0.0,
                'avg_shots': 0.0,
                'avg_points': 0.0,
            }
            
        return {
            'games_played': len(vs_team_games),
            'avg_goals': np.mean([g['goals'] for g in vs_team_games]),
            'avg_assists': np.mean([g['assists'] for g in vs_team_games]),
            'avg_shots': np.mean([g['shots'] for g in vs_team_games]),
            'avg_points': np.mean([g['points'] for g in vs_team_games]),
            'last_5_goals': sum(g['goals'] for g in vs_team_games[:5]),
            'last_5_points': sum(g['points'] for g in vs_team_games[:5]),
        }
        
    def get_team_roster(self, team_abbrev: str) -> List[Dict]:
        """
        Get current team roster with player IDs.
        
        Returns
        -------
        roster : list of dict
            Players with IDs, names, positions
        """
        # Map team abbreviation to ID
        team_id = self._get_team_id(team_abbrev)
        if not team_id:
            return []
            
        url = f"{self.stats_api}/teams/{team_id}/roster"
        data = self._get_json(url)
        
        if data and 'roster' in data:
            roster = []
            for player in data['roster']:
                roster.append({
                    'player_id': player['person']['id'],
                    'player_name': player['person']['fullName'],
                    'position': player['position']['abbreviation'],
                    'jersey_number': player.get('jerseyNumber', ''),
                })
            return roster
            
        return []
        
    def get_starting_goalies(self, date: str = None) -> Dict[str, Dict]:
        """
        Get projected starting goalies for date.
        
        Returns
        -------
        starters : dict
            {team: {goalie_id, goalie_name, confirmed}}
        """
        # This would typically come from a specialized source
        # like DailyFaceoff or LeftWingLock
        # For now, return empty dict
        return {}
        
    def get_player_injuries(self) -> Dict[int, Dict]:
        """
        Get current injury report.
        
        Returns
        -------
        injuries : dict
            {player_id: {status, description, return_date}}
        """
        # Would need specialized injury feed
        # For now, return empty dict
        return {}
        
    def calculate_player_form(self, game_logs: List[Dict]) -> Dict:
        """
        Calculate recent form metrics from game logs.
        
        Returns
        -------
        form : dict
            Recent performance trends and streaks
        """
        if not game_logs:
            return {
                'last_5_avg_goals': 0.0,
                'last_5_avg_assists': 0.0,
                'last_5_avg_shots': 0.0,
                'trend': 'neutral',
                'hot_streak': False,
            }
            
        # Recent averages
        last_5 = game_logs[:5]
        last_10 = game_logs[:10]
        
        l5_goals = np.mean([g['goals'] for g in last_5])
        l10_goals = np.mean([g['goals'] for g in last_10])
        
        l5_points = np.mean([g['points'] for g in last_5])
        l10_points = np.mean([g['points'] for g in last_10])
        
        # Determine trend
        if l5_goals > l10_goals * 1.2:
            trend = 'hot'
        elif l5_goals < l10_goals * 0.8:
            trend = 'cold'
        else:
            trend = 'neutral'
            
        # Check for hot streak (3+ games with points)
        point_streak = 0
        for game in game_logs:
            if game['points'] > 0:
                point_streak += 1
            else:
                break
                
        return {
            'last_5_avg_goals': l5_goals,
            'last_5_avg_assists': np.mean([g['assists'] for g in last_5]),
            'last_5_avg_shots': np.mean([g['shots'] for g in last_5]),
            'last_10_avg_points': l10_points,
            'trend': trend,
            'hot_streak': point_streak >= 3,
            'point_streak_games': point_streak,
            'goals_last_5': sum(g['goals'] for g in last_5),
            'multi_point_games_l10': sum(1 for g in last_10 if g['points'] >= 2),
        }
        
    def _get_team_id(self, team_abbrev: str) -> Optional[int]:
        """Map team abbreviation to ID"""
        team_map = {
            'ANA': 24, 'ARI': 53, 'BOS': 6, 'BUF': 7, 'CGY': 20,
            'CAR': 12, 'CHI': 16, 'COL': 21, 'CBJ': 29, 'DAL': 25,
            'DET': 17, 'EDM': 22, 'FLA': 13, 'LAK': 26, 'MIN': 30,
            'MTL': 8, 'NSH': 18, 'NJD': 1, 'NYI': 2, 'NYR': 3,
            'OTT': 9, 'PHI': 4, 'PIT': 5, 'SJS': 28, 'SEA': 55,
            'STL': 19, 'TBL': 14, 'TOR': 10, 'VAN': 23, 'VGK': 54,
            'WSH': 15, 'WPG': 52
        }
        return team_map.get(team_abbrev)
        
    def _parse_player_stats(self, stats: Dict, player_id: int) -> Dict:
        """Parse stats dict into standardized format"""
        games = stats.get('games', 0)
        if games == 0:
            games = 1  # Avoid division by zero
            
        return {
            'player_id': player_id,
            'games_played': games,
            'goals': stats.get('goals', 0),
            'assists': stats.get('assists', 0),
            'points': stats.get('points', 0),
            'shots': stats.get('shots', 0),
            'shooting_pct': stats.get('shootingPct', 0.0),
            'plus_minus': stats.get('plusMinus', 0),
            'pim': stats.get('pim', 0),
            'toi_per_game': stats.get('timeOnIcePerGame', "0:00"),
            'goals_per_game': stats.get('goals', 0) / games,
            'assists_per_game': stats.get('assists', 0) / games,
            'shots_per_game': stats.get('shots', 0) / games,
            'powerplay_goals': stats.get('powerPlayGoals', 0),
            'powerplay_points': stats.get('powerPlayPoints', 0),
        }
        
    def collect_players_for_game(self, home_team: str, away_team: str, 
                                 top_n_players: int = 8) -> Dict:
        """
        Collect top players for prop betting from both teams.
        
        Parameters
        ----------
        home_team : str
            Home team abbreviation
        away_team : str
            Away team abbreviation
        top_n_players : int
            Number of top players per team to collect
            
        Returns
        -------
        players : dict
            {team: [player_data]}
        """
        players = {'home': [], 'away': []}
        
        for team, side in [(home_team, 'home'), (away_team, 'away')]:
            roster = self.get_team_roster(team)
            
            # Filter to skaters only (exclude goalies for now)
            skaters = [p for p in roster if p['position'] not in ['G']]
            
            # Get season stats for all players
            player_stats = []
            for player in skaters[:20]:  # Top 20 to filter from
                stats = self.get_player_season_stats(player['player_id'])
                if stats and stats.get('games_played', 0) >= 10:
                    player_stats.append(stats)
                    
            # Sort by points and take top N
            player_stats.sort(key=lambda x: x.get('points', 0), reverse=True)
            
            for player in player_stats[:top_n_players]:
                # Get recent game logs
                game_logs = self.get_player_game_logs(player['player_id'], last_n_games=10)
                
                # Calculate form
                form = self.calculate_player_form(game_logs)
                
                # Get matchup history
                opponent = away_team if side == 'home' else home_team
                vs_stats = self.get_player_vs_team_stats(player['player_id'], opponent)
                
                players[side].append({
                    'player_id': player['player_id'],
                    'player_name': player['player_name'],
                    'position': player['position'],
                    'season_stats': player,
                    'recent_form': form,
                    'vs_opponent': vs_stats,
                    'last_5_games': game_logs[:5],
                })
                
        return players
        
    def save_player_data(self, players: Dict, game_id: str):
        """Save collected player data for game"""
        output_file = self.cache_dir / f"players_{game_id}.json"
        
        with open(output_file, 'w') as f:
            json.dump(players, f, indent=2)
            
        print(f"Saved player data to {output_file}")
        
    def load_player_data(self, game_id: str) -> Optional[Dict]:
        """Load cached player data"""
        input_file = self.cache_dir / f"players_{game_id}.json"
        
        if input_file.exists():
            with open(input_file, 'r') as f:
                return json.load(f)
                
        return None


def main():
    """Example usage"""
    print("NHL Player Data Collector - Prop Betting System")
    print("=" * 80)
    
    collector = NHLPlayerDataCollector()
    
    # Example: Collect data for upcoming game
    home_team = "TOR"
    away_team = "BOS"
    game_id = f"20241120-{away_team}-{home_team}"
    
    print(f"\nCollecting player data for {away_team} @ {home_team}")
    print("-" * 80)
    
    # Collect top players from both teams
    players = collector.collect_players_for_game(home_team, away_team, top_n_players=6)
    
    # Display collected data
    for side in ['home', 'away']:
        team = home_team if side == 'home' else away_team
        print(f"\n{team} Top Players:")
        
        for player in players[side]:
            print(f"\n  {player['player_name']} ({player['position']})")
            print(f"    Season: {player['season_stats']['goals']}G, "
                  f"{player['season_stats']['assists']}A, "
                  f"{player['season_stats']['shots']}S")
            print(f"    Last 5: {player['recent_form']['goals_last_5']}G, "
                  f"{player['recent_form']['last_5_avg_shots']:.1f} shots/game")
            print(f"    Form: {player['recent_form']['trend']}, "
                  f"Streak: {player['recent_form']['point_streak_games']} games")
            print(f"    vs {team}: {player['vs_opponent']['avg_goals']:.2f}G, "
                  f"{player['vs_opponent']['avg_points']:.2f}P per game")
                  
    # Save data
    collector.save_player_data(players, game_id)
    
    print(f"\n✓ Player data collection complete")
    

if __name__ == "__main__":
    main()
