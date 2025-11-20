"""
NHL Live Prop Monitor

Monitors live NHL games and adjusts prop predictions based on:
- Current game state (score, period, time remaining)
- Player performance so far (goals/assists/shots already)
- Momentum shifts and game flow
- Live odds movements

Critical for in-game prop betting where lines adjust dynamically.

Author: Live Prop Betting System
Date: November 20, 2024
"""

import json
import time
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np


class NHLLivePropMonitor:
    """
    Monitor live NHL games for prop betting opportunities.
    
    Tracks:
    - Player stats accumulation during game
    - Remaining prop opportunities
    - Live odds changes
    - Game state impact on props
    """
    
    def __init__(self, update_interval: int = 60):
        """
        Parameters
        ----------
        update_interval : int
            Seconds between live updates
        """
        self.update_interval = update_interval
        self.base_url = "https://statsapi.web.nhl.com/api/v1"
        self.live_games = {}
        self.prop_adjustments = {}
        
    def fetch_live_games(self) -> List[Dict]:
        """Fetch all currently live NHL games"""
        url = f"{self.base_url}/schedule"
        
        params = {
            'expand': 'schedule.linescore',
            'date': datetime.now().strftime('%Y-%m-%d')
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            live_games = []
            
            for date in data.get('dates', []):
                for game in date.get('games', []):
                    # Check if game is live
                    status = game['status']['detailedState']
                    
                    if status in ['Live', 'In Progress', 'In Progress - Critical']:
                        live_games.append({
                            'game_id': game['gamePk'],
                            'home_team': game['teams']['home']['team']['abbreviation'],
                            'away_team': game['teams']['away']['team']['abbreviation'],
                            'period': game['linescore']['currentPeriod'],
                            'time_remaining': game['linescore']['currentPeriodTimeRemaining'],
                            'home_score': game['teams']['home']['score'],
                            'away_score': game['teams']['away']['score'],
                            'status': status,
                        })
                        
            return live_games
            
        except Exception as e:
            print(f"Error fetching live games: {e}")
            return []
            
    def fetch_live_player_stats(self, game_id: int) -> Dict:
        """
        Fetch current player statistics for a live game.
        
        Returns
        -------
        stats : dict
            {player_id: {goals, assists, shots, saves, toi}}
        """
        url = f"{self.base_url}/game/{game_id}/boxscore"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            player_stats = {}
            
            # Process both teams
            for side in ['home', 'away']:
                team_data = data['teams'][side]
                
                # Skaters
                for player_id, player_data in team_data['players'].items():
                    if 'skaterStats' in player_data['stats']:
                        stats = player_data['stats']['skaterStats']
                        
                        player_stats[int(player_id.replace('ID', ''))] = {
                            'name': player_data['person']['fullName'],
                            'position': player_data['position']['abbreviation'],
                            'goals': stats.get('goals', 0),
                            'assists': stats.get('assists', 0),
                            'shots': stats.get('shots', 0),
                            'points': stats.get('goals', 0) + stats.get('assists', 0),
                            'toi': stats.get('timeOnIce', '0:00'),
                            'plus_minus': stats.get('plusMinus', 0),
                            'team': side,
                        }
                        
                # Goalies
                for player_id, player_data in team_data['players'].items():
                    if 'goalieStats' in player_data['stats']:
                        stats = player_data['stats']['goalieStats']
                        
                        player_stats[int(player_id.replace('ID', ''))] = {
                            'name': player_data['person']['fullName'],
                            'position': 'G',
                            'saves': stats.get('saves', 0),
                            'shots_against': stats.get('shots', 0),
                            'goals_against': stats.get('goalsAgainst', 0),
                            'save_pct': stats.get('savePercentage', 0.0),
                            'toi': stats.get('timeOnIce', '0:00'),
                            'team': side,
                        }
                        
            return player_stats
            
        except Exception as e:
            print(f"Error fetching player stats for game {game_id}: {e}")
            return {}
            
    def calculate_prop_adjustments(self, game_state: Dict, 
                                 player_stats: Dict) -> Dict:
        """
        Calculate live adjustments to prop probabilities.
        
        Parameters
        ----------
        game_state : dict
            Current game state (period, time, score)
        player_stats : dict
            Current player statistics
            
        Returns
        -------
        adjustments : dict
            {player_id: {prop_type: adjustment_factor}}
        """
        adjustments = {}
        
        period = game_state['period']
        time_str = game_state['time_remaining']
        
        # Calculate game progress (0-1)
        if period >= 3 and time_str == 'Final':
            game_progress = 1.0
        else:
            # Parse time remaining
            if ':' in time_str:
                minutes, seconds = map(int, time_str.split(':'))
                period_time_elapsed = 20 - (minutes + seconds/60)
            else:
                period_time_elapsed = 20
                
            total_elapsed = (period - 1) * 20 + period_time_elapsed
            game_progress = min(total_elapsed / 60, 1.0)  # 60 minute game
            
        # Calculate time remaining factor
        time_remaining_factor = 1 - game_progress
        
        for player_id, stats in player_stats.items():
            player_adjustments = {}
            
            if stats['position'] == 'G':
                # Goalie adjustments
                current_saves = stats.get('saves', 0)
                
                # Adjust save props based on current pace
                if game_progress > 0:
                    projected_saves = current_saves / game_progress
                    
                    # More weight to actual performance as game progresses
                    player_adjustments['saves'] = {
                        'current': current_saves,
                        'projected': projected_saves,
                        'certainty': game_progress,  # More certain later in game
                    }
            else:
                # Skater adjustments
                current_goals = stats.get('goals', 0)
                current_assists = stats.get('assists', 0)
                current_shots = stats.get('shots', 0)
                current_points = current_goals + current_assists
                
                # Parse time on ice
                toi_str = stats.get('toi', '0:00')
                if ':' in toi_str:
                    toi_minutes = int(toi_str.split(':')[0]) + int(toi_str.split(':')[1])/60
                else:
                    toi_minutes = 0
                    
                # Calculate scoring rate
                if toi_minutes > 0:
                    goals_per_60 = (current_goals / toi_minutes) * 60
                    points_per_60 = (current_points / toi_minutes) * 60
                    shots_per_60 = (current_shots / toi_minutes) * 60
                else:
                    goals_per_60 = 0
                    points_per_60 = 0
                    shots_per_60 = 0
                    
                # Project remaining production
                expected_remaining_toi = (1 - game_progress) * 18  # Assume 18 min/game
                
                projected_goals = current_goals + (goals_per_60 * expected_remaining_toi / 60)
                projected_assists = current_assists + ((points_per_60 - goals_per_60) * expected_remaining_toi / 60)
                projected_shots = current_shots + (shots_per_60 * expected_remaining_toi / 60)
                projected_points = projected_goals + projected_assists
                
                player_adjustments.update({
                    'goals': {
                        'current': current_goals,
                        'projected': projected_goals,
                        'certainty': game_progress,
                        'rate_per_60': goals_per_60,
                    },
                    'assists': {
                        'current': current_assists,
                        'projected': projected_assists,
                        'certainty': game_progress,
                    },
                    'shots': {
                        'current': current_shots,
                        'projected': projected_shots,
                        'certainty': game_progress,
                        'rate_per_60': shots_per_60,
                    },
                    'points': {
                        'current': current_points,
                        'projected': projected_points,
                        'certainty': game_progress,
                    }
                })
                
            adjustments[player_id] = player_adjustments
            
        return adjustments
        
    def adjust_prop_probability(self, base_prob: float, current: float, 
                              line: float, certainty: float) -> float:
        """
        Adjust prop probability based on live performance.
        
        Parameters
        ----------
        base_prob : float
            Pre-game probability of going over
        current : float
            Current stat value
        line : float
            Prop line (e.g., 0.5 for goals)
        certainty : float
            Game progress (0-1)
            
        Returns
        -------
        adjusted_prob : float
            Live adjusted probability
        """
        # If already over, probability is 1
        if current > line:
            return 1.0
            
        # Calculate remaining needed
        remaining_needed = line - current + 0.5  # Add 0.5 for "over"
        
        # If impossible to reach, probability is 0
        if certainty >= 0.95 and remaining_needed > 0:  # Game almost over
            return 0.0
            
        # Blend pre-game and live projections
        # More weight to live data as game progresses
        live_weight = certainty
        pre_weight = 1 - certainty
        
        # Simple linear adjustment for now
        # Could be enhanced with more sophisticated models
        if remaining_needed <= 0.5:
            # Need 1 more to go over
            live_prob = base_prob * (1 + certainty)  # Boost if performing well
        else:
            # Need multiple more
            live_prob = base_prob * (1 - certainty * 0.5)  # Decrease probability
            
        adjusted_prob = pre_weight * base_prob + live_weight * live_prob
        
        return np.clip(adjusted_prob, 0.0, 1.0)
        
    def find_live_edges(self, game_id: int, pre_game_predictions: List[Dict],
                       live_odds: List[Dict]) -> List[Dict]:
        """
        Find prop betting edges in live games.
        
        Parameters
        ----------
        game_id : int
            NHL game ID
        pre_game_predictions : list
            Pre-game prop predictions
        live_odds : list
            Current live prop odds
            
        Returns
        -------
        edges : list
            Live prop betting opportunities
        """
        # Fetch current game state
        live_games = self.fetch_live_games()
        game_state = next((g for g in live_games if g['game_id'] == game_id), None)
        
        if not game_state:
            return []
            
        # Fetch current player stats
        player_stats = self.fetch_live_player_stats(game_id)
        
        # Calculate adjustments
        adjustments = self.calculate_prop_adjustments(game_state, player_stats)
        
        # Find edges
        edges = []
        
        for pred in pre_game_predictions:
            player_id = pred['player_id']
            
            if player_id not in adjustments:
                continue
                
            adj = adjustments[player_id].get(pred['prop_type'], {})
            
            if not adj:
                continue
                
            # Adjust probability based on live data
            adjusted_prob = self.adjust_prop_probability(
                pred['prob_over'],
                adj['current'],
                pred['line'],
                adj['certainty']
            )
            
            # Find corresponding live odds
            live_prop = next(
                (p for p in live_odds 
                 if p['player_name'] == pred['player_name'] 
                 and p['prop_type'] == pred['prop_type']
                 and p['line'] == pred['line']),
                None
            )
            
            if live_prop:
                # Calculate edge
                implied_prob = self._american_to_probability(live_prop['odds'])
                edge = adjusted_prob - implied_prob
                
                if edge > 0.05:  # Minimum 5% edge for live
                    edges.append({
                        **pred,
                        'live_adjusted_prob': adjusted_prob,
                        'implied_prob': implied_prob,
                        'edge': edge,
                        'edge_pct': edge * 100,
                        'current_value': adj['current'],
                        'projected_final': adj.get('projected', adj['current']),
                        'game_progress': adj['certainty'],
                        'live_odds': live_prop['odds'],
                        'recommendation': 'BET' if edge > 0.08 else 'WATCH',
                    })
                    
        # Sort by edge
        edges.sort(key=lambda x: x['edge'], reverse=True)
        
        return edges
        
    def _american_to_probability(self, odds: int) -> float:
        """Convert American odds to probability"""
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)
            
    def monitor_games(self, games_to_monitor: List[int], 
                     pre_game_predictions: Dict[int, List[Dict]],
                     duration: int = 3600):
        """
        Monitor multiple games for prop opportunities.
        
        Parameters
        ----------
        games_to_monitor : list
            Game IDs to monitor
        pre_game_predictions : dict
            {game_id: [predictions]}
        duration : int
            How long to monitor (seconds)
        """
        print("\nNHL LIVE PROP MONITOR")
        print("=" * 80)
        print(f"Monitoring {len(games_to_monitor)} games")
        print(f"Update interval: {self.update_interval} seconds")
        print(f"Duration: {duration/60:.0f} minutes")
        print("=" * 80)
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Checking games...")
            
            for game_id in games_to_monitor:
                # Mock live odds for demo
                mock_live_odds = self._generate_mock_live_odds(
                    pre_game_predictions.get(game_id, [])
                )
                
                # Find edges
                edges = self.find_live_edges(
                    game_id,
                    pre_game_predictions.get(game_id, []),
                    mock_live_odds
                )
                
                if edges:
                    print(f"\n  Game {game_id}: Found {len(edges)} opportunities")
                    
                    for edge in edges[:3]:  # Top 3
                        print(f"    {edge['player_name']} {edge['prop_type']} "
                              f"o{edge['line']}: {edge['edge_pct']:.1f}% edge")
                        print(f"      Current: {edge['current_value']}, "
                              f"Projected: {edge['projected_final']:.1f}")
                        print(f"      Live odds: {edge['live_odds']:+d} "
                              f"({edge['implied_prob']:.1%})")
                        print(f"      Recommendation: {edge['recommendation']}")
                        
            # Wait for next update
            time.sleep(self.update_interval)
            
        print("\n✓ Monitoring complete")
        
    def _generate_mock_live_odds(self, predictions: List[Dict]) -> List[Dict]:
        """Generate mock live odds for testing"""
        live_odds = []
        
        for pred in predictions:
            # Simulate odds movement
            base_odds = -110
            movement = np.random.normal(0, 20)
            
            live_odds.append({
                'player_name': pred['player_name'],
                'prop_type': pred['prop_type'],
                'line': pred['line'],
                'odds': int(base_odds + movement),
            })
            
        return live_odds


def main():
    """Example usage"""
    monitor = NHLLivePropMonitor(update_interval=30)
    
    # Get live games
    live_games = monitor.fetch_live_games()
    
    print(f"\nFound {len(live_games)} live NHL games")
    
    if live_games:
        # Monitor first game
        game = live_games[0]
        game_id = game['game_id']
        
        print(f"\nMonitoring: {game['away_team']} @ {game['home_team']}")
        print(f"Period {game['period']}, {game['time_remaining']}")
        print(f"Score: {game['away_team']} {game['away_score']} - "
              f"{game['home_score']} {game['home_team']}")
              
        # Get player stats
        stats = monitor.fetch_live_player_stats(game_id)
        
        print(f"\nTracking {len(stats)} players")
        
        # Show top scorers
        scorers = sorted(
            [(p, s) for p, s in stats.items() if s.get('points', 0) > 0],
            key=lambda x: x[1].get('points', 0),
            reverse=True
        )
        
        print("\nCurrent scoring leaders:")
        for player_id, pstats in scorers[:5]:
            print(f"  {pstats['name']}: {pstats.get('goals', 0)}G, "
                  f"{pstats.get('assists', 0)}A, {pstats.get('shots', 0)}S")
    else:
        print("\nNo live games at the moment")
        print("Live monitoring would track:")
        print("- Player performance vs pre-game projections")
        print("- Prop line movements during game")
        print("- Opportunities when odds haven't caught up")
        print("- Time-sensitive edges (e.g., player close to milestone)")
        

if __name__ == "__main__":
    main()
