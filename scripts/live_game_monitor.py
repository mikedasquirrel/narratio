"""
Live Game Monitor with In-Game Feature Extraction
==================================================

Real-time game tracking system with:
- 2-minute update frequency
- Live score monitoring
- In-game feature extraction (momentum, score differential, etc.)
- Integration with live prediction system

Supports NBA and NFL games.

Author: AI Coding Assistant
Date: November 16, 2025
"""

import time
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class LiveGameMonitor:
    """
    Monitors live NBA/NFL games and extracts real-time features.
    """
    
    def __init__(
        self,
        update_frequency: int = 120,  # 2 minutes
        data_dir: Optional[Path] = None
    ):
        """
        Initialize live game monitor.
        
        Args:
            update_frequency: Seconds between updates (default: 120 = 2 min)
            data_dir: Directory to save live game data
        """
        self.update_frequency = update_frequency
        self.data_dir = data_dir or Path(__file__).parent.parent / 'data' / 'live'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.active_games = {}
        self.game_history = {}
        
    def fetch_live_nba_scores(self) -> List[Dict]:
        """
        Fetch current NBA scores.
        
        Returns:
            List of live game data
        """
        # In production, use ESPN API, NBA API, or similar
        # For now, return mock data
        return self._mock_nba_scores()
    
    def fetch_live_nfl_scores(self) -> List[Dict]:
        """
        Fetch current NFL scores.
        
        Returns:
            List of live game data
        """
        # In production, use ESPN API or similar
        return self._mock_nfl_scores()
    
    def extract_live_features(self, game_state: Dict, league: str = 'nba') -> Dict[str, float]:
        """
        Extract features from current game state.
        
        Args:
            game_state: Current game data
            league: 'nba' or 'nfl'
            
        Returns:
            Dict of live features
        """
        features = {}
        
        # Basic state
        home_score = game_state.get('home_score', 0)
        away_score = game_state.get('away_score', 0)
        time_remaining = game_state.get('time_remaining_minutes', 0)
        
        # Score differential
        features['score_differential'] = home_score - away_score
        features['score_differential_abs'] = abs(home_score - away_score)
        
        # Scoring pace
        if league == 'nba':
            total_minutes = 48
            quarter = game_state.get('quarter', 1)
        else:  # nfl
            total_minutes = 60
            quarter = game_state.get('quarter', 1)
        
        elapsed = total_minutes - time_remaining
        features['elapsed_pct'] = elapsed / total_minutes if total_minutes > 0 else 0
        
        # Total points (pace indicator)
        total_points = home_score + away_score
        features['total_points'] = total_points
        features['points_per_minute'] = total_points / elapsed if elapsed > 0 else 0
        
        # Momentum (requires game history)
        game_id = game_state.get('game_id')
        if game_id in self.game_history:
            features.update(self._calculate_momentum(game_id, game_state))
        else:
            features['momentum_5min'] = 0.0
            features['momentum_home'] = 0.0
            features['run_length'] = 0
        
        # Game situation
        features['close_game'] = 1.0 if features['score_differential_abs'] <= 5 else 0.0
        features['blowout'] = 1.0 if features['score_differential_abs'] >= 20 else 0.0
        features['clutch_time'] = 1.0 if (time_remaining <= 5 and features['close_game']) else 0.0
        
        # Quarter-specific
        features['quarter'] = quarter
        features['is_final_quarter'] = 1.0 if quarter >= 4 else 0.0
        
        # Lead changes (from history)
        if game_id in self.game_history:
            features['lead_changes'] = self._count_lead_changes(game_id)
        else:
            features['lead_changes'] = 0
        
        return features
    
    def _calculate_momentum(self, game_id: str, current_state: Dict) -> Dict[str, float]:
        """Calculate momentum from recent scoring."""
        history = self.game_history.get(game_id, [])
        
        if len(history) < 2:
            return {
                'momentum_5min': 0.0,
                'momentum_home': 0.0,
                'run_length': 0
            }
        
        # Get last 5 minutes of data
        current_time = current_state.get('time_remaining_minutes', 0)
        recent = [h for h in history if abs(h['time_remaining_minutes'] - current_time) <= 5]
        
        if len(recent) < 2:
            return {
                'momentum_5min': 0.0,
                'momentum_home': 0.0,
                'run_length': 0
            }
        
        # Calculate point differential change
        old_diff = recent[0]['home_score'] - recent[0]['away_score']
        new_diff = current_state['home_score'] - current_state['away_score']
        momentum = new_diff - old_diff
        
        # Detect runs (consecutive scoring)
        run_length = 0
        last_scorer = None
        
        for i in range(len(recent) - 1, 0, -1):
            curr = recent[i]
            prev = recent[i-1]
            
            home_scored = curr['home_score'] > prev['home_score']
            away_scored = curr['away_score'] > prev['away_score']
            
            if home_scored and not away_scored:
                if last_scorer == 'home' or last_scorer is None:
                    run_length += 1
                    last_scorer = 'home'
                else:
                    break
            elif away_scored and not home_scored:
                if last_scorer == 'away' or last_scorer is None:
                    run_length += 1
                    last_scorer = 'away'
                else:
                    break
        
        return {
            'momentum_5min': momentum,
            'momentum_home': 1.0 if momentum > 0 else 0.0,
            'run_length': run_length
        }
    
    def _count_lead_changes(self, game_id: str) -> int:
        """Count number of lead changes from history."""
        history = self.game_history.get(game_id, [])
        
        if len(history) < 2:
            return 0
        
        lead_changes = 0
        prev_leader = None
        
        for state in history:
            diff = state['home_score'] - state['away_score']
            if diff > 0:
                leader = 'home'
            elif diff < 0:
                leader = 'away'
            else:
                leader = 'tied'
            
            if prev_leader and leader != prev_leader and leader != 'tied':
                lead_changes += 1
            
            prev_leader = leader
        
        return lead_changes
    
    def update_game_history(self, game_id: str, game_state: Dict):
        """Store game state in history for momentum calculation."""
        if game_id not in self.game_history:
            self.game_history[game_id] = []
        
        # Add timestamp
        game_state['timestamp'] = datetime.now().isoformat()
        
        # Store
        self.game_history[game_id].append(game_state.copy())
        
        # Keep last 50 updates (covers entire game)
        if len(self.game_history[game_id]) > 50:
            self.game_history[game_id] = self.game_history[game_id][-50:]
    
    def monitor_games(self, league: str = 'nba', duration_minutes: Optional[int] = None):
        """
        Main monitoring loop.
        
        Args:
            league: 'nba' or 'nfl'
            duration_minutes: How long to monitor (None = forever)
        """
        print("=" * 80)
        print(f"LIVE GAME MONITOR - {league.upper()}")
        print("=" * 80)
        print(f"Update frequency: {self.update_frequency} seconds")
        print(f"Data directory: {self.data_dir}")
        
        if duration_minutes:
            print(f"Monitoring for: {duration_minutes} minutes")
            end_time = datetime.now() + timedelta(minutes=duration_minutes)
        else:
            print("Monitoring: Continuous (Ctrl+C to stop)")
            end_time = None
        
        print("\nStarting monitoring...\n")
        
        iteration = 0
        
        try:
            while True:
                iteration += 1
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Fetch live scores
                if league == 'nba':
                    games = self.fetch_live_nba_scores()
                else:
                    games = self.fetch_live_nfl_scores()
                
                print(f"[{timestamp}] Update #{iteration}: {len(games)} active games")
                
                # Process each game
                for game in games:
                    game_id = game['game_id']
                    
                    # Update history
                    self.update_game_history(game_id, game)
                    
                    # Extract features
                    features = self.extract_live_features(game, league)
                    
                    # Store active game with features
                    self.active_games[game_id] = {
                        'game': game,
                        'features': features,
                        'last_update': timestamp
                    }
                    
                    # Print summary
                    self._print_game_summary(game, features)
                    
                    # Check for betting opportunities
                    opportunities = self._check_opportunities(game, features)
                    if opportunities:
                        self._alert_opportunity(game, features, opportunities)
                
                # Save state
                self._save_state()
                
                # Check if should continue
                if end_time and datetime.now() >= end_time:
                    print(f"\nMonitoring duration complete ({duration_minutes} minutes)")
                    break
                
                # Wait for next update
                print(f"\nWaiting {self.update_frequency}s for next update...\n")
                time.sleep(self.update_frequency)
                
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")
            self._save_state()
        
        print("\n" + "=" * 80)
        print("MONITORING SESSION COMPLETE")
        print("=" * 80)
        print(f"Total updates: {iteration}")
        print(f"Games monitored: {len(self.active_games)}")
    
    def _print_game_summary(self, game: Dict, features: Dict):
        """Print concise game summary."""
        away = game.get('away_team', 'AWAY')
        home = game.get('home_team', 'HOME')
        away_score = game.get('away_score', 0)
        home_score = game.get('home_score', 0)
        quarter = game.get('quarter', 1)
        time_left = game.get('time_remaining_minutes', 0)
        
        # Determine leader
        if home_score > away_score:
            leader = f"{home} by {home_score - away_score}"
        elif away_score > home_score:
            leader = f"{away} by {away_score - home_score}"
        else:
            leader = "TIED"
        
        momentum = features.get('momentum_5min', 0)
        momentum_str = f"{'↑' if momentum > 0 else '↓' if momentum < 0 else '→'} {abs(momentum):.1f}"
        
        print(f"  {away} {away_score} @ {home} {home_score} | Q{quarter} {time_left:.1f}min | {leader} | Momentum: {momentum_str}")
    
    def _check_opportunities(self, game: Dict, features: Dict) -> Optional[List[str]]:
        """Check for betting opportunities based on live features."""
        opportunities = []
        
        # Example opportunity detection logic
        
        # 1. Big momentum shift + close game
        if features.get('momentum_5min', 0) >= 10 and features.get('close_game', 0) == 1:
            opportunities.append('momentum_shift')
        
        # 2. Long scoring run
        if features.get('run_length', 0) >= 3:
            opportunities.append('scoring_run')
        
        # 3. Blowout potential (for live totals)
        if features.get('points_per_minute', 0) > 2.5:  # NBA
            opportunities.append('high_pace')
        
        # 4. Clutch time + tight game
        if features.get('clutch_time', 0) == 1:
            opportunities.append('clutch_situation')
        
        return opportunities if opportunities else None
    
    def _alert_opportunity(self, game: Dict, features: Dict, opportunities: List[str]):
        """Alert on detected opportunity."""
        print(f"\n  🎯 OPPORTUNITY DETECTED: {', '.join(opportunities)}")
        print(f"     Features: diff={features['score_differential']:.0f}, " +
              f"momentum={features['momentum_5min']:.1f}, " +
              f"run={features['run_length']:.0f}")
    
    def _save_state(self):
        """Save current monitoring state."""
        state = {
            'timestamp': datetime.now().isoformat(),
            'active_games': self.active_games,
            'n_games': len(self.active_games)
        }
        
        filepath = self.data_dir / 'monitor_state.json'
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def _mock_nba_scores(self) -> List[Dict]:
        """Return mock NBA scores for testing."""
        # Simulate a live game with changing scores
        base_time = time.time()
        elapsed = (base_time % 2880) / 60  # Simulate 48 min game in loop
        
        quarter = min(4, int(elapsed / 12) + 1)
        time_remaining = 48 - elapsed
        
        # Simulate scoring
        home_score = int(25 * elapsed / 12)
        away_score = int(23 * elapsed / 12)
        
        return [
            {
                'game_id': 'nba_mock_1',
                'away_team': 'Lakers',
                'home_team': 'Warriors',
                'away_score': away_score,
                'home_score': home_score,
                'quarter': quarter,
                'time_remaining_minutes': time_remaining,
                'status': 'in_progress'
            }
        ]
    
    def _mock_nfl_scores(self) -> List[Dict]:
        """Return mock NFL scores for testing."""
        base_time = time.time()
        elapsed = (base_time % 3600) / 60  # 60 min game
        
        quarter = min(4, int(elapsed / 15) + 1)
        time_remaining = 60 - elapsed
        
        home_score = int(7 * elapsed / 15)
        away_score = int(6 * elapsed / 15)
        
        return [
            {
                'game_id': 'nfl_mock_1',
                'away_team': 'Chiefs',
                'home_team': 'Bills',
                'away_score': away_score,
                'home_score': home_score,
                'quarter': quarter,
                'time_remaining_minutes': time_remaining,
                'status': 'in_progress'
            }
        ]


def main():
    """Test live game monitor."""
    print("Testing Live Game Monitor...")
    print("\nThis will run for 5 minutes with 30-second updates")
    print("Press Ctrl+C to stop early\n")
    
    # Create monitor
    monitor = LiveGameMonitor(update_frequency=30)
    
    # Monitor for 5 minutes
    monitor.monitor_games(league='nba', duration_minutes=5)
    
    print("\nTest complete!")
    print(f"Check {monitor.data_dir} for saved state")


if __name__ == '__main__':
    main()
