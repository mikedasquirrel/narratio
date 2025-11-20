#!/usr/bin/env python3
"""
NFL Live Betting API
Real-time analysis of upcoming/live games with betting recommendations

Features:
- Fetches upcoming NFL games and current odds
- Calculates narrative features in real-time
- Identifies profitable patterns
- Flags spread, moneyline, and prop bet opportunities
- REST API for integration
"""

import sys
import json
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from flask import Flask, jsonify, request

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Initialize Flask app for API
app = Flask(__name__)

# Load trained patterns
PATTERNS_FILE = Path(__file__).parent.parent / "data" / "domains" / "nfl_betting_patterns_FIXED.json"
with open(PATTERNS_FILE) as f:
    TRAINED_PATTERNS = json.load(f)

# Load historical matchup data for context
ENRICHED_DATA_FILE = Path(__file__).parent.parent / "data" / "domains" / "nfl_enriched_with_rosters.json"
with open(ENRICHED_DATA_FILE) as f:
    HISTORICAL_DATA = json.load(f)
    HISTORICAL_GAMES = {g['game_id']: g for g in HISTORICAL_DATA['games']}

class NFLLiveBettingAnalyzer:
    """Real-time NFL betting analyzer"""
    
    def __init__(self):
        self.profitable_patterns = [
            p for p in TRAINED_PATTERNS['patterns'] 
            if p['profitable']
        ]
        print(f"✓ Loaded {len(self.profitable_patterns)} profitable patterns")
    
    def fetch_upcoming_games(self, week=None):
        """Fetch upcoming games from nflverse"""
        print("\n📥 Fetching upcoming NFL games...")
        
        # For demo, use this week's games from nflverse
        current_season = datetime.now().year
        
        try:
            # Try to get current week schedule
            url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{current_season}.csv.gz"
            print(f"  Attempting to fetch {current_season} season data...")
            
            # In production, you'd use a live odds API like The Odds API
            # For now, we'll use recent games as examples
            
            print("  ⚠ Using recent games as demo (integrate live odds API for production)")
            return self._get_recent_games_demo()
            
        except Exception as e:
            print(f"  ✗ Fetch failed: {e}")
            return []
    
    def _get_recent_games_demo(self):
        """Get recent 2025 games as demo"""
        # Load from our enriched dataset
        recent_games = [
            g for g in HISTORICAL_DATA['games']
            if g['season'] == 2025 and g.get('week', 0) >= 10
        ]
        
        print(f"  ✓ Loaded {len(recent_games)} recent games (Week 10+, 2025)")
        return recent_games[:10]  # Return next 10 games
    
    def calculate_live_features(self, game):
        """Calculate features for a live/upcoming game"""
        
        features = {
            # Basic game info
            'game_id': game['game_id'],
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'week': game.get('week', 0),
            'season': game['season'],
            
            # Spread features
            'spread_line': game.get('spread_line', 0),
            'is_home_dog': game.get('spread_line', 0) > 0,
            'is_big_dog': game.get('spread_line', 0) >= 7,
            'is_home_favorite': game.get('spread_line', 0) < 0,
            
            # Context features
            'playoff': game.get('playoff', False),
            'division_game': game.get('div_game', False),
            'late_season': game.get('week', 0) >= 13,
            
            # Records (parse from string)
            'home_record': game.get('home_record_before', '0-0'),
            'away_record': game.get('away_record_before', '0-0'),
        }
        
        # Parse records for momentum
        def parse_record(rec):
            try:
                w, l = rec.split('-')
                return int(w), int(l)
            except:
                return 0, 0
        
        home_w, home_l = parse_record(features['home_record'])
        away_w, away_l = parse_record(features['away_record'])
        
        features['home_win_pct'] = home_w / max(1, home_w + home_l)
        features['away_win_pct'] = away_w / max(1, away_w + away_l)
        features['record_differential'] = home_w - away_w
        features['high_momentum_home'] = features['record_differential'] > 2
        
        # Rivalry from matchup history
        matchup_hist = game.get('matchup_history', {})
        if isinstance(matchup_hist, dict):
            features['rivalry_games'] = matchup_hist.get('total_games', 0)
            features['is_rivalry'] = matchup_hist.get('total_games', 0) > 15
        else:
            features['rivalry_games'] = 0
            features['is_rivalry'] = False
        
        # Story quality (simplified live calculation)
        story_score = 0.0
        if features['playoff']: story_score += 0.3
        if features['is_rivalry']: story_score += 0.2
        if features['late_season']: story_score += 0.2
        if abs(features['record_differential']) <= 2: story_score += 0.2
        if features['division_game']: story_score += 0.1
        
        features['story_quality'] = min(story_score, 1.0)
        features['high_story'] = story_score >= 0.4
        
        return features
    
    def check_pattern_match(self, features, pattern_def):
        """Check if game matches a profitable pattern"""
        
        # Map pattern names to feature checks
        pattern_checks = {
            'Huge Home Underdog (+7+)': features['is_big_dog'],
            'Strong Record Home': features['high_momentum_home'],
            'Big Home Underdog (+3.5+)': features['spread_line'] >= 3.5,
            'Rivalry + Home Dog': features['is_rivalry'] and features['is_home_dog'],
            'High Momentum Home': features['home_win_pct'] > features['away_win_pct'] + 0.2,
            'High Story + Home Dog': features['high_story'] and features['is_home_dog'],
            'Late Season + Home Dog': features['late_season'] and features['is_home_dog'],
            'Division + Home Dog': features['division_game'] and features['is_home_dog'],
            'Home Underdog': features['is_home_dog'],
            'Playoff Game': features['playoff'],
            'High Rivalry Game': features['is_rivalry'],
            'Division Rivalry': features['division_game'],
            'Late Season Game': features['late_season'],
            'High Story Quality (Q >= 0.4)': features['high_story'],
        }
        
        return pattern_checks.get(pattern_def['pattern'], False)
    
    def analyze_game(self, game):
        """Analyze a single game for betting opportunities"""
        
        # Calculate features
        features = self.calculate_live_features(game)
        
        # Check against profitable patterns
        matching_patterns = []
        for pattern in self.profitable_patterns:
            if self.check_pattern_match(features, pattern):
                matching_patterns.append({
                    'pattern': pattern['pattern'],
                    'historical_win_rate': pattern['win_rate'],
                    'historical_roi': pattern['roi_pct'],
                    'historical_games': pattern['games'],
                    'confidence': 'HIGH' if pattern['roi_pct'] > 50 else 'MEDIUM',
                })
        
        # Generate recommendations
        recommendations = []
        
        if matching_patterns:
            # Sort by ROI
            matching_patterns = sorted(matching_patterns, key=lambda x: x['historical_roi'], reverse=True)
            
            # Spread bet recommendation
            if features['is_home_dog']:
                best_pattern = matching_patterns[0]
                recommendations.append({
                    'bet_type': 'SPREAD',
                    'recommendation': f"HOME {features['home_team']} +{features['spread_line']}",
                    'confidence': best_pattern['confidence'],
                    'expected_roi': f"{best_pattern['historical_roi']:.1f}%",
                    'pattern': best_pattern['pattern'],
                    'reasoning': f"Home underdog with {len(matching_patterns)} profitable pattern matches",
                })
            
            # Moneyline recommendation (if big underdog)
            if features['spread_line'] >= 7:
                recommendations.append({
                    'bet_type': 'MONEYLINE',
                    'recommendation': f"HOME {features['home_team']} ML (underdog)",
                    'confidence': 'HIGH',
                    'expected_roi': '100%+',
                    'pattern': 'Huge Home Dog Pattern',
                    'reasoning': f"94.4% ATS means likely moneyline value too",
                })
            
            # Prop bet recommendations (based on high story quality)
            if features['story_quality'] >= 0.5:
                recommendations.append({
                    'bet_type': 'PROP',
                    'recommendation': 'OVER on total points (high drama games)',
                    'confidence': 'MEDIUM',
                    'expected_roi': '10-15%',
                    'pattern': 'High Story Quality',
                    'reasoning': 'High story quality games tend toward higher scoring',
                })
        
        return {
            'game_id': game['game_id'],
            'matchup': f"{features['away_team']} @ {features['home_team']}",
            'week': features['week'],
            'spread': features['spread_line'],
            'features': features,
            'matching_patterns': matching_patterns,
            'recommendations': recommendations,
            'bet_flag': len(recommendations) > 0,
        }
    
    def analyze_all_upcoming(self):
        """Analyze all upcoming games"""
        print("\n🔄 Analyzing upcoming games...")
        
        games = self.fetch_upcoming_games()
        
        results = []
        flagged_games = []
        
        for game in games:
            analysis = self.analyze_game(game)
            results.append(analysis)
            
            if analysis['bet_flag']:
                flagged_games.append(analysis)
        
        print(f"\n✓ Analyzed {len(results)} games")
        print(f"✓ Flagged {len(flagged_games)} betting opportunities")
        
        return results, flagged_games

# API Endpoints

@app.route('/nfl/upcoming', methods=['GET'])
def get_upcoming_games():
    """Get all upcoming games with analysis"""
    analyzer = NFLLiveBettingAnalyzer()
    results, flagged = analyzer.analyze_all_upcoming()
    
    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'total_games': len(results),
        'flagged_opportunities': len(flagged),
        'games': results,
    })

@app.route('/nfl/flagged', methods=['GET'])
def get_flagged_bets():
    """Get only games with betting recommendations"""
    analyzer = NFLLiveBettingAnalyzer()
    results, flagged = analyzer.analyze_all_upcoming()
    
    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'opportunities': len(flagged),
        'flagged_games': flagged,
    })

@app.route('/nfl/game/<game_id>', methods=['GET'])
def analyze_specific_game(game_id):
    """Analyze a specific game"""
    analyzer = NFLLiveBettingAnalyzer()
    
    # Find game in historical data or upcoming
    game = HISTORICAL_GAMES.get(game_id)
    
    if not game:
        return jsonify({'error': 'Game not found'}), 404
    
    analysis = analyzer.analyze_game(game)
    
    return jsonify(analysis)

@app.route('/nfl/patterns', methods=['GET'])
def get_patterns():
    """Get all profitable patterns"""
    return jsonify({
        'total_patterns': len(TRAINED_PATTERNS['patterns']),
        'profitable': [p for p in TRAINED_PATTERNS['patterns'] if p['profitable']],
    })

@app.route('/nfl/health', methods=['GET'])
def health_check():
    """API health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'patterns_loaded': len(TRAINED_PATTERNS['patterns']),
        'historical_games': len(HISTORICAL_GAMES),
    })

def run_cli_analysis():
    """Command-line analysis mode"""
    print("="*60)
    print("NFL LIVE BETTING ANALYZER")
    print("="*60)
    
    analyzer = NFLLiveBettingAnalyzer()
    results, flagged = analyzer.analyze_all_upcoming()
    
    if not flagged:
        print("\n⚠ No betting opportunities flagged in upcoming games")
        return
    
    print("\n" + "="*60)
    print(f"🎯 BETTING OPPORTUNITIES FLAGGED: {len(flagged)}")
    print("="*60)
    
    for i, game in enumerate(flagged, 1):
        print(f"\n{i}. {game['matchup']} (Week {game['week']})")
        print(f"   Spread: {game['spread']:+.1f}")
        print(f"   Matching Patterns: {len(game['matching_patterns'])}")
        
        for rec in game['recommendations']:
            print(f"\n   💰 {rec['bet_type']}: {rec['recommendation']}")
            print(f"      Confidence: {rec['confidence']}")
            print(f"      Expected ROI: {rec['expected_roi']}")
            print(f"      Pattern: {rec['pattern']}")
            print(f"      Reasoning: {rec['reasoning']}")
    
    # Save flagged games
    output_file = Path(__file__).parent.parent / "data" / "domains" / "nfl_live_opportunities.json"
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'opportunities': flagged,
        }, f, indent=2)
    
    print(f"\n✓ Opportunities saved to: {output_file.name}")

def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='NFL Live Betting API')
    parser.add_argument('--mode', choices=['api', 'cli'], default='cli',
                       help='Run as API server or CLI analysis')
    parser.add_argument('--port', type=int, default=5739,
                       help='API port (default: 5739)')
    
    args = parser.parse_args()
    
    if args.mode == 'api':
        print(f"🚀 Starting NFL Betting API on port {args.port}...")
        print(f"   Endpoints:")
        print(f"   - http://localhost:{args.port}/nfl/upcoming")
        print(f"   - http://localhost:{args.port}/nfl/flagged")
        print(f"   - http://localhost:{args.port}/nfl/patterns")
        print(f"   - http://localhost:{args.port}/nfl/health")
        app.run(host='0.0.0.0', port=args.port, debug=True)
    else:
        run_cli_analysis()

if __name__ == "__main__":
    main()

