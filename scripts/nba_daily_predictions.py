"""
NBA Daily Predictions Script
==============================

Generates daily betting predictions for today's NBA games.
Uses trained ensemble model to identify high-confidence opportunities.

Usage:
    python scripts/nba_daily_predictions.py
    python scripts/nba_daily_predictions.py --date 2024-11-16
    python scripts/nba_daily_predictions.py --dry-run

Author: AI Coding Assistant
Date: November 16, 2025
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from narrative_optimization.betting.nba_ensemble_model import NBAEnsembleBettingModel
from narrative_optimization.betting.betting_utils import format_odds_display


def print_header(text, char='='):
    print()
    print(char * 80)
    print(text)
    print(char * 80)
    print()


def load_todays_games(date: Optional[str] = None, dry_run: bool = False) -> List[Dict]:
    """
    Load today's NBA games.
    
    For now, loads from live data file. In production, would fetch from API.
    
    Parameters
    ----------
    date : str, optional
        Date in YYYY-MM-DD format (default: today)
    dry_run : bool
        Use sample test data instead of live
        
    Returns
    -------
    games : list of dict
        Today's games with betting odds
    """
    if dry_run:
        print("[Data] DRY RUN mode - using sample test data")
        # Load from test set
        data_path = Path('data/domains/nba_complete_with_players.json')
        with open(data_path) as f:
            all_games = json.load(f)
        
        # Get recent games from 2023-24 season
        recent_games = [g for g in all_games if g['season'] == '2023-24'][:10]
        print(f"[Data] ✓ Loaded {len(recent_games)} sample games for testing")
        return recent_games
    
    # Try to load from live data file
    if date is None:
        date = datetime.now().strftime('%Y%m%d')
    else:
        date = date.replace('-', '')
    
    live_path = Path(f'data/live/nba_{date}.json')
    
    if live_path.exists():
        print(f"[Data] Loading from: {live_path}")
        with open(live_path) as f:
            games = json.load(f)
        print(f"[Data] ✓ Loaded {len(games)} games for {date}")
        return games
    else:
        print(f"[Data] ⚠️  No live data found for {date}")
        print(f"[Data] Run: python scripts/nba_fetch_today.py")
        print(f"[Data] Or use --dry-run flag for testing")
        return []


def build_clean_narrative(game: Dict) -> str:
    """Build clean pre-game narrative"""
    parts = []
    
    parts.append(f"Team {game.get('team_name', 'Unknown')}")
    parts.append(f"Matchup {game.get('matchup', 'vs Opponent')}")
    parts.append(f"Location {'home' if game.get('home_game', False) else 'away'}")
    
    if game.get('player_data', {}).get('available'):
        agg = game['player_data']['team_aggregates']
        if agg.get('top1_name'):
            parts.append(f"Star {agg['top1_name']}")
    
    tc = game.get('temporal_context', {})
    if tc.get('season_record_prior'):
        parts.append(f"Record {tc['season_record_prior']}")
    
    betting = game.get('betting_odds', {})
    if betting.get('moneyline'):
        parts.append(f"Line {betting['moneyline']}")
    
    return ". ".join(parts) + "."


def main():
    """Generate daily predictions"""
    
    parser = argparse.ArgumentParser(description='Generate NBA daily betting predictions')
    parser.add_argument('--date', type=str, help='Date in YYYY-MM-DD format (default: today)')
    parser.add_argument('--dry-run', action='store_true', help='Use sample test data')
    parser.add_argument('--model-path', type=str, 
                       default='narrative_optimization/betting/nba_ensemble_trained.pkl',
                       help='Path to trained model')
    args = parser.parse_args()
    
    print_header("NBA DAILY BETTING PREDICTIONS", "█")
    print(f"Date: {args.date or datetime.now().strftime('%Y-%m-%d')}")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    
    # Load model
    model_path = Path(args.model_path)
    
    if not model_path.exists():
        print(f"\n❌ Model not found: {model_path}")
        print(f"   Run: python narrative_optimization/betting/nba_backtest.py")
        return
    
    print(f"\n[Model] Loading from: {model_path}")
    model = NBAEnsembleBettingModel.load_model(str(model_path))
    
    # Load today's games
    games = load_todays_games(date=args.date, dry_run=args.dry_run)
    
    if len(games) == 0:
        print("\n❌ No games found for today")
        return
    
    print_header("GENERATING PREDICTIONS", "=")
    
    # Build narratives
    print(f"[Predict] Building narratives for {len(games)} games...")
    X = pd.Series([build_clean_narrative(g) for g in games])
    odds = np.array([g.get('betting_odds', {}).get('moneyline', 0) for g in games])
    
    # Get predictions
    print(f"[Predict] Running ensemble model...")
    predictions = model.predict_with_confidence(X, market_odds=odds, verbose=True)
    
    # Filter for high confidence
    high_conf = model.get_high_confidence_bets(predictions, min_confidence=model.min_confidence)
    
    print_header("HIGH-CONFIDENCE BETTING OPPORTUNITIES", "=")
    
    if len(high_conf) == 0:
        print("❌ No high-confidence bets found today")
        print(f"   (Confidence threshold: {model.min_confidence:.0%})")
        print(f"   (Edge threshold: {model.min_edge:.0%})")
    else:
        print(f"✅ Found {len(high_conf)} high-confidence opportunities!")
        print()
        
        for i, pred in enumerate(high_conf, 1):
            game = games[pred['game_index']]
            betting = pred['betting']
            
            print(f"\n{'─'*80}")
            print(f"BET #{i}")
            print('─'*80)
            print(f"Matchup: {game.get('matchup', 'Unknown')}")
            print(f"Team: {game.get('team_name', 'Unknown')}")
            print(f"Location: {'HOME' if game.get('home_game', False) else 'AWAY'}")
            print()
            print(f"MODEL PREDICTION:")
            print(f"  Win Probability: {pred['win_probability']:.1%}")
            print(f"  Confidence: {pred['confidence_level']}")
            print()
            print(f"BETTING ANALYSIS:")
            print(f"  Market Odds: {format_odds_display(betting['market_odds'])}")
            print(f"  Edge: {betting['edge']:+.1%}")
            print(f"  Expected Value: {betting['expected_value']:+.3f} units")
            print(f"  Recommended Units: {betting['recommended_units']:.1f}")
            print()
            print(f"REASONING: {betting['reason']}")
    
    # Save predictions
    date_str = args.date or datetime.now().strftime('%Y%m%d')
    output_path = Path(f'data/predictions/nba_daily_{date_str}.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'date': args.date or datetime.now().strftime('%Y-%m-%d'),
            'generated_at': datetime.now().isoformat(),
            'model_path': str(model_path),
            'n_games_analyzed': len(games),
            'n_high_confidence_bets': len(high_conf),
            'high_confidence_bets': high_conf,
            'all_predictions': predictions,
            'model_config': {
                'min_confidence': model.min_confidence,
                'min_edge': model.min_edge
            }
        }, f, indent=2)
    
    print(f"\n[Output] ✓ Predictions saved to: {output_path}")
    
    # Summary
    print_header("DAILY SUMMARY", "=")
    print(f"Date: {args.date or datetime.now().strftime('%Y-%m-%d')}")
    print(f"Games analyzed: {len(games)}")
    print(f"High-confidence bets: {len(high_conf)}")
    
    if len(high_conf) > 0:
        total_ev = sum(p['betting']['expected_value'] for p in high_conf)
        total_units = sum(p['betting']['recommended_units'] for p in high_conf)
        print(f"Total EV: {total_ev:+.2f} units")
        print(f"Total units recommended: {total_units:.1f}")
        print(f"Avg EV per bet: {total_ev/len(high_conf):+.3f}")
    
    print()


if __name__ == "__main__":
    main()

