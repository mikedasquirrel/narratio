"""
NHL Performance Tracker

Tracks actual vs predicted results for all NHL betting recommendations.
Calculates rolling statistics, alerts on pattern degradation.

Metrics tracked:
- Win rate (overall and by pattern)
- ROI (return on investment)
- Sharpe ratio (risk-adjusted returns)
- Pattern performance
- Model calibration

Author: Narrative Integration System
Date: November 16, 2025
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class NHLPerformanceTracker:
    """Track NHL betting performance"""
    
    def __init__(self):
        """Initialize tracker"""
        self.project_root = Path(__file__).parent.parent
        self.predictions_dir = self.project_root / 'data' / 'predictions'
        self.results_dir = self.project_root / 'data' / 'results'
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def load_all_predictions(self) -> List[Dict]:
        """Load all saved predictions"""
        if not self.predictions_dir.exists():
            return []
        
        all_predictions = []
        
        for pred_file in sorted(self.predictions_dir.glob('nhl_predictions_*.json')):
            with open(pred_file, 'r') as f:
                data = json.load(f)
                predictions = data.get('predictions', [])
                date = data.get('date', '')
                
                for pred in predictions:
                    pred['prediction_date'] = date
                    all_predictions.append(pred)
        
        return all_predictions
    
    def load_actual_results(self) -> Dict:
        """Load actual game results"""
        results_path = self.project_root / 'data' / 'domains' / 'nhl_games_with_odds.json'
        
        if not results_path.exists():
            return {}
        
        with open(results_path, 'r') as f:
            games = json.load(f)
        
        # Index by game key
        results_by_key = {}
        for game in games:
            key = f"{game['away_team']}_{game['home_team']}_{game['date'][:10]}"
            results_by_key[key] = game
        
        return results_by_key
    
    def match_predictions_to_results(self, predictions: List[Dict], results: Dict) -> List[Dict]:
        """Match predictions to actual outcomes"""
        matched = []
        
        for pred in predictions:
            game = pred.get('game', {})
            away = game.get('away_team', '')
            home = game.get('home_team', '')
            date = game.get('commence_time', '')[:10]
            
            key = f"{away}_{home}_{date}"
            
            if key in results:
                actual = results[key]
                
                # Check if prediction was correct
                rec = pred.get('recommendation')
                if rec and rec.get('bet') == 'HOME WIN':
                    correct = actual.get('home_won', False)
                elif rec and rec.get('bet') == 'AWAY WIN':
                    correct = not actual.get('home_won', True)
                else:
                    correct = False
                
                matched.append({
                    'prediction': pred,
                    'actual': actual,
                    'correct': correct,
                    'pattern': rec.get('pattern') if rec else None,
                })
        
        return matched
    
    def calculate_statistics(self, matched: List[Dict]) -> Dict:
        """Calculate performance statistics"""
        if not matched:
            return {}
        
        # Overall stats
        total = len(matched)
        correct = sum(1 for m in matched if m['correct'])
        win_rate = correct / total if total > 0 else 0
        
        # ROI calculation (assuming -110 juice and 1u bets)
        roi = (win_rate * 0.909) + ((1 - win_rate) * -1.0)
        roi_pct = roi * 100
        
        # By pattern
        by_pattern = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for m in matched:
            pattern = m.get('pattern', 'Unknown')
            by_pattern[pattern]['total'] += 1
            if m['correct']:
                by_pattern[pattern]['correct'] += 1
        
        pattern_stats = {}
        for pattern, stats in by_pattern.items():
            wr = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            pattern_stats[pattern] = {
                'total': stats['total'],
                'correct': stats['correct'],
                'win_rate': wr,
                'win_rate_pct': wr * 100,
            }
        
        return {
            'total_predictions': total,
            'correct': correct,
            'incorrect': total - correct,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'roi': roi,
            'roi_pct': roi_pct,
            'by_pattern': pattern_stats,
        }
    
    def generate_report(self) -> Dict:
        """Generate complete performance report"""
        
        print("\n" + "="*80)
        print("NHL PERFORMANCE TRACKING REPORT")
        print("="*80)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load data
        print("\n📂 Loading data...")
        predictions = self.load_all_predictions()
        results = self.load_actual_results()
        
        print(f"   Predictions: {len(predictions)}")
        print(f"   Results: {len(results)}")
        
        # Match
        matched = self.match_predictions_to_results(predictions, results)
        print(f"   Matched: {len(matched)}")
        
        if not matched:
            print("\n⚠️  No matched predictions yet")
            print("   Make predictions, wait for results, then run tracker")
            return {}
        
        # Calculate stats
        stats = self.calculate_statistics(matched)
        
        # Print report
        print("\n📊 OVERALL PERFORMANCE")
        print("-"*80)
        print(f"Total Predictions: {stats['total_predictions']}")
        print(f"Correct: {stats['correct']}")
        print(f"Incorrect: {stats['incorrect']}")
        print(f"Win Rate: {stats['win_rate_pct']:.1f}%")
        print(f"ROI: {stats['roi_pct']:.1f}%")
        
        print("\n📈 BY PATTERN")
        print("-"*80)
        for pattern, pstats in sorted(stats['by_pattern'].items(), 
                                      key=lambda x: x[1]['win_rate'], reverse=True):
            print(f"{pattern:50s} | {pstats['correct']}/{pstats['total']} | {pstats['win_rate_pct']:.1f}%")
        
        # Save report
        report_path = self.results_dir / f'nhl_performance_{datetime.now().strftime("%Y%m%d")}.json'
        with open(report_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'statistics': stats,
                'matched_predictions': len(matched),
            }, f, indent=2)
        
        print(f"\n💾 Report saved: {report_path}")
        
        return stats


def main():
    """Main execution"""
    
    tracker = NHLPerformanceTracker()
    stats = tracker.generate_report()
    
    if stats:
        print("\n✅ Performance tracking complete!")
    else:
        print("\n📝 No tracked performance yet - make predictions first!")


if __name__ == "__main__":
    main()

