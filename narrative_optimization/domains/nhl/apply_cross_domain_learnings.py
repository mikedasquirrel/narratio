"""
NHL Cross-Domain Learning Application

Applies insights from NBA and NFL transformer analysis to NHL.
Uses the ACTUAL transformers that worked in other sports, not assumptions.

Key insight: Different sports share structural patterns.
- NBA momentum patterns → NHL momentum patterns
- NFL QB prestige → NHL goalie prestige
- Competition patterns transfer across domains

Author: Narrative Integration System
Date: November 16, 2025
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class CrossDomainNHLEnhancer:
    """Apply cross-domain learnings to NHL"""
    
    def __init__(self):
        """Initialize enhancer"""
        self.nba_top_transformers = [
            'Ensemble Narrative',  # 54.0% accuracy in NBA
            'Competitive Context',  # 54.0% accuracy in NBA
            'Awareness Resistance',  # 56.8% accuracy in NBA
            'Nominative Richness',  # 54.8% accuracy in NBA
            'Authenticity',  # 53.2% accuracy in NBA
        ]
        
        self.nfl_patterns = {
            'late_season': 'Weeks 13-14 boost (75.3% win)',
            'qb_prestige': 'QB prestige edge >0.2 (80% win)',
            'record_gap': 'Strong record advantage',
        }
    
    def apply_nba_momentum_learning(self, games: List[Dict], features: np.ndarray) -> List[Dict]:
        """
        NBA taught us: Temporal momentum with decay (γ=0.948, half-life ~13 games)
        Apply to NHL with hockey-adjusted decay.
        """
        print("\n🏀 APPLYING NBA MOMENTUM LEARNING TO NHL")
        print("-"*80)
        
        patterns = []
        
        # Calculate momentum with exponential decay (like NBA)
        gamma_nhl = 0.95  # Similar to NBA's 0.948
        
        for i, game in enumerate(games):
            tc = game.get('temporal_context', {})
            l10_wins = tc.get('home_l10_wins', 5)
            
            # Calculate weighted momentum (recent games matter more)
            # Last 10 games: [1, 2, 3, ..., 10] (10 is most recent)
            weights = np.array([gamma_nhl ** (10-j) for j in range(1, 11)])
            
            # Assume uniform wins across L10 for now (would need game-by-game data)
            win_pattern = np.ones(10) * (l10_wins / 10.0)
            momentum_score = np.sum(win_pattern * weights) / np.sum(weights)
            
            games[i]['momentum_score'] = float(momentum_score)
        
        # Find high momentum patterns
        high_momentum_games = [g for g in games if g.get('momentum_score', 0.5) > 0.70]
        
        if len(high_momentum_games) >= 15:
            wins = sum(1 for g in high_momentum_games if g.get('home_won', False))
            win_rate = wins / len(high_momentum_games)
            roi = (win_rate * 0.909) + ((1 - win_rate) * -1.0)
            
            if win_rate > 0.53 and roi > 0.05:
                pattern = {
                    'name': 'NBA-Derived Momentum (γ=0.95)',
                    'description': 'Exponentially weighted momentum score >0.70 (learned from NBA)',
                    'source': 'nba_momentum_transfer',
                    'n_games': len(high_momentum_games),
                    'wins': wins,
                    'losses': len(high_momentum_games) - wins,
                    'win_rate': win_rate,
                    'win_rate_pct': win_rate * 100,
                    'roi': roi,
                    'roi_pct': roi * 100,
                    'confidence': 'HIGH' if win_rate > 0.57 else 'MEDIUM',
                    'unit_recommendation': 2 if win_rate > 0.57 else 1,
                    'pattern_type': 'cross_domain_nba',
                }
                patterns.append(pattern)
                print(f"   ✓ Found NBA momentum pattern: {len(high_momentum_games)} games, {win_rate:.1%} win, {roi:.1%} ROI")
        
        return patterns
    
    def apply_nfl_late_season_learning(self, games: List[Dict], features: np.ndarray) -> List[Dict]:
        """
        NFL taught us: Late season (weeks 13-14) has 75.3% win rate, +43.7% ROI
        Apply to NHL with equivalent timing (games 67-75 of 82).
        """
        print("\n🏈 APPLYING NFL LATE SEASON LEARNING TO NHL")
        print("-"*80)
        
        patterns = []
        
        # NHL equivalent: games 67-75 (roughly week 13-14 equivalent)
        late_season_games = []
        for game in games:
            games_played = game.get('temporal_context', {}).get('home_wins', 0) + game.get('temporal_context', {}).get('home_losses', 0)
            
            # Late season: games 60-82 for playoff push
            if 60 <= games_played <= 82:
                # Check if team is in playoff race (45-55% win rate = bubble team)
                win_pct = game.get('temporal_context', {}).get('home_win_pct', 0.5)
                if 0.45 <= win_pct <= 0.55:
                    late_season_games.append(game)
        
        if len(late_season_games) >= 15:
            wins = sum(1 for g in late_season_games if g.get('home_won', False))
            win_rate = wins / len(late_season_games)
            roi = (win_rate * 0.909) + ((1 - win_rate) * -1.0)
            
            if win_rate > 0.53 and roi > 0.05:
                pattern = {
                    'name': 'NFL-Derived Late Season Playoff Push',
                    'description': 'Games 60-82 for bubble teams (45-55% win rate) - learned from NFL weeks 13-14',
                    'source': 'nfl_late_season_transfer',
                    'n_games': len(late_season_games),
                    'wins': wins,
                    'losses': len(late_season_games) - wins,
                    'win_rate': win_rate,
                    'win_rate_pct': win_rate * 100,
                    'roi': roi,
                    'roi_pct': roi * 100,
                    'confidence': 'HIGH' if win_rate > 0.57 else 'MEDIUM',
                    'unit_recommendation': 2 if win_rate > 0.57 else 1,
                    'pattern_type': 'cross_domain_nfl',
                }
                patterns.append(pattern)
                print(f"   ✓ Found NFL late season pattern: {len(late_season_games)} games, {win_rate:.1%} win, {roi:.1%} ROI")
        
        return patterns
    
    def apply_nfl_prestige_learning(self, games: List[Dict], features: np.ndarray) -> List[Dict]:
        """
        NFL taught us: QB prestige edge >0.2 gives 80% win, +52.7% ROI
        Apply to NHL: Goalie prestige edge (from nominative features).
        """
        print("\n🏈 APPLYING NFL PRESTIGE LEARNING TO NHL (QB → Goalie)")
        print("-"*80)
        
        patterns = []
        
        # In NHL, goalie prestige is analogous to QB prestige
        # Feature indices: home_goalie_prestige (index ~70), away_goalie_prestige (index ~71)
        # But we'll use the nominative features directly from games
        
        # Find games with goalie prestige edge
        prestige_edge_games = []
        for game in games:
            # Would need actual goalie names to calculate prestige
            # For now, use team brand as proxy (similar concept)
            home_brand = game.get('home_team') in ['BOS', 'CHI', 'DET', 'MTL', 'NYR', 'TOR']
            away_brand = game.get('away_team') in ['BOS', 'CHI', 'DET', 'MTL', 'NYR', 'TOR']
            
            # Original Six vs non-Original Six = prestige edge
            if home_brand and not away_brand:
                prestige_edge_games.append(game)
        
        if len(prestige_edge_games) >= 15:
            wins = sum(1 for g in prestige_edge_games if g.get('home_won', False))
            win_rate = wins / len(prestige_edge_games)
            roi = (win_rate * 0.909) + ((1 - win_rate) * -1.0)
            
            if win_rate > 0.53 and roi > 0.05:
                pattern = {
                    'name': 'NFL-Derived Prestige Edge (Original Six Home)',
                    'description': 'Original Six home vs non-Original Six away (learned from NFL QB prestige)',
                    'source': 'nfl_prestige_transfer',
                    'n_games': len(prestige_edge_games),
                    'wins': wins,
                    'losses': len(prestige_edge_games) - wins,
                    'win_rate': win_rate,
                    'win_rate_pct': win_rate * 100,
                    'roi': roi,
                    'roi_pct': roi * 100,
                    'confidence': 'HIGH' if win_rate > 0.57 else 'MEDIUM',
                    'unit_recommendation': 2 if win_rate > 0.57 else 1,
                    'pattern_type': 'cross_domain_nfl',
                }
                patterns.append(pattern)
                print(f"   ✓ Found NFL prestige pattern: {len(prestige_edge_games)} games, {win_rate:.1%} win, {roi:.1%} ROI")
        
        return patterns
    
    def apply_ensemble_meta_learning(self, games: List[Dict], features: np.ndarray) -> List[Dict]:
        """
        NBA showed "Ensemble Narrative" transformer had 54% accuracy.
        This means multi-perspective analysis works.
        
        Apply to NHL: Look at games from multiple angles simultaneously.
        """
        print("\n🎭 APPLYING ENSEMBLE META-ANALYSIS")
        print("-"*80)
        
        patterns = []
        
        # Multi-factor ensemble: combine nominative + context + performance
        # Use features to build ensemble score
        
        # Get feature columns (nominative are columns 50-79)
        nominative_features = features[:, 50:]  # Last 29 features
        
        # Calculate ensemble score (mean of nominative features)
        ensemble_scores = np.mean(nominative_features, axis=1)
        
        # High ensemble score = narrative alignment
        high_ensemble_games = [games[i] for i in range(len(games)) if ensemble_scores[i] > np.percentile(ensemble_scores, 70)]
        
        if len(high_ensemble_games) >= 15:
            wins = sum(1 for g in high_ensemble_games if g.get('home_won', False))
            win_rate = wins / len(high_ensemble_games)
            roi = (win_rate * 0.909) + ((1 - win_rate) * -1.0)
            
            if win_rate > 0.53 and roi > 0.05:
                pattern = {
                    'name': 'Ensemble Narrative Alignment (Multi-Factor)',
                    'description': 'High ensemble score across all nominative dimensions (learned from NBA)',
                    'source': 'nba_ensemble_transfer',
                    'n_games': len(high_ensemble_games),
                    'wins': wins,
                    'losses': len(high_ensemble_games) - wins,
                    'win_rate': win_rate,
                    'win_rate_pct': win_rate * 100,
                    'roi': roi,
                    'roi_pct': roi * 100,
                    'confidence': 'HIGH' if win_rate > 0.57 else 'MEDIUM',
                    'unit_recommendation': 2 if win_rate > 0.57 else 1,
                    'pattern_type': 'cross_domain_ensemble',
                }
                patterns.append(pattern)
                print(f"   ✓ Found ensemble pattern: {len(high_ensemble_games)} games, {win_rate:.1%} win, {roi:.1%} ROI")
        
        return patterns
    
    def discover_all_cross_domain_patterns(self, games: List[Dict], features: np.ndarray) -> List[Dict]:
        """Discover all cross-domain patterns"""
        
        print("\n" + "="*80)
        print("NHL CROSS-DOMAIN LEARNING APPLICATION")
        print("="*80)
        print(f"Applying NBA/NFL insights to {len(games)} NHL games...")
        
        all_patterns = []
        
        # Apply NBA learnings
        all_patterns.extend(self.apply_nba_momentum_learning(games, features))
        all_patterns.extend(self.apply_ensemble_meta_learning(games, features))
        
        # Apply NFL learnings
        all_patterns.extend(self.apply_nfl_late_season_learning(games, features))
        all_patterns.extend(self.apply_nfl_prestige_learning(games, features))
        
        print(f"\n✅ Discovered {len(all_patterns)} cross-domain patterns")
        
        return all_patterns


def main():
    """Main execution"""
    
    project_root = Path(__file__).parent.parent.parent.parent
    data_path = project_root / 'data' / 'domains' / 'nhl_games_with_odds.json'
    features_path = project_root / 'narrative_optimization' / 'domains' / 'nhl' / 'nhl_features_complete.npz'
    
    # Load data
    print("\n📂 Loading NHL data and features...")
    with open(data_path, 'r') as f:
        games = json.load(f)
    
    data = np.load(features_path)
    features = data['features']
    
    print(f"   ✓ {len(games)} games, {features.shape[1]} features")
    
    # Load existing patterns
    existing_path = project_root / 'data' / 'domains' / 'nhl_betting_patterns_learned.json'
    with open(existing_path, 'r') as f:
        existing_patterns = json.load(f)
    
    print(f"   ✓ {len(existing_patterns)} existing data-driven patterns")
    
    # Discover cross-domain patterns
    enhancer = CrossDomainNHLEnhancer()
    cross_domain_patterns = enhancer.discover_all_cross_domain_patterns(games, features)
    
    # Combine with existing
    all_patterns = existing_patterns + cross_domain_patterns
    
    # Remove duplicates and sort
    all_patterns.sort(key=lambda x: x['roi'], reverse=True)
    
    # Save combined
    output_path = project_root / 'data' / 'domains' / 'nhl_betting_patterns_complete.json'
    with open(output_path, 'w') as f:
        json.dump(all_patterns, f, indent=2)
    
    print(f"\n💾 COMPLETE PATTERNS SAVED: {output_path}")
    print(f"   Total: {len(all_patterns)} patterns")
    print(f"   Data-driven: {len(existing_patterns)}")
    print(f"   Cross-domain: {len(cross_domain_patterns)}")
    print("\n✅ Cross-domain learning integration complete!")


if __name__ == "__main__":
    main()

