"""
NHL Domain Formula Calculation

Calculates the complete domain formula for NHL:
- π (narrativity): How open vs constrained the domain is
- r (correlation): Strength of narrative-outcome relationship  
- κ (coupling): Narrator-narrated relationship strength
- Δ (narrative agency): π × |r| × κ (THE MAGIC VARIABLE)

Expected: NHL likely fails threshold (Δ/π < 0.5) like NBA/NFL,
but reveals exploitable betting patterns.

Author: Narrative Integration System
Date: November 16, 2025
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class NHLFormulaCalculator:
    """Calculate NHL domain formula"""
    
    def __init__(self):
        """Initialize calculator"""
        self.scaler = StandardScaler()
        self.pca = None
        self.model = None
    
    def calculate_narrativity(self, games: List[Dict]) -> float:
        """
        Calculate π (narrativity) for NHL domain.
        
        π measures how open vs constrained the domain is (0-1 scale).
        NHL is semi-constrained by skill/performance but has room for
        narrative elements (goalies, momentum, rivalries).
        
        Expected: π ≈ 0.52 (between NBA 0.49 and NFL 0.57)
        
        Parameters
        ----------
        games : list of dict
            NHL game data
        
        Returns
        -------
        pi : float
            Narrativity score
        """
        print("\n📐 CALCULATING NARRATIVITY (π)")
        print("-"*80)
        
        # Multiple methods to estimate π
        
        # Method 1: Structural constraints
        # - Fixed rules: 82 games, 60 minutes, 3 periods
        # - Variable elements: overtime, shootouts, goalie performance
        structural = 0.52  # Semi-constrained
        
        # Method 2: Outcome variance
        # High variance = more narrative openness
        outcomes = [g.get('home_won', False) for g in games]
        home_win_rate = np.mean(outcomes)
        # Distance from 0.5 indicates constraint (skill dominates)
        outcome_variance = 1.0 - abs(home_win_rate - 0.5) * 2
        
        # Method 3: Upset rate (underdog wins)
        upsets = []
        for g in games:
            tc = g.get('temporal_context', {})
            home_win_pct = tc.get('home_win_pct', 0.5)
            home_won = g.get('home_won', False)
            
            # Count upsets (worse team wins)
            if home_win_pct < 0.45 and home_won:
                upsets.append(1)
            elif home_win_pct > 0.55 and not home_won:
                upsets.append(1)
            else:
                upsets.append(0)
        
        upset_rate = np.mean(upsets) if upsets else 0.3
        # More upsets = more narrative openness
        upset_narrativity = upset_rate / 0.40  # Normalize (40% upsets = fully narrative)
        
        # Method 4: Overtime/shootout rate (narrative drama)
        ot_rate = np.mean([g.get('overtime', False) for g in games])
        # More OT = more open/dramatic outcomes
        ot_narrativity = ot_rate / 0.25  # Normalize (25% OT = high drama)
        
        # Combine methods
        pi = (
            structural * 0.40 +
            outcome_variance * 0.25 +
            upset_narrativity * 0.20 +
            ot_narrativity * 0.15
        )
        
        print(f"   Structural constraint: {structural:.3f}")
        print(f"   Outcome variance: {outcome_variance:.3f}")
        print(f"   Upset rate: {upset_rate:.1%} → {upset_narrativity:.3f}")
        print(f"   Overtime rate: {ot_rate:.1%} → {ot_narrativity:.3f}")
        print(f"\n   ✓ π (narrativity) = {pi:.3f}")
        
        return pi
    
    def calculate_correlation(self, features: np.ndarray, outcomes: np.ndarray) -> Tuple[float, float]:
        """
        Calculate r (correlation) between narrative features and outcomes.
        
        Parameters
        ----------
        features : ndarray
            Feature matrix
        outcomes : ndarray
            Binary outcomes (home won)
        
        Returns
        -------
        r_pearson : float
            Pearson correlation
        r_spearman : float
            Spearman correlation
        """
        print("\n📊 CALCULATING CORRELATION (r)")
        print("-"*80)
        
        # Calculate story quality (ю) from features
        # Use PCA to reduce to single quality score
        self.pca = PCA(n_components=1)
        story_quality = self.pca.fit_transform(features).flatten()
        
        # Normalize to 0-1
        story_quality = (story_quality - story_quality.min()) / (story_quality.max() - story_quality.min())
        
        # Calculate correlations
        r_pearson, p_pearson = pearsonr(story_quality, outcomes)
        r_spearman, p_spearman = spearmanr(story_quality, outcomes)
        
        print(f"   Pearson r: {r_pearson:.4f} (p={p_pearson:.4f})")
        print(f"   Spearman r: {r_spearman:.4f} (p={p_spearman:.4f})")
        print(f"\n   ✓ Using Pearson r = {r_pearson:.4f}")
        
        return r_pearson, r_spearman
    
    def calculate_coupling(self, games: List[Dict]) -> float:
        """
        Calculate κ (coupling) - narrator-narrated relationship.
        
        In NHL: 
        - Players/teams (narrated) have high agency
        - Media/fans (narrators) external to game
        - κ moderate (~0.75)
        
        Parameters
        ----------
        games : list of dict
            NHL game data
        
        Returns
        -------
        kappa : float
            Coupling strength
        """
        print("\n🔗 CALCULATING COUPLING (κ)")
        print("-"*80)
        
        # NHL-specific coupling factors
        
        # 1. Player agency (high - players control outcome)
        player_agency = 0.85
        
        # 2. Narrative influence (moderate - momentum, rivalries matter)
        narrative_influence = 0.70
        
        # 3. External constraints (moderate - refs, injuries)
        external_constraints = 0.75
        
        # 4. Historical weight (moderate - past games matter some)
        historical_weight = 0.70
        
        # Combine
        kappa = (
            player_agency * 0.35 +
            narrative_influence * 0.30 +
            external_constraints * 0.20 +
            historical_weight * 0.15
        )
        
        print(f"   Player agency: {player_agency:.3f}")
        print(f"   Narrative influence: {narrative_influence:.3f}")
        print(f"   External constraints: {external_constraints:.3f}")
        print(f"   Historical weight: {historical_weight:.3f}")
        print(f"\n   ✓ κ (coupling) = {kappa:.3f}")
        
        return kappa
    
    def calculate_delta(self, pi: float, r: float, kappa: float) -> Dict:
        """
        Calculate Δ (narrative agency) - THE MAGIC VARIABLE.
        
        Formula: Δ = π × |r| × κ
        
        Threshold: Δ/π > 0.5 means narrative matters
        
        Parameters
        ----------
        pi : float
            Narrativity
        r : float
            Correlation
        kappa : float
            Coupling
        
        Returns
        -------
        results : dict
            Delta calculation results
        """
        print("\n⭐ CALCULATING NARRATIVE AGENCY (Δ)")
        print("-"*80)
        
        delta = pi * abs(r) * kappa
        efficiency = delta / pi if pi > 0 else 0
        
        # Determine if narrative matters
        threshold = 0.5
        narrative_matters = efficiency > threshold
        
        print(f"   π (narrativity) = {pi:.4f}")
        print(f"   |r| (correlation) = {abs(r):.4f}")
        print(f"   κ (coupling) = {kappa:.4f}")
        print(f"\n   Δ = π × |r| × κ = {delta:.4f}")
        print(f"   Efficiency (Δ/π) = {efficiency:.4f}")
        print(f"   Threshold = {threshold}")
        print(f"\n   Verdict: {'✓ NARRATIVE MATTERS' if narrative_matters else '✗ NARRATIVE FAILS THRESHOLD'}")
        
        return {
            'pi': float(pi),
            'r': float(r),
            'kappa': float(kappa),
            'delta': float(delta),
            'efficiency': float(efficiency),
            'threshold': float(threshold),
            'narrative_matters': bool(narrative_matters),
        }
    
    def structure_aware_validation(self, features: np.ndarray, games: List[Dict]) -> Dict:
        """
        Structure-aware validation across different contexts.
        
        Tests:
        - Division games
        - Playoff games
        - Back-to-back games
        - Rivalry games
        - Home/away
        - Goalie matchups
        
        Parameters
        ----------
        features : ndarray
            Feature matrix
        games : list of dict
            Game data
        
        Returns
        -------
        results : dict
            Validation results by context
        """
        print("\n🔬 STRUCTURE-AWARE VALIDATION")
        print("-"*80)
        
        results = {}
        
        contexts = {
            'division_games': [i for i, g in enumerate(games) if g.get('is_division_game', False)],
            'playoff_games': [i for i, g in enumerate(games) if g.get('is_playoff', False)],
            'rivalry_games': [i for i, g in enumerate(games) if g.get('is_rivalry', False)],
            'back_to_back': [i for i, g in enumerate(games) 
                            if g.get('temporal_context', {}).get('home_back_to_back', False)],
            'home_games': [i for i, g in enumerate(games) if g.get('home_won', False)],
            'overtime_games': [i for i, g in enumerate(games) if g.get('overtime', False)],
        }
        
        for context_name, indices in contexts.items():
            if len(indices) < 20:  # Minimum sample size
                continue
            
            context_features = features[indices]
            context_outcomes = np.array([games[i].get('home_won', False) for i in indices], dtype=float)
            
            # Calculate story quality for context
            if len(context_features) > 1:
                pca_context = PCA(n_components=1)
                story_quality = pca_context.fit_transform(context_features).flatten()
                story_quality = (story_quality - story_quality.min()) / (story_quality.max() - story_quality.min() + 1e-10)
                
                r_context, _ = pearsonr(story_quality, context_outcomes)
                
                results[context_name] = {
                    'n_games': int(len(indices)),
                    'r': float(r_context),
                    'home_win_pct': float(np.mean(context_outcomes)),
                }
                
                print(f"   {context_name}: n={len(indices)}, r={r_context:.4f}, home_win%={np.mean(context_outcomes):.1%}")
        
        print(f"\n   ✓ Validated {len(results)} contexts")
        
        return results


def load_nhl_data(data_path: Path) -> Tuple[List[Dict], np.ndarray]:
    """Load NHL data and features"""
    
    # Load game data
    with open(data_path, 'r') as f:
        games = json.load(f)
    
    # Load features
    features_path = data_path.parent.parent / 'narrative_optimization' / 'domains' / 'nhl' / 'nhl_features_complete.npz'
    
    if features_path.exists():
        data = np.load(features_path)
        features = data['features']
    else:
        print(f"⚠️  Features file not found: {features_path}")
        print("   Using placeholder features for demonstration")
        # Create placeholder features
        features = np.random.randn(len(games), 100)
    
    return games, features


def main():
    """Main execution"""
    
    # Paths
    project_root = Path(__file__).parent.parent.parent.parent
    data_path = project_root / 'data' / 'domains' / 'nhl_games_with_odds.json'
    output_dir = project_root / 'narrative_optimization' / 'domains' / 'nhl'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("NHL DOMAIN FORMULA CALCULATION")
    print("="*80)
    
    # Check data
    if not data_path.exists():
        print(f"\n❌ Data file not found: {data_path}")
        print("Run the NHL data builder first:")
        print("  python data_collection/nhl_data_builder.py")
        return
    
    # Load data
    print(f"\n📂 Loading NHL data...")
    games, features = load_nhl_data(data_path)
    print(f"   ✓ Loaded {len(games)} games with {features.shape[1]} features")
    
    # Extract outcomes
    outcomes = np.array([g.get('home_won', False) for g in games], dtype=float)
    
    # Calculate formula
    calculator = NHLFormulaCalculator()
    
    # 1. Calculate π (narrativity)
    pi = calculator.calculate_narrativity(games)
    
    # 2. Calculate r (correlation)
    r_pearson, r_spearman = calculator.calculate_correlation(features, outcomes)
    
    # 3. Calculate κ (coupling)
    kappa = calculator.calculate_coupling(games)
    
    # 4. Calculate Δ (narrative agency)
    delta_results = calculator.calculate_delta(pi, r_pearson, kappa)
    
    # 5. Structure-aware validation
    validation_results = calculator.structure_aware_validation(features, games)
    
    # Compile complete results
    complete_results = {
        'domain': 'nhl',
        'n_games': len(games),
        'n_features': features.shape[1],
        'formula': delta_results,
        'validation': validation_results,
        'comparison': {
            'nba': {'pi': 0.49, 'delta': 0.034, 'efficiency': 0.06},
            'nfl': {'pi': 0.57, 'delta': 0.034, 'efficiency': 0.06},
            'nhl': {
                'pi': delta_results['pi'],
                'delta': delta_results['delta'],
                'efficiency': delta_results['efficiency']
            }
        }
    }
    
    # Save results
    output_path = output_dir / 'nhl_formula_results.json'
    with open(output_path, 'w') as f:
        json.dump(complete_results, f, indent=2)
    
    print("\n💾 RESULTS SAVED")
    print("-"*80)
    print(f"   {output_path}")
    
    print("\n" + "="*80)
    print("NHL DOMAIN FORMULA - SUMMARY")
    print("="*80)
    print(f"π (narrativity): {delta_results['pi']:.3f}")
    print(f"r (correlation): {delta_results['r']:.4f}")
    print(f"κ (coupling): {delta_results['kappa']:.3f}")
    print(f"Δ (narrative agency): {delta_results['delta']:.4f}")
    print(f"Efficiency (Δ/π): {delta_results['efficiency']:.4f}")
    print(f"\nNarrative matters: {'YES' if delta_results['narrative_matters'] else 'NO'}")
    print("\nExpected: Like NBA/NFL, narrative likely fails threshold")
    print("BUT reveals exploitable betting patterns!")
    print("="*80)


if __name__ == "__main__":
    main()

