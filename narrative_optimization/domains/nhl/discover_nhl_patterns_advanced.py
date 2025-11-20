"""
NHL Advanced Pattern Discovery - Using ACTUAL Feature Analysis

This version uses the extracted features (79 dimensions) to find sophisticated
multi-factor patterns like we did for NBA/NFL, not just obvious stats.

Key difference: We cluster games by FEATURE SIMILARITY, then find which clusters
are profitable - this reveals hidden narrative patterns.

Author: Narrative Integration System
Date: November 16, 2025
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class AdvancedNHLPatternDiscoverer:
    """Advanced pattern discovery using actual extracted features"""
    
    def __init__(self, min_sample_size: int = 15, min_win_rate: float = 0.53, min_roi: float = 0.05):
        """
        Initialize advanced discoverer with RELAXED thresholds for 400 games.
        
        Parameters
        ----------
        min_sample_size : int
            Minimum games for pattern (15 for smaller dataset)
        min_win_rate : float
            Minimum win rate (53% relaxed from 55%)
        min_roi : float
            Minimum ROI (5% relaxed from 10%)
        """
        self.min_sample_size = min_sample_size
        self.min_win_rate = min_win_rate
        self.min_roi = min_roi
        
        self.patterns = []
    
    def discover_feature_based_patterns(self, games: List[Dict], features: np.ndarray) -> List[Dict]:
        """
        Discover patterns by clustering games based on their feature vectors,
        then identifying which clusters are profitable.
        
        This is how we find HIDDEN patterns that simple stats miss.
        """
        print("\n🧬 DISCOVERING FEATURE-BASED PATTERNS")
        print("-"*80)
        print("   Using 79-dimensional feature space to find hidden patterns...")
        
        patterns = []
        
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Try different numbers of clusters
        for n_clusters in [8, 12, 16, 20]:
            print(f"\n   Testing {n_clusters} clusters...")
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(features_scaled)
            
            # Evaluate each cluster
            for cluster_id in range(n_clusters):
                cluster_games = [games[i] for i in range(len(games)) if clusters[i] == cluster_id]
                
                if len(cluster_games) < self.min_sample_size:
                    continue
                
                # Calculate profitability
                wins = sum(1 for g in cluster_games if g.get('home_won', False))
                win_rate = wins / len(cluster_games)
                roi = (win_rate * 0.909) + ((1 - win_rate) * -1.0)
                
                if win_rate >= self.min_win_rate and roi >= self.min_roi:
                    # Find what makes this cluster unique
                    cluster_indices = [i for i in range(len(games)) if clusters[i] == cluster_id]
                    cluster_features = features[cluster_indices]
                    
                    # Get feature importance (distance from global mean)
                    global_mean = features.mean(axis=0)
                    cluster_mean = cluster_features.mean(axis=0)
                    feature_diff = np.abs(cluster_mean - global_mean)
                    
                    # Top 5 distinguishing features
                    top_features = np.argsort(feature_diff)[-5:][::-1]
                    
                    pattern = {
                        'name': f'Feature Cluster {n_clusters}-{cluster_id}',
                        'description': self._describe_cluster(cluster_games, top_features, cluster_mean),
                        'n_games': len(cluster_games),
                        'wins': wins,
                        'losses': len(cluster_games) - wins,
                        'win_rate': win_rate,
                        'win_rate_pct': win_rate * 100,
                        'roi': roi,
                        'roi_pct': roi * 100,
                        'cluster_id': cluster_id,
                        'n_clusters': n_clusters,
                        'top_features': top_features.tolist(),
                        'confidence': 'HIGH' if win_rate > 0.57 else 'MEDIUM',
                        'unit_recommendation': 2 if win_rate > 0.57 else 1,
                    }
                    
                    patterns.append(pattern)
        
        print(f"\n   ✓ Found {len(patterns)} feature-based patterns")
        return patterns
    
    def _describe_cluster(self, games: List[Dict], top_features: np.ndarray, cluster_mean: np.ndarray) -> str:
        """Generate human-readable description of what makes cluster unique"""
        
        # Analyze game characteristics
        rivalry_rate = sum(1 for g in games if g.get('is_rivalry', False)) / len(games)
        ot_rate = sum(1 for g in games if g.get('overtime', False)) / len(games)
        b2b_rate = sum(1 for g in games if g.get('temporal_context', {}).get('home_back_to_back', False)) / len(games)
        
        avg_rest = np.mean([g.get('temporal_context', {}).get('rest_advantage', 0) for g in games])
        
        # Build description
        parts = []
        
        if rivalry_rate > 0.15:
            parts.append("High rivalry rate")
        if ot_rate > 0.35:
            parts.append("Overtime-prone")
        if b2b_rate > 0.20:
            parts.append("Back-to-back situations")
        if avg_rest > 1.5:
            parts.append("Rest advantage")
        elif avg_rest < -1.5:
            parts.append("Rest disadvantage")
        
        # Check Original Six involvement
        original_six = ['BOS', 'CHI', 'DET', 'MTL', 'NYR', 'TOR']
        o6_rate = sum(1 for g in games 
                     if g.get('home_team') in original_six or g.get('away_team') in original_six) / len(games)
        
        if o6_rate > 0.30:
            parts.append("Original Six heavy")
        
        if not parts:
            parts.append("Mixed game situations")
        
        return " + ".join(parts)
    
    def discover_combination_patterns(self, games: List[Dict]) -> List[Dict]:
        """
        Discover multi-factor combination patterns.
        Like NFL: combine 2-3 factors to find edges.
        """
        print("\n🔬 DISCOVERING COMBINATION PATTERNS")
        print("-"*80)
        
        patterns = []
        
        # Pattern: Rivalry + Home Underdog
        rivalry_dog_games = []
        for g in games:
            if g.get('is_rivalry', False):
                tc = g.get('temporal_context', {})
                home_win_pct = tc.get('home_win_pct', 0.5)
                away_win_pct = tc.get('away_win_pct', 0.5)
                if home_win_pct < away_win_pct - 0.05:
                    rivalry_dog_games.append(g)
        
        if len(rivalry_dog_games) >= self.min_sample_size:
            pattern = self._evaluate_pattern(
                rivalry_dog_games,
                name="Rivalry Home Underdog",
                description="Worse team at home in rivalry game (extra motivation)"
            )
            if pattern:
                patterns.append(pattern)
        
        # Pattern: Original Six + Home
        o6_home_games = []
        original_six = ['BOS', 'CHI', 'DET', 'MTL', 'NYR', 'TOR']
        for g in games:
            if g.get('home_team') in original_six:
                o6_home_games.append(g)
        
        if len(o6_home_games) >= self.min_sample_size:
            pattern = self._evaluate_pattern(
                o6_home_games,
                name="Original Six at Home",
                description="Historic franchise with home ice advantage"
            )
            if pattern:
                patterns.append(pattern)
        
        # Pattern: Rest Advantage + Underdog
        rest_dog_games = []
        for g in games:
            tc = g.get('temporal_context', {})
            rest_adv = tc.get('rest_advantage', 0)
            home_win_pct = tc.get('home_win_pct', 0.5)
            away_win_pct = tc.get('away_win_pct', 0.5)
            
            if rest_adv >= 2 and home_win_pct < away_win_pct:
                rest_dog_games.append(g)
        
        if len(rest_dog_games) >= self.min_sample_size:
            pattern = self._evaluate_pattern(
                rest_dog_games,
                name="Rested Underdog",
                description="Worse team with 2+ days rest advantage"
            )
            if pattern:
                patterns.append(pattern)
        
        # Pattern: High-scoring overtime games
        high_scoring_ot = []
        for g in games:
            if g.get('overtime', False) and g.get('total_goals', 0) >= 6:
                high_scoring_ot.append(g)
        
        if len(high_scoring_ot) >= self.min_sample_size:
            pattern = self._evaluate_pattern(
                high_scoring_ot,
                name="High-Scoring Overtime",
                description="Games with 6+ goals going to OT (offensive edge)"
            )
            if pattern:
                patterns.append(pattern)
        
        # Pattern: Low-scoring games (tight defense)
        low_scoring = []
        for g in games:
            if g.get('total_goals', 10) <= 4:
                low_scoring.append(g)
        
        if len(low_scoring) >= self.min_sample_size:
            pattern = self._evaluate_pattern(
                low_scoring,
                name="Defensive Battle",
                description="Low-scoring games (≤4 goals total)"
            )
            if pattern:
                patterns.append(pattern)
        
        # Pattern: Strong form differential (L10)
        form_adv_games = []
        for g in games:
            tc = g.get('temporal_context', {})
            home_l10 = tc.get('home_l10_wins', 5)
            away_l10 = tc.get('away_l10_wins', 5)
            
            if home_l10 >= away_l10 + 3:
                form_adv_games.append(g)
        
        if len(form_adv_games) >= self.min_sample_size:
            pattern = self._evaluate_pattern(
                form_adv_games,
                name="Strong Recent Form Advantage",
                description="Home team 3+ more wins in L10 than visitor"
            )
            if pattern:
                patterns.append(pattern)
        
        # Pattern: Shootout tendency (teams that go to SO)
        shootout_games = []
        for g in games:
            if g.get('shootout', False):
                shootout_games.append(g)
        
        if len(shootout_games) >= self.min_sample_size:
            pattern = self._evaluate_pattern(
                shootout_games,
                name="Shootout Games",
                description="Games decided by shootout (50/50 skill)"
            )
            if pattern:
                patterns.append(pattern)
        
        # Pattern: Home favorite (strong team at home)
        home_fav_games = []
        for g in games:
            tc = g.get('temporal_context', {})
            home_win_pct = tc.get('home_win_pct', 0.5)
            away_win_pct = tc.get('away_win_pct', 0.5)
            
            if home_win_pct > away_win_pct + 0.10:
                home_fav_games.append(g)
        
        if len(home_fav_games) >= self.min_sample_size:
            pattern = self._evaluate_pattern(
                home_fav_games,
                name="Strong Home Favorite",
                description="Home team with 10%+ better record"
            )
            if pattern:
                patterns.append(pattern)
        
        # Pattern: Even matchup (50/50 games)
        even_games = []
        for g in games:
            tc = g.get('temporal_context', {})
            home_win_pct = tc.get('home_win_pct', 0.5)
            away_win_pct = tc.get('away_win_pct', 0.5)
            
            if abs(home_win_pct - away_win_pct) <= 0.05:
                even_games.append(g)
        
        if len(even_games) >= self.min_sample_size:
            pattern = self._evaluate_pattern(
                even_games,
                name="Evenly Matched",
                description="Teams within 5% win rate (toss-up + home ice)"
            )
            if pattern:
                patterns.append(pattern)
        
        # Pattern: Playoff-type intensity (rivalry + even)
        playoff_intensity = []
        for g in games:
            if g.get('is_rivalry', False):
                tc = g.get('temporal_context', {})
                home_win_pct = tc.get('home_win_pct', 0.5)
                away_win_pct = tc.get('away_win_pct', 0.5)
                if abs(home_win_pct - away_win_pct) <= 0.08:
                    playoff_intensity.append(g)
        
        if len(playoff_intensity) >= self.min_sample_size:
            pattern = self._evaluate_pattern(
                playoff_intensity,
                name="Rivalry + Evenly Matched",
                description="Intense rivalry between evenly matched teams"
            )
            if pattern:
                patterns.append(pattern)
        
        print(f"   ✓ Found {len(patterns)} combination patterns")
        return patterns
    
    def _evaluate_pattern(self, games: List[Dict], name: str, description: str) -> Optional[Dict]:
        """Evaluate a pattern for profitability"""
        if len(games) < self.min_sample_size:
            return None
        
        # Calculate win rate
        home_wins = sum(1 for g in games if g.get('home_won', False))
        win_rate = home_wins / len(games)
        
        # Calculate ROI (assuming -110 juice)
        roi = (win_rate * 0.909) + ((1 - win_rate) * -1.0)
        roi_pct = roi * 100
        
        # Check profitability thresholds
        if win_rate < self.min_win_rate:
            return None
        
        if roi < self.min_roi:
            return None
        
        # Pattern passes - compile stats
        pattern = {
            'name': name,
            'description': description,
            'n_games': len(games),
            'wins': home_wins,
            'losses': len(games) - home_wins,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'roi': roi,
            'roi_pct': roi_pct,
            'ats_rate': win_rate,
            'ats_pct': win_rate * 100,
            'profitable': roi > 0,
            'confidence': 'HIGH' if win_rate > 0.57 else 'MEDIUM' if win_rate > 0.54 else 'LOW',
            'unit_recommendation': 2 if win_rate > 0.57 else 1,
        }
        
        return pattern
    
    def discover_all_patterns(self, games: List[Dict], features: Optional[np.ndarray] = None) -> List[Dict]:
        """Discover all pattern types"""
        
        print("\n" + "="*80)
        print("NHL ADVANCED PATTERN DISCOVERY")
        print("="*80)
        print(f"Analyzing {len(games)} games with 79 features...")
        print(f"Relaxed thresholds: {self.min_sample_size}+ games, {self.min_win_rate:.1%}+ win, {self.min_roi:.1%}+ ROI")
        
        all_patterns = []
        
        # 1. Feature-based clustering patterns
        if features is not None:
            all_patterns.extend(self.discover_feature_based_patterns(games, features))
        
        # 2. Combination patterns
        all_patterns.extend(self.discover_combination_patterns(games))
        
        # Sort by ROI
        all_patterns.sort(key=lambda x: x['roi'], reverse=True)
        
        # Remove duplicates (keep best version)
        seen_games = set()
        unique_patterns = []
        for pattern in all_patterns:
            pattern_key = (pattern['n_games'], int(pattern['win_rate'] * 1000))
            if pattern_key not in seen_games:
                seen_games.add(pattern_key)
                unique_patterns.append(pattern)
        
        print("\n" + "="*80)
        print(f"✅ DISCOVERED {len(unique_patterns)} PROFITABLE PATTERNS")
        print("="*80)
        
        # Print summary
        for i, pattern in enumerate(unique_patterns[:15], 1):  # Top 15
            print(f"\n{i}. {pattern['name']}")
            print(f"   {pattern['description']}")
            print(f"   Games: {pattern['n_games']}, Win Rate: {pattern['win_rate_pct']:.1f}%, ROI: {pattern['roi_pct']:.1f}%")
        
        self.patterns = unique_patterns
        return unique_patterns


def main():
    """Main execution"""
    
    # Paths
    project_root = Path(__file__).parent.parent.parent.parent
    data_path = project_root / 'data' / 'domains' / 'nhl_games_with_odds.json'
    features_path = project_root / 'narrative_optimization' / 'domains' / 'nhl' / 'nhl_features_complete.npz'
    output_dir = project_root / 'data' / 'domains'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check data
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return
    
    # Load data
    print(f"\n📂 Loading NHL data...")
    with open(data_path, 'r') as f:
        games = json.load(f)
    print(f"   ✓ Loaded {len(games)} games")
    
    # Load features
    features = None
    if features_path.exists():
        print(f"📂 Loading extracted features...")
        data = np.load(features_path)
        features = data['features']
        print(f"   ✓ Loaded {features.shape[1]} features")
    else:
        print(f"   ⚠️  Features not found, using game-level patterns only")
    
    # Discover patterns with RELAXED thresholds
    discoverer = AdvancedNHLPatternDiscoverer(
        min_sample_size=15,  # Lower for 400 games
        min_win_rate=0.53,    # 53% instead of 55%
        min_roi=0.05         # 5% instead of 10%
    )
    
    patterns = discoverer.discover_all_patterns(games, features)
    
    # Save patterns
    output_path = output_dir / 'nhl_betting_patterns_advanced.json'
    with open(output_path, 'w') as f:
        json.dump(patterns, f, indent=2)
    
    print(f"\n💾 PATTERNS SAVED: {output_path}")
    print(f"✅ Advanced pattern discovery complete!")
    print(f"\n📊 Found {len(patterns)} sophisticated patterns vs 2 basic patterns")
    print(f"   Using actual 79-dimensional feature space")
    print(f"   Multi-factor combinations")
    print(f"   Feature-based clustering")


if __name__ == "__main__":
    main()

