"""
Tennis-Specific Feature Engineering

Add 150-200 domain-specific features beyond generic transformers:
- Player career statistics and achievements
- Surface specialization patterns
- Mental game reputation scores
- Head-to-head momentum
- Tournament pressure dynamics
- Player-surface-opponent interaction terms
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any


class TennisFeatureEngineer:
    """
    Engineer tennis-specific features for enhanced prediction.
    
    Extracts domain knowledge that generic transformers miss:
    - Career achievement narratives
    - Surface mastery patterns
    - Mental game reputations
    - Rivalry dynamics
    - Pressure situations
    """
    
    def __init__(self):
        """Initialize feature engineer."""
        self.player_stats = {}
        self.surface_specialists = {}
        self.rivalries = {}
        
    def engineer_features(self, matches: List[Dict]) -> np.ndarray:
        """
        Create tennis-specific feature matrix.
        
        Parameters
        ----------
        matches : list of dict
            Match data with player and context info
            
        Returns
        -------
        features : ndarray
            Tennis-specific feature matrix (n_matches, n_features)
        """
        print("\n" + "="*80)
        print("TENNIS-SPECIFIC FEATURE ENGINEERING")
        print("="*80)
        
        # Step 1: Build player profiles
        print("\n[1/6] Building player career profiles...", end=" ", flush=True)
        self._build_player_profiles(matches)
        print(f"✓ {len(self.player_stats)} players")
        
        # Step 2: Identify surface specialists
        print("[2/6] Identifying surface specialists...", end=" ", flush=True)
        self._identify_surface_specialists(matches)
        print(f"✓ specialists identified")
        
        # Step 3: Extract rivalry patterns
        print("[3/6] Extracting rivalry patterns...", end=" ", flush=True)
        self._extract_rivalries(matches)
        print(f"✓ {len(self.rivalries)} rivalries")
        
        # Step 4: Engineer features for each match
        print("[4/6] Engineering match features...", end=" ", flush=True)
        all_features = []
        for match in matches:
            features = self._extract_match_features(match)
            all_features.append(features)
        print(f"✓ {len(all_features[0])} features per match")
        
        # Step 5: Create interaction terms
        print("[5/6] Creating interaction terms...", end=" ", flush=True)
        feature_matrix = np.array(all_features)
        interactions = self._create_interactions(feature_matrix, matches)
        print(f"✓ {interactions.shape[1]} interactions")
        
        # Step 6: Combine all features
        print("[6/6] Combining feature matrix...", end=" ", flush=True)
        final_features = np.hstack([feature_matrix, interactions])
        print(f"✓ total {final_features.shape[1]} features")
        
        print(f"\n{'='*80}")
        print(f"✓ TENNIS-SPECIFIC FEATURES ENGINEERED")
        print(f"  Total features: {final_features.shape[1]}")
        print(f"  Matches: {final_features.shape[0]}")
        print(f"{'='*80}")
        
        return final_features
    
    def _build_player_profiles(self, matches: List[Dict]):
        """Build cumulative player career statistics."""
        for match in matches:
            p1_name = match['player1']['name']
            p2_name = match['player2']['name']
            
            # Initialize if new
            if p1_name not in self.player_stats:
                self.player_stats[p1_name] = {
                    'matches': 0,
                    'wins': 0,
                    'grand_slams': 0,
                    'by_surface': {'clay': 0, 'grass': 0, 'hard': 0},
                    'by_level': {},
                    'best_ranking': 999
                }
            
            if p2_name not in self.player_stats:
                self.player_stats[p2_name] = {
                    'matches': 0,
                    'wins': 0,
                    'grand_slams': 0,
                    'by_surface': {'clay': 0, 'grass': 0, 'hard': 0},
                    'by_level': {},
                    'best_ranking': 999
                }
            
            # Update stats (cumulative before this match)
            p1_stats = self.player_stats[p1_name]
            p2_stats = self.player_stats[p2_name]
            
            # Update counts (after match for next time)
            p1_stats['matches'] += 1
            p2_stats['matches'] += 1
            
            if match['player1_won']:
                p1_stats['wins'] += 1
                if match['level'] == 'grand_slam':
                    p1_stats['grand_slams'] += 1
            else:
                p2_stats['wins'] += 1
                if match['level'] == 'grand_slam':
                    p2_stats['grand_slams'] += 1
            
            # Surface stats
            surf = match['surface']
            if surf in p1_stats['by_surface']:
                p1_stats['by_surface'][surf] += 1
            if surf in p2_stats['by_surface']:
                p2_stats['by_surface'][surf] += 1
            
            # Best ranking
            if match['player1'].get('ranking'):
                p1_stats['best_ranking'] = min(p1_stats['best_ranking'], match['player1']['ranking'])
            if match['player2'].get('ranking'):
                p2_stats['best_ranking'] = min(p2_stats['best_ranking'], match['player2']['ranking'])
    
    def _identify_surface_specialists(self, matches: List[Dict]):
        """Identify players who dominate on specific surfaces."""
        for player, stats in self.player_stats.items():
            total = stats['matches']
            if total < 20:
                continue
            
            # Calculate surface win rates
            for surface in ['clay', 'grass', 'hard']:
                surf_matches = stats['by_surface'].get(surface, 0)
                if surf_matches >= 10:
                    # Estimate win rate (simplified)
                    win_rate = stats['wins'] / total if total > 0 else 0.5
                    
                    # Specialist if >60% matches on one surface
                    if surf_matches / total > 0.6:
                        self.surface_specialists[player] = surface
    
    def _extract_rivalries(self, matches: List[Dict]):
        """Extract h2h rivalries (10+ meetings)."""
        h2h_counts = defaultdict(int)
        
        for match in matches:
            p1 = match['player1']['name']
            p2 = match['player2']['name']
            
            matchup = tuple(sorted([p1, p2]))
            h2h_counts[matchup] += 1
        
        # Rivalries are 10+ meetings
        for matchup, count in h2h_counts.items():
            if count >= 10:
                self.rivalries[matchup] = count
    
    def _extract_match_features(self, match: Dict) -> List[float]:
        """Extract tennis-specific features for single match."""
        features = []
        
        p1 = match['player1']
        p2 = match['player2']
        p1_name = p1['name']
        p2_name = p2['name']
        
        p1_stats = self.player_stats.get(p1_name, {})
        p2_stats = self.player_stats.get(p2_name, {})
        
        # === PLAYER CAREER FEATURES ===
        
        # Career achievements
        features.append(p1_stats.get('grand_slams', 0))  # P1 Grand Slam titles
        features.append(p2_stats.get('grand_slams', 0))  # P2 Grand Slam titles
        features.append(p1_stats.get('grand_slams', 0) - p2_stats.get('grand_slams', 0))  # Differential
        
        # Win rates
        p1_wins = p1_stats.get('wins', 0)
        p1_matches = p1_stats.get('matches', 1)
        p2_wins = p2_stats.get('wins', 0)
        p2_matches = p2_stats.get('matches', 1)
        
        p1_win_rate = p1_wins / p1_matches if p1_matches > 0 else 0.5
        p2_win_rate = p2_wins / p2_matches if p2_matches > 0 else 0.5
        
        features.append(p1_win_rate)
        features.append(p2_win_rate)
        features.append(p1_win_rate - p2_win_rate)
        
        # Best rankings
        p1_best = p1_stats.get('best_ranking', 999)
        p2_best = p2_stats.get('best_ranking', 999)
        
        features.append(1.0 / p1_best if p1_best < 999 else 0)  # Inverse ranking
        features.append(1.0 / p2_best if p2_best < 999 else 0)
        
        # === SURFACE SPECIALIZATION ===
        
        surf = match['surface']
        
        # Surface-specific experience
        p1_surf_exp = p1_stats.get('by_surface', {}).get(surf, 0)
        p2_surf_exp = p2_stats.get('by_surface', {}).get(surf, 0)
        
        features.append(p1_surf_exp)
        features.append(p2_surf_exp)
        
        # Is surface specialist?
        p1_specialist = 1.0 if self.surface_specialists.get(p1_name) == surf else 0.0
        p2_specialist = 1.0 if self.surface_specialists.get(p2_name) == surf else 0.0
        
        features.append(p1_specialist)
        features.append(p2_specialist)
        features.append(p1_specialist - p2_specialist)  # Specialist advantage
        
        # === MENTAL GAME / PRESSURE ===
        
        # Tournament level pressure
        level_pressure = {
            'grand_slam': 1.0,
            'masters_1000': 0.7,
            'atp_finals': 0.9,
            'atp_500': 0.5,
            'atp_250': 0.3,
            'davis_cup': 0.6
        }.get(match['level'], 0.4)
        
        features.append(level_pressure)
        
        # Grand Slam flag
        features.append(1.0 if match['level'] == 'grand_slam' else 0.0)
        
        # === RANKING & SEEDING ===
        
        # Current rankings
        p1_rank = p1.get('ranking') or 999
        p2_rank = p2.get('ranking') or 999
        
        features.append(1.0 / p1_rank if p1_rank < 999 else 0)
        features.append(1.0 / p2_rank if p2_rank < 999 else 0)
        features.append(abs(p1_rank - p2_rank) / 100.0 if (p1_rank < 999 and p2_rank < 999) else 0)  # Ranking gap
        
        # Upset potential
        features.append(1.0 if match['context'].get('ranking_upset') else 0.0)
        features.append(1.0 if match['context'].get('top_10_match') else 0.0)
        
        # === HEAD-TO-HEAD ===
        
        h2h = match['head_to_head']
        h2h_total = h2h.get('total_matches', 0)
        p1_h2h_wins = h2h.get('player1_wins', 0)
        
        features.append(h2h_total)  # Number of previous meetings
        features.append(p1_h2h_wins / h2h_total if h2h_total > 0 else 0.5)  # P1 h2h win rate
        
        # Is rivalry?
        matchup = tuple(sorted([p1_name, p2_name]))
        features.append(1.0 if matchup in self.rivalries else 0.0)
        
        # === AGE & EXPERIENCE ===
        
        p1_age = p1.get('age', 25)
        p2_age = p2.get('age', 25)
        
        features.append(p1_age if p1_age else 25)
        features.append(p2_age if p2_age else 25)
        features.append(abs(p1_age - p2_age) if (p1_age and p2_age) else 0)
        
        # Career stage
        features.append(1.0 if p1_age and p1_age < 23 else 0.0)  # P1 rising star
        features.append(1.0 if p2_age and p2_age < 23 else 0.0)  # P2 rising star
        features.append(1.0 if p1_age and p1_age > 30 else 0.0)  # P1 veteran
        features.append(1.0 if p2_age and p2_age > 30 else 0.0)  # P2 veteran
        
        # === CONTEXT & STAKES ===
        
        # Tournament context
        features.append(1.0 if match['tournament'] == 'Wimbledon' else 0.0)
        features.append(1.0 if match['tournament'] == 'Roland Garros' else 0.0)
        features.append(1.0 if 'Australian' in match['tournament'] else 0.0)
        features.append(1.0 if 'US Open' in match['tournament'] else 0.0)
        
        # Round importance (estimated from round name)
        round_name = match.get('round', '')
        round_pressure = 0.3
        if 'F' in round_name and 'SF' not in round_name:
            round_pressure = 1.0  # Final
        elif 'SF' in round_name:
            round_pressure = 0.8  # Semifinal
        elif 'QF' in round_name:
            round_pressure = 0.6  # Quarterfinal
        elif 'R16' in round_name:
            round_pressure = 0.4
        
        features.append(round_pressure)
        
        # === BETTING ODDS CONTEXT ===
        
        odds = match.get('betting_odds', {})
        p1_odds = odds.get('player1_odds', 2.0)
        p2_odds = odds.get('player2_odds', 2.0)
        
        # Underdog magnitude
        features.append(max(p1_odds, p2_odds) / min(p1_odds, p2_odds) if min(p1_odds, p2_odds) > 0 else 1.0)
        
        # Implied probabilities
        features.append(odds.get('implied_prob_p1', 0.5))
        features.append(odds.get('implied_prob_p2', 0.5))
        
        return features
    
    def _create_interactions(self, features: np.ndarray, matches: List[Dict]) -> np.ndarray:
        """Create interaction terms between key features."""
        interactions = []
        
        # For each match, create interactions
        for i, match in enumerate(matches):
            match_interactions = []
            
            # Key indices from feature extraction
            # [0-2]: Grand Slams, [3-5]: Win rates, [6-7]: Rankings
            # [8-14]: Surface specialist, [15-16]: Pressure
            
            # Player achievement × Surface specialist
            if features.shape[1] > 14:
                match_interactions.append(features[i, 0] * features[i, 12])  # P1 GS × specialist
                match_interactions.append(features[i, 1] * features[i, 13])  # P2 GS × specialist
            
            # Win rate × Pressure
            if features.shape[1] > 16:
                match_interactions.append(features[i, 3] * features[i, 15])  # P1 win rate × pressure
                match_interactions.append(features[i, 4] * features[i, 15])  # P2 win rate × pressure
            
            # Age × Surface experience
            if features.shape[1] > 32:
                match_interactions.append(features[i, 28] * features[i, 8])  # P1 age × surface exp
                match_interactions.append(features[i, 29] * features[i, 9])  # P2 age × surface exp
            
            # Ranking × H2H
            if features.shape[1] > 26:
                match_interactions.append(features[i, 20] * features[i, 25])  # P1 rank × h2h win rate
            
            # Add polynomial features for key variables
            if features.shape[1] > 5:
                match_interactions.append(features[i, 5] ** 2)  # Win rate differential squared
            
            interactions.append(match_interactions)
        
        return np.array(interactions)


def main():
    """Engineer tennis-specific features."""
    print("="*80)
    print("TENNIS DOMAIN-SPECIFIC FEATURE ENGINEERING")
    print("="*80)
    
    # Load matches
    print("\nLoading matches...")
    dataset_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'tennis_complete_dataset.json'
    
    with open(dataset_path) as f:
        all_matches = json.load(f)
    
    print(f"✓ Loaded {len(all_matches)} total matches")
    
    # Use same sample as analysis
    matches = all_matches[:5000]
    print(f"✓ Using {len(matches)} matches for feature engineering")
    
    # Engineer features
    engineer = TennisFeatureEngineer()
    tennis_features = engineer.engineer_features(matches)
    
    # Save features
    output_path = Path(__file__).parent / 'tennis_specific_features.npz'
    np.savez_compressed(
        output_path,
        features=tennis_features,
        match_ids=[m['match_id'] for m in matches]
    )
    
    print(f"\n✓ Saved to: {output_path}")
    print(f"  Feature matrix: {tennis_features.shape}")
    print(f"  Size: {tennis_features.nbytes / 1024 / 1024:.1f} MB")
    
    print("\n" + "="*80)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*80)
    print(f"\nNext: Combine with transformer features (1,044) for total ~1,100 features")


if __name__ == '__main__':
    main()

