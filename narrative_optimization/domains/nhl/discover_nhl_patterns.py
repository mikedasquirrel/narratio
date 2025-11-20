"""
NHL Pattern Discovery - Betting System

Discovers profitable betting patterns in NHL games following the
NFL methodology that found 16 patterns with 55-96% win rates.

Pattern Categories:
1. Goalie-based patterns (hot goalie, backup, matchup history)
2. Underdog patterns (home dog, rest advantage, division dog)
3. Special teams patterns (PP%, PK%, differential)
4. Rivalry patterns (Original Six, playoff rematches)
5. Momentum patterns (win streaks, loss revenge)
6. Contextual patterns (back-to-back, rest, late season)

Validation Criteria:
- Win rate > 55% (significant edge)
- Sample size > 20 games (statistical confidence)
- ROI > 10% (profitable after juice)
- Temporal stability (works across seasons)

Author: Narrative Integration System
Date: November 16, 2025
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class NHLPatternDiscoverer:
    """Discover profitable betting patterns in NHL games"""
    
    def __init__(self, min_sample_size: int = 20, min_win_rate: float = 0.55, min_roi: float = 0.10):
        """
        Initialize pattern discoverer.
        
        Parameters
        ----------
        min_sample_size : int
            Minimum games for pattern validation
        min_win_rate : float
            Minimum win rate for profitability
        min_roi : float
            Minimum ROI for profitability
        """
        self.min_sample_size = min_sample_size
        self.min_win_rate = min_win_rate
        self.min_roi = min_roi
        
        self.patterns = []
    
    def discover_goalie_patterns(self, games: List[Dict]) -> List[Dict]:
        """
        Discover goalie-based patterns.
        
        Hockey-specific: Goalies are THE most critical narrative element.
        A hot goalie can carry a team. Backup goalies create opportunity.
        """
        print("\n🥅 DISCOVERING GOALIE PATTERNS")
        print("-"*80)
        
        patterns = []
        
        # Pattern: Hot home goalie (SV% > .920 L5 games)
        hot_goalie_games = []
        for g in games:
            goalie_recent_sv = g.get('goalie_recent_sv_pct', 0.905)
            if goalie_recent_sv > 0.920:
                hot_goalie_games.append(g)
        
        if len(hot_goalie_games) >= self.min_sample_size:
            pattern = self._evaluate_pattern(
                hot_goalie_games,
                name="Hot Home Goalie (SV% > .920 L5)",
                description="Home team with goalie on hot streak"
            )
            if pattern:
                patterns.append(pattern)
        
        # Pattern: Backup goalie advantage (starter rested)
        backup_games = []
        for g in games:
            is_starter = g.get('goalie_is_starter', True)
            if not is_starter:
                backup_games.append(g)
        
        if len(backup_games) >= self.min_sample_size:
            pattern = self._evaluate_pattern(
                backup_games,
                name="Backup Goalie Start",
                description="Backup goalie getting start (starter rested)"
            )
            if pattern:
                patterns.append(pattern)
        
        # Pattern: Goalie vs opponent dominance
        goalie_vs_opp_games = []
        for g in games:
            goalie_vs_opp = g.get('goalie_vs_opponent_sv', 0.905)
            if goalie_vs_opp > 0.930:  # Career dominance vs this opponent
                goalie_vs_opp_games.append(g)
        
        if len(goalie_vs_opp_games) >= self.min_sample_size:
            pattern = self._evaluate_pattern(
                goalie_vs_opp_games,
                name="Goalie Career Dominance vs Opponent",
                description="Goalie with career SV% > .930 vs this opponent"
            )
            if pattern:
                patterns.append(pattern)
        
        print(f"   ✓ Found {len(patterns)} goalie patterns")
        return patterns
    
    def discover_underdog_patterns(self, games: List[Dict]) -> List[Dict]:
        """Discover underdog betting patterns"""
        
        print("\n🐕 DISCOVERING UNDERDOG PATTERNS")
        print("-"*80)
        
        patterns = []
        
        # Pattern: Home underdog (based on win %)
        home_dog_games = []
        for g in games:
            tc = g.get('temporal_context', {})
            home_win_pct = tc.get('home_win_pct', 0.5)
            away_win_pct = tc.get('away_win_pct', 0.5)
            
            # Home is underdog
            if home_win_pct < away_win_pct - 0.10:  # Significant underdog
                home_dog_games.append(g)
        
        if len(home_dog_games) >= self.min_sample_size:
            pattern = self._evaluate_pattern(
                home_dog_games,
                name="Home Underdog (Win% Gap > .100)",
                description="Home team with significantly worse record"
            )
            if pattern:
                patterns.append(pattern)
        
        # Pattern: Underdog with rest advantage
        rest_dog_games = []
        for g in games:
            tc = g.get('temporal_context', {})
            home_win_pct = tc.get('home_win_pct', 0.5)
            away_win_pct = tc.get('away_win_pct', 0.5)
            rest_advantage = tc.get('rest_advantage', 0)
            
            # Home underdog with rest edge
            if home_win_pct < away_win_pct and rest_advantage >= 2:
                rest_dog_games.append(g)
        
        if len(rest_dog_games) >= self.min_sample_size:
            pattern = self._evaluate_pattern(
                rest_dog_games,
                name="Underdog with Rest Advantage (2+ days)",
                description="Worse team but well-rested"
            )
            if pattern:
                patterns.append(pattern)
        
        # Pattern: Division underdog (familiarity advantage)
        div_dog_games = []
        for g in games:
            tc = g.get('temporal_context', {})
            home_win_pct = tc.get('home_win_pct', 0.5)
            away_win_pct = tc.get('away_win_pct', 0.5)
            is_division = g.get('is_division_game', False)
            
            if is_division and home_win_pct < away_win_pct:
                div_dog_games.append(g)
        
        if len(div_dog_games) >= self.min_sample_size:
            pattern = self._evaluate_pattern(
                div_dog_games,
                name="Division Game Underdog",
                description="Underdog in familiar division matchup"
            )
            if pattern:
                patterns.append(pattern)
        
        print(f"   ✓ Found {len(patterns)} underdog patterns")
        return patterns
    
    def discover_special_teams_patterns(self, games: List[Dict]) -> List[Dict]:
        """Discover special teams patterns"""
        
        print("\n⚡ DISCOVERING SPECIAL TEAMS PATTERNS")
        print("-"*80)
        
        patterns = []
        
        # Pattern: Hot power play (>25% L10)
        hot_pp_games = []
        for g in games:
            pp_pct = g.get('recent_pp_pct', g.get('power_play_pct', 0.20))
            if pp_pct > 0.25:
                hot_pp_games.append(g)
        
        if len(hot_pp_games) >= self.min_sample_size:
            pattern = self._evaluate_pattern(
                hot_pp_games,
                name="Hot Power Play (>25% recent)",
                description="Team with elite recent PP performance"
            )
            if pattern:
                patterns.append(pattern)
        
        # Pattern: Elite penalty kill (>85% season)
        elite_pk_games = []
        for g in games:
            pk_pct = g.get('penalty_kill_pct', 0.80)
            if pk_pct > 0.85:
                elite_pk_games.append(g)
        
        if len(elite_pk_games) >= self.min_sample_size:
            pattern = self._evaluate_pattern(
                elite_pk_games,
                name="Elite Penalty Kill (>85%)",
                description="Team with elite PK shuts down opponents"
            )
            if pattern:
                patterns.append(pattern)
        
        # Pattern: Special teams differential advantage
        st_diff_games = []
        for g in games:
            pp_pct = g.get('power_play_pct', 0.20)
            opp_pk_pct = g.get('opponent_pk_pct', 0.80)
            differential = pp_pct - (1 - opp_pk_pct)
            
            if differential > 0.10:  # Significant advantage
                st_diff_games.append(g)
        
        if len(st_diff_games) >= self.min_sample_size:
            pattern = self._evaluate_pattern(
                st_diff_games,
                name="Special Teams Differential >10%",
                description="PP% significantly better than opponent PK%"
            )
            if pattern:
                patterns.append(pattern)
        
        print(f"   ✓ Found {len(patterns)} special teams patterns")
        return patterns
    
    def discover_rivalry_patterns(self, games: List[Dict]) -> List[Dict]:
        """Discover rivalry patterns"""
        
        print("\n🔥 DISCOVERING RIVALRY PATTERNS")
        print("-"*80)
        
        patterns = []
        
        # Pattern: Original Six matchups
        original_six_games = []
        for g in games:
            if g.get('is_rivalry', False):
                # Check if Original Six
                home_team = g.get('home_team', '')
                away_team = g.get('away_team', '')
                original_six = ['BOS', 'CHI', 'DET', 'MTL', 'NYR', 'TOR']
                
                if home_team in original_six and away_team in original_six:
                    original_six_games.append(g)
        
        if len(original_six_games) >= self.min_sample_size:
            pattern = self._evaluate_pattern(
                original_six_games,
                name="Original Six Rivalry",
                description="Classic Original Six matchup"
            )
            if pattern:
                patterns.append(pattern)
        
        # Pattern: Rivalry + home underdog
        rivalry_dog_games = []
        for g in games:
            if g.get('is_rivalry', False):
                tc = g.get('temporal_context', {})
                home_win_pct = tc.get('home_win_pct', 0.5)
                away_win_pct = tc.get('away_win_pct', 0.5)
                
                if home_win_pct < away_win_pct:
                    rivalry_dog_games.append(g)
        
        if len(rivalry_dog_games) >= self.min_sample_size:
            pattern = self._evaluate_pattern(
                rivalry_dog_games,
                name="Rivalry Home Underdog",
                description="Home underdog in rivalry game (extra motivation)"
            )
            if pattern:
                patterns.append(pattern)
        
        print(f"   ✓ Found {len(patterns)} rivalry patterns")
        return patterns
    
    def discover_momentum_patterns(self, games: List[Dict]) -> List[Dict]:
        """Discover momentum patterns"""
        
        print("\n📈 DISCOVERING MOMENTUM PATTERNS")
        print("-"*80)
        
        patterns = []
        
        # Pattern: Win streak (3+ games)
        win_streak_games = []
        for g in games:
            tc = g.get('temporal_context', {})
            l5_wins = tc.get('home_l10_wins', 5)
            
            if l5_wins >= 7:  # 7+ wins in L10 = hot streak
                win_streak_games.append(g)
        
        if len(win_streak_games) >= self.min_sample_size:
            pattern = self._evaluate_pattern(
                win_streak_games,
                name="Hot Streak (7+ wins in L10)",
                description="Team riding momentum"
            )
            if pattern:
                patterns.append(pattern)
        
        # Pattern: Loss revenge (after 3+ losses)
        revenge_games = []
        for g in games:
            tc = g.get('temporal_context', {})
            l5_wins = tc.get('home_l10_wins', 5)
            
            if l5_wins <= 3:  # 3 or fewer wins in L10 = struggling, due for bounce
                revenge_games.append(g)
        
        if len(revenge_games) >= self.min_sample_size:
            pattern = self._evaluate_pattern(
                revenge_games,
                name="Bounce-Back Spot (≤3 wins in L10)",
                description="Struggling team due for bounce-back"
            )
            if pattern:
                patterns.append(pattern)
        
        # Pattern: Form differential advantage
        form_diff_games = []
        for g in games:
            tc = g.get('temporal_context', {})
            home_l10 = tc.get('home_l10_wins', 5)
            away_l10 = tc.get('away_l10_wins', 5)
            
            if home_l10 - away_l10 >= 3:  # Significant form advantage
                form_diff_games.append(g)
        
        if len(form_diff_games) >= self.min_sample_size:
            pattern = self._evaluate_pattern(
                form_diff_games,
                name="Form Differential (3+ wins advantage)",
                description="Team significantly hotter in recent form"
            )
            if pattern:
                patterns.append(pattern)
        
        print(f"   ✓ Found {len(patterns)} momentum patterns")
        return patterns
    
    def discover_contextual_patterns(self, games: List[Dict]) -> List[Dict]:
        """Discover contextual patterns"""
        
        print("\n🗓️  DISCOVERING CONTEXTUAL PATTERNS")
        print("-"*80)
        
        patterns = []
        
        # Pattern: Back-to-back fade (opponent on B2B)
        b2b_fade_games = []
        for g in games:
            tc = g.get('temporal_context', {})
            away_b2b = tc.get('away_back_to_back', False)
            
            if away_b2b:
                b2b_fade_games.append(g)
        
        if len(b2b_fade_games) >= self.min_sample_size:
            pattern = self._evaluate_pattern(
                b2b_fade_games,
                name="Opponent Back-to-Back",
                description="Facing team on back-to-back (fatigue edge)"
            )
            if pattern:
                patterns.append(pattern)
        
        # Pattern: Rest advantage (3+ days vs <2)
        rest_adv_games = []
        for g in games:
            tc = g.get('temporal_context', {})
            rest_advantage = tc.get('rest_advantage', 0)
            
            if rest_advantage >= 3:
                rest_adv_games.append(g)
        
        if len(rest_adv_games) >= self.min_sample_size:
            pattern = self._evaluate_pattern(
                rest_adv_games,
                name="Rest Advantage (3+ days)",
                description="Well-rested team vs tired opponent"
            )
            if pattern:
                patterns.append(pattern)
        
        # Pattern: Late season playoff push (last 15 games)
        late_season_games = []
        for g in games:
            games_played = g.get('games_played', 41)
            if games_played >= 67:  # Last 15 games of 82
                # Check if team is near playoff race
                tc = g.get('temporal_context', {})
                win_pct = tc.get('home_win_pct', 0.5)
                if 0.45 <= win_pct <= 0.55:  # Bubble team
                    late_season_games.append(g)
        
        if len(late_season_games) >= self.min_sample_size:
            pattern = self._evaluate_pattern(
                late_season_games,
                name="Late Season Playoff Push",
                description="Bubble team in final 15 games (extra motivation)"
            )
            if pattern:
                patterns.append(pattern)
        
        print(f"   ✓ Found {len(patterns)} contextual patterns")
        return patterns
    
    def _evaluate_pattern(self, games: List[Dict], name: str, description: str) -> Optional[Dict]:
        """
        Evaluate a pattern for profitability.
        
        Returns pattern dict if profitable, None otherwise.
        """
        if len(games) < self.min_sample_size:
            return None
        
        # Calculate win rate
        home_wins = sum(1 for g in games if g.get('home_won', False))
        win_rate = home_wins / len(games)
        
        # Calculate ROI (assuming -110 juice)
        # Win: profit = 0.909 units
        # Loss: profit = -1.0 unit
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
            'ats_rate': win_rate,  # Against the spread (puck line)
            'ats_pct': win_rate * 100,
            'profitable': roi > 0,
            'confidence': 'HIGH' if win_rate > 0.60 else 'MEDIUM',
            'unit_recommendation': 2 if win_rate > 0.60 else 1,
        }
        
        return pattern
    
    def discover_all_patterns(self, games: List[Dict]) -> List[Dict]:
        """Discover all pattern categories"""
        
        print("\n" + "="*80)
        print("NHL PATTERN DISCOVERY")
        print("="*80)
        print(f"Analyzing {len(games)} games...")
        print(f"Minimum sample size: {self.min_sample_size}")
        print(f"Minimum win rate: {self.min_win_rate:.1%}")
        print(f"Minimum ROI: {self.min_roi:.1%}")
        
        all_patterns = []
        
        # Discover each category
        all_patterns.extend(self.discover_goalie_patterns(games))
        all_patterns.extend(self.discover_underdog_patterns(games))
        all_patterns.extend(self.discover_special_teams_patterns(games))
        all_patterns.extend(self.discover_rivalry_patterns(games))
        all_patterns.extend(self.discover_momentum_patterns(games))
        all_patterns.extend(self.discover_contextual_patterns(games))
        
        # Sort by ROI
        all_patterns.sort(key=lambda x: x['roi'], reverse=True)
        
        print("\n" + "="*80)
        print(f"✅ DISCOVERED {len(all_patterns)} PROFITABLE PATTERNS")
        print("="*80)
        
        # Print summary
        for i, pattern in enumerate(all_patterns, 1):
            print(f"\n{i}. {pattern['name']}")
            print(f"   Games: {pattern['n_games']}, Win Rate: {pattern['win_rate_pct']:.1f}%, ROI: {pattern['roi_pct']:.1f}%")
        
        self.patterns = all_patterns
        return all_patterns


def main():
    """Main execution"""
    
    # Paths
    project_root = Path(__file__).parent.parent.parent.parent
    data_path = project_root / 'data' / 'domains' / 'nhl_games_with_odds.json'
    output_dir = project_root / 'data' / 'domains'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check data
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        print("Run the NHL data builder first:")
        print("  python data_collection/nhl_data_builder.py")
        return
    
    # Load data
    print(f"\n📂 Loading NHL data...")
    with open(data_path, 'r') as f:
        games = json.load(f)
    print(f"   ✓ Loaded {len(games)} games")
    
    # Discover patterns
    discoverer = NHLPatternDiscoverer(
        min_sample_size=20,
        min_win_rate=0.55,
        min_roi=0.10
    )
    
    patterns = discoverer.discover_all_patterns(games)
    
    # Save patterns
    output_path = output_dir / 'nhl_betting_patterns.json'
    with open(output_path, 'w') as f:
        json.dump(patterns, f, indent=2)
    
    print(f"\n💾 PATTERNS SAVED: {output_path}")
    print(f"✅ Pattern discovery complete!")
    print("\nNext step: Validate patterns with temporal split")
    print("  python narrative_optimization/domains/nhl/validate_nhl_patterns.py")


if __name__ == "__main__":
    main()

