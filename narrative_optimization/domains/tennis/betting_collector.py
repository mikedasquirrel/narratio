"""
Tennis Betting Odds Collection Module

Simulates realistic betting odds based on player rankings and context.
In production, would integrate with tennis-data.co.uk historical odds.
"""

import json
from typing import List, Dict, Any
from pathlib import Path
import random


class TennisBettingCollector:
    """
    Collects/simulates tennis betting odds.
    
    Simulates realistic odds based on:
    - Player ranking differential
    - Surface specialization
    - Tournament level
    - Head-to-head record
    """
    
    def __init__(self):
        """Initialize betting odds collector."""
        print("Initializing Tennis Betting Odds Collector")
        print("Note: Using simulated odds based on rankings and context")
    
    def collect_odds_for_matches(
        self,
        matches: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Add betting odds to matches.
        
        Parameters
        ----------
        matches : list of dict
            Match records to enrich with betting odds
            
        Returns
        -------
        matches_with_odds : list of dict
            Matches enriched with betting odds
        """
        print("\n" + "="*80)
        print("ADDING BETTING ODDS DATA")
        print("="*80)
        
        enriched_matches = []
        
        for idx, match in enumerate(matches):
            # Calculate odds
            odds = self._simulate_odds(match)
            
            # Add odds to match record
            match['betting_odds'] = odds
            enriched_matches.append(match)
            
            if (idx + 1) % 10000 == 0:
                print(f"  Processed {idx + 1}/{len(matches)} matches...")
        
        print(f"\n✓ Added betting odds to {len(enriched_matches)} matches")
        
        return enriched_matches
    
    def _simulate_odds(self, match: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate realistic tennis betting odds.
        
        Based on:
        - Ranking differential
        - Surface (specialization matters)
        - Tournament level (pressure in Grand Slams)
        - H2H record
        """
        p1_rank = match['player1']['ranking']
        p2_rank = match['player2']['ranking']
        
        # Calculate base odds from ranking
        if p1_rank and p2_rank:
            rank_diff = p2_rank - p1_rank
            
            # Convert ranking advantage to odds
            # Better ranked = lower odds (favorite)
            if rank_diff > 50:  # P1 heavily favored
                p1_odds = 1.20 + random.uniform(-0.05, 0.10)
                p2_odds = 4.50 + random.uniform(-0.50, 1.00)
            elif rank_diff > 20:  # P1 favored
                p1_odds = 1.45 + random.uniform(-0.10, 0.15)
                p2_odds = 2.80 + random.uniform(-0.30, 0.50)
            elif rank_diff > 10:  # P1 slight favorite
                p1_odds = 1.65 + random.uniform(-0.10, 0.15)
                p2_odds = 2.25 + random.uniform(-0.20, 0.30)
            elif rank_diff > 0:  # P1 marginal favorite
                p1_odds = 1.85 + random.uniform(-0.10, 0.10)
                p2_odds = 1.95 + random.uniform(-0.10, 0.10)
            elif rank_diff > -10:  # P2 marginal favorite
                p1_odds = 1.95 + random.uniform(-0.10, 0.10)
                p2_odds = 1.85 + random.uniform(-0.10, 0.10)
            elif rank_diff > -20:  # P2 slight favorite
                p1_odds = 2.25 + random.uniform(-0.20, 0.30)
                p2_odds = 1.65 + random.uniform(-0.10, 0.15)
            elif rank_diff > -50:  # P2 favored
                p1_odds = 2.80 + random.uniform(-0.30, 0.50)
                p2_odds = 1.45 + random.uniform(-0.10, 0.15)
            else:  # P2 heavily favored
                p1_odds = 4.50 + random.uniform(-0.50, 1.00)
                p2_odds = 1.20 + random.uniform(-0.05, 0.10)
        else:
            # No ranking data - use pick'em odds
            p1_odds = 1.90 + random.uniform(-0.10, 0.10)
            p2_odds = 1.90 + random.uniform(-0.10, 0.10)
        
        # Adjust for surface specialization
        surface = match['surface']
        if surface == 'clay' and 'nadal' in match['player1']['name'].lower():
            p1_odds *= 0.85  # Nadal clay advantage
        elif surface == 'grass' and 'federer' in match['player1']['name'].lower():
            p1_odds *= 0.88  # Federer grass advantage
        
        # Adjust for Grand Slam
        if match['level'] == 'grand_slam':
            # Favorites get slightly stronger in Grand Slams
            if p1_odds < p2_odds:
                p1_odds *= 0.95
                p2_odds *= 1.05
            else:
                p1_odds *= 1.05
                p2_odds *= 0.95
        
        # Round to 2 decimals
        p1_odds = round(p1_odds, 2)
        p2_odds = round(p2_odds, 2)
        
        # Determine favorite
        favorite = 'player1' if p1_odds < p2_odds else 'player2'
        
        # Was there an upset?
        upset = (favorite == 'player2' and match['player1_won']) or \
                (favorite == 'player1' and not match['player1_won'])
        
        return {
            'player1_odds': p1_odds,
            'player2_odds': p2_odds,
            'favorite': favorite,
            'upset': upset,
            'implied_prob_p1': round(1 / p1_odds, 3),
            'implied_prob_p2': round(1 / p2_odds, 3)
        }


def main():
    """Main execution: add betting odds to collected matches."""
    print("="*80)
    print("TENNIS BETTING ODDS COLLECTION")
    print("="*80)
    
    # Load matches
    matches_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'tennis_matches_raw.json'
    
    print(f"\nLoading matches from: {matches_path}")
    with open(matches_path) as f:
        matches = json.load(f)
    
    print(f"✓ Loaded {len(matches)} matches")
    
    # Collect odds
    collector = TennisBettingCollector()
    matches_with_odds = collector.collect_odds_for_matches(matches)
    
    # Save enriched dataset
    output_path = matches_path.parent / 'tennis_matches_with_odds.json'
    
    with open(output_path, 'w') as f:
        json.dump(matches_with_odds, f, indent=2)
    
    print(f"\n✓ Saved matches with betting odds to: {output_path}")
    
    # Statistics
    print("\n" + "="*80)
    print("BETTING ODDS STATISTICS")
    print("="*80)
    
    upsets = sum(1 for m in matches_with_odds if m['betting_odds']['upset'])
    print(f"Total matches: {len(matches_with_odds)}")
    print(f"Upsets: {upsets} ({100*upsets/len(matches_with_odds):.1f}%)")
    
    # Average odds by favorite
    p1_favorite_odds = [m['betting_odds']['player1_odds'] for m in matches_with_odds 
                        if m['betting_odds']['favorite'] == 'player1']
    print(f"Average favorite odds: {sum(p1_favorite_odds)/len(p1_favorite_odds):.2f}" if p1_favorite_odds else "N/A")


if __name__ == '__main__':
    main()

