"""
Fix Tennis Data - Restructure from Player-Identity Perspective

Problem: All matches show player1_won=True (100% bias)
Solution: Restructure each match from both players' perspectives with proper outcomes

This creates balanced dataset where outcome varies based on who actually won.
"""

import json
from pathlib import Path
from typing import Dict, List


def restructure_tennis_match(match: Dict) -> List[Dict]:
    """
    Restructure match from player-identity perspective instead of positional.
    
    Returns 2 records (one from each player's perspective) with proper outcomes.
    """
    player1 = match['player1']
    player2 = match['player2']
    player1_won = match['player1_won']
    
    # Match from player1's perspective
    match1 = match.copy()
    match1['focal_player'] = player1['name']
    match1['focal_player_id'] = player1['id']
    match1['opponent_player'] = player2['name']
    match1['opponent_player_id'] = player2['id']
    match1['focal_won'] = player1_won  # True if player1 won
    match1['focal_ranking'] = player1.get('ranking')
    match1['opponent_ranking'] = player2.get('ranking')
    match1['focal_seed'] = player1.get('seed')
    match1['opponent_seed'] = player2.get('seed')
    match1['ranking_advantage'] = (player2.get('ranking', 999) - player1.get('ranking', 999))
    
    # Match from player2's perspective  
    match2 = match.copy()
    match2['focal_player'] = player2['name']
    match2['focal_player_id'] = player2['id']
    match2['opponent_player'] = player1['name']
    match2['opponent_player_id'] = player1['id']
    match2['focal_won'] = not player1_won  # False if player1 won (player2 lost)
    match2['focal_ranking'] = player2.get('ranking')
    match2['opponent_ranking'] = player1.get('ranking')
    match2['focal_seed'] = player2.get('seed')
    match2['opponent_seed'] = player1.get('seed')
    match2['ranking_advantage'] = (player1.get('ranking', 999) - player2.get('ranking', 999))
    
    # Update narrative text to use focal player language
    if 'narrative' in match1:
        # Replace player1/player2 language with focal/opponent
        match1['narrative'] = match1['narrative'].replace('player1', 'focal_player').replace('player2', 'opponent')
        match2['narrative'] = match2['narrative'].replace('player1', 'opponent').replace('player2', 'focal_player')
    
    return [match1, match2]


def main():
    print("="*80)
    print("TENNIS DATA RESTRUCTURING")
    print("="*80)
    
    # Load original data
    input_path = Path('data/domains/tennis_with_temporal_context.json')
    print(f"\nLoading: {input_path}")
    
    with open(input_path) as f:
        original_data = json.load(f)
    
    if isinstance(original_data, dict):
        matches = list(original_data.values())
    else:
        matches = original_data
    
    print(f"Original matches: {len(matches)}")
    print(f"All player1_won=True: {all(m.get('player1_won') == True for m in matches)}")
    
    # Restructure
    print("\nRestructuring from player-identity perspective...")
    restructured = []
    
    for match in matches:
        try:
            restructured.extend(restructure_tennis_match(match))
        except Exception as e:
            print(f"  ⚠️ Error restructuring match {match.get('match_id')}: {e}")
            continue
    
    print(f"\n✓ Restructured to {len(restructured)} records")
    print(f"  (Each match represented from both players' perspectives)")
    
    # Verify outcome distribution
    focal_wins = [r['focal_won'] for r in restructured]
    true_count = focal_wins.count(True)
    false_count = focal_wins.count(False)
    
    print(f"\nOutcome distribution:")
    print(f"  focal_won=True: {true_count} ({true_count/len(focal_wins)*100:.1f}%)")
    print(f"  focal_won=False: {false_count} ({false_count/len(focal_wins)*100:.1f}%)")
    
    if abs(true_count - false_count) < 100:
        print("  ✓ BALANCED - Good for prediction!")
    else:
        print("  ⚠️ Still imbalanced")
    
    # Save restructured data
    output_path = Path('data/domains/tennis_player_perspective.json')
    print(f"\nSaving to: {output_path}")
    
    with open(output_path, 'w') as f:
        json.dump(restructured, f, indent=2)
    
    print(f"✓ Saved {len(restructured)} player-perspective records")
    
    # Sample verification
    print("\n" + "="*80)
    print("SAMPLE RECORDS (verify proper structure):")
    print("="*80)
    
    for i, record in enumerate(restructured[:4], 1):
        print(f"\nRecord {i}:")
        print(f"  Focal: {record['focal_player']} (rank {record['focal_ranking']})")
        print(f"  vs {record['opponent_player']} (rank {record['opponent_ranking']})")
        print(f"  Focal won: {record['focal_won']}")
        print(f"  Ranking advantage: {record['ranking_advantage']}")
    
    print("\n" + "="*80)
    print("✓ TENNIS DATA RESTRUCTURED SUCCESSFULLY")
    print("="*80)
    print(f"\nNext steps:")
    print(f"1. Update domain_registry.py:")
    print(f"   - data_path: 'data/domains/tennis_player_perspective.json'")
    print(f"   - outcome_field: 'focal_won'")
    print(f"2. Re-run: python3 narrative_optimization/universal_domain_processor.py --domain tennis --sample_size 5000")
    print(f"3. Should now show ~50/50 outcome distribution")
    

if __name__ == '__main__':
    main()

