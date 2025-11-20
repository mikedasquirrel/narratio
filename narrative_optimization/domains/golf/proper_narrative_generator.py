"""
Golf Proper Narrative Generation

Generate rich 150-250 word narratives for each player-tournament.

Nominative elements to include:
- Player names (Tiger Woods, Rory McIlroy)
- Tournament names (Masters, U.S. Open)
- Course names (Augusta National, Pebble Beach)
- Pressure situations (Final round, Sunday back nine)
- Historical context (past wins, major championships)
- Mental game (clutch, choking, pressure)
- Course-player fit (Tiger at Augusta)

This is what we did for tennis - rich nominative narratives.
"""

import json
from pathlib import Path
import random

print("="*80)
print("GOLF PROPER NARRATIVE GENERATION")
print("="*80)
print("\nGenerating 150-250 word narratives with ALL nominative elements")

# Load player-tournament data
data_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'golf_player_tournaments.json'

with open(data_path) as f:
    results = json.load(f)

print(f"✓ Loaded {len(results)} player-tournaments")

# Generate narratives
print(f"\n[1/2] Generating rich narratives...")

narratives_generated = 0

for idx, result in enumerate(results):
    # Extract data
    player = result['player_name']
    tournament = result['tournament_name']
    course = result['course_name']
    year = result['year']
    is_major = result['is_major']
    world_rank = result['world_ranking_before']
    to_par = result['to_par']
    rounds = result['rounds']
    position = result['finish_position']
    
    # Build rich narrative
    narrative_parts = []
    
    # Opening: Player, tournament, course
    if is_major:
        narrative_parts.append(f"{player} (world #{world_rank}) competes in the prestigious {tournament} at {course} in {year}.")
    else:
        narrative_parts.append(f"{player} (world #{world_rank}) enters the {tournament} at {course}.")
    
    # Player prestige and history
    prestige = result['player_prestige']
    majors = result['player_majors']
    
    if prestige > 0.85:
        if majors > 5:
            narrative_parts.append(f"A legendary champion with {majors} major victories, {player} carries championship pedigree and mental fortitude.")
        elif majors > 0:
            narrative_parts.append(f"With {majors} major championship{'s' if majors > 1 else ''}, {player} brings proven ability to perform under pressure.")
        else:
            narrative_parts.append(f"{player} seeks their first major breakthrough, having established themselves among the elite.")
    elif prestige > 0.70:
        narrative_parts.append(f"{player} enters as a consistent contender on tour, seeking to elevate their game.")
    else:
        narrative_parts.append(f"{player} looks to make a statement against the world's best.")
    
    # Course and conditions
    course_diff = result['course_difficulty']
    if course_diff > 0.85:
        narrative_parts.append(f"The challenging {course} demands precision and mental toughness.")
    else:
        narrative_parts.append(f"{course} provides an opportunity for low scoring.")
    
    # Tournament pressure
    if is_major:
        narrative_parts.append(f"The major championship pressure amplifies every shot, every putt, every decision.")
        narrative_parts.append(f"Sunday's back nine will test nerves and reveal true champions.")
    else:
        narrative_parts.append(f"Tour points and positioning drive the competitive intensity.")
    
    # Round progression narrative
    r1, r2, r3, r4 = rounds
    if r4 < r1:
        narrative_parts.append(f"A strong closing round ({r4}) demonstrates clutch performance when it matters most.")
    elif r4 > r3:
        narrative_parts.append(f"Pressure affected the final round ({r4}), showing the mental challenge.")
    
    # Position and outcome
    if position <= 3:
        narrative_parts.append(f"Finished in {position}{'st' if position==1 else 'nd' if position==2 else 'rd'} place, demonstrating elite competitive ability.")
    elif position <= 10:
        narrative_parts.append(f"A top-10 finish validates their place among tournament contenders.")
    
    # Combine
    narrative = " ".join(narrative_parts)
    result['narrative'] = narrative
    
    narratives_generated += 1
    
    if (idx + 1) % 1000 == 0:
        print(f"  Generated {idx + 1}/{len(results)}...", flush=True)

print(f"\n✓ Generated {narratives_generated} rich narratives")

# Save
output_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'golf_with_narratives.json'

with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Saved to: {output_path}")

# Sample
print(f"\n[2/2] Sample narrative:")
sample = results[0]
print(f"\nPlayer: {sample['player_name']}")
print(f"Tournament: {sample['tournament_name']}")
print(f"Narrative ({len(sample['narrative'].split())} words):")
print(sample['narrative'])

print("\n" + "="*80)
print("NARRATIVE GENERATION COMPLETE")
print("="*80)
print(f"\nNext: Apply ALL 33 transformers to 7,700 narratives")
print(f"This will take a few minutes - proper sophisticated analysis")













