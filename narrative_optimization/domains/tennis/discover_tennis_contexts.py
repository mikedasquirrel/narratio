"""
Tennis Context Discovery - With 93% R² Baseline!

Measure |r| across all contexts to find where narrative is strongest.
Starting from MUCH higher baseline than NFL (93.1% vs 0.01%).
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

print("="*80)
print("TENNIS CONTEXT DISCOVERY")
print("="*80)
print("\nLoading data...")

# Load matches
dataset_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'tennis_complete_dataset.json'
with open(dataset_path) as f:
    all_matches = json.load(f)

# Load genome
genome_path = Path(__file__).parent / 'tennis_genome_data.npz'
genome_data = np.load(genome_path, allow_pickle=True)

ю = genome_data['story_quality']
outcomes = genome_data['outcomes']

sample_matches = all_matches[:5000]  # Same sample as analysis

print(f"✓ Loaded {len(sample_matches)} matches (sample)")
print(f"✓ Story quality and outcomes loaded")

# Baseline
r_overall = np.corrcoef(ю, outcomes)[0, 1]
abs_r_overall = abs(r_overall)
print(f"\nBaseline |r|: {abs_r_overall:.4f}")

# ============================================================================
# MEASURE CONTEXTS
# ============================================================================

print("\n" + "="*80)
print("MEASURING CONTEXTS")
print("="*80)

all_contexts = []

# Surface contexts
print("\n[1/6] By Surface...", end=" ", flush=True)
for surface in ['clay', 'grass', 'hard']:
    indices = [i for i, m in enumerate(sample_matches) if m['surface'] == surface]
    if len(indices) >= 50:
        r = np.corrcoef(ю[indices], outcomes[indices])[0, 1]
        all_contexts.append({
            'dimension': 'surface',
            'value': surface,
            'abs_r': abs(r),
            'r': r,
            'n': len(indices)
        })
print(f"✓ {3} surfaces")

# Tournament level
print("[2/6] By Tournament Level...", end=" ", flush=True)
for level in ['grand_slam', 'masters_1000', 'atp_500']:
    indices = [i for i, m in enumerate(sample_matches) if m['level'] == level]
    if len(indices) >= 50:
        r = np.corrcoef(ю[indices], outcomes[indices])[0, 1]
        all_contexts.append({
            'dimension': 'tournament_level',
            'value': level,
            'abs_r': abs(r),
            'r': r,
            'n': len(indices)
        })
print(f"✓ {3} levels")

# Year
print("[3/6] By Year...", end=" ", flush=True)
years = set(m['year'] for m in sample_matches)
for year in sorted(years):
    indices = [i for i, m in enumerate(sample_matches) if m['year'] == year]
    if len(indices) >= 50:
        r = np.corrcoef(ю[indices], outcomes[indices])[0, 1]
        all_contexts.append({
            'dimension': 'year',
            'value': str(year),
            'abs_r': abs(r),
            'r': r,
            'n': len(indices)
        })
print(f"✓ {len(years)} years")

# Top players
print("[4/6] By Player 1...", end=" ", flush=True)
player_counts = defaultdict(int)
for m in sample_matches:
    player_counts[m['player1']['name']] += 1

top_players = [p for p, c in player_counts.items() if c >= 20][:50]  # Top 50 by match count

for player in top_players:
    indices = [i for i, m in enumerate(sample_matches) if m['player1']['name'] == player]
    if len(indices) >= 20:
        r = np.corrcoef(ю[indices], outcomes[indices])[0, 1]
        all_contexts.append({
            'dimension': 'player',
            'value': player,
            'abs_r': abs(r),
            'r': r,
            'n': len(indices)
        })
print(f"✓ {len(top_players)} players")

# Ranking upset potential
print("[5/6] By Ranking Upset...", end=" ", flush=True)
for upset in [True, False]:
    indices = [i for i, m in enumerate(sample_matches) if m['context'].get('ranking_upset') == upset]
    if len(indices) >= 50:
        r = np.corrcoef(ю[indices], outcomes[indices])[0, 1]
        all_contexts.append({
            'dimension': 'ranking_upset',
            'value': 'upset' if upset else 'expected',
            'abs_r': abs(r),
            'r': r,
            'n': len(indices)
        })
print(f"✓ 2 categories")

# Top 10 matches
print("[6/6] By Top 10 Match...", end=" ", flush=True)
for top10 in [True, False]:
    indices = [i for i, m in enumerate(sample_matches) if m['context'].get('top_10_match') == top10]
    if len(indices) >= 50:
        r = np.corrcoef(ю[indices], outcomes[indices])[0, 1]
        all_contexts.append({
            'dimension': 'top_10_match',
            'value': 'top10' if top10 else 'other',
            'abs_r': abs(r),
            'r': r,
            'n': len(indices)
        })
print(f"✓ 2 categories")

# ============================================================================
# RANK CONTEXTS
# ============================================================================

print("\n" + "="*80)
print("RANKING CONTEXTS BY |r|")
print("="*80)

all_contexts.sort(key=lambda x: x['abs_r'], reverse=True)

print(f"\nTOP 30 CONTEXTS:")
print(f"{'Rank':<6} {'Dimension':<20} {'Value':<30} {'|r|':<10} {'n':<8}")
print("-" * 80)

for i, ctx in enumerate(all_contexts[:30], 1):
    print(f"{i:<6} {ctx['dimension']:<20} {ctx['value']:<30} {ctx['abs_r']:<10.4f} {ctx['n']:<8}")

# Save
output = {
    'baseline': {'abs_r': float(abs_r_overall), 'r': float(r_overall), 'n': len(sample_matches)},
    'ranked_contexts': all_contexts,
    'summary': {
        'total_measured': len(all_contexts),
        'strongest': all_contexts[0] if all_contexts else None
    }
}

output_path = Path(__file__).parent / 'tennis_context_discoveries.json'
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✓ Saved to: {output_path}")
print(f"  Total contexts: {len(all_contexts)}")

print("\n" + "="*80)
print("CONTEXT DISCOVERY COMPLETE")
print("="*80)

