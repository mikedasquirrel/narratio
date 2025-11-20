"""
MLB Context Discovery - Exhaustive Search

PRESUME: Narrative effects exist somewhere in MLB
PROVE: Exhaustively search all contexts to find where they're strongest

Strategy:
1. Test all possible context subdivisions
2. Measure correlation in each context
3. Rank contexts by narrative strength
4. Identify top contexts where narrative matters most
"""

import json
import numpy as np
from pathlib import Path
import sys
from collections import defaultdict
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

print("="*80)
print("MLB CONTEXT DISCOVERY - EXHAUSTIVE SEARCH")
print("="*80)
print("\nPRESUME: Narrative effects exist somewhere")
print("PROVE: Find where they're strongest through exhaustive search")
print("\nLoading data...")

# Load data
dataset_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'mlb_complete_dataset.json'
with open(dataset_path) as f:
    all_games = json.load(f)

genome_path = Path(__file__).parent / 'mlb_genome_data.npz'
genome_data = np.load(genome_path, allow_pickle=True)

ж = genome_data['genome']
ю = genome_data['story_quality']
outcomes = genome_data['outcomes']

print(f"✓ Loaded {len(all_games)} games")
print(f"✓ Loaded genome: {ж.shape[0]} samples")

# Match games to outcomes
sample_games = all_games[:len(outcomes)]

# ============================================================================
# CONTEXT DISCOVERY FUNCTIONS
# ============================================================================

def measure_context_correlation(games, ю, outcomes, context_name, context_extractor, min_samples=50):
    """Measure correlation in a specific context."""
    context_indices = []
    context_values = []
    
    for i, game in enumerate(games):
        context_value = context_extractor(game)
        if context_value is not None:
            context_indices.append(i)
            context_values.append(context_value)
    
    if len(context_indices) < min_samples:
        return None
    
    ю_context = ю[context_indices]
    outcomes_context = outcomes[context_indices]
    
    if len(np.unique(outcomes_context)) < 2:
        return None
    
    # Measure correlation
    r = np.corrcoef(ю_context, outcomes_context)[0, 1]
    abs_r = abs(r) if not np.isnan(r) else 0.0
    
    # Statistical significance
    if abs_r > 0:
        try:
            _, p_value = stats.pearsonr(ю_context, outcomes_context)
        except:
            p_value = 1.0
    else:
        p_value = 1.0
    
    return {
        'context_name': context_name,
        'n_samples': len(context_indices),
        'r': float(r),
        'abs_r': float(abs_r),
        'p_value': float(p_value),
        'significant': bool(p_value < 0.05)
    }

# Context extractors
def extract_rivalry(game):
    """Extract rivalry type."""
    if game['context'].get('rivalry', False):
        home_abbr = game['home_team']['abbreviation']
        away_abbr = game['away_team']['abbreviation']
        pair = tuple(sorted([home_abbr, away_abbr]))
        return f"{pair[0]}-{pair[1]}"
    return None

def extract_month(game):
    """Extract month from date."""
    date_str = game.get('date', '')
    if '-' in date_str:
        try:
            month = int(date_str.split('-')[1])
            return f"Month_{month}"
        except:
            pass
    return None

def extract_season(game):
    """Extract season."""
    return f"Season_{game['season']}"

def extract_stadium(game):
    """Extract stadium name."""
    stadium = game['venue']['name']
    # Group similar stadiums
    if 'Fenway' in stadium:
        return 'Fenway_Park'
    elif 'Wrigley' in stadium:
        return 'Wrigley_Field'
    elif 'Yankee' in stadium:
        return 'Yankee_Stadium'
    elif 'Dodger' in stadium:
        return 'Dodger_Stadium'
    else:
        return None  # Only track historic ones

def extract_team_combination(game):
    """Extract home-away team combination."""
    home = game['home_team']['abbreviation']
    away = game['away_team']['abbreviation']
    return f"{home}_vs_{away}"

def extract_playoff_context(game):
    """Extract playoff race context."""
    if game['context'].get('playoff_race', False):
        home_record = game['home_team']['record']
        away_record = game['away_team']['record']
        home_pct = home_record['wins'] / (home_record['wins'] + home_record['losses']) if (home_record['wins'] + home_record['losses']) > 0 else 0.5
        away_pct = away_record['wins'] / (away_record['wins'] + away_record['losses']) if (away_record['wins'] + away_record['losses']) > 0 else 0.5
        
        if home_pct > 0.55 and away_pct > 0.55:
            return 'Both_Strong'
        elif home_pct > 0.55 or away_pct > 0.55:
            return 'One_Strong'
        else:
            return 'Both_Moderate'
    return None

# ============================================================================
# EXHAUSTIVE CONTEXT DISCOVERY
# ============================================================================

print("\n" + "="*80)
print("DISCOVERING CONTEXTS")
print("="*80)

all_contexts = []

# 1. Rivalry types
print("\n[1/6] Discovering rivalry contexts...")
rivalry_types = defaultdict(list)
for i, game in enumerate(sample_games):
    rivalry = extract_rivalry(game)
    if rivalry:
        rivalry_types[rivalry].append(i)

for rivalry_name, indices in rivalry_types.items():
    if len(indices) >= 30:
        result = measure_context_correlation(
            [sample_games[i] for i in indices],
            ю[indices],
            outcomes[indices],
            f"Rivalry_{rivalry_name}",
            lambda g: True,
            min_samples=30
        )
        if result:
            all_contexts.append(result)
            print(f"  {rivalry_name}: |r|={result['abs_r']:.4f}, n={result['n_samples']}")

# 2. Months
print("\n[2/6] Discovering month contexts...")
month_types = defaultdict(list)
for i, game in enumerate(sample_games):
    month = extract_month(game)
    if month:
        month_types[month].append(i)

for month_name, indices in month_types.items():
    if len(indices) >= 100:
        result = measure_context_correlation(
            [sample_games[i] for i in indices],
            ю[indices],
            outcomes[indices],
            month_name,
            lambda g: True,
            min_samples=100
        )
        if result:
            all_contexts.append(result)
            print(f"  {month_name}: |r|={result['abs_r']:.4f}, n={result['n_samples']}")

# 3. Seasons
print("\n[3/6] Discovering season contexts...")
season_types = defaultdict(list)
for i, game in enumerate(sample_games):
    season = extract_season(game)
    season_types[season].append(i)

for season_name, indices in season_types.items():
    if len(indices) >= 200:
        result = measure_context_correlation(
            [sample_games[i] for i in indices],
            ю[indices],
            outcomes[indices],
            season_name,
            lambda g: True,
            min_samples=200
        )
        if result:
            all_contexts.append(result)
            print(f"  {season_name}: |r|={result['abs_r']:.4f}, n={result['n_samples']}")

# 4. Historic stadiums
print("\n[4/6] Discovering stadium contexts...")
stadium_types = defaultdict(list)
for i, game in enumerate(sample_games):
    stadium = extract_stadium(game)
    if stadium:
        stadium_types[stadium].append(i)

for stadium_name, indices in stadium_types.items():
    if len(indices) >= 50:
        result = measure_context_correlation(
            [sample_games[i] for i in indices],
            ю[indices],
            outcomes[indices],
            f"Stadium_{stadium_name}",
            lambda g: True,
            min_samples=50
        )
        if result:
            all_contexts.append(result)
            print(f"  {stadium_name}: |r|={result['abs_r']:.4f}, n={result['n_samples']}")

# 5. Team combinations (top matchups)
print("\n[5/6] Discovering team combination contexts...")
team_combos = defaultdict(list)
for i, game in enumerate(sample_games):
    combo = extract_team_combination(game)
    team_combos[combo].append(i)

# Only test top 20 most frequent matchups
top_combos = sorted(team_combos.items(), key=lambda x: len(x[1]), reverse=True)[:20]

for combo_name, indices in top_combos:
    if len(indices) >= 30:
        result = measure_context_correlation(
            [sample_games[i] for i in indices],
            ю[indices],
            outcomes[indices],
            f"Matchup_{combo_name}",
            lambda g: True,
            min_samples=30
        )
        if result:
            all_contexts.append(result)
            print(f"  {combo_name}: |r|={result['abs_r']:.4f}, n={result['n_samples']}")

# 6. Playoff race contexts
print("\n[6/6] Discovering playoff race contexts...")
playoff_types = defaultdict(list)
for i, game in enumerate(sample_games):
    playoff = extract_playoff_context(game)
    if playoff:
        playoff_types[playoff].append(i)

for playoff_name, indices in playoff_types.items():
    if len(indices) >= 100:
        result = measure_context_correlation(
            [sample_games[i] for i in indices],
            ю[indices],
            outcomes[indices],
            f"Playoff_{playoff_name}",
            lambda g: True,
            min_samples=100
        )
        if result:
            all_contexts.append(result)
            print(f"  {playoff_name}: |r|={result['abs_r']:.4f}, n={result['n_samples']}")

# ============================================================================
# RANK AND REPORT
# ============================================================================

print("\n" + "="*80)
print("TOP CONTEXTS BY NARRATIVE STRENGTH")
print("="*80)

# Sort by absolute correlation
all_contexts_sorted = sorted(all_contexts, key=lambda x: x['abs_r'], reverse=True)

print(f"\nTotal contexts discovered: {len(all_contexts)}")
print(f"\nTop 15 contexts:")

for i, ctx in enumerate(all_contexts_sorted[:15], 1):
    sig_marker = "***" if ctx['significant'] else ""
    print(f"  {i:2d}. {ctx['context_name']:<40} |r|={ctx['abs_r']:.4f}  n={ctx['n_samples']:4d}  p={ctx['p_value']:.4f} {sig_marker}")

# Calculate efficiency for top contexts
print("\n" + "="*80)
print("EFFICIENCY CALCULATION (Top Contexts)")
print("="*80)

π = 0.25
κ = 0.35
threshold = 0.5

top_contexts_efficiency = []
for ctx in all_contexts_sorted[:10]:
    Δ = π * ctx['abs_r'] * κ
    efficiency = Δ / π
    passed = efficiency > threshold
    
    top_contexts_efficiency.append({
        **ctx,
        'Δ': float(Δ),
        'efficiency': float(efficiency),
        'passed': bool(passed)
    })
    
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {ctx['context_name']:<40} Δ/π={efficiency:.4f} {status}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

results = {
    'total_contexts_discovered': len(all_contexts),
    'top_contexts': all_contexts_sorted[:20],
    'top_contexts_efficiency': top_contexts_efficiency,
    'contexts_passing_threshold': [c for c in top_contexts_efficiency if c['passed']],
    'summary': {
        'strongest_context': all_contexts_sorted[0]['context_name'] if all_contexts_sorted else None,
        'strongest_abs_r': all_contexts_sorted[0]['abs_r'] if all_contexts_sorted else 0.0,
        'contexts_with_significant_correlation': sum(1 for c in all_contexts if c['significant']),
        'contexts_passing_efficiency': len([c for c in top_contexts_efficiency if c['passed']])
    }
}

output_path = Path(__file__).parent / 'mlb_context_discovery.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to: {output_path}")

if all_contexts_sorted:
    print(f"\n✓ Strongest context: {all_contexts_sorted[0]['context_name']} (|r|={all_contexts_sorted[0]['abs_r']:.4f})")
    print(f"✓ Contexts passing threshold: {len([c for c in top_contexts_efficiency if c['passed']])}")

print("\n" + "="*80)
print("CONTEXT DISCOVERY COMPLETE")
print("="*80)

