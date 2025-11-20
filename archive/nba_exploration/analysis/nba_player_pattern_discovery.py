"""
NBA Player Pattern Discovery - What the Transformer Will Find

Demonstrates what types of player-level narrative patterns will emerge
when Context Pattern Transformer analyzes raw player data.

NO HARD-CODED CATEGORIES - just shows what the discovery will look like.

Author: Narrative Optimization Framework
Date: November 16, 2025
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from transformers.context_pattern import ContextPatternTransformer

print("="*80)
print("NBA PLAYER PATTERN DISCOVERY - WHAT WILL EMERGE")
print("="*80)

print("\n💡 PHILOSOPHY:")
print("   We collect RAW player stats (PPG, APG, experience, etc.)")
print("   Transformer discovers thresholds and patterns automatically")
print("   NO pre-defined 'star' or 'role player' labels")
print()

# Create synthetic player-level data to demonstrate
print("🎲 Creating synthetic player-level game data...")
print("   (Replace with real data from nba_raw_player_data_collector.py)")
print()

np.random.seed(42)
n_games = 1000

# Generate raw player stats (what we'd get from NBA API)
game_features = pd.DataFrame({
    # Team-level context (existing)
    'home': np.random.choice([0, 1], n_games),
    'season_win_pct': np.random.uniform(0.2, 0.8, n_games),
    'games_played': np.random.uniform(0, 1, n_games),
    
    # Player distributions (RAW numbers, no categories)
    'top1_ppg': np.random.uniform(15, 35, n_games),  # Top scorer
    'top2_ppg': np.random.uniform(10, 25, n_games),  # Second scorer
    'top3_ppg': np.random.uniform(8, 20, n_games),   # Third scorer
    
    'players_20plus_pts': np.random.randint(0, 4, n_games),  # How many scored 20+
    'players_15plus_pts': np.random.randint(1, 5, n_games),
    
    'top1_assists': np.random.uniform(2, 12, n_games),
    'top1_experience': np.random.uniform(0, 15, n_games),  # Years
    
    'experienced_players': np.random.randint(3, 10, n_games),  # Players with 5+ years
    
    'top1_scoring_share': np.random.uniform(0.2, 0.5, n_games),  # Scoring concentration
    'top3_scoring_share': np.random.uniform(0.5, 0.8, n_games),
    
    'bench_points': np.random.uniform(15, 45, n_games),
})

# Create outcome - RANDOM, no injected patterns
# We're just demonstrating the DATA STRUCTURE, not pre-determining patterns
# In real data, patterns exist naturally or they don't - we discover, not inject

y = np.random.choice([0, 1], n_games, p=[0.5, 0.5])

print(f"✓ Generated {len(game_features):,} games with player-level features")
print(f"  Features: {len(game_features.columns)}")
print(f"  Outcomes: Random (real data will have real patterns)")
print(f"  NOTE: This is just demonstrating data structure")
print()

# Discover patterns
print("="*80)
print("PATTERN DISCOVERY - PLAYER-LEVEL CONTEXTS")
print("="*80)

transformer = ContextPatternTransformer(
    min_accuracy=0.62,
    min_samples=40,
    max_patterns=15,
    feature_combinations=3
)

print("\n🔍 Discovering player-driven narrative patterns...")
transformer.fit(game_features, y)

print(f"\n✓ Patterns discovered: {len(transformer.patterns_)}")

# Show what was discovered
print("\n" + "="*80)
print("DISCOVERED PLAYER PATTERNS")
print("="*80)

for i, pattern in enumerate(transformer.patterns_[:10], 1):
    print(f"\n{i}. {pattern}")
    
    # Interpret what this pattern represents
    features = pattern.features
    conditions = pattern.conditions
    
    print(f"   Interpretation:")
    
    # Analyze what the transformer found
    if 'top1_ppg' in features or 'top1_scoring_share' in features:
        print(f"     → Scoring hierarchy matters")
    if 'players_20plus_pts' in features or 'players_15plus_pts' in features:
        print(f"     → Scoring distribution/balance matters")
    if 'experienced_players' in features or 'top1_experience' in features:
        print(f"     → Experience/veteran presence matters")
    if 'top1_assists' in features:
        print(f"     → Playmaking/ball movement matters")
    if 'bench_points' in features:
        print(f"     → Depth/bench contribution matters")
    if 'top1_scoring_share' in features or 'top3_scoring_share' in features:
        print(f"     → Scoring concentration matters")

print("\n" + "="*80)
print("WHAT THESE PATTERNS REVEAL")
print("="*80)

print("\n🏀 PLAYER HIERARCHY NARRATIVES:")
print("   The transformer discovers (without being told):")
print("   - When scoring concentration helps vs hurts")
print("   - When balanced attack predicts wins")
print("   - When superstar dominance matters")
print("   - When bench depth creates edge")

print("\n👴 EXPERIENCE NARRATIVES:")
print("   The transformer discovers:")
print("   - When veteran teams excel (e.g., late season)")
print("   - When young teams struggle (e.g., playoffs)")
print("   - Experience thresholds that matter")

print("\n🎯 PLAYMAKING NARRATIVES:")
print("   The transformer discovers:")
print("   - When elite playmakers change outcomes")
print("   - Ball movement vs iso-heavy patterns")
print("   - Assist thresholds that predict wins")

print("\n📊 WHAT THIS ENABLES:")
print()
print("1. BETTER GAME PREDICTIONS")
print("   - 'Superstar home game' context → bet with confidence")
print("   - 'Balanced young team' context → fade the hype")
print("   - 'Veteran team late season' → trust experience")
print()
print("2. PLAYER PROPS BETTING")
print("   - 'Top scorer in concentration game' → OVER")
print("   - 'Playmaker with balanced attack' → ASSISTS OVER")
print("   - 'Bench points in deep roster' → BENCH POINTS OVER")
print()
print("3. LIVE BETTING EDGES")
print("   - If pattern says 'superstar game' but trailing → value bet")
print("   - If pattern says 'balanced attack' and clicking → bet momentum")
print()
print("4. COMPLETELY NEW NARRATIVES")
print("   - Whatever patterns exist in NBA player dynamics")
print("   - Not limited to what we think matters")
print("   - Discovered from data, not theory")

print("\n" + "="*80)
print("NEXT STEP: COLLECT REAL PLAYER DATA")
print("="*80)

print("\nRun: python data_collection/nba_raw_player_data_collector.py")
print("\nThen run Context Pattern Transformer on enhanced data")
print("Let it discover player patterns automatically!")

print("\n" + "="*80)

