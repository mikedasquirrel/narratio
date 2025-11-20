"""
NHL Temporal Feature Integration

Integrates the three-scale temporal framework into the existing NHL pipeline:
1. Load historical season data
2. Extract macro/meso/micro temporal features
3. Append to existing 79-feature baseline
4. Retrain models with expanded feature set

This demonstrates how to add temporal depth to any production domain.

Author: Temporal Integration System
Date: November 19, 2025
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
import sys

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from narrative_optimization.domains.nhl.temporal_narrative_features import NHLTemporalExtractor


def load_nhl_historical_data(seasons: List[str] = ['2023-24', '2024-25']) -> List[Dict]:
    """
    Load historical NHL game data for temporal context.
    
    Parameters
    ----------
    seasons : list of str
        Seasons to load (e.g., ['2023-24', '2024-25'])
    
    Returns
    -------
    games : list of dict
        All historical games
    """
    games = []
    
    # Load from existing NHL data file
    data_path = Path('data/domains/nhl_games_with_odds.json')
    if data_path.exists():
        with open(data_path) as f:
            all_games = json.load(f)
        
        # Filter to requested seasons
        for game in all_games:
            season = game.get('season', '')
            if any(s in season for s in seasons):
                games.append(game)
    
    print(f"✓ Loaded {len(games)} historical NHL games for temporal context")
    return games


def enrich_games_with_temporal_features(games: List[Dict], historical_data: List[Dict]) -> List[Dict]:
    """
    Enrich games with three-scale temporal features.
    
    Parameters
    ----------
    games : list of dict
        Games to enrich (could be upcoming games or historical for training)
    historical_data : list of dict
        Full season data for context
    
    Returns
    -------
    enriched_games : list of dict
        Games with temporal features added
    """
    extractor = NHLTemporalExtractor()
    enriched = []
    
    print(f"\n[Temporal Enrichment] Processing {len(games)} games...")
    
    for i, game in enumerate(games):
        if i % 100 == 0 and i > 0:
            print(f"  Progress: {i}/{len(games)} games enriched")
        
        # Extract temporal features
        temporal_features = extractor.extract_all_temporal_features(game, historical_data)
        
        # Add to game dict
        enriched_game = game.copy()
        enriched_game['temporal_features'] = temporal_features
        enriched.append(enriched_game)
    
    print(f"✓ Enriched {len(enriched)} games with {len(temporal_features)} temporal features")
    return enriched


def build_temporal_feature_matrix(games: List[Dict]) -> pd.DataFrame:
    """
    Build feature matrix from temporal features.
    
    Parameters
    ----------
    games : list of dict
        Games with temporal_features dict
    
    Returns
    -------
    df : DataFrame
        Temporal feature matrix
    """
    rows = []
    for game in games:
        temporal = game.get('temporal_features', {})
        rows.append(temporal)
    
    return pd.DataFrame(rows)


def integrate_with_existing_pipeline(games: List[Dict], historical_data: List[Dict]) -> pd.DataFrame:
    """
    Integrate temporal features with existing NHL pipeline.
    
    This shows how to ADD temporal depth to an existing production model
    without breaking the current 79-feature baseline.
    
    Parameters
    ----------
    games : list of dict
        Games to score
    historical_data : list of dict
        Season context
    
    Returns
    -------
    full_features : DataFrame
        79 baseline features + 50 temporal features = 129 total
    """
    from narrative_optimization.domains.nhl.score_upcoming_games import build_feature_matrix
    
    # Get baseline 79 features (existing pipeline)
    print("\n[Integration] Building baseline features (79 features)...")
    
    # Load feature columns from metadata
    metadata_path = Path('narrative_optimization/domains/nhl/nhl_narrative_betting_metadata.json')
    with open(metadata_path) as f:
        metadata = json.load(f)
    feature_columns = metadata['columns']
    
    baseline_features = build_feature_matrix(games, feature_columns)
    print(f"✓ Baseline features: {baseline_features.shape}")
    
    # Get temporal features (50 new features)
    print("\n[Integration] Extracting temporal features (50 features)...")
    enriched_games = enrich_games_with_temporal_features(games, historical_data)
    temporal_features = build_temporal_feature_matrix(enriched_games)
    print(f"✓ Temporal features: {temporal_features.shape}")
    
    # Combine
    full_features = pd.concat([baseline_features, temporal_features], axis=1)
    print(f"\n✓ Combined feature matrix: {full_features.shape} (79 baseline + {temporal_features.shape[1]} temporal)")
    
    return full_features


def demonstrate_temporal_impact():
    """
    Demonstrate the value of temporal features on a sample game.
    
    Shows how macro/meso/micro features capture different narrative layers.
    """
    print("\n" + "="*80)
    print("TEMPORAL FEATURE DEMONSTRATION")
    print("="*80)
    
    # Load sample data
    historical = load_nhl_historical_data(['2024-25'])
    
    if not historical:
        print("✗ No historical data available")
        return
    
    # Take a recent game as example
    sample_game = historical[-1]
    
    print(f"\nSample Game: {sample_game.get('away_team')} @ {sample_game.get('home_team')}")
    print(f"Date: {sample_game.get('date')}")
    
    # Extract temporal features
    extractor = NHLTemporalExtractor()
    temporal = extractor.extract_all_temporal_features(sample_game, historical)
    
    print("\n--- MACRO-TEMPORAL (Season-Long) ---")
    macro_keys = [k for k in temporal.keys() if any(x in k for x in ['playoff', 'expectation', 'trade', 'coach', 'trajectory', 'desperation'])]
    for key in macro_keys[:10]:
        print(f"  {key}: {temporal[key]:.3f}")
    
    print("\n--- MESO-TEMPORAL (Recent Form) ---")
    meso_keys = [k for k in temporal.keys() if any(x in k for x in ['l5', 'l10', 'l20', 'goals', 'pp_', 'goalie'])]
    for key in meso_keys[:10]:
        print(f"  {key}: {temporal[key]:.3f}")
    
    print("\n--- MICRO-TEMPORAL (In-Game) ---")
    micro_keys = [k for k in temporal.keys() if any(x in k for x in ['comeback', 'momentum', 'period', 'leading'])]
    for key in micro_keys[:10]:
        print(f"  {key}: {temporal[key]:.3f}")
    
    print(f"\nTotal temporal features: {len(temporal)}")


if __name__ == '__main__':
    demonstrate_temporal_impact()

