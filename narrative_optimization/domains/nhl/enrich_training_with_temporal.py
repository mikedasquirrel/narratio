"""
NHL Training Dataset Temporal Enrichment

Adds 50 three-scale temporal features to the existing 900-feature training set.

Input: nhl_narrative_betting_dataset.parquet (15,927 games × 900 features)
Output: nhl_narrative_betting_temporal_dataset.parquet (15,927 games × 950 features)

Progress tracking: Prints every 500 games to monitor enrichment.

Author: Temporal Integration System
Date: November 19, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
import sys
from datetime import datetime
import json

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from narrative_optimization.domains.nhl.temporal_narrative_features import NHLTemporalExtractor


def load_training_dataset() -> pd.DataFrame:
    """Load existing NHL training dataset"""
    path = Path('narrative_optimization/domains/nhl/nhl_narrative_betting_dataset.parquet')
    df = pd.read_parquet(path)
    print(f"✓ Loaded training dataset: {len(df):,} games × {len(df.columns):,} features")
    return df


def convert_df_to_game_dicts(df: pd.DataFrame) -> List[Dict]:
    """Convert DataFrame rows to game dictionaries for temporal extractor"""
    games = []
    
    print(f"\n[Conversion] Converting DataFrame to game dicts...")
    for idx, row in df.iterrows():
        game = {
            'game_id': row.get('game_id', f'game_{idx}'),
            'date': row.get('date', ''),
            'season': row.get('season', ''),
            'home_team': row.get('home_team', ''),
            'away_team': row.get('away_team', ''),
            'home_won': row.get('home_won', False),
            'home_score': row.get('home_score', 0),
            'away_score': row.get('away_score', 0),
            'home_goalie': row.get('home_goalie', ''),
            'away_goalie': row.get('away_goalie', ''),
            # Include existing temporal context
            'temporal_context': {
                'home_wins': row.get('ctx_home_wins', 0),
                'home_losses': row.get('ctx_home_losses', 0),
                'away_wins': row.get('ctx_away_wins', 0),
                'away_losses': row.get('ctx_away_losses', 0),
                'home_l10_wins': row.get('ctx_home_l10_wins', 5),
                'away_l10_wins': row.get('ctx_away_l10_wins', 5),
                'home_rest_days': row.get('ctx_home_rest_days', 1),
                'away_rest_days': row.get('ctx_away_rest_days', 1),
            }
        }
        games.append(game)
        
        if (idx + 1) % 5000 == 0:
            print(f"  Converted {idx + 1:,} games...")
    
    print(f"✓ Converted {len(games):,} games to dict format")
    return games


def enrich_with_temporal_features(games: List[Dict], progress_interval: int = 500) -> pd.DataFrame:
    """
    Enrich games with three-scale temporal features.
    
    Parameters
    ----------
    games : list of dict
        Game data
    progress_interval : int
        Print progress every N games
    
    Returns
    -------
    temporal_df : DataFrame
        50 temporal features × N games
    """
    extractor = NHLTemporalExtractor()
    
    print(f"\n[Temporal Enrichment] Extracting features for {len(games):,} games...")
    print(f"Progress updates every {progress_interval} games\n")
    
    # Group games by season for context
    season_groups = {}
    for game in games:
        season = game.get('season', '')
        if season not in season_groups:
            season_groups[season] = []
        season_groups[season].append(game)
    
    print(f"✓ Grouped into {len(season_groups)} seasons")
    
    # Extract temporal features
    all_temporal_features = []
    
    for idx, game in enumerate(games):
        # Get season context (all games in same season up to this game's date)
        season = game.get('season', '')
        game_date = game.get('date', '')
        
        season_context = [
            g for g in season_groups.get(season, [])
            if g.get('date', '') < game_date
        ]
        
        # Extract temporal features
        temporal = extractor.extract_all_temporal_features(game, season_context)
        all_temporal_features.append(temporal)
        
        # Progress tracking
        if (idx + 1) % progress_interval == 0:
            print(f"  [{idx + 1:,}/{len(games):,}] Processed {(idx + 1) / len(games) * 100:.1f}%")
    
    print(f"\n✓ Extracted temporal features for all {len(games):,} games")
    
    # Convert to DataFrame
    temporal_df = pd.DataFrame(all_temporal_features)
    print(f"✓ Temporal feature matrix: {temporal_df.shape}")
    
    return temporal_df


def combine_and_save(baseline_df: pd.DataFrame, temporal_df: pd.DataFrame, output_path: Path):
    """
    Combine baseline and temporal features, save enriched dataset.
    
    Parameters
    ----------
    baseline_df : DataFrame
        Original 900-feature dataset
    temporal_df : DataFrame
        50 temporal features
    output_path : Path
        Where to save enriched dataset
    """
    print(f"\n[Combination] Merging feature sets...")
    print(f"  Baseline: {baseline_df.shape}")
    print(f"  Temporal: {temporal_df.shape}")
    
    # Combine
    enriched_df = pd.concat([baseline_df, temporal_df], axis=1)
    
    print(f"✓ Combined: {enriched_df.shape}")
    
    # Save
    enriched_df.to_parquet(output_path, index=False)
    print(f"✓ Saved to: {output_path}")
    
    # Generate metadata
    metadata = {
        'created': datetime.now().isoformat(),
        'n_games': len(enriched_df),
        'n_features': len(enriched_df.columns),
        'baseline_features': len(baseline_df.columns),
        'temporal_features': len(temporal_df.columns),
        'temporal_feature_names': list(temporal_df.columns),
        'seasons': sorted(baseline_df['season'].unique().tolist()),
        'date_range': {
            'start': baseline_df['date'].min(),
            'end': baseline_df['date'].max()
        }
    }
    
    metadata_path = output_path.parent / 'nhl_temporal_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved metadata to: {metadata_path}")
    
    return enriched_df


def main():
    """Main enrichment pipeline"""
    print("\n" + "="*80)
    print("NHL TEMPORAL ENRICHMENT PIPELINE")
    print("="*80)
    
    # Step 1: Load baseline dataset
    print("\n[Step 1/4] Loading baseline dataset...")
    baseline_df = load_training_dataset()
    
    # Step 2: Convert to game dicts
    print("\n[Step 2/4] Converting to game format...")
    games = convert_df_to_game_dicts(baseline_df)
    
    # Step 3: Extract temporal features
    print("\n[Step 3/4] Extracting temporal features...")
    temporal_df = enrich_with_temporal_features(games, progress_interval=500)
    
    # Step 4: Combine and save
    print("\n[Step 4/4] Combining and saving...")
    output_path = Path('narrative_optimization/domains/nhl/nhl_narrative_betting_temporal_dataset.parquet')
    enriched_df = combine_and_save(baseline_df, temporal_df, output_path)
    
    # Summary
    print("\n" + "="*80)
    print("ENRICHMENT COMPLETE")
    print("="*80)
    print(f"Input:  {baseline_df.shape[0]:,} games × {baseline_df.shape[1]:,} features")
    print(f"Output: {enriched_df.shape[0]:,} games × {enriched_df.shape[1]:,} features")
    print(f"Added:  {temporal_df.shape[1]} temporal features")
    print(f"\nNew features include:")
    print(f"  - 18 macro-temporal (season-long narratives)")
    print(f"  - 22 meso-temporal (recent form patterns)")
    print(f"  - 10 micro-temporal (in-game dynamics)")
    print(f"\nDataset ready for model retraining.")
    print("="*80)


if __name__ == '__main__':
    main()

