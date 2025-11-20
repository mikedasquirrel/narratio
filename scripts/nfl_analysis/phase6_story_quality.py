#!/usr/bin/env python3
"""
Phase 6: Story Quality Calculation
Calculates comprehensive narrative quality scores for each game
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def calculate_story_quality(df, features):
    """Calculate multi-component story quality score"""
    
    print("  📊 Component 1/6: QB Prestige...")
    qb_prestige = (
        features['home_qb_prestige_score'] + 
        features['away_qb_prestige_score']
    ) / 2.0
    
    print("  📊 Component 2/6: Rivalry Intensity...")
    rivalry = features['rivalry_intensity']
    
    print("  📊 Component 3/6: Stakes...")
    # Stakes from playoff + late season + record quality
    stakes = (
        features['playoff_game'] * 1.0 +
        features['late_season'] * 0.5 +
        (features['combined_wins'] / 20.0) * 0.5
    ) / 2.0
    
    print("  📊 Component 4/6: Drama...")
    # Drama from close games + overtime
    drama = (
        features['game_closeness'] * 0.7 +
        features['overtime_drama'] * 0.3
    )
    
    print("  📊 Component 5/6: Star Power...")
    # Star power from QB data presence + elite brands
    star_power = (
        features['has_qb_data'] * 0.5 +
        (features['home_elite_brand'] + features['away_elite_brand']) / 4.0
    )
    
    print("  📊 Component 6/6: Underdog Factor...")
    # Underdog narrative strength
    underdog = features['underdog_intensity']
    
    # Weighted combination
    weights = {
        'qb_prestige': 0.20,
        'rivalry': 0.15,
        'stakes': 0.25,
        'drama': 0.20,
        'star_power': 0.10,
        'underdog': 0.10,
    }
    
    story_quality = (
        weights['qb_prestige'] * qb_prestige +
        weights['rivalry'] * rivalry +
        weights['stakes'] * stakes +
        weights['drama'] * drama +
        weights['star_power'] * star_power +
        weights['underdog'] * underdog
    )
    
    return story_quality, {
        'qb_prestige': qb_prestige,
        'rivalry': rivalry,
        'stakes': stakes,
        'drama': drama,
        'star_power': star_power,
        'underdog': underdog,
    }

def main():
    print("="*60)
    print(f"PHASE 6: STORY QUALITY - {datetime.now().strftime('%H:%M:%S')}")
    print("="*60)
    
    data_dir = Path(__file__).parent.parent.parent / "data" / "domains"
    
    # Load enriched games
    print("\n📂 Loading enriched games...")
    with open(data_dir / "nfl_enriched_with_rosters.json", 'r') as f:
        data = json.load(f)
    games = data['games']
    print(f"  ✓ {len(games):,} games")
    
    # Load complete features
    print("\n📂 Loading complete feature matrix...")
    features = pd.read_csv(data_dir / "nfl_complete_features.csv")
    print(f"  ✓ {features.shape[1]} features")
    
    # Calculate story quality
    print("\n🔄 Calculating story quality scores...")
    story_quality, components = calculate_story_quality(pd.DataFrame(games), features)
    
    print(f"\n✓ Story quality calculated for {len(story_quality)} games")
    print(f"  Mean: {story_quality.mean():.3f}")
    print(f"  Std:  {story_quality.std():.3f}")
    print(f"  Range: {story_quality.min():.3f} - {story_quality.max():.3f}")
    
    # Add to games
    for i, game in enumerate(games):
        game['story_quality'] = float(story_quality.iloc[i])
        game['story_components'] = {k: float(v.iloc[i]) for k, v in components.items()}
    
    # Save with story scores
    output_data = {
        'metadata': {
            'created': datetime.now().isoformat(),
            'total_games': len(games),
            'story_quality_stats': {
                'mean': float(story_quality.mean()),
                'std': float(story_quality.std()),
                'min': float(story_quality.min()),
                'max': float(story_quality.max()),
            },
            'components': list(components.keys()),
        },
        'games': games
    }
    
    output_path = data_dir / "nfl_story_scores.json"
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Story scores saved: {output_path.name}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Show top stories
    print("\n🏆 Top 5 Story Quality Games (All-Time):")
    games_sorted = sorted(games, key=lambda g: g['story_quality'], reverse=True)
    for i, game in enumerate(games_sorted[:5], 1):
        print(f"  {i}. {game['away_team']} @ {game['home_team']} (Week {game.get('week')}, {game['season']})")
        print(f"      Score: {game['away_score']}-{game['home_score']}, Quality: {game['story_quality']:.3f}")
    
    print(f"\n{'='*60}")
    print("PHASE 6 COMPLETE ✓")
    print(f"{'='*60}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

