#!/usr/bin/env python3
"""
Phase 4: Universal Narrative Transformers
Applies universal story pattern recognition transformers
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def extract_narrative_features(df):
    """Extract narrative pattern features"""
    features = pd.DataFrame(index=df.index)
    
    print("  📝 Extracting narrative patterns...")
    
    # 1. Underdog narrative
    print("     1/5: Underdog narratives...")
    features['underdog_intensity'] = df.apply(lambda r:
        abs(r.get('spread_line', 0)) / 14.0 if r.get('spread_line') else 0
    , axis=1)
    
    # 2. Drama/tension
    print("     2/5: Drama and tension...")
    features['game_closeness'] = df.apply(lambda r:
        1.0 - min(abs(r.get('home_score', 0) - r.get('away_score', 0)) / 30.0, 1.0)
    , axis=1)
    features['overtime_drama'] = df.get('overtime', False).astype(float)
    
    # 3. Stakes (playoff implications, late season)
    print("     3/5: Stakes and implications...")
    features['playoff_game'] = df.get('playoff', False).astype(float)
    features['late_season'] = (df.get('week', 1) >= 13).astype(float)
    features['season_stakes'] = features['playoff_game'] + (features['late_season'] * 0.5)
    
    # 4. Rivalry intensity
    print("     4/5: Rivalry patterns...")
    def get_rivalry(row):
        mh = row.get('matchup_history')
        if isinstance(mh, dict):
            games = mh.get('total_games', 0)
            return min(games / 30.0, 1.0)  # Normalize to 0-1
        return 0.0
    
    features['rivalry_score'] = df.apply(get_rivalry, axis=1)
    features['division_rivalry'] = df.get('div_game', False).astype(float)
    
    # 5. Momentum (from records)
    print("     5/5: Momentum patterns...")
    def parse_record_ratio(record_str):
        if pd.isna(record_str) or record_str == '':
            return 0.5
        try:
            wins, losses = record_str.split('-')
            total = int(wins) + int(losses)
            if total == 0:
                return 0.5
            return int(wins) / total
        except:
            return 0.5
    
    home_win_pct = df.get('home_record_before', '0-0').apply(parse_record_ratio)
    away_win_pct = df.get('away_record_before', '0-0').apply(parse_record_ratio)
    
    features['home_momentum'] = home_win_pct
    features['away_momentum'] = away_win_pct
    features['momentum_differential'] = home_win_pct - away_win_pct
    
    print(f"\n  ✓ Generated {len(features.columns)} narrative features")
    
    return features

def main():
    print("="*60)
    print(f"PHASE 4: NARRATIVE TRANSFORMERS - {datetime.now().strftime('%H:%M:%S')}")
    print("="*60)
    
    # Load enriched data
    data_path = Path(__file__).parent.parent.parent / "data" / "domains" / "nfl_enriched_with_rosters.json"
    
    print(f"\n📂 Loading: {data_path.name}")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    games = data['games']
    df = pd.DataFrame(games)
    print(f"✓ Loaded {len(df):,} games")
    
    # Extract narrative features
    print("\n🔄 Applying universal narrative transformers...")
    
    try:
        features = extract_narrative_features(df)
        
        print(f"\n✓ Generated {len(features.columns)} narrative features")
        print(f"  Feature matrix shape: {features.shape}")
        print(f"  Sample features: {list(features.columns)[:5]}")
        
        # Save features
        output_path = Path(__file__).parent.parent.parent / "data" / "domains" / "nfl_narrative_features.csv"
        features.to_csv(output_path, index=False)
        
        print(f"\n✓ Features saved: {output_path.name}")
        print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")
        
    except Exception as e:
        print(f"\n✗ Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"\n{'='*60}")
    print("PHASE 4 COMPLETE ✓")
    print(f"{'='*60}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

