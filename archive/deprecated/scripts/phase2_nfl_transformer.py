#!/usr/bin/env python3
"""
Phase 2: NFL Domain Transformer
Extracts NFL-specific features using NFLPerformanceTransformer
"""

import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def extract_nfl_features(df):
    """Extract NFL-specific features manually"""
    features = pd.DataFrame(index=df.index)
    
    # Team performance features
    features['home_score'] = df['home_score']
    features['away_score'] = df['away_score']
    features['point_differential'] = df['home_score'] - df['away_score']
    features['total_points'] = df['home_score'] + df['away_score']
    features['close_game'] = (abs(df['home_score'] - df['away_score']) <= 7).astype(int)
    
    # Game context
    features['week'] = df['week'].fillna(1)
    features['season_progress'] = df.get('season_progress', df['week'] / 18.0)
    features['playoff'] = df.get('playoff', False).astype(int)
    features['division_game'] = df.get('div_game', False).astype(int)
    features['overtime'] = df.get('overtime', False).astype(int)
    
    # Spread features
    features['spread_line'] = df.get('spread_line', 0).fillna(0)
    features['underdog_game'] = (abs(features['spread_line']) >= 7).astype(int)
    features['favorite_won'] = ((features['spread_line'] < 0) == (features['point_differential'] > 0)).astype(int)
    
    # Team records (parse wins-losses)
    def parse_record(record_str):
        if pd.isna(record_str) or record_str == '' or record_str == '0-0':
            return 0, 0
        try:
            wins, losses = record_str.split('-')
            return int(wins), int(losses)
        except:
            return 0, 0
    
    home_records = df.get('home_record_before', '0-0').apply(parse_record)
    away_records = df.get('away_record_before', '0-0').apply(parse_record)
    
    features['home_wins'] = [r[0] for r in home_records]
    features['away_wins'] = [r[0] for r in away_records]
    features['record_differential'] = features['home_wins'] - features['away_wins']
    features['combined_wins'] = features['home_wins'] + features['away_wins']
    
    # Rivalry intensity (handle nested dict carefully)
    def get_rivalry_games(row):
        mh = row.get('matchup_history')
        if isinstance(mh, dict):
            return mh.get('total_games', 0)
        return 0
    
    features['rivalry_games'] = df.apply(get_rivalry_games, axis=1)
    features['rivalry_intensity'] = features['rivalry_games'] / 50.0  # Normalize
    
    # QB presence (handle nested dict carefully)
    def has_both_qbs(row):
        home_qb = row.get('home_qb')
        away_qb = row.get('away_qb')
        return int(isinstance(home_qb, dict) and isinstance(away_qb, dict))
    
    features['has_qb_data'] = df.apply(has_both_qbs, axis=1)
    
    # Weather/stadium
    features['indoor_game'] = df.get('roof', '').apply(lambda x: 1 if x in ['dome', 'closed'] else 0)
    features['has_weather'] = df.get('temp').notna().astype(int)
    
    return features

def main():
    print("="*60)
    print(f"PHASE 2: NFL DOMAIN FEATURES - {datetime.now().strftime('%H:%M:%S')}")
    print("="*60)
    
    # Load enriched data
    data_path = Path(__file__).parent.parent.parent / "data" / "domains" / "nfl_enriched_with_rosters.json"
    
    print(f"\n📂 Loading: {data_path.name}")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    games = data['games']
    df = pd.DataFrame(games)
    print(f"✓ Loaded {len(df):,} games")
    
    # Extract NFL features
    print("\n🔄 Extracting NFL-specific features...")
    print("   - Team performance metrics")
    print("   - Game context (week, playoffs, division)")
    print("   - Spread and betting lines")
    print("   - Team records and differentials")
    print("   - Rivalry intensity")
    print("   - QB and player data presence")
    
    try:
        features = extract_nfl_features(df)
        
        print(f"\n✓ Generated {len(features.columns)} NFL-specific features")
        print(f"  Feature matrix shape: {features.shape}")
        print(f"  Features: {list(features.columns)[:8]}...")
        
        # Save features
        output_path = Path(__file__).parent.parent.parent / "data" / "domains" / "nfl_domain_features.csv"
        features.to_csv(output_path, index=False)
        
        print(f"\n✓ Features saved: {output_path.name}")
        print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")
        
    except Exception as e:
        print(f"\n✗ Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"\n{'='*60}")
    print("PHASE 2 COMPLETE ✓")
    print(f"{'='*60}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

