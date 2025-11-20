#!/usr/bin/env python3
"""
Phase 3: Nominative Analysis
Analyzes team names and QB names for semantic fields and prestige
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def calculate_name_features(name):
    """Calculate nominative features for a name"""
    if not name or name == 'Unknown' or pd.isna(name):
        return {
            'length': 0,
            'syllables': 0,
            'vowel_ratio': 0,
            'consonant_ratio': 0,
            'prestige_score': 0
        }
    
    name = str(name)
    vowels = 'aeiouAEIOU'
    
    features = {
        'length': len(name),
        'syllables': max(1, sum(1 for c in name if c in vowels)),
        'vowel_ratio': sum(1 for c in name if c in vowels) / max(1, len(name)),
        'consonant_ratio': sum(1 for c in name if c.isalpha() and c not in vowels) / max(1, len(name)),
    }
    
    # Simple prestige score (can be enhanced with actual career stats)
    features['prestige_score'] = min(1.0, features['length'] / 15.0)
    
    return features

def extract_qb_nominative_features(df):
    """Extract QB name features"""
    features = pd.DataFrame(index=df.index)
    
    print("  📝 Analyzing QB names...")
    
    # Extract QB names
    def get_qb_name(row, side):
        qb = row.get(f'{side}_qb')
        if isinstance(qb, dict):
            return qb.get('qb_name', 'Unknown')
        return 'Unknown'
    
    home_qbs = df.apply(lambda r: get_qb_name(r, 'home'), axis=1)
    away_qbs = df.apply(lambda r: get_qb_name(r, 'away'), axis=1)
    
    # Calculate features for each QB
    for side, qbs in [('home', home_qbs), ('away', away_qbs)]:
        for i, qb_name in enumerate(qbs):
            if i % 500 == 0:
                print(f"     Processing {side} QBs: {i}/{len(qbs)}")
            
            qb_features = calculate_name_features(qb_name)
            for feat_name, feat_val in qb_features.items():
                features.loc[i, f'{side}_qb_{feat_name}'] = feat_val
    
    # QB differential features
    features['qb_prestige_diff'] = features['home_qb_prestige_score'] - features['away_qb_prestige_score']
    features['qb_name_length_diff'] = features['home_qb_length'] - features['away_qb_length']
    
    print(f"  ✓ Generated {len([c for c in features.columns if 'qb' in c])} QB features")
    
    return features

def extract_team_nominative_features(df):
    """Extract team name features"""
    features = pd.DataFrame(index=df.index)
    
    print("  📝 Analyzing team names...")
    
    # Simple team prestige (based on team abbreviation commonality)
    # Elite brands: DAL, NE, GB, SF, PIT
    elite_teams = ['DAL', 'NE', 'GB', 'SF', 'PIT', 'KC', 'BUF']
    
    features['home_elite_brand'] = df['home_team'].isin(elite_teams).astype(int)
    features['away_elite_brand'] = df['away_team'].isin(elite_teams).astype(int)
    features['brand_matchup'] = features['home_elite_brand'] & features['away_elite_brand']
    
    print(f"  ✓ Generated {len([c for c in features.columns if 'brand' in c])} team brand features")
    
    return features

def main():
    print("="*60)
    print(f"PHASE 3: NOMINATIVE ANALYSIS - {datetime.now().strftime('%H:%M:%S')}")
    print("="*60)
    
    # Load enriched data
    data_path = Path(__file__).parent.parent.parent / "data" / "domains" / "nfl_enriched_with_rosters.json"
    
    print(f"\n📂 Loading: {data_path.name}")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    games = data['games']
    df = pd.DataFrame(games)
    print(f"✓ Loaded {len(df):,} games")
    
    # Extract nominative features
    print("\n🔄 Extracting nominative features...")
    
    all_features = []
    
    # QB features
    qb_features = extract_qb_nominative_features(df)
    all_features.append(qb_features)
    
    # Team features
    team_features = extract_team_nominative_features(df)
    all_features.append(team_features)
    
    # Combine
    combined = pd.concat(all_features, axis=1)
    
    print(f"\n✓ Generated {len(combined.columns)} total nominative features")
    print(f"  Feature matrix shape: {combined.shape}")
    
    # Save features
    output_path = Path(__file__).parent.parent.parent / "data" / "domains" / "nfl_nominative_features.csv"
    combined.to_csv(output_path, index=False)
    
    print(f"\n✓ Features saved: {output_path.name}")
    print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")
    
    print(f"\n{'='*60}")
    print("PHASE 3 COMPLETE ✓")
    print(f"{'='*60}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

