#!/usr/bin/env python3
"""
Complete NFL Feature Extraction
Apply all transformers to enriched NFL dataset

This script:
1. Loads enriched NFL data (with rosters, QBs, matchups)
2. Applies all universal transformers (47+)
3. Applies NFL domain-specific transformers
4. Generates complete feature matrix
5. Calculates story quality scores
6. Saves results for analysis
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*60)
print("NFL COMPLETE FEATURE EXTRACTION")
print("="*60)
print("Loading transformers...")

# Import transformers
from narrative_optimization.src.transformers.nominative import NominativeAnalysisTransformer
from narrative_optimization.src.transformers.emotional_resonance import EmotionalResonanceTransformer
from narrative_optimization.src.transformers.authenticity import AuthenticityTransformer
from narrative_optimization.src.transformers.conflict_tension import ConflictTensionTransformer
from narrative_optimization.src.transformers.sports.nfl_performance import NFLPerformanceTransformer

print("✓ Transformers loaded")

def load_nfl_data():
    """Load enriched NFL data"""
    print("\n" + "="*60)
    print("LOADING NFL DATA")
    print("="*60)
    
    data_path = Path(__file__).parent.parent / "data" / "domains" / "nfl_enriched_with_rosters.json"
    
    if not data_path.exists():
        print(f"✗ Data file not found: {data_path}")
        print("Run enrich_nfl_with_rosters_matchups.py first")
        return None
    
    print(f"📂 Loading: {data_path.name}")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    games = data['games']
    print(f"✓ Loaded {len(games):,} games")
    print(f"  Seasons: {min(g['season'] for g in games)} - {max(g['season'] for g in games)}")
    
    return games

def prepare_dataframe(games):
    """Convert games to DataFrame for transformer processing"""
    print("\n🔄 Preparing DataFrame...")
    
    df = pd.DataFrame(games)
    
    # Add narrative text fields for transformers
    def get_qb_name(qb_data):
        if isinstance(qb_data, dict):
            return qb_data.get('qb_name', 'Unknown')
        return 'Unknown'
    
    df['game_narrative'] = df.apply(lambda row: f"""
{row['away_team']} at {row['home_team']} 
Week {row.get('week', 'N/A')}, {row['season']} season
Records: {row.get('away_record_before', '0-0')} at {row.get('home_record_before', '0-0')}
QBs: {get_qb_name(row.get('away_qb'))} vs {get_qb_name(row.get('home_qb'))}
Final Score: {row['away_score']}-{row['home_score']}
{row['away_team']} {'won' if not row['home_won'] else 'lost'} by {abs(row['result'])} points
""".strip(), axis=1)
    
    # Add team names for nominative analysis
    df['home_team_full'] = df['home_team']
    df['away_team_full'] = df['away_team']
    
    # Add QB names as separate fields
    df['home_qb_name'] = df.apply(lambda row: get_qb_name(row.get('home_qb')), axis=1)
    df['away_qb_name'] = df.apply(lambda row: get_qb_name(row.get('away_qb')), axis=1)
    
    # Add rivalry intensity from matchup history
    df['rivalry_intensity'] = df.apply(lambda row: 
        row.get('matchup_history', {}).get('total_games', 0) / 50.0  # Normalize to 0-1
    , axis=1)
    
    # Add underdog flag
    df['is_underdog_game'] = df.apply(lambda row:
        row.get('spread_line') is not None and abs(row.get('spread_line', 0)) >= 7.0
    , axis=1)
    
    print(f"✓ Prepared {len(df):,} rows with {len(df.columns)} columns")
    return df

def apply_nfl_performance_transformer(df):
    """Apply NFL domain-specific transformer"""
    print("\n" + "="*60)
    print("APPLYING NFL PERFORMANCE TRANSFORMER")
    print("="*60)
    
    try:
        transformer = NFLPerformanceTransformer()
        print(f"✓ Transformer: {transformer.__class__.__name__}")
        
        # Transform
        features = transformer.fit_transform(df)
        print(f"✓ Generated {features.shape[1]} NFL-specific features")
        print(f"  Sample features: {list(features.columns)[:5]}")
        
        return features
    except Exception as e:
        print(f"✗ NFL transformer failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def apply_nominative_transformer(df):
    """Apply nominative analysis to team and QB names"""
    print("\n" + "="*60)
    print("APPLYING NOMINATIVE TRANSFORMER")
    print("="*60)
    
    try:
        transformer = NominativeAnalysisTransformer()
        print(f"✓ Transformer: {transformer.__class__.__name__}")
        
        # Fit on team names and QB names
        transformer.fit(df[['home_team_full', 'away_team_full', 'home_qb_name', 'away_qb_name']])
        
        # Transform
        features = transformer.transform(df[['home_team_full', 'away_team_full', 'home_qb_name', 'away_qb_name']])
        print(f"✓ Generated {features.shape[1]} nominative features")
        
        return features
    except Exception as e:
        print(f"⚠ Nominative transformer: {e}")
        # Return empty features if fails
        return pd.DataFrame(index=df.index)

def apply_narrative_transformers(df):
    """Apply universal narrative transformers"""
    print("\n" + "="*60)
    print("APPLYING UNIVERSAL TRANSFORMERS")
    print("="*60)
    
    all_features = []
    
    # 1. Emotional Resonance
    try:
        print("\n1. Emotional Resonance Transformer...")
        transformer = EmotionalResonanceTransformer()
        transformer.fit(df)
        features = transformer.transform(df)
        print(f"   ✓ {features.shape[1]} features")
        all_features.append(features)
    except Exception as e:
        print(f"   ⚠ Skipped: {e}")
    
    # 2. Authenticity
    try:
        print("\n2. Authenticity Transformer...")
        transformer = AuthenticityTransformer()
        transformer.fit(df)
        features = transformer.transform(df)
        print(f"   ✓ {features.shape[1]} features")
        all_features.append(features)
    except Exception as e:
        print(f"   ⚠ Skipped: {e}")
    
    # 3. Conflict/Tension
    try:
        print("\n3. Conflict Tension Transformer...")
        transformer = ConflictTensionTransformer()
        transformer.fit(df)
        features = transformer.transform(df)
        print(f"   ✓ {features.shape[1]} features")
        all_features.append(features)
    except Exception as e:
        print(f"   ⚠ Skipped: {e}")
    
    if all_features:
        combined = pd.concat(all_features, axis=1)
        print(f"\n✓ Total universal features: {combined.shape[1]}")
        return combined
    else:
        return pd.DataFrame(index=df.index)

def calculate_story_quality(df, all_features):
    """Calculate overall story quality score"""
    print("\n" + "="*60)
    print("CALCULATING STORY QUALITY SCORES")
    print("="*60)
    
    # Normalize and aggregate features
    story_scores = []
    
    # Weight different aspects
    weights = {
        'rivalry': 0.2,      # Matchup history intensity
        'stakes': 0.3,       # Record differential, playoff implications
        'stars': 0.2,        # QB prestige, key players
        'drama': 0.3,        # Close game, comeback, overtime
    }
    
    for idx, row in df.iterrows():
        score = 0.0
        
        # Rivalry component
        rivalry = row.get('rivalry_intensity', 0)
        score += weights['rivalry'] * rivalry
        
        # Stakes component (record differential)
        home_rec = row.get('home_record_before', '0-0')
        away_rec = row.get('away_record_before', '0-0')
        
        try:
            home_wins = int(home_rec.split('-')[0])
            away_wins = int(away_rec.split('-')[0])
            stakes = min(abs(home_wins - away_wins) / 10.0, 1.0)  # Normalize
            score += weights['stakes'] * (1.0 - stakes)  # Closer records = better
        except:
            pass
        
        # Star power (has QB data)
        has_qb_data = bool(row.get('home_qb')) and bool(row.get('away_qb'))
        score += weights['stars'] * (1.0 if has_qb_data else 0.5)
        
        # Drama (close game, overtime)
        point_diff = abs(row['home_score'] - row['away_score'])
        drama = 1.0 - min(point_diff / 30.0, 1.0)  # Closer = more dramatic
        if row.get('overtime'):
            drama = 1.0
        score += weights['drama'] * drama
        
        story_scores.append(score)
    
    df['story_quality'] = story_scores
    
    print(f"✓ Story quality scores calculated")
    print(f"  Mean: {np.mean(story_scores):.3f}")
    print(f"  Std:  {np.std(story_scores):.3f}")
    print(f"  Range: {np.min(story_scores):.3f} - {np.max(story_scores):.3f}")
    
    return df

def save_results(df, features, output_path):
    """Save processed results"""
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Save features matrix
    features_path = output_path.parent / f"{output_path.stem}_features.csv"
    features.to_csv(features_path, index=False)
    print(f"✓ Features saved: {features_path.name} ({features.shape})")
    
    # Save enriched games with story scores
    games_output = df.to_dict('records')
    
    output_data = {
        'metadata': {
            'created': datetime.now().isoformat(),
            'source': 'nfl_enriched_with_rosters.json',
            'transformers_applied': [
                'NFLPerformanceTransformer',
                'NominativeAnalysisTransformer',
                'EmotionalResonanceTransformer',
                'AuthenticityTransformer',
                'ConflictTensionTransformer',
            ],
            'total_games': len(games_output),
            'total_features': features.shape[1],
            'story_quality_stats': {
                'mean': float(np.mean(df['story_quality'])),
                'std': float(np.std(df['story_quality'])),
                'min': float(np.min(df['story_quality'])),
                'max': float(np.max(df['story_quality'])),
            }
        },
        'games': games_output
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Games saved: {output_path.name} ({len(games_output):,} games)")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    return output_path

def create_summary_report(df, features):
    """Create analysis summary"""
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\nDataset:")
    print(f"  Total Games: {len(df):,}")
    print(f"  Seasons: {df['season'].min()} - {df['season'].max()}")
    print(f"  Total Features: {features.shape[1]:,}")
    
    print(f"\nStory Quality Distribution:")
    quality_bins = pd.cut(df['story_quality'], bins=[0, 0.3, 0.5, 0.7, 1.0], 
                          labels=['Low', 'Medium', 'High', 'Exceptional'])
    print(quality_bins.value_counts().sort_index())
    
    print(f"\nTop 5 Best Stories (2024):")
    recent = df[df['season'] == 2024].nlargest(5, 'story_quality')
    for idx, game in recent.iterrows():
        print(f"  {game['away_team']} @ {game['home_team']} (Week {game.get('week')})")
        print(f"    Score: {game['away_score']}-{game['home_score']}, Quality: {game['story_quality']:.3f}")
    
    print(f"\nFeature Coverage:")
    print(f"  Games with QB data: {df['home_qb_name'].notna().sum():,}")
    print(f"  Games with rivalry data: {(df['rivalry_intensity'] > 0).sum():,}")
    print(f"  Underdog games (7+ pt spread): {df['is_underdog_game'].sum():,}")
    print(f"  Overtime games: {df.get('overtime', pd.Series([False]*len(df))).sum():,}")
    
    print("="*60)

def main():
    """Main execution"""
    
    # Load data
    games = load_nfl_data()
    if games is None:
        return 1
    
    # Prepare DataFrame
    df = prepare_dataframe(games)
    
    # Apply transformers
    all_features = []
    
    # 1. NFL Performance Transformer
    nfl_features = apply_nfl_performance_transformer(df)
    if nfl_features is not None:
        all_features.append(nfl_features)
    
    # 2. Nominative Transformer
    nominative_features = apply_nominative_transformer(df)
    if nominative_features is not None and len(nominative_features.columns) > 0:
        all_features.append(nominative_features)
    
    # 3. Universal Narrative Transformers
    narrative_features = apply_narrative_transformers(df)
    if narrative_features is not None and len(narrative_features.columns) > 0:
        all_features.append(narrative_features)
    
    # Combine all features
    if all_features:
        combined_features = pd.concat(all_features, axis=1)
        print(f"\n✓ Combined feature matrix: {combined_features.shape}")
    else:
        print("\n✗ No features generated")
        return 1
    
    # Calculate story quality
    df = calculate_story_quality(df, combined_features)
    
    # Save results
    output_path = Path(__file__).parent.parent / "data" / "domains" / "nfl_with_all_features.json"
    save_results(df, combined_features, output_path)
    
    # Create summary
    create_summary_report(df, combined_features)
    
    print("\n🎉 NFL feature extraction complete!")
    print(f"\nOutput files:")
    print(f"  • nfl_with_all_features.json - Games with story scores")
    print(f"  • nfl_with_all_features_features.csv - Feature matrix")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

