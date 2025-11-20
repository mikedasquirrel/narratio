"""
NCAA Discovery Analysis - LIGHTWEIGHT VERSION

No spaCy, no sentence transformers, no mutex locks.
Just pure feature extraction from text patterns.

Author: Narrative Optimization Framework
Date: November 17, 2025
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import re

print("NCAA DISCOVERY ANALYSIS - LIGHTWEIGHT")
print("="*80)


def load_ncaa_data():
    """Load NCAA games."""
    data_path = Path('data/domains/ncaa_basketball_complete.json')
    
    with open(data_path) as f:
        games = json.load(f)
    
    print(f"✅ Loaded {len(games)} NCAA games")
    return games


def extract_lightweight_features(games):
    """Extract features without heavy models."""
    print("\nExtracting lightweight features...")
    
    narratives = [g['narrative'] for g in games]
    
    features_list = []
    feature_names = []
    
    # 1. TF-IDF (lightweight, no models needed)
    print("  1. TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=100, stop_words='english')
    tfidf_feat = tfidf.fit_transform(narratives).toarray()
    features_list.append(tfidf_feat)
    feature_names.extend([f"tfidf_{i}" for i in range(100)])
    print(f"     ✅ 100 features")
    
    # 2. Basic text statistics
    print("  2. Text statistics...")
    text_stats = []
    for narrative in narratives:
        words = narrative.split()
        stats = [
            len(words),  # Length
            len(set(words)) / (len(words) + 1),  # Diversity
            narrative.count('championship'),  # Championship mentions
            narrative.count('coach'),  # Coach mentions
            narrative.count('tournament'),  # Tournament mentions
            sum(1 for w in words if w[0].isupper()) / (len(words) + 1),  # Proper noun density
            narrative.count('wins'),  # Win mentions
            narrative.count('titles'),  # Title mentions
            len(re.findall(r'\d+-\d+', narrative)),  # Record mentions
            len(re.findall(r'\d{3,}', narrative))  # Big number mentions (wins, etc)
        ]
        text_stats.append(stats)
    
    features_list.append(np.array(text_stats))
    feature_names.extend(['length', 'diversity', 'championship_mentions', 'coach_mentions', 
                         'tournament_mentions', 'proper_noun_density', 'win_mentions', 
                         'title_mentions', 'record_mentions', 'big_number_mentions'])
    print(f"     ✅ 10 features")
    
    # 3. Program legacy features (from metadata)
    print("  3. Program legacy features...")
    legacy_features = []
    for game in games:
        leg1 = game.get('team1_legacy', {})
        leg2 = game.get('team2_legacy', {})
        
        feats = [
            leg1.get('championships', 0),
            leg2.get('championships', 0),
            leg1.get('championships', 0) - leg2.get('championships', 0),  # Championship differential
            leg1.get('wins', 0) / 1000,  # Normalize
            leg2.get('wins', 0) / 1000,
            (leg1.get('wins', 0) - leg2.get('wins', 0)) / 1000,  # Win differential
            leg1.get('final_fours', 0),
            leg2.get('final_fours', 0)
        ]
        legacy_features.append(feats)
    
    features_list.append(np.array(legacy_features))
    feature_names.extend(['team1_champ', 'team2_champ', 'champ_diff', 'team1_wins_k', 
                         'team2_wins_k', 'wins_diff', 'team1_ff', 'team2_ff'])
    print(f"     ✅ 8 features")
    
    # 4. Coach features (from metadata)
    print("  4. Coach features...")
    coach_features = []
    for game in games:
        coach1_name = game.get('team1_coach')
        coach2_name = game.get('team2_coach')
        
        # Binary: has legendary coach
        feats = [
            1.0 if coach1_name else 0.0,
            1.0 if coach2_name else 0.0,
            1.0 if (coach1_name and not coach2_name) else 0.0,  # Coach advantage
        ]
        coach_features.append(feats)
    
    features_list.append(np.array(coach_features))
    feature_names.extend(['team1_has_coach', 'team2_has_coach', 'coach_advantage'])
    print(f"     ✅ 3 features")
    
    # 5. Context features
    print("  5. Context features...")
    context_features = []
    for game in games:
        ctx = game.get('context', {})
        feats = [
            1.0 if ctx.get('game_type') == 'tournament' else 0.0,
            1.0 if ctx.get('rivalry', False) else 0.0,
            ctx.get('seed1', 8) if ctx.get('seed1') else 8,  # Default mid-seed
            ctx.get('seed2', 8) if ctx.get('seed2') else 8,
            abs(ctx.get('seed1', 8) - ctx.get('seed2', 8)) if ctx.get('seed1') and ctx.get('seed2') else 0
        ]
        context_features.append(feats)
    
    features_list.append(np.array(context_features))
    feature_names.extend(['is_tournament', 'is_rivalry', 'seed1', 'seed2', 'seed_diff'])
    print(f"     ✅ 5 features")
    
    # Combine
    features = np.hstack(features_list)
    
    print(f"\n✅ TOTAL: {features.shape[1]} features extracted")
    print(f"   No heavy models, no mutex locks")
    
    return features, feature_names


def discover_correlations(features, feature_names, games):
    """Find natural correlations."""
    print("\n" + "="*80)
    print("DISCOVERING CORRELATIONS")
    print("="*80)
    
    # Outcomes
    margins = np.array([g['outcome']['margin'] for g in games])
    upsets = np.array([1 if g['outcome'].get('upset', False) else 0 for g in games])
    
    # Find correlations
    correlations = []
    
    for i, fname in enumerate(feature_names):
        try:
            r_margin, p_margin = pearsonr(features[:, i], margins)
            
            correlations.append({
                'feature': fname,
                'r_margin': float(r_margin),
                'p_margin': float(p_margin),
                'abs_r': abs(r_margin)
            })
        except:
            continue
    
    # Sort by strength
    correlations.sort(key=lambda x: x['abs_r'], reverse=True)
    
    print("\nTop 15 features by correlation with margin:")
    for i, corr in enumerate(correlations[:15], 1):
        print(f"  {i:2d}. {corr['feature']:30s} r={corr['r_margin']:7.3f}")
    
    return correlations


def main():
    # Load data
    games = load_ncaa_data()
    
    # Extract features (lightweight)
    features, feature_names = extract_lightweight_features(games)
    
    # Discover correlations
    correlations = discover_correlations(features, feature_names, games)
    
    # Save results
    results = {
        'n_games': len(games),
        'n_features': features.shape[1],
        'top_correlations': correlations[:20],
        'feature_names': feature_names
    }
    
    output_file = Path('narrative_optimization/domains/ncaa/discovery_results_lightweight.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    print("\n" + "="*80)
    print(f"NCAA Discovery Complete: {features.shape[1]} features analyzed")
    print("="*80)


if __name__ == '__main__':
    main()



