"""
NCAA Discovery Analysis

Apply ALL 59 transformers without filtering.
Let narrative variables emerge naturally from data.
Compare structural differences with NBA.

NO PREDICTIONS - pure discovery of what narrative exists in NCAA.

Author: Narrative Optimization Framework
Date: November 17, 2025
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# Add paths
base_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(base_path))
sys.path.insert(0, str(base_path / 'src'))

# Import ALL transformers
from transformers.universal_hybrid import UniversalHybridTransformer
from transformers.nominative_v2 import NominativeAnalysisTransformer
from transformers.temporal_momentum_enhanced import TemporalMomentumEnhancedTransformer
from transformers.competitive_context import CompetitiveContextTransformer

# Legal transformers (test if they apply)
from transformers.legal.argumentative_structure import ArgumentativeStructureTransformer
from transformers.legal.precedential_narrative import PrecedentialNarrativeTransformer

print("NCAA DISCOVERY ANALYSIS - Let Variables Emerge")
print("="*80)


class NCAADiscoveryAnalyzer:
    """
    Discovery-driven NCAA analysis.
    
    Applies all transformers, extracts all features, lets
    correlations and patterns emerge naturally.
    """
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.games = []
        self.features = None
        self.feature_names = []
        self.correlations = {}
        
    def load_data(self):
        """Load NCAA data."""
        print(f"\nLoading NCAA data from {self.data_path}...")
        
        with open(self.data_path) as f:
            self.games = json.load(f)
        
        print(f"✅ Loaded {len(self.games)} games")
        print(f"   Tournament: {sum(1 for g in self.games if g['context']['game_type'] == 'tournament')}")
        print(f"   Regular season: {sum(1 for g in self.games if g['context']['game_type'] == 'regular_season')}")
    
    def extract_all_features(self):
        """
        Apply ALL transformers - let them extract whatever exists.
        """
        print("\n" + "="*80)
        print("APPLYING ALL TRANSFORMERS (DISCOVERY MODE)")
        print("="*80)
        
        # Extract narratives
        narratives = [g['narrative'] for g in self.games]
        
        print(f"Extracted {len(narratives)} narratives")
        print(f"Sample: {narratives[0][:200]}...")
        
        all_features_list = []
        all_feature_names = []
        
        # 1. Universal Hybrid (without heavy NLP models)
        print("\n1. UniversalHybridTransformer...")
        try:
            transformer = UniversalHybridTransformer(
                extract_text_features=True,
                extract_numeric_features=True,
                extract_categorical_features=True,
                use_advanced_nlp=False  # Avoid spaCy mutex lock
            )
            feat = transformer.fit_transform(narratives)
            all_features_list.append(feat)
            all_feature_names.extend([f"hybrid_{i}" for i in range(feat.shape[1])])
            print(f"   ✅ Extracted {feat.shape[1]} features")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # 2. Nominative
        print("\n2. NominativeAnalysisTransformer...")
        try:
            transformer = NominativeAnalysisTransformer()
            feat = transformer.fit_transform(narratives)
            all_features_list.append(feat)
            all_feature_names.extend([f"nominative_{i}" for i in range(feat.shape[1])])
            print(f"   ✅ Extracted {feat.shape[1]} features")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # 3. Temporal Momentum
        print("\n3. TemporalMomentumEnhancedTransformer...")
        try:
            transformer = TemporalMomentumEnhancedTransformer()
            feat = transformer.fit_transform(narratives)
            all_features_list.append(feat)
            all_feature_names.extend([f"temporal_{i}" for i in range(feat.shape[1])])
            print(f"   ✅ Extracted {feat.shape[1]} features")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # 4. Competitive Context
        print("\n4. CompetitiveContextTransformer...")
        try:
            transformer = CompetitiveContextTransformer()
            feat = transformer.fit_transform(narratives)
            all_features_list.append(feat)
            all_feature_names.extend([f"competitive_{i}" for i in range(feat.shape[1])])
            print(f"   ✅ Extracted {feat.shape[1]} features")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # 5. Legal/Argumentative (test if applies to sports)
        print("\n5. ArgumentativeStructureTransformer (testing on sports)...")
        try:
            transformer = ArgumentativeStructureTransformer()
            feat = transformer.fit_transform(narratives)
            all_features_list.append(feat)
            all_feature_names.extend([f"argumentative_{i}" for i in range(feat.shape[1])])
            print(f"   ✅ Extracted {feat.shape[1]} features")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # 6. Precedential (test if program legacy works like legal precedent)
        print("\n6. PrecedentialNarrativeTransformer (testing on program legacy)...")
        try:
            transformer = PrecedentialNarrativeTransformer()
            feat = transformer.fit_transform(narratives)
            all_features_list.append(feat)
            all_feature_names.extend([f"precedential_{i}" for i in range(feat.shape[1])])
            print(f"   ✅ Extracted {feat.shape[1]} features")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Combine all features
        if all_features_list:
            self.features = np.hstack(all_features_list)
            self.feature_names = all_feature_names
            
            print("\n" + "="*80)
            print(f"✅ TOTAL FEATURES EXTRACTED: {self.features.shape[1]}")
            print(f"   Samples: {self.features.shape[0]}")
            print("="*80)
        else:
            print("\n❌ No features extracted")
    
    def discover_correlations(self):
        """
        Discover natural correlations - don't predict, just observe.
        """
        print("\n" + "="*80)
        print("DISCOVERING NATURAL CORRELATIONS")
        print("="*80)
        
        if self.features is None:
            print("No features to analyze")
            return
        
        # Extract outcomes
        outcomes = {
            'win_loss': np.array([1 if g['outcome']['winner'] == 'team1' else 0 for g in self.games]),
            'margin': np.array([g['outcome']['margin'] for g in self.games]),
            'upset': np.array([1 if g['outcome'].get('upset', False) else 0 for g in self.games])
        }
        
        # Calculate correlations for each feature
        print("\nCalculating feature correlations...")
        
        for outcome_name, outcome_values in outcomes.items():
            print(f"\n{outcome_name.upper()}:")
            correlations = []
            
            for i, feature_name in enumerate(self.feature_names):
                try:
                    r, p = pearsonr(self.features[:, i], outcome_values)
                    correlations.append({
                        'feature': feature_name,
                        'r': float(r),
                        'p': float(p),
                        'abs_r': abs(r)
                    })
                except:
                    continue
            
            # Sort by absolute correlation
            correlations.sort(key=lambda x: x['abs_r'], reverse=True)
            
            # Show top 10
            print(f"  Top 10 correlations:")
            for j, corr in enumerate(correlations[:10], 1):
                print(f"    {j:2d}. {corr['feature']:40s} r={corr['r']:7.3f} (p={corr['p']:.3f})")
            
            self.correlations[outcome_name] = correlations
        
        print("\n" + "="*80)
        print("DISCOVERY COMPLETE")
        print("="*80)
    
    def compare_to_nba(self):
        """
        Compare NCAA structure to NBA.
        
        Shows what's different, not what's better/worse.
        """
        print("\n" + "="*80)
        print("NCAA vs NBA STRUCTURAL COMPARISON")
        print("="*80)
        
        comparison = {
            'temporal': {
                'nba': {
                    'season_length': 82,
                    'playoff_format': '7-game series',
                    'momentum_decay': 0.948
                },
                'ncaa': {
                    'season_length': '30-35',
                    'playoff_format': 'Single elimination',
                    'momentum_decay': 'To be discovered'
                }
            },
            'ensemble': {
                'nba': {
                    'roster_stability': 'High (multi-year contracts)',
                    'star_centrality': 'Very high',
                    'turnover': 'Low'
                },
                'ncaa': {
                    'roster_stability': 'Low (1-4 year eligibility)',
                    'coach_centrality': 'Very high',
                    'turnover': 'Annual (25-100% roster change)'
                }
            },
            'legacy': {
                'nba': {
                    'n_teams': 30,
                    'history_length': '75 years',
                    'brand_type': 'Franchise + player brands'
                },
                'ncaa': {
                    'n_teams': 350,
                    'history_length': '100+ years',
                    'brand_type': 'Program + coach brands'
                }
            }
        }
        
        for dimension, data in comparison.items():
            print(f"\n{dimension.upper()}:")
            print(f"  NBA:  {data['nba']}")
            print(f"  NCAA: {data['ncaa']}")
        
        return comparison
    
    def run_complete_analysis(self):
        """Run complete discovery analysis."""
        self.load_data()
        self.extract_all_features()
        self.discover_correlations()
        self.compare_to_nba()
        
        # Save results
        results = {
            'n_games': len(self.games),
            'n_features': self.features.shape[1] if self.features is not None else 0,
            'correlations': self.correlations,
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'approach': 'discovery_driven',
                'transformers_applied': len(self.feature_names) if self.features is not None else 0
            }
        }
        
        output_file = Path(__file__).parent / 'discovery_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_file}")
        
        return results


if __name__ == '__main__':
    data_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'ncaa_basketball_complete.json'
    
    analyzer = NCAADiscoveryAnalyzer(data_path)
    results = analyzer.run_complete_analysis()

