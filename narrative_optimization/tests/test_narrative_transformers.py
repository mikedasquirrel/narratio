"""
Test script for enhanced narrative transformers with NHL data.

Verifies that all new transformers work correctly and produce
expected features for sports domain data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import all the new transformers
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transformers.narrative.deep_archetype import DeepArchetypeTransformer
from src.transformers.linguistic.resonance_patterns import LinguisticResonanceTransformer
from src.transformers.narrative.completion_pressure import NarrativeCompletionPressureTransformer
from src.transformers.temporal.narrative_cycles import TemporalNarrativeCyclesTransformer
from src.transformers.cultural.zeitgeist_sync import CulturalZeitgeistTransformer
from src.transformers.ritual.ceremony_effects import RitualCeremonyTransformer
from src.transformers.meta.narrative_awareness import MetaNarrativeAwarenessTransformer
from src.transformers.geographic.location_narrative import GeographicNarrativeTransformer


def create_sample_nhl_game():
    """Create a sample NHL game with narrative features."""
    return {
        # Basic game info
        'home_team': 'Boston Bruins',
        'away_team': 'Montreal Canadiens',
        'home_team_code': 'BOS',
        'away_team_code': 'MTL',
        'game_date': datetime.now(),
        'venue': 'TD Garden',
        'is_home': True,
        'is_playoffs': False,
        
        # Performance context
        'win_pct': 0.65,
        'opponent_win_pct': 0.55,
        'last_10_record': [7, 3],
        'games_played': 40,
        'playoff_position': 3,
        
        # Narrative elements
        'description': 'Historic rivalry game between Original Six teams',
        'is_rival': True,
        'division_rival': True,
        'geographic_rival': True,
        'playoff_meetings_last_10_years': 3,
        
        # Milestone/ceremony info
        'jersey_retirement_ceremony': False,
        'player_milestones': [
            {'type': 'goals', 'current': 398, 'player': 'Patrice Bergeron'}
        ],
        
        # Cultural context
        'current_cultural_events': ['winter_classic_week'],
        'media_narratives': [
            {'themes': ['rivalry', 'tradition']},
            {'themes': ['playoff_race', 'rivalry']}
        ],
        
        # Geographic info
        'travel_distance_miles': 280,
        'time_zone_difference': 0,
        'cities_distance_miles': 280,
        'us_vs_canada_game': True,
        
        # Meta-narrative
        'narrative_media_mentions': 45,
        'total_media_mentions': 60,
        'recent_player_quotes': [
            "This rivalry never gets old",
            "Just another game for us",
            "We know what this means to the fans"
        ]
    }


def test_transformer(transformer, name, sample_data):
    """Test a single transformer."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    try:
        # Fit the transformer
        transformer.fit([sample_data])
        
        # Transform the data
        features = transformer.transform([sample_data])
        
        # Get feature names
        feature_names = transformer.get_feature_names()
        
        print(f"✓ Transformer works correctly")
        print(f"✓ Produces {features.shape[1]} features")
        print(f"✓ Feature names: {len(feature_names)} names")
        print(f"✓ Feature shape: {features.shape}")
        
        # Show first few features
        print(f"\nFirst 10 features:")
        for i in range(min(10, len(feature_names))):
            print(f"  {feature_names[i]}: {features[0, i]:.3f}")
            
        # Check for NaN or inf values
        if np.any(np.isnan(features)):
            print("⚠️  WARNING: NaN values detected!")
        if np.any(np.isinf(features)):
            print("⚠️  WARNING: Inf values detected!")
            
        return True
        
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Test all enhanced narrative transformers."""
    print("ENHANCED NARRATIVE TRANSFORMERS TEST")
    print("====================================")
    
    # Create sample data
    sample_game = create_sample_nhl_game()
    
    # Initialize all transformers
    transformers = [
        (DeepArchetypeTransformer(), "DeepArchetypeTransformer"),
        (LinguisticResonanceTransformer(), "LinguisticResonanceTransformer"),
        (NarrativeCompletionPressureTransformer(), "NarrativeCompletionPressureTransformer"),
        (TemporalNarrativeCyclesTransformer(), "TemporalNarrativeCyclesTransformer"),
        (CulturalZeitgeistTransformer(), "CulturalZeitgeistTransformer"),
        (RitualCeremonyTransformer(), "RitualCeremonyTransformer"),
        (MetaNarrativeAwarenessTransformer(), "MetaNarrativeAwarenessTransformer"),
        (GeographicNarrativeTransformer(), "GeographicNarrativeTransformer")
    ]
    
    # Test each transformer
    results = []
    for transformer, name in transformers:
        success = test_transformer(transformer, name, sample_game)
        results.append((name, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    successful = sum(1 for _, success in results if success)
    print(f"\n✓ {successful}/{len(results)} transformers passed")
    
    if successful < len(results):
        print("\nFailed transformers:")
        for name, success in results:
            if not success:
                print(f"  - {name}")
    
    # Test combined feature extraction
    print(f"\n{'='*60}")
    print("COMBINED FEATURE EXTRACTION TEST")
    print(f"{'='*60}")
    
    total_features = 0
    all_features = []
    
    for transformer, name in transformers:
        try:
            features = transformer.transform([sample_game])
            total_features += features.shape[1]
            all_features.append(features)
        except:
            pass
    
    if all_features:
        combined = np.hstack(all_features)
        print(f"✓ Total combined features: {combined.shape[1]}")
        print(f"✓ Combined shape: {combined.shape}")
        
        # Check data quality
        nan_count = np.sum(np.isnan(combined))
        inf_count = np.sum(np.isinf(combined))
        
        print(f"\nData quality:")
        print(f"  - NaN values: {nan_count}")
        print(f"  - Inf values: {inf_count}")
        print(f"  - Valid values: {combined.size - nan_count - inf_count}")


if __name__ == '__main__':
    main()
