"""
NHL Feature Extraction Pipeline

Combines all feature extraction methods:
1. Universal transformers (47 transformers, ~200-300 features)
2. NHL-specific performance features (50 features)
3. NHL nominative features (29 features)

Total genome: ~280-380 features (ж)

This follows the same pattern as NBA/NFL feature extraction.

Author: Narrative Integration System
Date: November 16, 2025
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from narrative_optimization.src.transformers.sports.nhl_performance import NHLPerformanceTransformer
from narrative_optimization.domains.nhl.nhl_nominative_features import (
    NHLNominativeExtractor, extract_nominative_features_batch
)
from narrative_optimization.src.pipelines.domain_config import DomainConfig
from narrative_optimization.src.transformers.transformer_selector import TransformerSelector
from narrative_optimization.src.pipelines.feature_extraction_pipeline import FeatureExtractionPipeline

# Transformers that require structured schedule/player data (handled outside the universal text pipeline)
INVISIBLE_TRANSFORMERS = {
    'ScheduleNarrativeTransformer',
    'MilestoneProximityTransformer',
    'CalendarRhythmTransformer',
    'BroadcastNarrativeTransformer',
    'NarrativeInterferenceTransformer',
    'OpponentContextTransformer',
    'SeasonSeriesNarrativeTransformer',
    'EliminationProximityTransformer',
}


class NHLFeatureExtractor:
    """Complete NHL feature extraction pipeline"""
    
    def __init__(self):
        """Initialize all extractors"""
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.nhl_performance = NHLPerformanceTransformer()
        self.nominative_extractor = NHLNominativeExtractor()
        self.transformer_selector = TransformerSelector()
        self.domain_config = self._load_domain_config()
        self.universal_metadata: Dict[str, Any] = {}
        self.pipeline_cache_dir = self.project_root / "narrative_optimization" / "cache" / "features"
        self.pipeline_cache_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def domain_name(self) -> str:
        return "NHL"
    
    def _load_domain_config(self) -> Optional[DomainConfig]:
        """Load domain configuration (π, type) from config.yaml."""
        config_path = Path(__file__).with_name("config.yaml")
        try:
            return DomainConfig.from_yaml(config_path)
        except Exception as exc:
            print(f"⚠️  Could not load NHL domain config ({exc}). Using defaults.")
            return None
    
    def _select_transformers(self, narratives: List[str]) -> List[str]:
        """Select transformer suite for the NHL domain."""
        pi_value = self.domain_config.pi if self.domain_config else 0.6
        domain_type = self.domain_config.type.value if self.domain_config else "sports"
        sample = narratives[: min(len(narratives), 100)]
        
        transformer_names = self.transformer_selector.select_transformers(
            domain_name=self.domain_name,
            pi_value=pi_value,
            domain_type=domain_type,
            data_sample=sample,
            include_renovation=True,
            include_expensive=True,
        )
        
        # Filter out transformers that require structured schedule/player feeds
        filtered = [t for t in transformer_names if t not in INVISIBLE_TRANSFORMERS]
        return filtered
    
    def create_game_narrative(self, game: Dict) -> str:
        """
        Create narrative text from game data for universal transformers.
        
        This is critical: universal transformers work on text narratives,
        so we need to convert structured game data into rich narrative descriptions.
        
        Parameters
        ----------
        game : dict
            Game data dictionary
        
        Returns
        -------
        narrative : str
            Narrative description of the game
        """
        home_team = game.get('home_team', '')
        away_team = game.get('away_team', '')
        date = game.get('date', '')[:10]
        
        # Get temporal context
        tc = game.get('temporal_context', {})
        home_wins = tc.get('home_wins', 0)
        home_losses = tc.get('home_losses', 0)
        away_wins = tc.get('away_wins', 0)
        away_losses = tc.get('away_losses', 0)
        home_l10 = tc.get('home_l10_wins', 5)
        away_l10 = tc.get('away_l10_wins', 5)
        
        # Build narrative
        narrative_parts = []
        
        # Opening
        narrative_parts.append(f"{away_team} at {home_team} on {date}.")
        
        # Records
        narrative_parts.append(
            f"{away_team} comes in with a record of {away_wins}-{away_losses}, "
            f"while {home_team} is {home_wins}-{home_losses}."
        )
        
        # Recent form
        narrative_parts.append(
            f"In their last 10 games, {away_team} has won {away_l10} "
            f"and {home_team} has won {home_l10}."
        )
        
        # Goalie matchup
        home_goalie = game.get('home_goalie', 'the starting goalie')
        away_goalie = game.get('away_goalie', 'the starting goalie')
        narrative_parts.append(
            f"The goalie matchup features {away_goalie} for {away_team} "
            f"against {home_goalie} for {home_team}."
        )
        
        # Rivalry context
        if game.get('is_rivalry', False):
            narrative_parts.append(
                f"This is a heated rivalry game between {away_team} and {home_team}, "
                "adding extra intensity and narrative weight."
            )
        
        # Playoff context
        if game.get('is_playoff', False):
            narrative_parts.append(
                "This is a playoff game with championship implications and elevated stakes."
            )
        
        # Momentum context
        home_back_to_back = tc.get('home_back_to_back', False)
        away_back_to_back = tc.get('away_back_to_back', False)
        
        if home_back_to_back or away_back_to_back:
            if home_back_to_back:
                narrative_parts.append(
                    f"{home_team} is playing on back-to-back nights, "
                    "potentially facing fatigue."
                )
            if away_back_to_back:
                narrative_parts.append(
                    f"{away_team} is playing their second game in as many nights, "
                    "which could impact their performance."
                )
        
        # Rest advantage
        rest_adv = tc.get('rest_advantage', 0)
        if abs(rest_adv) >= 2:
            if rest_adv > 0:
                narrative_parts.append(
                    f"{home_team} has a significant rest advantage, "
                    f"coming in {rest_adv} days more rested."
                )
            else:
                narrative_parts.append(
                    f"{away_team} has the rest advantage, "
                    f"with {abs(rest_adv)} more days off."
                )
        
        # Special teams context (if available)
        if 'power_play_pct' in game:
            pp_pct = game.get('power_play_pct', 0.20)
            if pp_pct > 0.25:
                narrative_parts.append(
                    f"{home_team} brings a dangerous power play clicking at {pp_pct*100:.1f}%."
                )
        
        # Join all parts
        narrative = " ".join(narrative_parts)
        
        return narrative
    
    def extract_universal_features(self, games: List[Dict]) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Extract features using the full universal transformer pipeline (text-based).
        """
        print("\n🌐 APPLYING UNIVERSAL TRANSFORMER PIPELINE")
        print("-"*80)
        
        narratives = [self.create_game_narrative(game) for game in games]
        transformer_names = self._select_transformers(narratives)
        
        if not transformer_names:
            print("⚠️  No universal transformers selected. Skipping.")
            return None, {}
        
        pipeline = FeatureExtractionPipeline(
            transformer_names=transformer_names,
            domain_name=self.domain_name,
            cache_dir=self.pipeline_cache_dir,
            enable_caching=True,
            verbose=True,
        )
        
        try:
            features = pipeline.fit_transform(narratives)
        except Exception as exc:
            print(f"⚠️  Universal pipeline failed: {exc}")
            return None, {}
        
        report = pipeline.get_extraction_report()
        transformer_status = report.get('transformer_status', {})
        for status in transformer_status.values():
            status.pop('instance', None)
        successful_transformers = [
            name for name, status in transformer_status.items()
            if status.get('status') == 'initialized'
        ]
        feature_names = list(pipeline.feature_provenance.keys())
        metadata = {
            'transformers': successful_transformers,
            'feature_names': feature_names,
            'extraction_report': report,
        }
        self.universal_metadata = metadata
        
        print(f"\n   ✓ Universal feature count: {features.shape[1]}")
        return features, metadata
    
    def extract_performance_features(self, games: List[Dict]) -> np.ndarray:
        """
        Extract NHL-specific performance features.
        
        Parameters
        ----------
        games : list of dict
            Game data
        
        Returns
        -------
        features : ndarray
            Performance features (50 dimensions)
        """
        print("\n🏒 APPLYING NHL PERFORMANCE TRANSFORMER")
        print("-"*80)
        
        # Fit and transform
        self.nhl_performance.fit(games)
        features = self.nhl_performance.transform(games)
        
        print(f"   ✓ Extracted {features.shape[1]} performance features")
        
        return features
    
    def extract_nominative_features(self, games: List[Dict]) -> np.ndarray:
        """
        Extract nominative (name-based) features.
        
        Parameters
        ----------
        games : list of dict
            Game data
        
        Returns
        -------
        features : ndarray
            Nominative features (29 dimensions)
        """
        print("\n📛 APPLYING NOMINATIVE FEATURES")
        print("-"*80)
        
        features = extract_nominative_features_batch(games)
        
        print(f"   ✓ Extracted {features.shape[1]} nominative features")
        
        return features
    
    def extract_complete_genome(self, games: List[Dict]) -> Tuple[np.ndarray, Dict]:
        """
        Extract complete feature genome (ж) for NHL games.
        
        Combines:
        - Universal transformer features (~200-300)
        - NHL performance features (50)
        - Nominative features (29)
        
        Parameters
        ----------
        games : list of dict
            Game data
        
        Returns
        -------
        genome : ndarray
            Complete feature matrix
        metadata : dict
            Information about feature extraction
        """
        print("\n" + "="*80)
        print("NHL COMPLETE FEATURE EXTRACTION PIPELINE")
        print("="*80)
        print(f"Processing {len(games)} games...")
        
        feature_arrays: List[np.ndarray] = []
        feature_counts: Dict[str, int] = {}
        
        # 1. Universal transformers
        universal_features, universal_meta = self.extract_universal_features(games)
        if universal_features is not None:
            feature_arrays.append(universal_features)
            feature_counts['universal'] = universal_features.shape[1]
        else:
            print("   ⚠️  Skipping universal features")
            feature_counts['universal'] = 0
            universal_meta = {}
        
        # 2. NHL performance features
        performance_features = self.extract_performance_features(games)
        feature_arrays.append(performance_features)
        feature_counts['performance'] = performance_features.shape[1]
        
        # 3. Nominative features
        nominative_features = self.extract_nominative_features(games)
        feature_arrays.append(nominative_features)
        feature_counts['nominative'] = nominative_features.shape[1]
        
        # Combine all features
        print("\n🔗 COMBINING FEATURES")
        print("-"*80)
        
        complete_genome = np.concatenate(feature_arrays, axis=1)
        
        # Metadata
        metadata = {
            'n_games': len(games),
            'total_features': complete_genome.shape[1],
            'feature_breakdown': feature_counts,
            'universal': universal_meta,
            'feature_types': {
                'universal': list(range(0, feature_counts.get('universal', 0))),
                'performance': list(range(
                    feature_counts.get('universal', 0),
                    feature_counts.get('universal', 0) + feature_counts['performance']
                )),
                'nominative': list(range(
                    feature_counts.get('universal', 0) + feature_counts['performance'],
                    complete_genome.shape[1]
                )),
            }
        }
        
        print(f"   Feature breakdown:")
        for feat_type, count in feature_counts.items():
            print(f"   - {feat_type}: {count} features")
        print(f"\n   ✅ TOTAL GENOME SIZE (ж): {complete_genome.shape[1]} features")
        print("="*80)
        
        return complete_genome, metadata


def load_nhl_data(data_path: str) -> List[Dict]:
    """Load NHL game data from JSON file"""
    with open(data_path, 'r') as f:
        games = json.load(f)
    return games


def main():
    """Main execution - extract features from NHL dataset"""
    
    # Paths
    project_root = Path(__file__).parent.parent.parent.parent
    data_path = project_root / 'data' / 'domains' / 'nhl_games_with_odds.json'
    output_dir = project_root / 'narrative_optimization' / 'domains' / 'nhl'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if data exists
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        print("Run the NHL data builder first:")
        print("  python data_collection/nhl_data_builder.py")
        return
    
    # Load data
    print(f"\n📂 Loading NHL data from {data_path.name}...")
    games = load_nhl_data(data_path)
    print(f"   ✓ Loaded {len(games)} games")
    
    # Extract features
    extractor = NHLFeatureExtractor()
    genome, metadata = extractor.extract_complete_genome(games)
    
    # Save features
    features_path = output_dir / 'nhl_features_complete.npz'
    metadata_path = output_dir / 'nhl_features_metadata.json'
    
    print(f"\n💾 SAVING FEATURES")
    print("-"*80)
    print(f"   Features: {features_path}")
    print(f"   Metadata: {metadata_path}")
    
    # Save features as numpy compressed
    np.savez_compressed(
        features_path,
        features=genome,
        game_ids=[g.get('game_id', '') for g in games]
    )
    
    # Save metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   ✓ Saved successfully")
    print("\n✅ FEATURE EXTRACTION COMPLETE!")
    print("="*80)
    print(f"\nFeature genome (ж): {genome.shape[1]} dimensions")
    print(f"Ready for domain formula calculation and pattern discovery.")


if __name__ == "__main__":
    main()

