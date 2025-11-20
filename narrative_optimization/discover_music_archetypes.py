"""
Discover Music Archetypes

Applies all 7 discovery transformers to music dataset to learn:
- Ξ (Golden Narratio) for music
- Optimal α for music entertainment
- Universal song success patterns
- Cross-domain transferable patterns

This builds the MUSIC ARCHETYPE for comparison with movies.

Author: Narrative Optimization Framework
Date: November 2025
"""

import numpy as np
import json
import pickle
from pathlib import Path
from datetime import datetime
from scipy.stats import pearsonr


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Import discovery transformers
from src.transformers.universal_structural_pattern import UniversalStructuralPatternTransformer
from src.transformers.relational_topology import RelationalTopologyTransformer
from src.transformers.cross_domain_embedding import CrossDomainEmbeddingTransformer
from src.transformers.temporal_derivative import TemporalDerivativeTransformer
from src.transformers.meta_feature_interaction import MetaFeatureInteractionTransformer
from src.transformers.outcome_conditioned_archetype import OutcomeConditionedArchetypeTransformer
from src.transformers.anomaly_uniquity import AnomalyUniquityTransformer

# Import base transformers for genome extraction
from src.transformers.nominative import NominativeAnalysisTransformer
from src.transformers.linguistic_advanced import LinguisticPatternsTransformer
from src.transformers.narrative_potential import NarrativePotentialTransformer


class MusicArchetypeDiscoverer:
    """Discovers music narrative archetypes."""
    
    def __init__(self, max_samples=None):
        self.max_samples = max_samples  # None = use all samples
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'domain': 'music',
            'n_samples': 0,
            'genome_features': None,
            'discovered_archetypes': {},
            'discovered_patterns': [],
            'feature_correlations': {},
            'archetype_library': {}
        }
    
    def load_music(self):
        """Load music dataset."""
        print("="*80)
        print("STEP 1: LOADING MUSIC DATA")
        print("="*80)
        
        print("\nLoading Spotify songs dataset...")
        with open('../data/domains/spotify_songs.json', 'r') as f:
            data = json.load(f)
            songs = data['songs']
        
        total_songs = len(songs)
        print(f"Total songs in dataset: {total_songs}")
        
        # Use all samples or limit if specified
        if self.max_samples is None:
            samples_to_use = total_songs
            print(f"Using ALL {total_songs} songs")
        else:
            samples_to_use = min(self.max_samples, total_songs)
            print(f"Using {samples_to_use} of {total_songs} songs")
        
        # Extract texts and outcomes
        texts = []
        outcomes = []
        song_metadata = []
        
        print(f"Processing songs...")
        
        for i, song in enumerate(songs[:samples_to_use]):
            if i % 5000 == 0:
                print(f"  Processed {i}/{samples_to_use} songs...")
            
            # Create narrative text from song name + artist (songs don't have plots)
            text = f"{song['song_name']} by {song['artist_name']}"
            texts.append(text)
            
            # Outcome: popularity (high = successful)
            # Binary: success if popularity >= 70 (top ~30% of songs)
            popularity = song.get('popularity', 0)
            outcomes.append(1 if popularity >= 70 else 0)
            
            song_metadata.append({
                'song_name': song.get('song_name', 'Unknown'),
                'artist_name': song.get('artist_name', 'Unknown'),
                'popularity': popularity,
                'genre': song.get('genre', 'unknown'),
                'explicit': song.get('explicit', False)
            })
        
        self.texts = texts
        self.outcomes = np.array(outcomes)
        self.song_metadata = song_metadata
        self.results['n_samples'] = len(texts)
        
        success_rate = np.mean(outcomes)
        print(f"\n✓ Loaded {len(texts)} songs")
        print(f"  Success rate (popularity >= 70): {success_rate:.1%}")
        print(f"  Winners: {np.sum(outcomes)}")
        print(f"  Losers: {len(outcomes) - np.sum(outcomes)}")
        
        return self
    
    def extract_genome(self):
        """Extract genome features from songs."""
        print("\n" + "="*80)
        print("STEP 2: EXTRACTING GENOME FEATURES")
        print("="*80)
        
        print("\nApplying base transformers...")
        print("Note: Using song names + artists as narrative text")
        
        transformers = [
            ('Nominative', NominativeAnalysisTransformer()),
            ('Linguistic', LinguisticPatternsTransformer()),
            ('Narrative Potential', NarrativePotentialTransformer())
            # Fast transformers only for 50,000 samples
        ]
        
        genome_features = []
        feature_names = []
        
        for name, transformer in transformers:
            print(f"\n  [{name}]")
            try:
                features = transformer.fit_transform(self.texts)
                genome_features.append(features)
                feat_names = transformer.metadata.get('feature_names', 
                                                      [f'{name}_feat_{i}' for i in range(features.shape[1])])
                feature_names.extend(feat_names)
                print(f"    ✓ Extracted {features.shape[1]} features")
            except Exception as e:
                print(f"    ✗ Error: {e}")
        
        self.genome = np.hstack(genome_features)
        self.feature_names = feature_names
        self.results['genome_features'] = {
            'shape': self.genome.shape,
            'n_features': self.genome.shape[1]
        }
        
        print(f"\n✓ Total genome features: {self.genome.shape[1]}")
        print(f"  Shape: {self.genome.shape}")
        
        return self
    
    def discover_structural_patterns(self):
        """Apply Universal Structural Pattern Transformer."""
        print("\n" + "="*80)
        print("STEP 3: DISCOVERING STRUCTURAL PATTERNS")
        print("="*80)
        
        print("\n[1/7] Universal Structural Pattern Transformer")
        print("  Extracting song name patterns, title structure...")
        
        transformer = UniversalStructuralPatternTransformer(
            n_sequence_points=20,  # Shorter for song names
            use_fft=True,
            detect_changepoints=True
        )
        
        features = transformer.fit_transform(self.texts)
        
        print(f"  ✓ Extracted {features.shape[1]} structural features")
        
        # Analyze key patterns
        arc_slope = features[:, 0]
        tension = features[:, 12]
        
        # Correlation with success
        corr_slope = pearsonr(arc_slope, self.outcomes)[0]
        corr_tension = pearsonr(tension, self.outcomes)[0]
        
        print(f"\n  Discovered Patterns:")
        print(f"    Title structure → success: r = {corr_slope:+.3f}")
        print(f"    Name tension → success: r = {corr_tension:+.3f}")
        
        self.structural_features = features
        self.results['structural_patterns'] = {
            'arc_slope_correlation': corr_slope,
            'tension_correlation': corr_tension
        }
        
        return self
    
    def discover_embedding_space(self):
        """Apply Cross-Domain Embedding Transformer."""
        print("\n" + "="*80)
        print("STEP 4: DISCOVERING UNIVERSAL EMBEDDING SPACE")
        print("="*80)
        
        print("\n[2/7] Cross-Domain Embedding Transformer")
        print("  Projecting songs into universal narrative space...")
        
        # Sample for embedding (too many samples slows down)
        sample_size = min(5000, len(self.genome))
        sample_indices = np.random.choice(len(self.genome), sample_size, replace=False)
        
        print(f"  Using {sample_size} songs for embedding (sampled from {len(self.genome)})")
        
        # Prepare embedding data
        embedding_data = [
            {
                'genome_features': self.genome[i],
                'domain': 'music',
                'text': self.texts[i][:100]  # Truncate
            }
            for i in sample_indices
        ]
        
        transformer = CrossDomainEmbeddingTransformer(
            n_embedding_dims=10,
            n_clusters=8,
            embedding_method='pca',
            track_domains=True
        )
        
        print("  Fitting embedder...")
        sampled_outcomes = self.outcomes[sample_indices]
        features = transformer.fit_transform(embedding_data, y=sampled_outcomes)
        
        print(f"  ✓ Projected into {transformer.metadata['embedding_dim']}-dimensional space")
        print(f"  ✓ Discovered {transformer.metadata['n_clusters']} universal clusters")
        
        # Analyze cluster distributions
        cluster_ids = features[:, 0]
        print(f"\n  Cluster Distribution:")
        for cluster in range(transformer.metadata['n_clusters']):
            count = np.sum(cluster_ids == cluster)
            pct = count / len(cluster_ids) * 100
            print(f"    Cluster {cluster}: {count:4} songs ({pct:5.1f}%)")
        
        # Cluster win rates
        if 'cluster_win_rates' in transformer.metadata:
            print(f"\n  Cluster Success Rates:")
            for cluster, rate in transformer.metadata['cluster_win_rates'].items():
                print(f"    Cluster {cluster}: {rate:.1%}")
        
        self.embedding_features = features
        self.embedding_transformer = transformer
        self.results['embedding'] = {
            'n_clusters': transformer.metadata['n_clusters'],
            'embedding_dim': transformer.metadata['embedding_dim'],
            'sampled_for_embedding': sample_size
        }
        
        return self
    
    def discover_golden_narratio(self):
        """Apply Outcome-Conditioned Archetype Transformer."""
        print("\n" + "="*80)
        print("STEP 5: DISCOVERING GOLDEN NARRATIO (Ξ)")
        print("="*80)
        
        print("\n[3/7] Outcome-Conditioned Archetype Transformer")
        print("  Learning what POPULAR SONGS look like...")
        
        # Prepare data
        embedding_data = [
            {
                'genome_features': self.genome[i],
                'feature_names': self.feature_names,
                'domain': 'music'
            }
            for i in range(len(self.genome))
        ]
        
        transformer = OutcomeConditionedArchetypeTransformer(
            n_winner_clusters=3,
            min_winner_samples=5,
            use_pca=True,
            alpha_method='correlation',
            enable_transfer=True
        )
        
        print("  Clustering popular songs to find Ξ...")
        features = transformer.fit_transform(embedding_data, y=self.outcomes)
        
        print(f"  ✓ Discovered Ξ (Golden Narratio) for music")
        print(f"  ✓ Found {transformer.metadata['n_winner_clusters']} winner sub-archetypes")
        
        # Key discoveries
        optimal_alpha = transformer.optimal_alpha_
        char_eff = transformer.metadata['character_effectiveness']
        plot_eff = transformer.metadata['plot_effectiveness']
        
        print(f"\n  DISCOVERED OPTIMAL BALANCE (α):")
        print(f"    α = {optimal_alpha:.3f}")
        print(f"    Character effectiveness: {char_eff:.3f}")
        print(f"    Plot effectiveness: {plot_eff:.3f}")
        
        if optimal_alpha < 0.5:
            balance = "CHARACTER-HEAVY (traits > events)"
        elif optimal_alpha > 0.5:
            balance = "PLOT-HEAVY (events > traits)"
        else:
            balance = "BALANCED"
        
        print(f"    Interpretation: {balance}")
        
        # Distance to Ξ correlation
        dist_to_xi = features[:, 0]
        corr_xi = pearsonr(dist_to_xi, self.outcomes)[0]
        
        print(f"\n  PREDICTIVE POWER OF Ξ:")
        print(f"    Distance to Ξ → outcomes: r = {corr_xi:+.3f}")
        
        if abs(corr_xi) > 0.3:
            print(f"    🎯 STRONG PATTERN: Songs closer to Ξ are more popular!")
        elif abs(corr_xi) > 0.2:
            print(f"    ✓ MODERATE PATTERN: Ξ distance moderately predictive")
        else:
            print(f"    → WEAK PATTERN: Ξ distance weakly predictive")
        
        # Save archetype
        self.music_xi = transformer.xi_vector_
        self.music_alpha = optimal_alpha
        self.archetype_transformer = transformer
        self.archetype_features = features
        
        self.results['golden_narratio'] = {
            'optimal_alpha': float(optimal_alpha),
            'character_effectiveness': float(char_eff),
            'plot_effectiveness': float(plot_eff),
            'xi_correlation': float(corr_xi),
            'n_winner_clusters': transformer.metadata['n_winner_clusters'],
            'n_winners': transformer.metadata['n_winners'],
            'n_losers': transformer.metadata['n_losers']
        }
        
        self.results['discovered_patterns'].append({
            'type': 'golden_narratio',
            'pattern': f"Optimal α = {optimal_alpha:.3f} ({balance})",
            'strength': abs(corr_xi)
        })
        
        return self
    
    def discover_interactions(self):
        """Apply Meta-Feature Interaction Transformer."""
        print("\n" + "="*80)
        print("STEP 6: DISCOVERING FEATURE INTERACTIONS")
        print("="*80)
        
        print("\n[4/7] Meta-Feature Interaction Transformer")
        print("  Generating interaction features automatically...")
        
        transformer = MetaFeatureInteractionTransformer(
            interaction_degree=2,
            include_ratios=True,
            include_synergy=True,
            max_features=100,
            variance_threshold=0.01
        )
        
        print("  Generating multiplicative, ratio, and synergy features...")
        features = transformer.fit_transform(self.genome, y=self.outcomes)
        
        print(f"  ✓ Generated {features.shape[1]} interaction features")
        print(f"    (from {self.genome.shape[1]} base features)")
        
        # Check top interactions
        if features.shape[1] > 0:
            # Compute correlation of each interaction with outcomes
            correlations = []
            for i in range(min(features.shape[1], 1000)):  # Sample for speed
                if np.std(features[:, i]) > 0:
                    corr = pearsonr(features[:, i], self.outcomes)[0]
                    correlations.append(abs(corr))
                else:
                    correlations.append(0.0)
            
            # Top 5 interactions
            top_indices = np.argsort(correlations)[-5:][::-1]
            
            print(f"\n  Top 5 Discovered Interactions:")
            for rank, idx in enumerate(top_indices, 1):
                feat_name = transformer.metadata['feature_names'][idx] if idx < len(transformer.metadata.get('feature_names', [])) else f'interaction_{idx}'
                corr = correlations[idx]
                print(f"    {rank}. {feat_name[:60]:<60} r = {corr:+.3f}")
        
        self.interaction_features = features
        self.results['interactions'] = {
            'n_generated': features.shape[1],
            'from_base_features': self.genome.shape[1]
        }
        
        return self
    
    def discover_novelty_patterns(self):
        """Apply Anomaly Uniquity Transformer."""
        print("\n" + "="*80)
        print("STEP 7: DISCOVERING NOVELTY PATTERNS")
        print("="*80)
        
        print("\n[5/7] Anomaly Uniquity Transformer")
        print("  Detecting unusual/novel song names...")
        
        transformer = AnomalyUniquityTransformer(
            contamination=0.1,
            n_neighbors=10
        )
        
        features = transformer.fit_transform(self.genome)
        
        print(f"  ✓ Extracted {features.shape[1]} uniqueness features")
        
        # Analyze novelty effects
        novelty_score = features[:, -3]
        
        # Correlation with success
        corr_novelty = pearsonr(novelty_score, self.outcomes)[0]
        
        print(f"\n  Novelty Analysis:")
        print(f"    Mean novelty: {np.mean(novelty_score):.3f}")
        print(f"    Novelty range: [{np.min(novelty_score):.3f}, {np.max(novelty_score):.3f}]")
        print(f"    Novelty → success: r = {corr_novelty:+.3f}")
        
        # Test hypothesis: Does originality help or hurt?
        high_novelty = novelty_score > np.percentile(novelty_score, 75)
        low_novelty = novelty_score < np.percentile(novelty_score, 25)
        
        high_novelty_success = np.mean(self.outcomes[high_novelty])
        low_novelty_success = np.mean(self.outcomes[low_novelty])
        
        print(f"\n  Success by Novelty Level:")
        print(f"    High novelty (top 25%): {high_novelty_success:.1%} success")
        print(f"    Low novelty (bottom 25%): {low_novelty_success:.1%} success")
        print(f"    Difference: {(high_novelty_success - low_novelty_success)*100:+.1f} percentage points")
        
        if high_novelty_success > low_novelty_success + 0.1:
            discovery = "Novelty HELPS music success (originality rewarded)"
        elif high_novelty_success < low_novelty_success - 0.1:
            discovery = "Novelty HURTS music success (familiarity preferred)"
        else:
            discovery = "Novelty has neutral effect on success"
        
        print(f"\n  🔍 DISCOVERY: {discovery}")
        
        self.novelty_features = features
        self.results['novelty'] = {
            'correlation': float(corr_novelty),
            'high_novelty_success': float(high_novelty_success),
            'low_novelty_success': float(low_novelty_success),
            'effect': discovery
        }
        
        self.results['discovered_patterns'].append({
            'type': 'novelty_effect',
            'pattern': discovery,
            'strength': abs(corr_novelty)
        })
        
        return self
    
    def compare_with_movies(self):
        """Compare music archetype with movies archetype."""
        print("\n" + "="*80)
        print("STEP 8: COMPARING WITH MOVIES ARCHETYPE")
        print("="*80)
        
        try:
            with open('archetypes/movies_archetype_library.pkl', 'rb') as f:
                movie_archetype = pickle.load(f)
            
            print("\n✓ Loaded movies archetype for comparison")
            
            movie_alpha = movie_archetype['optimal_alpha']
            music_alpha = self.music_alpha
            
            print(f"\n  α Comparison:")
            print(f"    Movies α: {movie_alpha:.3f}")
            print(f"    Music α:  {music_alpha:.3f}")
            print(f"    Difference: {abs(movie_alpha - music_alpha):.3f}")
            
            if abs(movie_alpha - music_alpha) < 0.1:
                print(f"    → Very similar! Entertainment domains share α")
            elif abs(movie_alpha - music_alpha) < 0.2:
                print(f"    → Somewhat similar entertainment patterns")
            else:
                print(f"    → Different! Music has distinct narrative pattern")
            
            # Compare Ξ vectors (if same dimensionality)
            movie_xi = np.array(movie_archetype['xi_vector'])
            if len(movie_xi) == len(self.music_xi):
                xi_similarity = np.dot(movie_xi, self.music_xi) / (np.linalg.norm(movie_xi) * np.linalg.norm(self.music_xi))
                print(f"\n  Ξ Vector Similarity:")
                print(f"    Cosine similarity: {xi_similarity:.3f}")
                
                if xi_similarity > 0.7:
                    print(f"    → HIGH similarity! Movies archetype transfers to music")
                elif xi_similarity > 0.4:
                    print(f"    → MODERATE similarity. Partial transfer possible")
                else:
                    print(f"    → LOW similarity. Limited transferability")
                
                self.results['comparison_with_movies'] = {
                    'movie_alpha': float(movie_alpha),
                    'music_alpha': float(music_alpha),
                    'alpha_difference': float(abs(movie_alpha - music_alpha)),
                    'xi_similarity': float(xi_similarity)
                }
        
        except FileNotFoundError:
            print("\n  Movies archetype not found. Skipping comparison.")
        
        return self
    
    def save_archetype_library(self):
        """Save discovered archetypes for transfer learning."""
        print("\n" + "="*80)
        print("STEP 9: SAVING ARCHETYPE LIBRARY")
        print("="*80)
        
        # Create archetype package
        archetype_library = {
            'domain': 'music',
            'n_samples': len(self.texts),
            'timestamp': datetime.now().isoformat(),
            'xi_vector': self.music_xi.tolist(),
            'optimal_alpha': float(self.music_alpha),
            'genome_feature_names': self.feature_names,
            'cluster_centroids': self.embedding_transformer.cluster_centroids_.tolist(),
            'n_clusters': self.embedding_transformer.metadata['n_clusters']
        }
        
        # Save to file
        output_path = Path('archetypes/music_archetype_library.pkl')
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(archetype_library, f)
        
        print(f"✓ Saved music archetype library to: {output_path}")
        
        # Also save JSON version
        json_path = Path('archetypes/music_archetype_library.json')
        with open(json_path, 'w') as f:
            json.dump(archetype_library, f, indent=2, cls=NumpyEncoder)
        
        print(f"✓ Saved JSON version to: {json_path}")
        
        # Save full results
        results_path = Path('results/music_discovery_results.json')
        results_path.parent.mkdir(exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, cls=NumpyEncoder)
        
        print(f"✓ Saved full results to: {results_path}")
        
        return self
    
    def generate_report(self):
        """Generate comprehensive discovery report."""
        print("\n" + "="*80)
        print("DISCOVERY REPORT: MUSIC")
        print("="*80)
        
        print(f"\nDataset: {self.results['n_samples']:,} songs")
        print(f"Success rate: {np.mean(self.outcomes):.1%}")
        
        print(f"\n{'='*80}")
        print("KEY DISCOVERIES")
        print("="*80)
        
        # Golden Narratio
        gn = self.results['golden_narratio']
        print(f"\n1. GOLDEN NARRATIO (Ξ) FOR MUSIC:")
        print(f"   - Optimal α: {gn['optimal_alpha']:.3f}")
        print(f"   - Character effectiveness: {gn['character_effectiveness']:.3f}")
        print(f"   - Plot effectiveness: {gn['plot_effectiveness']:.3f}")
        print(f"   - Winner clusters: {gn['n_winner_clusters']}")
        print(f"   - Ξ predictive power: r = {gn['xi_correlation']:+.3f}")
        
        # Structural patterns
        if 'structural_patterns' in self.results:
            sp = self.results['structural_patterns']
            print(f"\n2. STRUCTURAL PATTERNS:")
            print(f"   - Title structure correlation: r = {sp['arc_slope_correlation']:+.3f}")
            print(f"   - Name tension correlation: r = {sp['tension_correlation']:+.3f}")
        
        # Novelty
        if 'novelty' in self.results:
            nov = self.results['novelty']
            print(f"\n3. NOVELTY EFFECTS:")
            print(f"   - Novelty correlation: r = {nov['correlation']:+.3f}")
            print(f"   - High novelty success: {nov['high_novelty_success']:.1%}")
            print(f"   - Low novelty success: {nov['low_novelty_success']:.1%}")
            print(f"   - Effect: {nov['effect']}")
        
        # Comparison with movies
        if 'comparison_with_movies' in self.results:
            comp = self.results['comparison_with_movies']
            print(f"\n4. COMPARISON WITH MOVIES:")
            print(f"   - Movies α: {comp['movie_alpha']:.3f}")
            print(f"   - Music α: {comp['music_alpha']:.3f}")
            print(f"   - Difference: {comp['alpha_difference']:.3f}")
            if 'xi_similarity' in comp:
                print(f"   - Ξ similarity: {comp['xi_similarity']:.3f}")
        
        print(f"\n{'='*80}")
        print("MUSIC ARCHETYPE DISCOVERY COMPLETE")
        print("="*80)
        print(f"\n✓ Processed {self.results['n_samples']:,} songs")
        print(f"✓ Discovered Ξ (Golden Narratio) for music")
        print(f"✓ Archetype library saved and ready")
        
        return self


def main():
    """Main discovery function."""
    discoverer = MusicArchetypeDiscoverer(max_samples=None)  # Use ALL samples
    
    discoverer.load_music()
    discoverer.extract_genome()
    discoverer.discover_structural_patterns()
    discoverer.discover_embedding_space()
    discoverer.discover_golden_narratio()
    discoverer.discover_interactions()
    discoverer.discover_novelty_patterns()
    discoverer.compare_with_movies()
    discoverer.save_archetype_library()
    discoverer.generate_report()
    
    print("\n" + "="*80)
    print("MUSIC ARCHETYPE DISCOVERY COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("  1. Review: results/music_discovery_results.json")
    print("  2. Use archetype: archetypes/music_archetype_library.pkl")
    print("  3. Compare with movies archetype")
    print("="*80)


if __name__ == '__main__':
    main()






