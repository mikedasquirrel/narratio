"""
Classical Archetype Discovery Tool

Discovers archetypal patterns from literary corpus using unsupervised and supervised methods.
Extends existing archetype discovery to classical narrative theory.

Author: Narrative Optimization Framework
Date: November 13, 2025
"""

import numpy as np
import argparse
import json
from typing import List, Dict, Tuple
from pathlib import Path
from collections import Counter
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from transformers.archetypes import (
    HeroJourneyTransformer,
    CharacterArchetypeTransformer,
    PlotArchetypeTransformer,
    discover_journey_patterns,
    discover_archetype_patterns,
    discover_plot_patterns
)


class ClassicalArchetypeDiscovery:
    """
    Discover archetypal patterns from literary corpus.
    
    Methods:
    1. Unsupervised: Cluster narratives in archetype space
    2. Supervised: Learn what predicts success
    3. Comparative: Compare to classical theory
    """
    
    def __init__(self):
        self.transformers = {
            'journey': HeroJourneyTransformer(),
            'character': CharacterArchetypeTransformer(),
            'plot': PlotArchetypeTransformer()
        }
    
    def discover_from_corpus(self, texts: List[str], outcomes: np.ndarray,
                            method='all') -> Dict:
        """
        Discover patterns from corpus.
        
        Args:
            texts: List of narrative texts
            outcomes: Success outcomes
            method: 'unsupervised', 'supervised', or 'all'
            
        Returns:
            Discovered patterns and validation of classical theory
        """
        results = {'discoveries': {}, 'validations': {}}
        
        if method in ['supervised', 'all']:
            print("Discovering journey patterns...")
            journey_results = discover_journey_patterns(texts, outcomes, method='correlation')
            results['discoveries']['journey'] = journey_results
            
            print("Discovering character patterns...")
            character_results = discover_archetype_patterns(texts, outcomes, method='correlation')
            results['discoveries']['character'] = character_results
            
            print("Discovering plot patterns...")
            plot_results = discover_plot_patterns(texts, outcomes, method='correlation')
            results['discoveries']['plot'] = plot_results
        
        if method in ['unsupervised', 'all']:
            print("Clustering narratives...")
            clusters = self.cluster_narratives(texts)
            results['discoveries']['clusters'] = clusters
        
        # Generate summary
        results['summary'] = self._generate_discovery_summary(results['discoveries'])
        
        return results
    
    def cluster_narratives(self, texts: List[str], n_clusters=5) -> Dict:
        """
        Cluster narratives in archetype space.
        
        Discovers natural groupings (may correspond to classical types or reveal new ones).
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        # Extract features
        all_features = []
        for transformer in self.transformers.values():
            transformer.fit(texts)
            features = transformer.transform(texts)
            all_features.append(features)
        
        # Concatenate all archetype features
        combined_features = np.concatenate(all_features, axis=1)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(combined_features)
        
        # Silhouette score
        silhouette = silhouette_score(combined_features, clusters)
        
        # Analyze each cluster
        cluster_profiles = {}
        for cluster_id in range(n_clusters):
            cluster_mask = clusters == cluster_id
            cluster_features = combined_features[cluster_mask]
            
            # Mean profile
            mean_profile = np.mean(cluster_features, axis=0)
            
            # Identify dominant characteristics
            # (In production, would use feature names)
            cluster_profiles[cluster_id] = {
                'size': int(cluster_mask.sum()),
                'mean_features': mean_profile.tolist(),
                'example_indices': np.where(cluster_mask)[0][:5].tolist()
            }
        
        return {
            'n_clusters': n_clusters,
            'silhouette_score': silhouette,
            'cluster_assignments': clusters.tolist(),
            'cluster_profiles': cluster_profiles
        }
    
    def compare_to_classical_theory(self, learned_patterns: Dict, 
                                   theory_name: str) -> Dict:
        """
        Compare discovered patterns to classical theory expectations.
        
        Args:
            learned_patterns: From discover_from_corpus
            theory_name: 'campbell', 'jung', 'booker', etc.
            
        Returns:
            Validation results
        """
        if theory_name == 'campbell' and 'journey' in learned_patterns:
            journey_data = learned_patterns['journey']
            return journey_data['theoretical_validation']
        
        elif theory_name == 'jung' and 'character' in learned_patterns:
            character_data = learned_patterns['character']
            return character_data['theoretical_validation']
        
        elif theory_name == 'booker' and 'plot' in learned_patterns:
            plot_data = learned_patterns['plot']
            return plot_data['theoretical_validation']
        
        return {'error': f'No data for theory {theory_name}'}
    
    def identify_new_patterns(self, discovered: Dict, classical: Dict) -> List[Dict]:
        """
        Identify patterns discovered empirically that aren't in classical theory.
        
        These are NEW ARCHETYPES!
        """
        new_patterns = []
        
        # Look for high-importance features not in classical theory
        if 'journey' in discovered:
            learned_weights = discovered['journey']['learned_weights']
            
            for stage, weight in learned_weights.items():
                # If high weight but Campbell gave low weight = discovery
                stage_info = discovered['journey']['theoretical_validation']['stages'].get(stage, {})
                if stage_info.get('campbell_undervalued', False):
                    new_patterns.append({
                        'pattern': stage,
                        'type': 'journey_stage',
                        'empirical_importance': weight,
                        'theoretical_importance': stage_info.get('theoretical_weight', 0),
                        'discovery': 'Undervalued by classical theory'
                    })
        
        return new_patterns
    
    def _generate_discovery_summary(self, discoveries: Dict) -> Dict:
        """Generate human-readable summary of discoveries."""
        summary = {
            'total_patterns_analyzed': 0,
            'classical_validated': [],
            'classical_challenged': [],
            'new_patterns': []
        }
        
        # Journey patterns
        if 'journey' in discoveries:
            journey = discoveries['journey']['theoretical_validation']
            summary['total_patterns_analyzed'] += 17  # Campbell stages
            
            if journey['summary'].get('campbell_validated', False):
                summary['classical_validated'].append('Campbell Hero\'s Journey')
            else:
                summary['classical_challenged'].append({
                    'theory': 'Campbell',
                    'reason': f"Correlation {journey['summary'].get('correlation', 0):.2f} < 0.80"
                })
        
        # Character patterns
        if 'character' in discoveries:
            summary['total_patterns_analyzed'] += 12  # Jung archetypes
        
        # Plot patterns
        if 'plot' in discoveries:
            summary['total_patterns_analyzed'] += 7  # Booker plots
        
        return summary


def main():
    """Command-line interface for archetype discovery."""
    parser = argparse.ArgumentParser(description='Discover classical archetypal patterns from corpus')
    
    parser.add_argument('--corpus', type=str, required=True,
                       help='Path to corpus JSON file with texts and outcomes')
    parser.add_argument('--domain', type=str, default='literature',
                       help='Domain name (literature, mythology, film, etc.)')
    parser.add_argument('--method', type=str, default='all',
                       choices=['supervised', 'unsupervised', 'all'],
                       help='Discovery method')
    parser.add_argument('--output', type=str, default='archetype_discoveries.json',
                       help='Output path for results')
    parser.add_argument('--validate', type=str, nargs='+',
                       choices=['campbell', 'jung', 'booker', 'aristotle', 'frye'],
                       help='Theories to validate')
    
    args = parser.parse_args()
    
    # Load corpus
    print(f"Loading corpus from {args.corpus}...")
    with open(args.corpus) as f:
        corpus_data = json.load(f)
    
    texts = corpus_data['texts']
    outcomes = np.array(corpus_data['outcomes'])
    
    print(f"Loaded {len(texts)} narratives from {args.domain} domain")
    
    # Discover patterns
    print(f"\nDiscovering patterns using {args.method} method...")
    discoverer = ClassicalArchetypeDiscovery()
    results = discoverer.discover_from_corpus(texts, outcomes, method=args.method)
    
    # Validate classical theories if requested
    if args.validate:
        print(f"\nValidating classical theories: {args.validate}")
        results['classical_validation'] = {}
        
        for theory in args.validate:
            validation = discoverer.compare_to_classical_theory(
                results['discoveries'], theory
            )
            results['classical_validation'][theory] = validation
            
            if validation.get('summary', {}).get('theory_validated', False):
                print(f"✓ {theory.capitalize()} validated!")
            else:
                print(f"✗ {theory.capitalize()} challenged")
    
    # Identify new patterns
    print("\nIdentifying new patterns...")
    new_patterns = discoverer.identify_new_patterns(
        results['discoveries'],
        {}  # Classical theory reference
    )
    
    if new_patterns:
        print(f"Found {len(new_patterns)} novel patterns!")
        for pattern in new_patterns[:5]:
            print(f"  - {pattern['pattern']}: {pattern['discovery']}")
    
    # Save results
    print(f"\nSaving results to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print("DISCOVERY COMPLETE")
    print("="*60)
    print(f"Patterns analyzed: {results['summary']['total_patterns_analyzed']}")
    print(f"Classical validated: {len(results['summary']['classical_validated'])}")
    print(f"Classical challenged: {len(results['summary']['classical_challenged'])}")
    print(f"New patterns found: {len(new_patterns)}")
    print("="*60)


if __name__ == '__main__':
    main()

