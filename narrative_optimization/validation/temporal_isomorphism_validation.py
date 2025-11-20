"""
Temporal Isomorphism Validation

Test hypothesis: Narratives at equivalent % completion show similar patterns
regardless of absolute duration.

Examples to test:
- NBA Game minute 35/48 (73%) ≈ Novel page 220/300 (73%)
- UFC Round 3/5 (60%) ≈ Film Act 2/3 (67%)
- Symphony mvt 3/4 (75%) ≈ TV Ep 17/22 (77%)

Uses AI embeddings to measure structural similarity WITHOUT presupposing
what the structure IS.

If validated: Enables transfer learning across temporal scales.

Author: Narrative Optimization Framework
Date: November 2025
"""

from typing import List, Dict, Tuple
import numpy as np
from pathlib import Path
import json

try:
    from ..src.transformers.utils.embeddings import EmbeddingManager
    from ..src.analysis.sequential_narrative_processor import SequentialNarrativeProcessor
except ImportError:
    from narrative_optimization.src.transformers.utils.embeddings import EmbeddingManager
    from narrative_optimization.src.analysis.sequential_narrative_processor import SequentialNarrativeProcessor


class TemporalIsomorphismValidator:
    """
    Test if narrative structure at X% completion is similar across domains.
    
    Method:
    1. Extract narratives from multiple domains
    2. Identify equivalent completion points (e.g., all at 73%)
    3. Analyze local structure (ς, ρ, semantic patterns) at those points
    4. Measure cross-domain correlation
    5. If r > 0.60: Isomorphism validated
    """
    
    def __init__(self):
        """Initialize validator."""
        self.embedder = EmbeddingManager()
        self.processor = SequentialNarrativeProcessor()
        
        # Test points (% completion)
        self.test_points = [0.25, 0.50, 0.73, 0.85]
        
    def validate_isomorphism(
        self,
        domain_pairs: List[Tuple[str, List[str], List[str]]],
        output_file: Optional[str] = None
    ) -> Dict:
        """
        Test isomorphism across domain pairs.
        
        Parameters
        ----------
        domain_pairs : list of (name, domain1_narratives, domain2_narratives)
        output_file : str, optional
            Where to save results
            
        Returns
        -------
        validation_results : dict
            Correlations at each test point,
            Overall isomorphism scores,
            Validated pairs (r > 0.60)
        """
        print(f"\n{'='*80}")
        print("TEMPORAL ISOMORPHISM VALIDATION")
        print(f"{'='*80}\n")
        print("Hypothesis: Structure at X% is similar across domains")
        print(f"Test points: {self.test_points}")
        print("Method: AI embeddings + unsupervised comparison\n")
        
        results = []
        
        for pair_name, domain1_narratives, domain2_narratives in domain_pairs:
            print(f"Testing pair: {pair_name}")
            
            pair_result = self._test_pair(
                pair_name,
                domain1_narratives,
                domain2_narratives
            )
            
            results.append(pair_result)
            
            print(f"  Result: r = {pair_result['avg_correlation']:.3f}")
            print(f"  Isomorphism: {'✓ VALIDATED' if pair_result['validated'] else '✗ Not validated'}\n")
        
        # Overall summary
        validated_count = sum(1 for r in results if r['validated'])
        
        summary = {
            'n_pairs_tested': len(results),
            'n_validated': validated_count,
            'validation_rate': validated_count / len(results) if results else 0.0,
            'pair_results': results,
            'conclusion': 'Temporal isomorphism VALIDATED' if validated_count >= len(results) * 0.7 else 'Temporal isomorphism NOT validated'
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"✓ Results saved: {output_file}")
        
        return summary
    
    def _test_pair(
        self,
        pair_name: str,
        narratives1: List[str],
        narratives2: List[str]
    ) -> Dict:
        """Test single domain pair."""
        
        # Process narratives from both domains
        features1_by_point = {point: [] for point in self.test_points}
        features2_by_point = {point: [] for point in self.test_points}
        
        # Extract features at test points for domain 1
        for narrative in narratives1[:100]:  # Sample for performance
            features_at_points = self._extract_features_at_points(narrative)
            for point, features in features_at_points.items():
                features1_by_point[point].append(features)
        
        # Extract features at test points for domain 2
        for narrative in narratives2[:100]:
            features_at_points = self._extract_features_at_points(narrative)
            for point, features in features_at_points.items():
                features2_by_point[point].append(features)
        
        # Compute correlations at each point
        correlations = {}
        for point in self.test_points:
            if features1_by_point[point] and features2_by_point[point]:
                # Average features at this point
                avg1 = np.mean(features1_by_point[point], axis=0)
                avg2 = np.mean(features2_by_point[point], axis=0)
                
                # Correlation between avg feature patterns
                if len(avg1) == len(avg2):
                    r = np.corrcoef(avg1, avg2)[0, 1]
                    correlations[point] = float(r)
        
        avg_correlation = np.mean(list(correlations.values())) if correlations else 0.0
        
        return {
            'pair_name': pair_name,
            'correlations_by_point': correlations,
            'avg_correlation': avg_correlation,
            'validated': avg_correlation > 0.60
        }
    
    def _extract_features_at_points(
        self,
        narrative: str
    ) -> Dict[float, np.ndarray]:
        """
        Extract features at specific completion points.
        
        Returns features WITHOUT interpreting them.
        """
        # Segment narrative
        segments = narrative.split('\n\n') if '\n\n' in narrative else [narrative]
        segments = [s.strip() for s in segments if s.strip()]
        
        if len(segments) < 5:
            return {}
        
        # Embed segments
        embeddings = self.embedder.encode(segments, show_progress=False)
        positions = np.linspace(0, 1, len(segments))
        
        # Extract features at each test point
        features_at_points = {}
        
        for test_point in self.test_points:
            # Find segment nearest to test point
            nearest_idx = np.argmin(np.abs(positions - test_point))
            
            # Extract local features (window around point)
            window_start = max(0, nearest_idx - 2)
            window_end = min(len(segments), nearest_idx + 3)
            
            window_embeddings = embeddings[window_start:window_end]
            
            # Features (what's happening at this point)
            if len(window_embeddings) >= 2:
                # Local semantic density
                centroid = window_embeddings.mean(axis=0)
                distances = np.linalg.norm(window_embeddings - centroid, axis=1)
                local_density = 1.0 / (1.0 + distances.mean())
                
                # Local variation
                local_variation = distances.std()
                
                # Direction (toward what?)
                if nearest_idx > 0 and nearest_idx < len(embeddings) - 1:
                    direction_from_prev = embeddings[nearest_idx] - embeddings[nearest_idx - 1]
                    direction_to_next = embeddings[nearest_idx + 1] - embeddings[nearest_idx]
                    
                    # Momentum (continuing same direction?)
                    momentum = np.dot(direction_from_prev, direction_to_next) / (
                        np.linalg.norm(direction_from_prev) * np.linalg.norm(direction_to_next) + 1e-8
                    )
                else:
                    momentum = 0.0
                
                # Combine into feature vector
                features = np.array([
                    local_density,
                    local_variation,
                    momentum,
                    test_point  # Position itself
                ])
                
                # Add first few dimensions of local embedding
                features = np.concatenate([features, centroid[:10]])
                
                features_at_points[test_point] = features
        
        return features_at_points


def run_temporal_isomorphism_validation(output_dir: str = 'results/validation'):
    """
    Run complete temporal isomorphism validation across 10+ domain pairs.
    
    Tests:
    - NBA vs Novel
    - UFC vs Short Story  
    - Golf vs Epic Novel
    - Tennis vs Novella
    - Film vs TV Episode
    - Pop Song vs Haiku
    - Symphony vs Novel
    - And more...
    """
    validator = TemporalIsomorphismValidator()
    
    # Define domain pairs to test
    # In practice, load actual narratives from data
    domain_pairs = [
        # Placeholder structure - actual implementation loads real data
        ('NBA_vs_Novel', [], []),  # Load from data
        ('UFC_vs_ShortStory', [], []),
        ('Film_vs_TVEpisode', [], []),
    ]
    
    output_path = Path(output_dir) / 'temporal_isomorphism_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = validator.validate_isomorphism(
        domain_pairs,
        output_file=str(output_path)
    )
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print(f"Validated: {results['n_validated']}/{results['n_pairs_tested']} pairs")
    print(f"Conclusion: {results['conclusion']}")
    print("="*80 + "\n")
    
    return results


if __name__ == '__main__':
    run_temporal_isomorphism_validation()

