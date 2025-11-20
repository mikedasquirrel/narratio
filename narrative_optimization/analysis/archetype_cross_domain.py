"""
Cross-Domain Archetype Analysis

Compares archetype distributions across all domains.
Validates universal patterns vs domain-specific variations.

Author: Narrative Optimization Framework
Date: November 13, 2025
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Import archetype transformers
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from transformers.archetypes import (
    HeroJourneyTransformer,
    CharacterArchetypeTransformer,
    PlotArchetypeTransformer,
    StructuralBeatTransformer,
    ThematicArchetypeTransformer
)
from config.domain_archetypes import DOMAIN_ARCHETYPES


class ArchetypeCrossDomainAnalyzer:
    """
    Analyze and compare archetype patterns across multiple domains.
    
    Key analyses:
    1. Universal vs domain-specific patterns
    2. Archetype importance by domain
    3. Theory validation across domains
    4. θ/λ clustering by mythos
    5. Temporal evolution of patterns
    """
    
    def __init__(self):
        self.transformers = {
            'hero_journey': HeroJourneyTransformer(),
            'character': CharacterArchetypeTransformer(),
            'plot': PlotArchetypeTransformer(),
            'structural': StructuralBeatTransformer(),
            'thematic': ThematicArchetypeTransformer()
        }
        
        self.domain_data = {}  # Will store {domain: {texts, outcomes, features}}
        self.learned_weights = {}  # {domain: {transformer: weights}}
        self.comparisons = {}
    
    def load_domain_data(self, domain_name: str, texts: List[str], 
                        outcomes: np.ndarray) -> None:
        """
        Load data for a domain.
        
        Args:
            domain_name: Name of domain (e.g., 'mythology', 'film_extended')
            texts: List of narrative texts
            outcomes: Success outcomes
        """
        self.domain_data[domain_name] = {
            'texts': texts,
            'outcomes': outcomes,
            'features': {},
            'learned_weights': {}
        }
        
        # Extract features with all transformers
        for transformer_name, transformer in self.transformers.items():
            transformer.fit(texts)
            features = transformer.transform(texts)
            self.domain_data[domain_name]['features'][transformer_name] = features
            
            # Learn empirical weights
            try:
                weights = transformer.learn_weights_from_data(texts, outcomes)
                self.domain_data[domain_name]['learned_weights'][transformer_name] = weights
            except:
                pass  # Some transformers may not support weight learning yet
    
    def compare_journey_patterns(self) -> Dict:
        """
        Compare Hero's Journey patterns across all loaded domains.
        
        Tests Campbell's universality hypothesis.
        """
        if not self.domain_data:
            raise ValueError("No domain data loaded")
        
        comparison = {
            'domains': {},
            'universal_stages': [],
            'domain_specific_stages': [],
            'campbell_validation_by_domain': {}
        }
        
        # Analyze each domain
        for domain_name, data in self.domain_data.items():
            if 'hero_journey' not in data['learned_weights']:
                continue
            
            weights = data['learned_weights']['hero_journey']
            
            # Get archetype config for this domain
            domain_config = DOMAIN_ARCHETYPES.get(domain_name, {})
            expected_journey = domain_config.get('classical_theory_expectations', {}).get(
                'campbell_journey_completion', 0.70
            )
            
            # Calculate actual journey completion
            features = data['features']['hero_journey']
            journey_features = features[:, :17]  # First 17 are Campbell stages
            actual_completion = np.mean(journey_features[journey_features > 0.3])
            
            comparison['domains'][domain_name] = {
                'expected_completion': expected_journey,
                'actual_completion': actual_completion,
                'learned_weights': weights,
                'meets_expectation': abs(actual_completion - expected_journey) < 0.15
            }
        
        # Find universal stages (high weight across ALL domains)
        if len(comparison['domains']) >= 2:
            stage_importances = {}  # {stage_name: [weights across domains]}
            
            for domain_name, domain_info in comparison['domains'].items():
                for stage, weight in domain_info['learned_weights'].items():
                    if stage not in stage_importances:
                        stage_importances[stage] = []
                    stage_importances[stage].append(weight)
            
            # Universal = consistently high across domains
            for stage, weights in stage_importances.items():
                mean_weight = np.mean(weights)
                std_weight = np.std(weights)
                
                if mean_weight > 0.70 and std_weight < 0.15:
                    comparison['universal_stages'].append({
                        'stage': stage,
                        'mean_importance': mean_weight,
                        'consistency': 1 - std_weight
                    })
                elif std_weight > 0.30:
                    comparison['domain_specific_stages'].append({
                        'stage': stage,
                        'mean_importance': mean_weight,
                        'variability': std_weight
                    })
        
        return comparison
    
    def compare_archetype_clarity(self) -> Dict:
        """
        Compare archetype clarity across domains.
        
        Tests: Mythology > Literature > Film > Music (expected hierarchy)
        """
        comparison = {
            'domains': {},
            'ranking': [],
            'hypothesis_test': {}
        }
        
        for domain_name, data in self.domain_data.items():
            if 'character' not in data['features']:
                continue
            
            features = data['features']['character']
            
            # Find archetype clarity features (indices 12-14 typically)
            # In production, would use feature names
            clarity_scores = features[:, 12:15] if features.shape[1] > 15 else features[:, 0:3]
            mean_clarity = np.mean(clarity_scores)
            
            comparison['domains'][domain_name] = {
                'mean_clarity': mean_clarity,
                'pi': DOMAIN_ARCHETYPES.get(domain_name, {}).get('pi', 0.5)
            }
        
        # Rank by clarity
        comparison['ranking'] = sorted(
            comparison['domains'].items(),
            key=lambda x: x[1]['mean_clarity'],
            reverse=True
        )
        
        # Test hypothesis: Higher π → Higher archetype clarity
        if len(comparison['domains']) >= 3:
            pis = [d['pi'] for d in comparison['domains'].values()]
            clarities = [d['mean_clarity'] for d in comparison['domains'].values()]
            
            corr, pval = pearsonr(pis, clarities)
            comparison['hypothesis_test'] = {
                'correlation_pi_clarity': corr,
                'p_value': pval,
                'hypothesis_validated': corr > 0.60 and pval < 0.05
            }
        
        return comparison
    
    def test_frye_theta_lambda_clustering(self) -> Dict:
        """
        Test if Frye's four mythoi cluster in θ/λ phase space.
        
        Expected clusters:
        - Comedy: θ≈0.30, λ≈0.50
        - Romance: θ≈0.20, λ≈0.30
        - Tragedy: θ≈0.55, λ≈0.75
        - Irony: θ≈0.85, λ≈0.50 (variable)
        """
        if len(self.domain_data) < 4:
            return {'error': 'Need at least 4 domains for clustering test'}
        
        # Extract θ/λ coordinates for each domain
        coordinates = []
        domain_names = []
        mythoi_scores = []
        
        for domain_name, data in self.domain_data.items():
            if 'thematic' not in data['features']:
                continue
            
            # Get θ and λ from domain config
            config = DOMAIN_ARCHETYPES.get(domain_name, {})
            theta = np.mean(config.get('theta_range', (0.5, 0.5)))
            lambda_val = np.mean(config.get('lambda_range', (0.5, 0.5)))
            
            # Get mythos scores from thematic transformer
            thematic_features = data['features']['thematic']
            # First 4 features are Frye's mythoi
            mythos = thematic_features[:, :4].mean(axis=0)
            
            coordinates.append([theta, lambda_val])
            domain_names.append(domain_name)
            mythoi_scores.append(mythos)
        
        if len(coordinates) < 4:
            return {'error': 'Insufficient data for clustering'}
        
        # K-means clustering (4 clusters for 4 mythoi)
        coordinates_array = np.array(coordinates)
        kmeans = KMeans(n_clusters=4, random_state=42)
        clusters = kmeans.fit_predict(coordinates_array)
        
        # Assign clusters to mythoi
        cluster_mythoi = {}
        for i, cluster_id in enumerate(clusters):
            dominant_mythos = np.argmax(mythoi_scores[i])
            mythos_names = ['comedy', 'romance', 'tragedy', 'irony']
            
            if cluster_id not in cluster_mythoi:
                cluster_mythoi[cluster_id] = []
            cluster_mythoi[cluster_id].append(mythos_names[dominant_mythos])
        
        return {
            'coordinates': coordinates,
            'domain_names': domain_names,
            'clusters': clusters.tolist(),
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'cluster_mythoi': cluster_mythoi,
            'silhouette_score': self._calculate_silhouette(coordinates_array, clusters)
        }
    
    def _calculate_silhouette(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette score for clustering quality."""
        from sklearn.metrics import silhouette_score
        try:
            return silhouette_score(X, labels)
        except:
            return 0.0
    
    def compare_theoretical_vs_empirical_all_domains(self) -> Dict:
        """
        Compare theoretical weights to empirical weights across all domains.
        
        Discovers where classical theories hold vs fail.
        """
        comparison = {
            'by_domain': {},
            'by_transformer': {},
            'universal_validations': [],
            'domain_specific_patterns': []
        }
        
        # Collect all transformer comparisons
        for domain_name, data in self.domain_data.items():
            comparison['by_domain'][domain_name] = {}
            
            for transformer_name, transformer in self.transformers.items():
                if transformer_name not in data['learned_weights']:
                    continue
                
                # Get theoretical vs empirical comparison
                try:
                    comp = transformer.compare_theoretical_vs_empirical()
                    comparison['by_domain'][domain_name][transformer_name] = comp
                    
                    # Track by transformer
                    if transformer_name not in comparison['by_transformer']:
                        comparison['by_transformer'][transformer_name] = {}
                    comparison['by_transformer'][transformer_name][domain_name] = comp
                except:
                    pass
        
        return comparison
    
    def analyze_temporal_evolution(self, domain_name: str, 
                                   temporal_metadata: List[int]) -> Dict:
        """
        Analyze how archetype patterns evolved over time within a domain.
        
        Args:
            domain_name: Domain to analyze
            temporal_metadata: Year/period for each sample
            
        Returns:
            Evolution analysis
        """
        if domain_name not in self.domain_data:
            raise ValueError(f"Domain {domain_name} not loaded")
        
        data = self.domain_data[domain_name]
        
        # Group by period
        periods = np.array(temporal_metadata)
        unique_periods = sorted(set(periods))
        
        evolution = {
            'periods': unique_periods,
            'journey_completion_over_time': [],
            'archetype_clarity_over_time': [],
            'theta_over_time': [],
            'patterns': {}
        }
        
        for period in unique_periods:
            period_mask = periods == period
            
            # Hero's Journey completion
            if 'hero_journey' in data['features']:
                journey_features = data['features']['hero_journey'][period_mask]
                completion = np.mean(journey_features[:, 2])  # journey_completion_mean
                evolution['journey_completion_over_time'].append(completion)
            
            # Archetype clarity
            if 'character' in data['features']:
                char_features = data['features']['character'][period_mask]
                clarity = np.mean(char_features[:, 12])  # jung_archetype_clarity
                evolution['archetype_clarity_over_time'].append(clarity)
            
            # θ estimation
            if 'thematic' in data['features']:
                thematic_features = data['features']['thematic'][period_mask]
                theta = np.mean(thematic_features[:, -2])  # estimated_theta
                evolution['theta_over_time'].append(theta)
        
        # Calculate trends
        if len(unique_periods) >= 3:
            # Linear trend in journey completion
            x = np.arange(len(unique_periods))
            if evolution['journey_completion_over_time']:
                y = evolution['journey_completion_over_time']
                slope, _ = np.polyfit(x, y, 1)
                evolution['patterns']['journey_trend'] = {
                    'slope': slope,
                    'direction': 'increasing' if slope > 0.01 else 'decreasing' if slope < -0.01 else 'stable'
                }
            
            # Trend in archetype clarity
            if evolution['archetype_clarity_over_time']:
                y = evolution['archetype_clarity_over_time']
                slope, _ = np.polyfit(x, y, 1)
                evolution['patterns']['clarity_trend'] = {
                    'slope': slope,
                    'direction': 'increasing' if slope > 0.01 else 'decreasing' if slope < -0.01 else 'stable'
                }
            
            # Trend in θ (awareness)
            if evolution['theta_over_time']:
                y = evolution['theta_over_time']
                slope, _ = np.polyfit(x, y, 1)
                evolution['patterns']['theta_trend'] = {
                    'slope': slope,
                    'direction': 'increasing' if slope > 0.01 else 'decreasing' if slope < -0.01 else 'stable'
                }
        
        return evolution
    
    def test_cross_domain_transfer(self, source_domain: str, 
                                   target_domain: str) -> Dict:
        """
        Test if patterns learned from one domain transfer to another.
        
        Args:
            source_domain: Domain to train on
            target_domain: Domain to test on
            
        Returns:
            Transfer analysis with R² scores
        """
        if source_domain not in self.domain_data or target_domain not in self.domain_data:
            raise ValueError("Both domains must be loaded")
        
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score
        
        source = self.domain_data[source_domain]
        target = self.domain_data[target_domain]
        
        results = {}
        
        for transformer_name in self.transformers.keys():
            if transformer_name not in source['features'] or transformer_name not in target['features']:
                continue
            
            # Train on source
            X_train = source['features'][transformer_name]
            y_train = source['outcomes']
            
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            
            # Test on target
            X_test = target['features'][transformer_name]
            y_test = target['outcomes']
            
            predictions = model.predict(X_test)
            r2 = r2_score(y_test, predictions)
            
            results[transformer_name] = {
                'r2_score': r2,
                'transfers_well': r2 > 0.40,
                'model_coefficients': model.coef_.tolist()
            }
        
        # Overall transfer quality
        r2_scores = [r['r2_score'] for r in results.values()]
        results['overall'] = {
            'mean_r2': np.mean(r2_scores),
            'transfer_success': np.mean(r2_scores) > 0.40,
            'best_transferring': max(results.items(), key=lambda x: x[1]['r2_score'])[0]
        }
        
        return results
    
    def identify_universal_patterns(self) -> Dict:
        """
        Identify archetype patterns that are universal across all domains.
        
        Returns:
            Dictionary of universal vs domain-specific patterns
        """
        if len(self.domain_data) < 3:
            raise ValueError("Need at least 3 domains for universal pattern detection")
        
        universal = {
            'universal_patterns': [],
            'domain_specific_patterns': [],
            'analysis': {}
        }
        
        # Collect learned weights for Hero's Journey across domains
        journey_weights_by_domain = {}
        for domain_name, data in self.domain_data.items():
            if 'hero_journey' in data['learned_weights']:
                journey_weights_by_domain[domain_name] = data['learned_weights']['hero_journey']
        
        if len(journey_weights_by_domain) >= 3:
            # Find stages with consistent importance
            all_stages = set()
            for weights in journey_weights_by_domain.values():
                all_stages.update(weights.keys())
            
            for stage in all_stages:
                stage_weights = []
                for domain_weights in journey_weights_by_domain.values():
                    if stage in domain_weights:
                        stage_weights.append(domain_weights[stage])
                
                if len(stage_weights) >= 3:
                    mean_weight = np.mean(stage_weights)
                    std_weight = np.std(stage_weights)
                    
                    if mean_weight > 0.65 and std_weight < 0.15:
                        # Universal pattern
                        universal['universal_patterns'].append({
                            'pattern': stage,
                            'mean_importance': mean_weight,
                            'consistency': 1 - std_weight,
                            'pattern_type': 'hero_journey_stage'
                        })
                    elif std_weight > 0.30:
                        # Domain-specific
                        universal['domain_specific_patterns'].append({
                            'pattern': stage,
                            'mean_importance': mean_weight,
                            'variability': std_weight,
                            'pattern_type': 'hero_journey_stage',
                            'domains_high': [d for d, w in journey_weights_by_domain.items() 
                                           if stage in w and w[stage] > mean_weight + std_weight],
                            'domains_low': [d for d, w in journey_weights_by_domain.items() 
                                          if stage in w and w[stage] < mean_weight - std_weight]
                        })
        
        # Sort by importance
        universal['universal_patterns'].sort(key=lambda x: x['mean_importance'], reverse=True)
        universal['domain_specific_patterns'].sort(key=lambda x: x['variability'], reverse=True)
        
        return universal
    
    def generate_archetype_comparison_matrix(self) -> pd.DataFrame:
        """
        Generate comparison matrix of archetype features across domains.
        
        Returns:
            DataFrame with domains as rows, archetype features as columns
        """
        data_rows = []
        
        for domain_name, data in self.domain_data.items():
            row = {'domain': domain_name}
            
            # Add π/λ/θ
            config = DOMAIN_ARCHETYPES.get(domain_name, {})
            row['pi'] = config.get('pi', 0.5)
            row['theta'] = np.mean(config.get('theta_range', (0.5, 0.5)))
            row['lambda'] = np.mean(config.get('lambda_range', (0.5, 0.5)))
            
            # Add archetype metrics
            if 'hero_journey' in data['features']:
                features = data['features']['hero_journey']
                row['journey_completion'] = np.mean(features[:, 2])  # Overall completion
            
            if 'character' in data['features']:
                features = data['features']['character']
                row['archetype_clarity'] = np.mean(features[:, 12])  # Jung clarity
            
            if 'plot' in data['features']:
                features = data['features']['plot']
                row['plot_purity'] = np.mean(features[:, 7])  # Plot purity
            
            if 'structural' in data['features']:
                features = data['features']['structural']
                row['beat_adherence'] = np.mean(features[:, 15])  # Beat adherence
            
            if 'thematic' in data['features']:
                features = data['features']['thematic']
                row['mythos_clarity'] = np.mean(features[:, 4])  # Mythos purity
            
            data_rows.append(row)
        
        df = pd.DataFrame(data_rows)
        return df
    
    def visualize_archetype_space(self, output_path: Optional[str] = None) -> None:
        """
        Create visualization of all domains in archetype space.
        
        Uses PCA to reduce ~225 archetype features to 3D for visualization.
        """
        # Collect all archetype features
        all_features = []
        all_labels = []
        
        for domain_name, data in self.domain_data.items():
            # Concatenate all transformer features
            domain_features = []
            for transformer_name in ['hero_journey', 'character', 'plot', 'structural', 'thematic']:
                if transformer_name in data['features']:
                    domain_features.append(data['features'][transformer_name])
            
            if domain_features:
                # Average across samples in domain
                combined = np.concatenate(domain_features, axis=1)
                mean_features = np.mean(combined, axis=0)
                all_features.append(mean_features)
                all_labels.append(domain_name)
        
        if len(all_features) < 3:
            print("Need at least 3 domains for visualization")
            return
        
        # PCA to 3D
        features_array = np.array(all_features)
        pca = PCA(n_components=3)
        features_3d = pca.fit_transform(features_array)
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot points
        ax.scatter(features_3d[:, 0], features_3d[:, 1], features_3d[:, 2], 
                  s=100, c=range(len(all_labels)), cmap='tab10')
        
        # Label points
        for i, label in enumerate(all_labels):
            ax.text(features_3d[i, 0], features_3d[i, 1], features_3d[i, 2], 
                   label, fontsize=9)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
        ax.set_title('Domains in Archetype Space (PCA)')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def generate_report(self, output_path: str) -> None:
        """
        Generate comprehensive cross-domain analysis report.
        """
        report = {
            'summary': {
                'domains_analyzed': len(self.domain_data),
                'total_samples': sum([len(d['texts']) for d in self.domain_data.values()]),
                'transformers_used': len(self.transformers)
            },
            'journey_comparison': self.compare_journey_patterns(),
            'clarity_comparison': self.compare_archetype_clarity(),
            'theta_lambda_clustering': self.test_frye_theta_lambda_clustering(),
            'universal_patterns': self.identify_universal_patterns(),
            'comparison_matrix': self.generate_archetype_comparison_matrix().to_dict()
        }
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Cross-domain analysis report saved to: {output_path}")
        
        return report


# Convenience functions
def quick_cross_domain_analysis(domain_datasets: Dict[str, Tuple[List[str], np.ndarray]]) -> Dict:
    """
    Quick cross-domain analysis.
    
    Args:
        domain_datasets: {domain_name: (texts, outcomes)}
        
    Returns:
        Analysis results
    """
    analyzer = ArchetypeCrossDomainAnalyzer()
    
    for domain_name, (texts, outcomes) in domain_datasets.items():
        analyzer.load_domain_data(domain_name, texts, outcomes)
    
    return {
        'journey_patterns': analyzer.compare_journey_patterns(),
        'archetype_clarity': analyzer.compare_archetype_clarity(),
        'universal_patterns': analyzer.identify_universal_patterns(),
        'matrix': analyzer.generate_archetype_comparison_matrix()
    }

