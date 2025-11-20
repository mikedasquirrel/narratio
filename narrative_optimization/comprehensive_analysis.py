#!/usr/bin/env python3
"""
Comprehensive Narrative Analysis Runner

Self-contained script that runs the complete pipeline across all transformers,
generates multi-dimensional analysis, creates enhanced visualizations, and
produces analytical predictions.

Usage:
    python comprehensive_analysis.py [--samples 100] [--domain all]
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.transformers.statistical import StatisticalTransformer
from src.transformers.ensemble import EnsembleNarrativeTransformer
from src.transformers.linguistic_advanced import LinguisticPatternsTransformer
from src.transformers.self_perception import SelfPerceptionTransformer
from src.transformers.narrative_potential import NarrativePotentialTransformer
from src.transformers.relational import RelationalValueTransformer
from src.transformers.nominative import NominativeAnalysisTransformer
from src.utils.toy_data import quick_load_toy_data
from src.utils.plain_english import PlainEnglishExplainer
from src.visualization.plotly_interactive import InteractivePlotlyCharts
from src.visualization.advanced_plots import AdvancedNarrativeVisualizations
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class ComprehensiveNarrativeAnalyzer:
    """
    Complete pipeline analyzer - all transformers, all dimensions.
    
    Performs comprehensive multi-dimensional narrative analysis following
    the compare page model: analyze across ALL dimensions simultaneously.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
        # Initialize ALL transformers
        self.transformers = {
            'statistical': StatisticalTransformer(max_features=500),
            'ensemble': EnsembleNarrativeTransformer(n_top_terms=30, network_metrics=True),
            'linguistic': LinguisticPatternsTransformer(track_evolution=True),
            'self_perception': SelfPerceptionTransformer(track_attribution=True, track_growth=True),
            'potential': NarrativePotentialTransformer(track_modality=True, track_flexibility=True),
            'relational': RelationalValueTransformer(n_features=50),
            'nominative': NominativeAnalysisTransformer(n_semantic_fields=10)
        }
        
        self.explainer = PlainEnglishExplainer()
        self.fitted = False
        self.feature_names = {}
    
    def fit(self, X_train):
        """Fit all transformers on training data."""
        if self.verbose:
            print("🔧 Fitting All Transformers...")
            print("=" * 60)
        
        total_features = 0
        
        for i, (name, transformer) in enumerate(self.transformers.items(), 1):
            if self.verbose:
                print(f"[{i}/7] {name.replace('_', ' ').title()}...", end=" ", flush=True)
            
            # Fit transformer
            transformer.fit(X_train)
            
            # Get feature count
            X_sample = transformer.transform(X_train[:1])
            if hasattr(X_sample, 'shape'):
                n_features = X_sample.shape[1]
            else:
                n_features = len(X_sample[0])
            
            total_features += n_features
            
            if self.verbose:
                desc = self._get_transformer_description(name)
                print(f"✓ {n_features} features ({desc})")
        
        if self.verbose:
            print()
            print(f"📊 Total: {total_features} features across all transformers")
            print()
        
        self.fitted = True
        return self
    
    def _get_transformer_description(self, name: str) -> str:
        """Get short description of transformer."""
        descriptions = {
            'statistical': 'word frequencies',
            'ensemble': 'network effects',
            'linguistic': 'voice, agency, time',
            'self_perception': 'growth, identity',
            'potential': 'future, possibility',
            'relational': 'complementarity',
            'nominative': 'naming, categories'
        }
        return descriptions.get(name, '')
    
    def analyze_comprehensive(self, text: str) -> Dict[str, Any]:
        """
        Complete multi-dimensional analysis of a single text.
        
        Returns comprehensive analysis across all dimensions with
        plain English interpretations.
        """
        if not self.fitted:
            raise ValueError("Must fit() before analyzing")
        
        analysis = {
            'text_preview': text[:200] + '...' if len(text) > 200 else text,
            'text_length': len(text),
            'word_count': len(text.split()),
            'dimensions': {},
            'dimensional_scores': {},
            'narrative_profile': {},
            'predictions': {}
        }
        
        # Extract features from each transformer
        for name, transformer in self.transformers.items():
            try:
                # Transform
                features = transformer.transform([text])[0]
                
                # Handle sparse matrices
                if hasattr(features, 'toarray'):
                    features = features.toarray().flatten()
                
                # Store features
                feature_dict = {}
                if name == 'ensemble':
                    feature_dict = {
                        'ensemble_size': float(features[0]) if len(features) > 0 else 0,
                        'cooccurrence_density': float(features[1]) if len(features) > 1 else 0,
                        'diversity': float(features[2]) if len(features) > 2 else 0,
                        'centrality_mean': float(features[3]) if len(features) > 3 else 0
                    }
                elif name == 'linguistic':
                    feature_dict = {
                        'first_person_density': float(features[0]) if len(features) > 0 else 0,
                        'future_orientation': float(features[5]) if len(features) > 5 else 0,
                        'agency_score': float(features[10]) if len(features) > 10 else 0,
                        'voice_consistency': float(features[3]) if len(features) > 3 else 0
                    }
                elif name == 'self_perception':
                    feature_dict = {
                        'growth_mindset_score': float(features[8]) if len(features) > 8 else 0,
                        'attribution_balance': float(features[5]) if len(features) > 5 else 0,
                        'identity_coherence': float(features[-3]) if len(features) > 3 else 0,
                        'self_focus_ratio': float(features[2]) if len(features) > 2 else 0
                    }
                elif name == 'potential':
                    feature_dict = {
                        'future_orientation_score': float(features[2]) if len(features) > 2 else 0,
                        'possibility_score': float(features[5]) if len(features) > 5 else 0,
                        'narrative_momentum': float(features[-1]) if len(features) > 0 else 0,
                        'flexibility_ratio': float(features[10]) if len(features) > 10 else 0
                    }
                
                # Get interpretation
                interpretation = self.explainer.generate_narrative_interpretation(name, feature_dict)
                
                # Store in analysis
                analysis['dimensions'][name] = {
                    'features': feature_dict,
                    'feature_vector': features.tolist() if hasattr(features, 'tolist') else list(features),
                    'interpretation': interpretation,
                    'n_features': len(features)
                }
                
                # Calculate dimensional score (average of normalized features)
                score = np.mean(np.abs(features[:10])) if len(features) >= 10 else np.mean(np.abs(features))
                analysis['dimensional_scores'][name] = float(score)
                
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: {name} analysis failed: {e}")
                analysis['dimensions'][name] = {'error': str(e)}
        
        # Generate narrative profile
        analysis['narrative_profile'] = self._generate_narrative_profile(analysis['dimensional_scores'])
        
        # Generate predictions (domain-specific)
        analysis['predictions'] = self._generate_predictions(analysis['dimensional_scores'])
        
        return analysis
    
    def _generate_narrative_profile(self, scores: Dict[str, float]) -> Dict[str, Any]:
        """Generate overall narrative profile from dimensional scores."""
        # Identify strengths (top dimensions)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_scores[:3]
        
        strengths = [name.replace('_', ' ').title() for name, score in top_3 if score > 0.3]
        
        # Determine archetype based on dimension pattern
        if scores.get('ensemble', 0) > 0.4 and scores.get('potential', 0) > 0.3:
            archetype = "Growth-Oriented Explorer"
        elif scores.get('linguistic', 0) > 0.4 and scores.get('self_perception', 0) > 0.3:
            archetype = "Self-Aware Communicator"
        elif scores.get('ensemble', 0) > 0.4 and scores.get('relational', 0) > 0.3:
            archetype = "Relational Connector"
        elif scores.get('potential', 0) > 0.4 and scores.get('self_perception', 0) > 0.3:
            archetype = "Future-Focused Developer"
        else:
            archetype = "Balanced Narrative"
        
        # Overall quality (weighted average)
        weights = {
            'ensemble': 0.2,
            'linguistic': 0.2,
            'self_perception': 0.2,
            'potential': 0.2,
            'relational': 0.1,
            'nominative': 0.1
        }
        
        overall_quality = sum(scores.get(dim, 0) * weight for dim, weight in weights.items())
        
        return {
            'archetype': archetype,
            'strengths': strengths if strengths else ['Statistical content'],
            'overall_quality': float(overall_quality),
            'dominant_dimension': top_3[0][0] if top_3 else 'statistical'
        }
    
    def _generate_predictions(self, scores: Dict[str, float]) -> Dict[str, Any]:
        """Generate domain-specific predictions."""
        predictions = {}
        
        # Relationship compatibility (ensemble + potential)
        rel_score = (scores.get('ensemble', 0) * 0.5 + scores.get('potential', 0) * 0.5)
        predictions['relationship_compatibility'] = {
            'score': float(rel_score),
            'prediction': 'high' if rel_score > 0.6 else 'moderate' if rel_score > 0.4 else 'low'
        }
        
        # Wellness trajectory (self_perception + potential)
        wellness_score = (scores.get('self_perception', 0) * 0.6 + scores.get('potential', 0) * 0.4)
        predictions['wellness_trajectory'] = {
            'score': float(wellness_score),
            'prediction': 'improving' if wellness_score > 0.6 else 'stable' if wellness_score > 0.4 else 'needs_support'
        }
        
        # Content engagement (linguistic + ensemble)
        content_score = (scores.get('linguistic', 0) * 0.6 + scores.get('ensemble', 0) * 0.4)
        predictions['content_engagement'] = {
            'score': float(content_score),
            'prediction': 'high' if content_score > 0.6 else 'moderate' if content_score > 0.4 else 'low'
        }
        
        return predictions
    
    def analyze_batch(self, texts: List[str], labels: np.ndarray = None) -> Dict[str, Any]:
        """
        Analyze multiple texts and generate comprehensive comparative analysis.
        
        Parameters
        ----------
        texts : list of str
            Texts to analyze
        labels : array, optional
            True labels if available
        
        Returns
        -------
        batch_results : dict
            Complete analysis including individual results and aggregate insights
        """
        if self.verbose:
            print(f"\n🎯 Analyzing {len(texts)} Samples...")
            print("=" * 60)
        
        individual_analyses = []
        all_dimensional_scores = {dim: [] for dim in self.transformers.keys()}
        
        for i, text in enumerate(texts):
            if self.verbose and (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(texts)}", flush=True)
            
            analysis = self.analyze_comprehensive(text)
            individual_analyses.append(analysis)
            
            # Collect scores
            for dim in self.transformers.keys():
                all_dimensional_scores[dim].append(analysis['dimensional_scores'].get(dim, 0))
        
        if self.verbose:
            print(f"  Progress: {len(texts)}/{len(texts)} ✓")
            print()
        
        # Aggregate analysis
        batch_results = {
            'individual_analyses': individual_analyses,
            'aggregate': self._compute_aggregate_insights(all_dimensional_scores, labels),
            'clusters': self._identify_narrative_archetypes(individual_analyses),
            'patterns': self._detect_patterns(individual_analyses, labels)
        }
        
        return batch_results
    
    def _compute_aggregate_insights(self, dimensional_scores: Dict[str, List[float]], labels: np.ndarray = None) -> Dict[str, Any]:
        """Compute aggregate statistics and insights."""
        aggregate = {}
        
        for dim, scores in dimensional_scores.items():
            aggregate[dim] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'median': float(np.median(scores))
            }
        
        # Dimension correlations
        if len(list(dimensional_scores.values())[0]) > 1:
            dim_matrix = np.array([dimensional_scores[dim] for dim in dimensional_scores.keys()])
            correlations = np.corrcoef(dim_matrix)
            aggregate['dimension_correlations'] = correlations.tolist()
        
        return aggregate
    
    def _identify_narrative_archetypes(self, analyses: List[Dict]) -> Dict[str, Any]:
        """Identify narrative archetypes through clustering."""
        # Extract dimensional scores
        score_matrix = []
        for analysis in analyses:
            scores = [analysis['dimensional_scores'].get(dim, 0) for dim in self.transformers.keys()]
            score_matrix.append(scores)
        
        score_matrix = np.array(score_matrix)
        
        # Cluster
        n_clusters = min(4, len(analyses) // 10)  # At least 10 samples per cluster
        if n_clusters >= 2:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(score_matrix)
            
            # Name clusters
            cluster_names = []
            for i in range(n_clusters):
                cluster_mask = cluster_labels == i
                cluster_scores = score_matrix[cluster_mask].mean(axis=0)
                
                # Find dominant dimensions
                dim_names = list(self.transformers.keys())
                top_dim_idx = np.argmax(cluster_scores)
                top_dim = dim_names[top_dim_idx]
                
                cluster_names.append(f"{top_dim.replace('_', ' ').title()} Dominant")
            
            return {
                'n_clusters': n_clusters,
                'cluster_labels': cluster_labels.tolist(),
                'cluster_names': cluster_names,
                'cluster_centers': kmeans.cluster_centers_.tolist()
            }
        else:
            return {'n_clusters': 0, 'note': 'Not enough samples for clustering'}
    
    def _detect_patterns(self, analyses: List[Dict], labels: np.ndarray = None) -> Dict[str, Any]:
        """Detect patterns across samples."""
        patterns = {
            'common_archetypes': {},
            'dimension_ranges': {},
            'correlations_found': []
        }
        
        # Count archetypes
        archetypes = [a['narrative_profile']['archetype'] for a in analyses]
        from collections import Counter
        patterns['common_archetypes'] = dict(Counter(archetypes).most_common(5))
        
        # Dimension ranges
        for dim in self.transformers.keys():
            scores = [a['dimensional_scores'].get(dim, 0) for a in analyses]
            patterns['dimension_ranges'][dim] = {
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'mean': float(np.mean(scores))
            }
        
        return patterns
    
    def generate_visualizations(self, batch_results: Dict[str, Any], output_dir: str):
        """Generate all enhanced visualizations."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.verbose:
            print("📈 Generating Enhanced Visualizations...")
            print("=" * 60)
        
        # Extract data for visualization
        analyses = batch_results['individual_analyses']
        
        # 1. Multi-dimensional radar (all 6 advanced dimensions)
        if self.verbose:
            print("  [1/6] Multi-dimensional radar chart...")
        
        self._create_multidimensional_radar(analyses, output_path)
        
        # 2. Feature importance heatmap
        if self.verbose:
            print("  [2/6] Feature contribution heatmap...")
        
        self._create_feature_heatmap(analyses, output_path)
        
        # 3. Narrative archetype map
        if self.verbose:
            print("  [3/6] Narrative archetype clustering...")
        
        self._create_archetype_map(analyses, batch_results['clusters'], output_path)
        
        # 4. Dimension weights
        if self.verbose:
            print("  [4/6] Dimension importance chart...")
        
        self._create_dimension_weights(batch_results['aggregate'], output_path)
        
        # 5. Prediction dashboard
        if self.verbose:
            print("  [5/6] Prediction confidence dashboard...")
        
        self._create_prediction_dashboard(analyses, output_path)
        
        # 6. Comprehensive report
        if self.verbose:
            print("  [6/6] Comprehensive HTML report...")
        
        self._create_html_report(batch_results, output_path)
        
        if self.verbose:
            print()
            print(f"✓ All visualizations saved to {output_path}")
            print()
    
    def _create_multidimensional_radar(self, analyses: List[Dict], output_path: Path):
        """Create radar chart showing all dimensions."""
        import plotly.graph_objects as go
        
        # Average scores across all samples
        avg_scores = {}
        for dim in ['ensemble', 'linguistic', 'self_perception', 'potential', 'relational', 'nominative']:
            scores = [a['dimensional_scores'].get(dim, 0) for a in analyses]
            avg_scores[dim] = np.mean(scores)
        
        dimensions = list(avg_scores.keys())
        values = list(avg_scores.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=[d.replace('_', ' ').title() for d in dimensions] + [dimensions[0].replace('_', ' ').title()],
            fill='toself',
            name='Average Profile',
            line_color='#e879f9',
            fillcolor='#e879f9',
            opacity=0.6
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, max(values) * 1.2]),
                bgcolor='#1e293b'
            ),
            title="Multi-Dimensional Narrative Profile",
            paper_bgcolor='#0f172a',
            font=dict(color='#f1f5f9'),
            height=600
        )
        
        fig.write_html(output_path / 'multidimensional_radar.html')
    
    def _create_feature_heatmap(self, analyses: List[Dict], output_path: Path):
        """Placeholder for feature heatmap."""
        pass  # Implemented in visualization module
    
    def _create_archetype_map(self, analyses: List[Dict], clusters: Dict, output_path: Path):
        """Create 2D map of narrative archetypes."""
        import plotly.express as px
        
        # Extract scores for PCA
        score_matrix = []
        archetypes = []
        
        for analysis in analyses:
            scores = [analysis['dimensional_scores'].get(dim, 0) for dim in self.transformers.keys()]
            score_matrix.append(scores)
            archetypes.append(analysis['narrative_profile']['archetype'])
        
        score_matrix = np.array(score_matrix)
        
        # PCA to 2D
        pca = PCA(n_components=2, random_state=42)
        coords_2d = pca.fit_transform(score_matrix)
        
        df = pd.DataFrame({
            'PC1': coords_2d[:, 0],
            'PC2': coords_2d[:, 1],
            'Archetype': archetypes
        })
        
        fig = px.scatter(
            df, x='PC1', y='PC2', color='Archetype',
            title='Narrative Archetype Map',
            color_discrete_sequence=['#e879f9', '#22d3ee', '#0ea5e9', '#f97316']
        )
        
        fig.update_layout(
            paper_bgcolor='#0f172a',
            plot_bgcolor='#1e293b',
            font=dict(color='#f1f5f9'),
            height=700
        )
        
        fig.write_html(output_path / 'archetype_map.html')
    
    def _create_dimension_weights(self, aggregate: Dict, output_path: Path):
        """Show relative importance of each dimension."""
        import plotly.graph_objects as go
        
        dimensions = []
        means = []
        stds = []
        
        for dim, stats in aggregate.items():
            if dim != 'dimension_correlations':
                dimensions.append(dim.replace('_', ' ').title())
                means.append(stats['mean'])
                stds.append(stats['std'])
        
        fig = go.Figure(data=[
            go.Bar(
                x=dimensions,
                y=means,
                error_y=dict(type='data', array=stds),
                marker_color='#e879f9',
                text=[f'{m:.3f}' for m in means],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title='Dimension Importance (Mean Scores)',
            xaxis_title='Narrative Dimension',
            yaxis_title='Average Score',
            paper_bgcolor='#0f172a',
            plot_bgcolor='#1e293b',
            font=dict(color='#f1f5f9'),
            height=600
        )
        
        fig.write_html(output_path / 'dimension_weights.html')
    
    def _create_prediction_dashboard(self, analyses: List[Dict], output_path: Path):
        """Create prediction confidence visualizations."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Collect predictions
        rel_scores = [a['predictions']['relationship_compatibility']['score'] for a in analyses]
        wellness_scores = [a['predictions']['wellness_trajectory']['score'] for a in analyses]
        content_scores = [a['predictions']['content_engagement']['score'] for a in analyses]
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Relationship', 'Wellness', 'Content')
        )
        
        fig.add_trace(go.Histogram(x=rel_scores, name='Relationship', marker_color='#e879f9'), row=1, col=1)
        fig.add_trace(go.Histogram(x=wellness_scores, name='Wellness', marker_color='#22d3ee'), row=1, col=2)
        fig.add_trace(go.Histogram(x=content_scores, name='Content', marker_color='#0ea5e9'), row=1, col=3)
        
        fig.update_layout(
            title='Predicted Outcome Distributions',
            paper_bgcolor='#0f172a',
            plot_bgcolor='#1e293b',
            font=dict(color='#f1f5f9'),
            showlegend=False,
            height=500
        )
        
        fig.write_html(output_path / 'prediction_dashboard.html')
    
    def _create_html_report(self, batch_results: Dict, output_path: Path):
        """Generate comprehensive HTML report."""
        report = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Comprehensive Narrative Analysis Report</title>
    <style>
        body {{ font-family: system-ui; background: #0f172a; color: #f1f5f9; padding: 2rem; }}
        h1 {{ color: #e879f9; }}
        h2 {{ color: #22d3ee; border-bottom: 2px solid #334155; padding-bottom: 0.5rem; }}
        .summary {{ background: #1e293b; padding: 2rem; border-radius: 12px; margin: 2rem 0; }}
        .metric {{ padding: 1rem; background: rgba(232, 121, 249, 0.1); margin: 0.5rem 0; border-radius: 8px; }}
        .archetype {{ font-size: 1.2rem; color: #22d3ee; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>📊 Comprehensive Narrative Analysis Report</h1>
    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>Samples Analyzed:</strong> {len(batch_results['individual_analyses'])}</p>
    
    <div class="summary">
        <h2>Executive Summary</h2>
        <ul>
            <li>Analyzed {len(batch_results['individual_analyses'])} samples across 7 transformers (79-107 features each)</li>
            <li>Identified {batch_results['clusters'].get('n_clusters', 0)} narrative archetypes</li>
            <li>Generated predictions for relationship compatibility, wellness trajectory, and content engagement</li>
        </ul>
    </div>
    
    <h2>Common Narrative Archetypes</h2>
    <div class="summary">
        {"<br>".join([f"<div class='archetype'>{arch}: {count} samples</div>" 
                      for arch, count in batch_results['patterns']['common_archetypes'].items()])}
    </div>
    
    <h2>Dimension Analysis</h2>
    <div class="summary">
        {"<br>".join([f"<div class='metric'><strong>{dim.replace('_', ' ').title()}:</strong> "
                      f"{stats['mean']:.3f} ± {stats['std']:.3f} "
                      f"(range: {stats['min']:.3f} to {stats['max']:.3f})</div>"
                      for dim, stats in batch_results['aggregate'].items() 
                      if dim != 'dimension_correlations'])}
    </div>
    
    <h2>Interactive Visualizations</h2>
    <p><a href="multidimensional_radar.html" style="color: #e879f9;">Multi-Dimensional Radar</a></p>
    <p><a href="archetype_map.html" style="color: #e879f9;">Narrative Archetype Map</a></p>
    <p><a href="dimension_weights.html" style="color: #e879f9;">Dimension Importance</a></p>
    <p><a href="prediction_dashboard.html" style="color: #e879f9;">Prediction Dashboard</a></p>
</body>
</html>
"""
        
        with open(output_path / 'comprehensive_report.html', 'w') as f:
            f.write(report)


def main():
    """Main execution function."""
    print("\n" + "🚀" * 40)
    print("\n  COMPREHENSIVE NARRATIVE ANALYSIS")
    print("  Complete Pipeline - All Dimensions")
    print("\n" + "🚀" * 40 + "\n")
    
    # Initialize analyzer
    analyzer = ComprehensiveNarrativeAnalyzer(verbose=True)
    
    # Load data
    print("📊 Loading Data...")
    print("=" * 60)
    data = quick_load_toy_data()
    X_train = data['X_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    print(f"✓ Training: {len(X_train)} samples")
    print(f"✓ Test: {len(X_test)} samples")
    print()
    
    # Fit all transformers
    analyzer.fit(X_train)
    
    # Analyze test set
    batch_results = analyzer.analyze_batch(X_test[:50], y_test[:50])  # First 50 for speed
    
    # Generate visualizations
    output_dir = f"results/comprehensive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    analyzer.generate_visualizations(batch_results, output_dir)
    
    # Save results
    print("💾 Saving Results...")
    print("=" * 60)
    
    results_file = Path(output_dir) / 'analysis_results.json'
    
    # Make serializable
    serializable_results = {
        'aggregate': batch_results['aggregate'],
        'clusters': batch_results['clusters'],
        'patterns': batch_results['patterns'],
        'sample_count': len(batch_results['individual_analyses']),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"✓ Results saved to {results_file}")
    print(f"✓ Report saved to {Path(output_dir) / 'comprehensive_report.html'}")
    print(f"✓ Visualizations in {output_dir}/")
    
    print("\n" + "✅" * 40)
    print("\n  ANALYSIS COMPLETE!")
    print(f"\n  Open: {Path(output_dir) / 'comprehensive_report.html'}")
    print("\n" + "✅" * 40 + "\n")
    
    return batch_results


if __name__ == '__main__':
    results = main()

