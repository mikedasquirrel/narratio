"""
Advanced Visualization Suite: Heatmaps, Clustering, Density, Correlation

Production-grade interactive visualizations with clear variable labeling.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import pearsonr
import json


class AdvancedNarrativeVisualizations:
    """
    Advanced visualizations for narrative analysis.
    
    Includes:
    - Correlation heatmaps (feature relationships)
    - Clustering visualizations (narrative groups)
    - Density plots (score distributions)
    - Feature importance heatmaps
    - Transformer comparison matrices
    """
    
    def __init__(self, theme='dark'):
        self.colors = {
            'primary': '#e879f9',
            'secondary': '#22d3ee',
            'accent': '#0ea5e9',
            'background': '#0f172a',
            'card': '#1e293b',
            'text': '#f1f5f9'
        }
    
    def create_correlation_heatmap(
        self,
        features: np.ndarray,
        feature_names: List[str],
        title: str = "Feature Correlation Matrix",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Interactive correlation heatmap showing relationships between features.
        
        Parameters
        ----------
        features : array
            Feature matrix (n_samples, n_features)
        feature_names : list
            Names for each feature
        title : str
            Chart title
        save_path : str, optional
            Path to save HTML
        
        Returns
        -------
        fig : Plotly Figure
            Interactive heatmap with hover details
        """
        # Compute correlation matrix
        df = pd.DataFrame(features, columns=feature_names)
        corr_matrix = df.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=feature_names,
            y=feature_names,
            colorscale=[
                [0, '#0ea5e9'],    # cerulean (negative)
                [0.5, '#1e293b'],  # dark neutral
                [1, '#e879f9']     # fuchsia (positive)
            ],
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(
                title="Correlation",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=['-1', '-0.5', '0', '0.5', '1']
            ),
            hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=20, color=self.colors['primary'])),
            xaxis_title="Features",
            yaxis_title="Features",
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['card'],
            font=dict(color=self.colors['text'], size=11),
            height=800,
            width=900,
            xaxis=dict(tickangle=45),
            yaxis=dict(autorange='reversed')
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_clustering_visualization(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        text_samples: List[str],
        n_clusters: int = 4,
        title: str = "Narrative Clustering Analysis",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Interactive clustering visualization with PCA projection.
        
        Shows how narratives group together in 2D/3D space.
        
        Parameters
        ----------
        features : array
            Feature matrix
        labels : array
            True labels (for color coding)
        text_samples : list
            Text of each sample (for hover)
        n_clusters : int
            Number of clusters to find
        title : str
            Chart title
        save_path : str, optional
            Save path
        
        Returns
        -------
        fig : Plotly Figure
            Interactive scatter plot with clusters
        """
        # Reduce to 2D with PCA
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        pca = PCA(n_components=2, random_state=42)
        features_2d = pca.fit_transform(features_scaled)
        
        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Create dataframe
        df = pd.DataFrame({
            'PC1': features_2d[:, 0],
            'PC2': features_2d[:, 1],
            'Cluster': [f'Cluster {c}' for c in cluster_labels],
            'True Label': labels,
            'Text': [text[:100] + '...' if len(text) > 100 else text for text in text_samples]
        })
        
        # Create scatter
        fig = px.scatter(
            df,
            x='PC1',
            y='PC2',
            color='Cluster',
            symbol='True Label',
            hover_data=['Text'],
            title=title,
            color_discrete_sequence=[
                self.colors['primary'],
                self.colors['secondary'],
                self.colors['accent'],
                '#f97316',
                '#10b981',
                '#f59e0b'
            ]
        )
        
        # Add cluster centers
        centers_2d = pca.transform(scaler.transform(kmeans.cluster_centers_))
        
        fig.add_trace(go.Scatter(
            x=centers_2d[:, 0],
            y=centers_2d[:, 1],
            mode='markers',
            marker=dict(
                size=20,
                symbol='x',
                color='white',
                line=dict(width=3, color=self.colors['accent'])
            ),
            name='Cluster Centers',
            hovertemplate='<b>Cluster Center</b><extra></extra>'
        ))
        
        fig.update_layout(
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['card'],
            font=dict(color=self.colors['text']),
            title_x=0.5,
            title_font_size=20,
            title_font_color=self.colors['primary'],
            height=700,
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
            legend=dict(
                bgcolor=self.colors['card'],
                bordercolor=self.colors['text'],
                borderwidth=1
            )
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_density_plot(
        self,
        results: Dict[str, Any],
        metric: str = 'f1_macro',
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Density plot showing score distributions across narratives.
        
        Parameters
        ----------
        results : dict
            Experiment results
        metric : str
            Metric to visualize
        title : str, optional
            Chart title
        save_path : str, optional
            Save path
        
        Returns
        -------
        fig : Plotly Figure
            Interactive density plot
        """
        if title is None:
            title = f"{metric.replace('_', ' ').title()} Score Density Distribution"
        
        narratives = results['narratives']
        
        fig = go.Figure()
        
        for name, narrative_data in narratives.items():
            scores = narrative_data['cv_scores'][metric]['test_scores']
            
            # Create density estimate
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(scores)
            x_range = np.linspace(min(scores) * 0.9, max(scores) * 1.1, 100)
            density = kde(x_range)
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=density,
                mode='lines',
                name=name,
                fill='tozeroy',
                opacity=0.6,
                line=dict(width=3),
                hovertemplate=f'<b>{name}</b><br>Score: %{{x:.4f}}<br>Density: %{{y:.3f}}<extra></extra>'
            ))
            
            # Add mean line
            mean_score = narrative_data['cv_scores'][metric]['test_mean']
            fig.add_vline(
                x=mean_score,
                line_dash="dash",
                line_width=2,
                annotation_text=f"{name}<br>mean",
                annotation_position="top"
            )
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=20, color=self.colors['primary'])),
            xaxis_title=metric.replace('_', ' ').title(),
            yaxis_title="Density",
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['card'],
            font=dict(color=self.colors['text']),
            height=600,
            hovermode='x unified',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="right",
                x=1.15,
                bgcolor=self.colors['card'],
                bordercolor=self.colors['text'],
                borderwidth=1
            )
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_transformer_performance_heatmap(
        self,
        results: Dict[str, Any],
        metrics: List[str],
        title: str = "Transformer × Metric Performance Matrix",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Heatmap showing each transformer's performance across metrics.
        
        Clearly shows: which transformers excel at which metrics.
        
        Parameters
        ----------
        results : dict
            Experiment results
        metrics : list
            Metrics to include
        title : str
            Chart title
        save_path : str, optional
            Save path
        
        Returns
        -------
        fig : Plotly Figure
            Interactive heatmap
        """
        narratives = results['narratives']
        narrative_names = list(narratives.keys())
        
        # Build matrix: narratives × metrics
        matrix = []
        for name in narrative_names:
            row = [narratives[name]['cv_scores'][m]['test_mean'] for m in metrics]
            matrix.append(row)
        
        matrix = np.array(matrix)
        
        # Create annotations with values
        annotations = []
        for i, name in enumerate(narrative_names):
            for j, metric in enumerate(metrics):
                value = matrix[i, j]
                annotations.append(
                    dict(
                        x=j,
                        y=i,
                        text=f'{value:.3f}',
                        showarrow=False,
                        font=dict(color='white' if value > 0.5 else 'black', size=12)
                    )
                )
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=[m.replace('_', ' ').title() for m in metrics],
            y=narrative_names,
            colorscale=[
                [0, '#1e293b'],      # dark
                [0.3, '#0ea5e9'],    # cerulean
                [0.6, '#22d3ee'],    # cyan
                [1, '#e879f9']       # fuchsia
            ],
            colorbar=dict(
                title="Score",
                thickness=20,
                len=0.7
            ),
            hovertemplate='<b>%{y}</b><br><b>%{x}</b><br>Score: %{z:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=20, color=self.colors['primary'])),
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['card'],
            font=dict(color=self.colors['text'], size=12),
            xaxis=dict(side='bottom', tickangle=0),
            yaxis=dict(autorange='reversed'),
            height=600,
            width=900,
            annotations=annotations
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_feature_importance_heatmap(
        self,
        transformer_name: str,
        feature_names: List[str],
        importance_scores: np.ndarray,
        top_n: int = 30,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Heatmap of feature importance scores.
        
        Shows which features matter most in a transformer.
        
        Parameters
        ----------
        transformer_name : str
            Name of transformer
        feature_names : list
            Feature names
        importance_scores : array
            Importance score for each feature
        top_n : int
            Show top N features
        title : str, optional
            Chart title
        save_path : str, optional
            Save path
        
        Returns
        -------
        fig : Plotly Figure
        """
        if title is None:
            title = f"{transformer_name} - Top {top_n} Feature Importance"
        
        # Get top features
        top_indices = np.argsort(np.abs(importance_scores))[-top_n:][::-1]
        top_features = [feature_names[i] for i in top_indices]
        top_scores = importance_scores[top_indices]
        
        # Create horizontal bar chart (clearer than heatmap for this)
        fig = go.Figure(data=[
            go.Bar(
                y=top_features,
                x=top_scores,
                orientation='h',
                marker=dict(
                    color=top_scores,
                    colorscale=[
                        [0, self.colors['accent']],
                        [0.5, self.colors['secondary']],
                        [1, self.colors['primary']]
                    ],
                    colorbar=dict(title="Importance"),
                    line=dict(color='white', width=1)
                ),
                hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=20, color=self.colors['primary'])),
            xaxis_title="Importance Score",
            yaxis_title="Features",
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['card'],
            font=dict(color=self.colors['text']),
            height=800,
            width=1000,
            yaxis=dict(autorange='reversed')
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_hierarchical_clustering(
        self,
        features: np.ndarray,
        sample_names: List[str],
        title: str = "Hierarchical Clustering of Narratives",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Interactive dendrogram showing hierarchical clustering.
        
        Shows which narratives are most similar.
        
        Parameters
        ----------
        features : array
            Feature matrix
        sample_names : list
            Names for each sample
        title : str
            Chart title
        save_path : str, optional
            Save path
        
        Returns
        -------
        fig : Plotly Figure
        """
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Compute linkage
        Z = linkage(features_scaled, method='ward')
        
        # Create dendrogram
        fig = ff.create_dendrogram(
            features_scaled,
            labels=sample_names,
            color_threshold=np.median(Z[:, 2]),
            colorscale=[self.colors['primary'], self.colors['secondary'], self.colors['accent']]
        )
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=20, color=self.colors['primary'])),
            xaxis_title="Narratives",
            yaxis_title="Distance (Ward Linkage)",
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['card'],
            font=dict(color=self.colors['text']),
            height=600,
            width=1200
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_density_comparison(
        self,
        results: Dict[str, Any],
        metrics: List[str],
        title: str = "Score Density Comparison Across Metrics",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Multi-panel density plots for multiple metrics.
        
        Shows distribution shape for each metric.
        """
        narratives = results['narratives']
        n_metrics = len(metrics)
        
        fig = make_subplots(
            rows=1, cols=n_metrics,
            subplot_titles=[m.replace('_', ' ').title() for m in metrics],
            horizontal_spacing=0.1
        )
        
        colors = [self.colors['primary'], self.colors['secondary'], self.colors['accent']]
        
        for col, metric in enumerate(metrics, 1):
            for idx, (name, narrative_data) in enumerate(narratives.items()):
                scores = narrative_data['cv_scores'][metric]['test_scores']
                
                # KDE
                from scipy.stats import gaussian_kde
                if len(scores) > 1:
                    kde = gaussian_kde(scores)
                    x_range = np.linspace(min(scores) * 0.9, max(scores) * 1.1, 50)
                    density = kde(x_range)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_range,
                            y=density,
                            mode='lines',
                            name=name if col == 1 else None,
                            line=dict(color=colors[idx % len(colors)], width=2),
                            fill='tozeroy',
                            opacity=0.5,
                            showlegend=(col == 1),
                            hovertemplate=f'{name}<br>Score: %{{x:.4f}}<extra></extra>'
                        ),
                        row=1, col=col
                    )
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=20, color=self.colors['primary'])),
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['card'],
            font=dict(color=self.colors['text']),
            height=500,
            showlegend=True,
            legend=dict(orientation="v", x=1.05, y=1)
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_transformer_correlation_matrix(
        self,
        all_results: List[Dict[str, Any]],
        title: str = "Inter-Transformer Correlation Matrix",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Show how different transformers correlate in their predictions.
        
        High correlation = transformers capture similar signals.
        Low correlation = transformers capture distinct signals.
        """
        # Extract scores from all experiments
        transformer_scores = {}
        
        for results in all_results:
            for name, narrative_data in results['narratives'].items():
                if name not in transformer_scores:
                    transformer_scores[name] = []
                
                # Get all fold scores
                scores = narrative_data['cv_scores']['accuracy']['test_scores']
                transformer_scores[name].extend(scores)
        
        # Build correlation matrix
        transformer_names = list(transformer_scores.keys())
        n_transformers = len(transformer_names)
        corr_matrix = np.zeros((n_transformers, n_transformers))
        
        for i, name_i in enumerate(transformer_names):
            for j, name_j in enumerate(transformer_names):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    # Compute correlation
                    scores_i = transformer_scores[name_i]
                    scores_j = transformer_scores[name_j]
                    
                    # Pad if needed
                    min_len = min(len(scores_i), len(scores_j))
                    if min_len > 1:
                        corr, _ = pearsonr(scores_i[:min_len], scores_j[:min_len])
                        corr_matrix[i, j] = corr
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=transformer_names,
            y=transformer_names,
            colorscale=[
                [0, '#0ea5e9'],
                [0.5, '#1e293b'],
                [1, '#e879f9']
            ],
            zmid=0,
            text=corr_matrix,
            texttemplate='%{text:.2f}',
            textfont={"size": 11, "color": "white"},
            colorbar=dict(title="Correlation"),
            hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=20, color=self.colors['primary'])),
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['card'],
            font=dict(color=self.colors['text']),
            height=700,
            width=800,
            xaxis=dict(tickangle=45),
            yaxis=dict(autorange='reversed')
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig


def generate_all_advanced_visualizations(experiment_dir: str):
    """Generate all advanced visualizations for an experiment."""
    from pathlib import Path
    import json
    
    exp_path = Path(experiment_dir)
    results_file = exp_path / 'results.json'
    
    if not results_file.exists():
        print(f"No results in {experiment_dir}")
        return
    
    with open(results_file) as f:
        results = json.load(f)
    
    print(f"📊 Generating advanced visualizations for {results['experiment_id']}")
    
    viz = AdvancedNarrativeVisualizations()
    
    # 1. Transformer performance heatmap
    print("  Creating performance heatmap...")
    first_narrative = list(results['narratives'].values())[0]
    metrics = list(first_narrative['cv_scores'].keys())
    
    viz.create_transformer_performance_heatmap(
        results,
        metrics,
        save_path=str(exp_path / 'heatmap_performance.html')
    )
    
    # 2. Density comparison
    print("  Creating density plots...")
    viz.create_density_comparison(
        results,
        metrics[:min(3, len(metrics))],
        save_path=str(exp_path / 'density_comparison.html')
    )
    
    # 3. Individual density for primary metric
    print("  Creating detailed density plot...")
    viz.create_density_plot(
        results,
        metrics[0],
        save_path=str(exp_path / 'density_detailed.html')
    )
    
    print(f"✓ Advanced visualizations saved to {exp_path}")
    print(f"  - heatmap_performance.html")
    print(f"  - density_comparison.html")
    print(f"  - density_detailed.html")


if __name__ == '__main__':
    print("Advanced Visualization Suite Ready")
    print("Heatmaps, Clustering, Density, Correlation - All Interactive")

