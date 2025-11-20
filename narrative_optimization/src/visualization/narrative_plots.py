"""
Visualization utilities for narrative analysis and comparison.

Provides interactive and static visualizations for understanding narrative
performance, feature importance, and experimental results.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class NarrativePlotter:
    """
    Create visualizations for narrative optimization experiments.
    
    Supports both static (matplotlib) and interactive (plotly) visualizations.
    
    Parameters
    ----------
    style : str
        Matplotlib style to use
    output_dir : str, optional
        Directory for saving plots
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid', output_dir: Optional[str] = None):
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        sns.set_palette("husl")
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_narrative_performance_comparison(
        self,
        results: Dict[str, Any],
        metrics: List[str],
        save_name: Optional[str] = None,
        interactive: bool = True
    ):
        """
        Compare performance across narratives.
        
        Parameters
        ----------
        results : dict
            Experiment results dictionary
        metrics : list
            Metrics to plot
        save_name : str, optional
            Filename to save plot
        interactive : bool
            Use plotly (True) or matplotlib (False)
        """
        narratives = results['narratives']
        
        if interactive:
            self._plot_performance_comparison_plotly(narratives, metrics, save_name)
        else:
            self._plot_performance_comparison_mpl(narratives, metrics, save_name)
    
    def _plot_performance_comparison_plotly(
        self,
        narratives: Dict[str, Any],
        metrics: List[str],
        save_name: Optional[str]
    ):
        """Create interactive performance comparison with Plotly."""
        n_metrics = len(metrics)
        fig = make_subplots(
            rows=1, cols=n_metrics,
            subplot_titles=[m.replace('_', ' ').title() for m in metrics]
        )
        
        narrative_names = list(narratives.keys())
        colors = px.colors.qualitative.Set2
        
        for col, metric in enumerate(metrics, 1):
            means = []
            stds = []
            
            for name in narrative_names:
                scores = narratives[name]['cv_scores'][metric]
                means.append(scores['test_mean'])
                stds.append(scores['test_std'])
            
            fig.add_trace(
                go.Bar(
                    name=metric,
                    x=narrative_names,
                    y=means,
                    error_y=dict(type='data', array=stds),
                    marker_color=colors[:len(narrative_names)],
                    showlegend=False
                ),
                row=1, col=col
            )
        
        fig.update_layout(
            title_text="Narrative Performance Comparison",
            height=400,
            showlegend=False
        )
        
        if save_name and self.output_dir:
            fig.write_html(self.output_dir / f"{save_name}.html")
        
        fig.show()
    
    def _plot_performance_comparison_mpl(
        self,
        narratives: Dict[str, Any],
        metrics: List[str],
        save_name: Optional[str]
    ):
        """Create static performance comparison with matplotlib."""
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        narrative_names = list(narratives.keys())
        
        for ax, metric in zip(axes, metrics):
            means = []
            stds = []
            
            for name in narrative_names:
                scores = narratives[name]['cv_scores'][metric]
                means.append(scores['test_mean'])
                stds.append(scores['test_std'])
            
            x = np.arange(len(narrative_names))
            ax.bar(x, means, yerr=stds, capsize=5)
            ax.set_xlabel('Narrative')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(narrative_names, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_name and self.output_dir:
            plt.savefig(self.output_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_cv_score_distributions(
        self,
        results: Dict[str, Any],
        metric: str,
        save_name: Optional[str] = None,
        interactive: bool = True
    ):
        """
        Plot distribution of cross-validation scores.
        
        Shows variability and consistency across folds.
        
        Parameters
        ----------
        results : dict
            Experiment results
        metric : str
            Metric to visualize
        save_name : str, optional
            Filename to save
        interactive : bool
            Use plotly or matplotlib
        """
        narratives = results['narratives']
        
        if interactive:
            fig = go.Figure()
            
            for name, narrative_data in narratives.items():
                scores = narrative_data['cv_scores'][metric]['test_scores']
                fig.add_trace(go.Box(
                    y=scores,
                    name=name,
                    boxmean='sd'
                ))
            
            fig.update_layout(
                title=f"Cross-Validation {metric.replace('_', ' ').title()} Distribution",
                yaxis_title=metric.replace('_', ' ').title(),
                xaxis_title='Narrative'
            )
            
            if save_name and self.output_dir:
                fig.write_html(self.output_dir / f"{save_name}.html")
            
            fig.show()
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            data = []
            labels = []
            for name, narrative_data in narratives.items():
                scores = narrative_data['cv_scores'][metric]['test_scores']
                data.append(scores)
                labels.append(name)
            
            ax.boxplot(data, labels=labels)
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_xlabel('Narrative')
            ax.set_title(f'Cross-Validation {metric.replace("_", " ").title()} Distribution')
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            
            if save_name and self.output_dir:
                plt.savefig(self.output_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
            
            plt.show()
    
    def plot_confusion_matrices(
        self,
        confusion_matrices: Dict[str, np.ndarray],
        target_names: List[str],
        save_name: Optional[str] = None
    ):
        """
        Plot confusion matrices for each narrative.
        
        Parameters
        ----------
        confusion_matrices : dict
            Dictionary mapping narrative names to confusion matrices
        target_names : list
            Class labels
        save_name : str, optional
            Filename to save
        """
        n_narratives = len(confusion_matrices)
        fig, axes = plt.subplots(1, n_narratives, figsize=(6 * n_narratives, 5))
        
        if n_narratives == 1:
            axes = [axes]
        
        for ax, (name, cm) in zip(axes, confusion_matrices.items()):
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names,
                ax=ax
            )
            ax.set_title(f'{name}\nConfusion Matrix')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        if save_name and self.output_dir:
            plt.savefig(self.output_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_narrative_quality_radar(
        self,
        evaluations: Dict[str, Dict[str, Any]],
        save_name: Optional[str] = None
    ):
        """
        Create radar plot comparing narrative quality dimensions.
        
        Parameters
        ----------
        evaluations : dict
            Dictionary mapping narrative names to comprehensive evaluations
        save_name : str, optional
            Filename to save
        """
        # Extract quality dimensions
        dimensions = ['Performance', 'Coherence', 'Interpretability', 'Robustness']
        
        fig = go.Figure()
        
        for name, evaluation in evaluations.items():
            values = [
                evaluation['performance'].get('f1', 0.0),
                evaluation['coherence']['coherence_score'],
                evaluation['interpretability']['interpretability_score'],
                evaluation.get('robustness', {}).get('robustness_score', 1.0)
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=dimensions,
                fill='toself',
                name=name
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Narrative Quality Comparison (Radar)",
            showlegend=True
        )
        
        if save_name and self.output_dir:
            fig.write_html(self.output_dir / f"{save_name}.html")
        
        fig.show()
    
    def plot_generalization_gap(
        self,
        results: Dict[str, Any],
        metric: str,
        save_name: Optional[str] = None
    ):
        """
        Plot train vs test performance to visualize overfitting.
        
        Parameters
        ----------
        results : dict
            Experiment results
        metric : str
            Metric to compare
        save_name : str, optional
            Filename to save
        """
        narratives = results['narratives']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        narrative_names = list(narratives.keys())
        train_means = []
        test_means = []
        
        for name in narrative_names:
            scores = narratives[name]['cv_scores'][metric]
            train_means.append(scores['train_mean'])
            test_means.append(scores['test_mean'])
        
        x = np.arange(len(narrative_names))
        width = 0.35
        
        ax.bar(x - width/2, train_means, width, label='Train', alpha=0.8)
        ax.bar(x + width/2, test_means, width, label='Test', alpha=0.8)
        
        # Draw lines connecting train to test
        for i in range(len(narrative_names)):
            ax.plot([i - width/2, i + width/2], [train_means[i], test_means[i]], 
                   'k--', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('Narrative')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Generalization Gap: {metric.replace("_", " ").title()}')
        ax.set_xticks(x)
        ax.set_xticklabels(narrative_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_name and self.output_dir:
            plt.savefig(self.output_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_experiment_summary_plot(
        self,
        results: Dict[str, Any],
        primary_metric: str = 'f1',
        save_name: Optional[str] = None
    ):
        """
        Create comprehensive summary visualization of experiment.
        
        Parameters
        ----------
        results : dict
            Experiment results
        primary_metric : str
            Main metric to highlight
        save_name : str, optional
            Filename to save
        """
        narratives = results['narratives']
        narrative_names = list(narratives.keys())
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Main performance comparison
        ax1 = fig.add_subplot(gs[0, :2])
        means = [narratives[name]['cv_scores'][primary_metric]['test_mean'] for name in narrative_names]
        stds = [narratives[name]['cv_scores'][primary_metric]['test_std'] for name in narrative_names]
        
        x = np.arange(len(narrative_names))
        bars = ax1.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
        ax1.set_xlabel('Narrative', fontsize=12)
        ax1.set_ylabel(primary_metric.replace('_', ' ').title(), fontsize=12)
        ax1.set_title(f'Primary Metric: {primary_metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(narrative_names, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        # Highlight best
        best_idx = np.argmax(means)
        bars[best_idx].set_color('gold')
        bars[best_idx].set_alpha(1.0)
        
        # 2. Score distribution
        ax2 = fig.add_subplot(gs[0, 2])
        data = [narratives[name]['cv_scores'][primary_metric]['test_scores'] for name in narrative_names]
        bp = ax2.boxplot(data, labels=narrative_names)
        ax2.set_ylabel(primary_metric.replace('_', ' ').title(), fontsize=10)
        ax2.set_title('Score Distribution', fontsize=12)
        ax2.tick_params(axis='x', rotation=45, labelsize=8)
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Generalization gap
        ax3 = fig.add_subplot(gs[1, 0])
        train_means = [narratives[name]['cv_scores'][primary_metric]['train_mean'] for name in narrative_names]
        test_means = [narratives[name]['cv_scores'][primary_metric]['test_mean'] for name in narrative_names]
        gaps = [t - te for t, te in zip(train_means, test_means)]
        
        ax3.barh(narrative_names, gaps, alpha=0.7)
        ax3.set_xlabel('Generalization Gap', fontsize=10)
        ax3.set_title('Overfitting Analysis', fontsize=12)
        ax3.axvline(x=0, color='k', linestyle='--', linewidth=1)
        ax3.grid(axis='x', alpha=0.3)
        
        # 4. Experiment info
        ax4 = fig.add_subplot(gs[1, 1:])
        ax4.axis('off')
        
        info_text = f"Experiment: {results['experiment_id']}\n\n"
        info_text += f"Description: {results['description']}\n\n"
        info_text += "Results Summary:\n"
        
        for i, name in enumerate(narrative_names):
            score = means[i]
            marker = "★ " if i == best_idx else "  "
            info_text += f"{marker}{name}: {score:.4f} ± {stds[i]:.4f}\n"
        
        ax4.text(0.1, 0.9, info_text, fontsize=11, verticalalignment='top',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        fig.suptitle(f"Experiment Summary: {results['experiment_id']}", 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_name and self.output_dir:
            plt.savefig(self.output_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        
        plt.show()


def quick_plot_results(results: Dict[str, Any], output_dir: Optional[str] = None):
    """
    Quick utility to generate standard visualizations for experiment results.
    
    Parameters
    ----------
    results : dict
        Experiment results dictionary
    output_dir : str, optional
        Directory to save plots
    """
    plotter = NarrativePlotter(output_dir=output_dir)
    
    # Get available metrics
    first_narrative = list(results['narratives'].values())[0]
    metrics = list(first_narrative['cv_scores'].keys())
    
    # Create summary plot
    primary_metric = 'f1' if 'f1' in metrics else metrics[0]
    plotter.create_experiment_summary_plot(
        results,
        primary_metric=primary_metric,
        save_name='experiment_summary'
    )
    
    # Create performance comparison
    plotter.plot_narrative_performance_comparison(
        results,
        metrics=metrics,
        save_name='performance_comparison',
        interactive=False
    )
    
    # Create CV distribution
    plotter.plot_cv_score_distributions(
        results,
        primary_metric,
        save_name='cv_distributions',
        interactive=False
    )
    
    print(f"Visualizations saved to {output_dir or 'current directory'}")

