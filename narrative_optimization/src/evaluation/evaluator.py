"""
Evaluation framework for narrative quality and performance metrics.

Provides multi-objective evaluation that considers both traditional ML performance
and novel narrative quality dimensions.
"""

from typing import Dict, Any, List, Optional, Callable
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix
)
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
import warnings


class NarrativeEvaluator:
    """
    Evaluates pipelines on both performance and narrative quality.
    
    Implements multi-objective evaluation framework that assesses:
    1. Traditional ML metrics (accuracy, F1, AUC, etc.)
    2. Narrative quality metrics (coherence, interpretability, stability)
    3. Meta-metrics (generalization, robustness, component attribution)
    
    Parameters
    ----------
    random_state : int, optional
        Random seed for reproducibility
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.evaluation_history: List[Dict[str, Any]] = []
    
    def evaluate_performance(
        self,
        pipeline,
        X_test,
        y_test,
        X_train=None,
        y_train=None
    ) -> Dict[str, float]:
        """
        Evaluate standard ML performance metrics.
        
        Parameters
        ----------
        pipeline : fitted Pipeline
            The fitted pipeline to evaluate
        X_test : array-like
            Test features
        y_test : array-like
            Test targets
        X_train : array-like, optional
            Training features (for computing train metrics)
        y_train : array-like, optional
            Training targets
        
        Returns
        -------
        metrics : dict
            Dictionary of performance metrics
        """
        metrics = {}
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Basic metrics
        metrics['accuracy'] = float(accuracy_score(y_test, y_pred))
        
        # Precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        metrics['precision'] = float(precision)
        metrics['recall'] = float(recall)
        metrics['f1'] = float(f1)
        
        # Macro-averaged F1 for imbalanced classes
        _, _, f1_macro, _ = precision_recall_fscore_support(
            y_test, y_pred, average='macro', zero_division=0
        )
        metrics['f1_macro'] = float(f1_macro)
        
        # ROC-AUC for probabilistic predictions
        if hasattr(pipeline, 'predict_proba'):
            try:
                y_proba = pipeline.predict_proba(X_test)
                # Handle binary and multiclass cases
                if y_proba.shape[1] == 2:
                    metrics['roc_auc'] = float(roc_auc_score(y_test, y_proba[:, 1]))
                else:
                    metrics['roc_auc'] = float(
                        roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
                    )
            except Exception as e:
                warnings.warn(f"Could not compute ROC-AUC: {e}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Training metrics if provided
        if X_train is not None and y_train is not None:
            y_train_pred = pipeline.predict(X_train)
            metrics['train_accuracy'] = float(accuracy_score(y_train, y_train_pred))
            metrics['generalization_gap'] = metrics['train_accuracy'] - metrics['accuracy']
        
        return metrics
    
    def evaluate_narrative_coherence(self, pipeline) -> Dict[str, Any]:
        """
        Assess internal consistency of the narrative.
        
        Evaluates whether the pipeline's components form a coherent story
        and align with the stated narrative hypothesis.
        
        Parameters
        ----------
        pipeline : Pipeline
            The pipeline to evaluate
        
        Returns
        -------
        coherence_metrics : dict
            Metrics assessing narrative coherence
        """
        coherence = {
            'has_narrative_metadata': False,
            'narrative_documentation_completeness': 0.0,
            'components_with_narrative_reports': 0,
            'total_components': 0,
            'coherence_score': 0.0
        }
        
        # Check for narrative metadata
        if hasattr(pipeline, '_narrative_metadata'):
            coherence['has_narrative_metadata'] = True
            metadata = pipeline._narrative_metadata
            
            # Check completeness of documentation
            required_fields = ['hypothesis', 'expected_outcome', 'domain_assumptions']
            completed_fields = sum(1 for field in required_fields if metadata.get(field))
            coherence['narrative_documentation_completeness'] = completed_fields / len(required_fields)
        
        # Check how many components have narrative reports
        steps = pipeline.named_steps if hasattr(pipeline, 'named_steps') else {}
        coherence['total_components'] = len(steps)
        
        for step_name, step in steps.items():
            if hasattr(step, 'get_narrative_report'):
                coherence['components_with_narrative_reports'] += 1
        
        # Compute overall coherence score
        if coherence['total_components'] > 0:
            component_ratio = coherence['components_with_narrative_reports'] / coherence['total_components']
            coherence['coherence_score'] = (
                0.3 * float(coherence['has_narrative_metadata']) +
                0.3 * coherence['narrative_documentation_completeness'] +
                0.4 * component_ratio
            )
        
        return coherence
    
    def evaluate_interpretability(self, pipeline) -> Dict[str, Any]:
        """
        Measure how understandable the narrative is.
        
        Assesses whether humans can understand what the pipeline does
        and why it makes its predictions.
        
        Parameters
        ----------
        pipeline : Pipeline
            The pipeline to evaluate
        
        Returns
        -------
        interpretability_metrics : dict
            Metrics assessing interpretability
        """
        interpretability = {
            'has_feature_names': False,
            'has_narrative_reports': False,
            'components_with_interpretation': 0,
            'total_components': 0,
            'interpretability_score': 0.0
        }
        
        # Check if pipeline preserves feature names
        steps = pipeline.named_steps if hasattr(pipeline, 'named_steps') else {}
        interpretability['total_components'] = len(steps)
        
        for step_name, step in steps.items():
            # Check for feature names
            if hasattr(step, 'get_feature_names_out') or hasattr(step, 'get_feature_names'):
                interpretability['has_feature_names'] = True
            
            # Check for narrative interpretation
            if hasattr(step, 'get_narrative_report'):
                interpretability['has_narrative_reports'] = True
                interpretability['components_with_interpretation'] += 1
        
        # Compute interpretability score
        if interpretability['total_components'] > 0:
            component_ratio = interpretability['components_with_interpretation'] / interpretability['total_components']
            interpretability['interpretability_score'] = (
                0.3 * float(interpretability['has_feature_names']) +
                0.7 * component_ratio
            )
        
        return interpretability
    
    def evaluate_robustness(
        self,
        pipeline,
        X_test,
        y_test,
        perturbation_strength: float = 0.1,
        n_perturbations: int = 10
    ) -> Dict[str, float]:
        """
        Test narrative stability under perturbation.
        
        Evaluates how sensitive the pipeline is to small changes in input data.
        More robust narratives maintain performance under perturbation.
        
        Parameters
        ----------
        pipeline : fitted Pipeline
            The fitted pipeline to test
        X_test : array-like
            Test features
        y_test : array-like
            Test targets
        perturbation_strength : float
            How much to perturb (fraction of std dev)
        n_perturbations : int
            Number of perturbation trials
        
        Returns
        -------
        robustness_metrics : dict
            Metrics assessing robustness
        """
        # Get baseline performance
        y_pred_baseline = pipeline.predict(X_test)
        baseline_acc = accuracy_score(y_test, y_pred_baseline)
        
        # Generate perturbations
        perturbed_accuracies = []
        
        for _ in range(n_perturbations):
            # Add Gaussian noise
            noise = np.random.normal(0, perturbation_strength, X_test.shape)
            X_perturbed = X_test + noise
            
            # Evaluate on perturbed data
            y_pred_perturbed = pipeline.predict(X_perturbed)
            perturbed_acc = accuracy_score(y_test, y_pred_perturbed)
            perturbed_accuracies.append(perturbed_acc)
        
        # Compute robustness metrics
        mean_perturbed_acc = np.mean(perturbed_accuracies)
        std_perturbed_acc = np.std(perturbed_accuracies)
        performance_drop = baseline_acc - mean_perturbed_acc
        
        return {
            'baseline_accuracy': float(baseline_acc),
            'mean_perturbed_accuracy': float(mean_perturbed_acc),
            'std_perturbed_accuracy': float(std_perturbed_acc),
            'performance_drop': float(performance_drop),
            'robustness_score': float(1.0 - performance_drop)  # Higher is better
        }
    
    def comprehensive_evaluation(
        self,
        pipeline,
        X_test,
        y_test,
        X_train=None,
        y_train=None,
        include_robustness: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete evaluation across all dimensions.
        
        Parameters
        ----------
        pipeline : fitted Pipeline
            The pipeline to evaluate
        X_test : array-like
            Test features
        y_test : array-like
            Test targets
        X_train : array-like, optional
            Training features
        y_train : array-like, optional
            Training targets
        include_robustness : bool
            Whether to run robustness tests (can be slow)
        
        Returns
        -------
        evaluation : dict
            Comprehensive evaluation results
        """
        evaluation = {
            'performance': self.evaluate_performance(pipeline, X_test, y_test, X_train, y_train),
            'coherence': self.evaluate_narrative_coherence(pipeline),
            'interpretability': self.evaluate_interpretability(pipeline)
        }
        
        if include_robustness and isinstance(X_test, np.ndarray):
            evaluation['robustness'] = self.evaluate_robustness(pipeline, X_test, y_test)
        
        # Compute overall quality score
        evaluation['overall_quality'] = self._compute_overall_quality(evaluation)
        
        # Store in history
        self.evaluation_history.append(evaluation)
        
        return evaluation
    
    def _compute_overall_quality(self, evaluation: Dict[str, Any]) -> float:
        """
        Compute weighted overall quality score.
        
        Combines performance, coherence, interpretability, and robustness
        into a single quality metric.
        """
        weights = {
            'performance': 0.5,
            'coherence': 0.2,
            'interpretability': 0.2,
            'robustness': 0.1
        }
        
        scores = {
            'performance': evaluation['performance'].get('f1', 0.0),
            'coherence': evaluation['coherence']['coherence_score'],
            'interpretability': evaluation['interpretability']['interpretability_score'],
            'robustness': evaluation.get('robustness', {}).get('robustness_score', 1.0)
        }
        
        overall = sum(weights[k] * scores[k] for k in weights)
        return float(overall)
    
    def compare_narratives(
        self,
        evaluations: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare multiple narrative evaluations.
        
        Parameters
        ----------
        evaluations : dict
            Dictionary mapping narrative names to their evaluation results
        
        Returns
        -------
        comparison : dict
            Comparative analysis across narratives
        """
        comparison = {
            'rankings': {},
            'best_by_dimension': {},
            'insights': []
        }
        
        # Rank by overall quality
        quality_scores = {
            name: eval_result['overall_quality']
            for name, eval_result in evaluations.items()
        }
        comparison['rankings']['overall_quality'] = sorted(
            quality_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Find best by each dimension
        dimensions = ['performance', 'coherence', 'interpretability', 'robustness']
        
        for dimension in dimensions:
            if dimension == 'performance':
                scores = {
                    name: eval_result['performance'].get('f1', 0.0)
                    for name, eval_result in evaluations.items()
                }
            elif dimension in ['coherence', 'interpretability']:
                scores = {
                    name: eval_result[dimension][f'{dimension}_score']
                    for name, eval_result in evaluations.items()
                }
            elif dimension == 'robustness':
                scores = {
                    name: eval_result.get('robustness', {}).get('robustness_score', 0.0)
                    for name, eval_result in evaluations.items()
                }
            
            best_name = max(scores.items(), key=lambda x: x[1])
            comparison['best_by_dimension'][dimension] = {
                'name': best_name[0],
                'score': best_name[1]
            }
        
        # Generate insights
        best_overall = comparison['rankings']['overall_quality'][0]
        comparison['insights'].append(
            f"Best overall narrative: '{best_overall[0]}' with quality score {best_overall[1]:.3f}"
        )
        
        for dimension, best_info in comparison['best_by_dimension'].items():
            comparison['insights'].append(
                f"Best {dimension}: '{best_info['name']}' ({best_info['score']:.3f})"
            )
        
        return comparison

