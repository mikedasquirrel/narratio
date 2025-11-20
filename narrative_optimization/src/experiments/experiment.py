"""
Experiment framework for testing narrative hypotheses.

Enables systematic comparison of competing narrative approaches with
comprehensive result tracking and analysis.
"""

from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from pathlib import Path
import json
import pickle
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd


class NarrativeExperiment:
    """
    Defines and executes experiments comparing narrative hypotheses.
    
    Manages multiple competing narrative pipelines, evaluates them systematically,
    and generates comprehensive analysis of which narratives perform best and why.
    
    Parameters
    ----------
    experiment_id : str
        Unique identifier for this experiment
    description : str
        What this experiment tests
    output_dir : str, optional
        Directory for saving results (default: experiments/{experiment_id})
    
    Attributes
    ----------
    narratives : list
        List of (pipeline, hypothesis, name) tuples
    metrics : list
        Evaluation metrics to compute
    cv_strategy : cross-validation strategy
    results : dict or None
        Experiment results (None until run() is called)
    """
    
    def __init__(
        self,
        experiment_id: str,
        description: str,
        output_dir: Optional[str] = None
    ):
        self.experiment_id = experiment_id
        self.description = description
        self.output_dir = Path(output_dir or f"experiments/{experiment_id}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.narratives: List[Dict[str, Any]] = []
        self.metrics: List[str] = []
        self.cv_strategy = None
        self.results: Optional[Dict[str, Any]] = None
        
        self.metadata = {
            'experiment_id': experiment_id,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'status': 'initialized'
        }
    
    def add_narrative(
        self,
        pipeline: Pipeline,
        hypothesis: str,
        name: str,
        narrative_metadata: Optional[Dict[str, Any]] = None
    ) -> 'NarrativeExperiment':
        """
        Add a narrative pipeline to test.
        
        Parameters
        ----------
        pipeline : Pipeline
            sklearn Pipeline to evaluate
        hypothesis : str
            What narrative hypothesis this pipeline tests
        name : str
            Short name for this narrative
        narrative_metadata : dict, optional
            Additional metadata about the narrative
        
        Returns
        -------
        self : NarrativeExperiment
            For method chaining
        """
        narrative_entry = {
            'name': name,
            'hypothesis': hypothesis,
            'pipeline': pipeline,
            'metadata': narrative_metadata or {}
        }
        
        # Try to extract narrative metadata from pipeline if available
        if hasattr(pipeline, '_narrative_metadata'):
            narrative_entry['metadata'].update(pipeline._narrative_metadata)
        
        self.narratives.append(narrative_entry)
        return self
    
    def define_evaluation(
        self,
        metrics: List[str],
        cv_strategy,
        additional_evaluators: Optional[List[Callable]] = None
    ) -> 'NarrativeExperiment':
        """
        Define how narratives will be evaluated.
        
        Parameters
        ----------
        metrics : list of str
            Metric names for sklearn cross_validate (e.g., ['accuracy', 'f1_macro'])
        cv_strategy
            Cross-validation strategy (e.g., StratifiedKFold(5))
        additional_evaluators : list of callable, optional
            Custom evaluation functions taking (pipeline, X, y) and returning dict
        
        Returns
        -------
        self : NarrativeExperiment
            For method chaining
        """
        self.metrics = metrics
        self.cv_strategy = cv_strategy
        self.additional_evaluators = additional_evaluators or []
        return self
    
    def run(self, X, y, verbose: bool = True) -> Dict[str, Any]:
        """
        Execute the experiment and collect results.
        
        Parameters
        ----------
        X : array-like or DataFrame
            Feature data
        y : array-like
            Target values
        verbose : bool
            Whether to print progress
        
        Returns
        -------
        results : dict
            Comprehensive experiment results
        """
        if not self.narratives:
            raise ValueError("No narratives added. Use add_narrative() first.")
        
        if not self.metrics or self.cv_strategy is None:
            raise ValueError("Evaluation not defined. Use define_evaluation() first.")
        
        self.metadata['status'] = 'running'
        self.metadata['run_started_at'] = datetime.now().isoformat()
        
        results = {
            'experiment_id': self.experiment_id,
            'description': self.description,
            'metadata': self.metadata,
            'narratives': {}
        }
        
        for narrative in self.narratives:
            if verbose:
                print(f"\nEvaluating narrative: {narrative['name']}")
                print(f"Hypothesis: {narrative['hypothesis']}")
            
            # Run cross-validation
            scoring = {metric: metric for metric in self.metrics}
            cv_results = cross_validate(
                narrative['pipeline'],
                X, y,
                cv=self.cv_strategy,
                scoring=scoring,
                return_train_score=True,
                return_estimator=True
            )
            
            # Process results
            narrative_results = {
                'name': narrative['name'],
                'hypothesis': narrative['hypothesis'],
                'metadata': narrative['metadata'],
                'cv_scores': {},
                'summary_statistics': {}
            }
            
            # Extract and summarize scores
            for metric in self.metrics:
                test_scores = cv_results[f'test_{metric}']
                train_scores = cv_results[f'train_{metric}']
                
                narrative_results['cv_scores'][metric] = {
                    'test_scores': test_scores.tolist(),
                    'train_scores': train_scores.tolist(),
                    'test_mean': float(np.mean(test_scores)),
                    'test_std': float(np.std(test_scores)),
                    'train_mean': float(np.mean(train_scores)),
                    'train_std': float(np.std(train_scores))
                }
            
            # Run additional evaluators
            if self.additional_evaluators:
                narrative_results['additional_metrics'] = {}
                for evaluator in self.additional_evaluators:
                    eval_result = evaluator(narrative['pipeline'], X, y)
                    narrative_results['additional_metrics'].update(eval_result)
            
            # Store fitted estimators for later analysis
            narrative_results['fitted_estimators'] = cv_results['estimator']
            
            results['narratives'][narrative['name']] = narrative_results
            
            if verbose:
                for metric in self.metrics:
                    scores = narrative_results['cv_scores'][metric]
                    print(f"  {metric}: {scores['test_mean']:.4f} (+/- {scores['test_std']:.4f})")
        
        self.metadata['status'] = 'completed'
        self.metadata['run_completed_at'] = datetime.now().isoformat()
        self.results = results
        
        # Save results
        self._save_results()
        
        return results
    
    def analyze(self) -> Dict[str, Any]:
        """
        Generate comprehensive analysis of results.
        
        Returns
        -------
        analysis : dict
            Comparative analysis including:
            - Best performing narrative per metric
            - Statistical comparisons
            - Interpretation and insights
        """
        if self.results is None:
            raise ValueError("Experiment not yet run. Call run() first.")
        
        analysis = {
            'experiment_id': self.experiment_id,
            'best_narratives': {},
            'comparative_analysis': {},
            'insights': []
        }
        
        # Find best narrative for each metric
        for metric in self.metrics:
            best_name = None
            best_score = -np.inf
            
            scores = {}
            for name, narrative_results in self.results['narratives'].items():
                score = narrative_results['cv_scores'][metric]['test_mean']
                scores[name] = score
                
                if score > best_score:
                    best_score = score
                    best_name = name
            
            analysis['best_narratives'][metric] = {
                'name': best_name,
                'score': best_score,
                'all_scores': scores
            }
        
        # Generate insights
        analysis['insights'].append(
            f"Experiment '{self.experiment_id}' tested {len(self.narratives)} narrative hypotheses."
        )
        
        for metric, best_info in analysis['best_narratives'].items():
            analysis['insights'].append(
                f"For {metric}, '{best_info['name']}' performed best with {best_info['score']:.4f}."
            )
        
        return analysis
    
    def _save_results(self):
        """Save experiment results to disk."""
        # Save JSON-serializable results
        results_json = self.results.copy()
        
        # Remove non-serializable estimators
        for name in results_json['narratives']:
            if 'fitted_estimators' in results_json['narratives'][name]:
                del results_json['narratives'][name]['fitted_estimators']
        
        results_path = self.output_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        # Save full results including estimators with pickle
        full_results_path = self.output_dir / 'results_full.pkl'
        with open(full_results_path, 'wb') as f:
            pickle.dump(self.results, f)
        
        # Save metadata
        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate markdown report of experiment results.
        
        Parameters
        ----------
        output_path : str, optional
            Where to save the report (default: {output_dir}/report.md)
        
        Returns
        -------
        report : str
            Markdown-formatted report
        """
        if self.results is None:
            raise ValueError("Experiment not yet run. Call run() first.")
        
        analysis = self.analyze()
        
        report_lines = [
            f"# Experiment: {self.experiment_id}",
            "",
            f"**Description**: {self.description}",
            "",
            f"**Run Date**: {self.metadata.get('run_completed_at', 'N/A')}",
            "",
            "## Hypotheses Tested",
            ""
        ]
        
        for narrative in self.narratives:
            report_lines.extend([
                f"### {narrative['name']}",
                f"{narrative['hypothesis']}",
                ""
            ])
        
        report_lines.extend([
            "## Results",
            ""
        ])
        
        for metric in self.metrics:
            report_lines.append(f"### {metric}")
            report_lines.append("")
            
            for name, narrative_results in self.results['narratives'].items():
                scores = narrative_results['cv_scores'][metric]
                report_lines.append(
                    f"- **{name}**: {scores['test_mean']:.4f} (+/- {scores['test_std']:.4f})"
                )
            
            report_lines.append("")
        
        report_lines.extend([
            "## Best Performing Narratives",
            ""
        ])
        
        for metric, best_info in analysis['best_narratives'].items():
            report_lines.append(
                f"- **{metric}**: {best_info['name']} ({best_info['score']:.4f})"
            )
        
        report_lines.extend([
            "",
            "## Insights",
            ""
        ])
        
        for insight in analysis['insights']:
            report_lines.append(f"- {insight}")
        
        report_lines.extend([
            "",
            "## Next Steps",
            "",
            "_To be filled in based on findings._",
            ""
        ])
        
        report = "\n".join(report_lines)
        
        # Save report
        if output_path is None:
            output_path = self.output_dir / 'report.md'
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        return report

