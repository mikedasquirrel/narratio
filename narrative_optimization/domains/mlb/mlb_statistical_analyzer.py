"""
MLB Statistical Analyzer

Performs statistical analysis on MLB narrative features:
- Univariate correlations
- Feature importance
- Context-specific analysis (rivalries, playoff race)
- Model performance metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split


class MLBStatisticalAnalyzer:
    """
    Statistical analysis for MLB narrative features.
    """
    
    def __init__(self):
        """Initialize analyzer."""
        pass
    
    def analyze_features(self, features: np.ndarray, outcomes: np.ndarray, 
                        feature_names: List[str]) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis.
        
        Parameters
        ----------
        features : np.ndarray
            Feature matrix (n_samples, n_features)
        outcomes : np.ndarray
            Binary outcomes (0/1)
        feature_names : list of str
            Names of features
        
        Returns
        -------
        results : dict
            Analysis results
        """
        results = {
            'univariate_correlations': {},
            'feature_importance': {},
            'model_performance': {},
            'context_analysis': {}
        }
        
        # Univariate correlations
        print("\n[1/4] Computing univariate correlations...")
        correlations = self._compute_correlations(features, outcomes, feature_names)
        results['univariate_correlations'] = correlations
        
        # Feature importance (Random Forest)
        print("\n[2/4] Computing feature importance...")
        importance = self._compute_feature_importance(features, outcomes, feature_names)
        results['feature_importance'] = importance
        
        # Model performance
        print("\n[3/4] Evaluating model performance...")
        model_perf = self._evaluate_models(features, outcomes)
        results['model_performance'] = model_perf
        
        # Context-specific analysis
        print("\n[4/4] Context-specific analysis...")
        context_results = self._analyze_contexts(features, outcomes, feature_names)
        results['context_analysis'] = context_results
        
        return results
    
    def _compute_correlations(self, features: np.ndarray, outcomes: np.ndarray,
                            feature_names: List[str]) -> Dict[str, float]:
        """Compute univariate correlations."""
        correlations = {}
        
        for i, name in enumerate(feature_names[:features.shape[1]]):
            try:
                feature_vec = features[:, i]
                if np.std(feature_vec) > 0:  # Avoid division by zero
                    corr = np.corrcoef(feature_vec, outcomes)[0, 1]
                    correlations[name] = float(corr) if not np.isnan(corr) else 0.0
            except:
                continue
        
        # Sort by absolute correlation
        sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        return {
            'all': dict(sorted_corrs[:50]),  # Top 50
            'top_10': dict(sorted_corrs[:10]),
            'strongest_positive': dict([(k, v) for k, v in sorted_corrs if v > 0.1][:10]),
            'strongest_negative': dict([(k, v) for k, v in sorted_corrs if v < -0.1][:10])
        }
    
    def _compute_feature_importance(self, features: np.ndarray, outcomes: np.ndarray,
                                   feature_names: List[str]) -> Dict[str, Any]:
        """Compute feature importance using Random Forest."""
        # Use subset of features if too many
        max_features = min(100, features.shape[1])
        feature_subset = features[:, :max_features]
        names_subset = feature_names[:max_features]
        
        # Remove NaN/inf
        mask = ~(np.isnan(feature_subset).any(axis=1) | np.isinf(feature_subset).any(axis=1))
        feature_subset = feature_subset[mask]
        outcomes_subset = outcomes[mask]
        
        if len(feature_subset) < 100:
            return {'error': 'Insufficient valid samples'}
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(feature_subset, outcomes_subset)
        
        # Get importance
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        top_features = {}
        for i in indices[:20]:  # Top 20
            top_features[names_subset[i]] = float(importances[i])
        
        return {
            'top_features': top_features,
            'mean_importance': float(np.mean(importances)),
            'max_importance': float(np.max(importances))
        }
    
    def _evaluate_models(self, features: np.ndarray, outcomes: np.ndarray) -> Dict[str, Any]:
        """Evaluate predictive models."""
        # Remove NaN/inf
        mask = ~(np.isnan(features).any(axis=1) | np.isinf(features).any(axis=1))
        features_clean = features[mask]
        outcomes_clean = outcomes[mask]
        
        if len(features_clean) < 100:
            return {'error': 'Insufficient valid samples'}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_clean, outcomes_clean, test_size=0.2, random_state=42
        )
        
        results = {}
        
        # Logistic Regression
        try:
            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(X_train, y_train)
            lr_pred = lr.predict(X_test)
            lr_proba = lr.predict_proba(X_test)[:, 1]
            
            results['logistic_regression'] = {
                'accuracy': float(accuracy_score(y_test, lr_pred)),
                'auc': float(roc_auc_score(y_test, lr_proba)) if len(np.unique(y_test)) > 1 else 0.0
            }
        except Exception as e:
            results['logistic_regression'] = {'error': str(e)}
        
        # Random Forest
        try:
            rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_test)
            rf_proba = rf.predict_proba(X_test)[:, 1]
            
            results['random_forest'] = {
                'accuracy': float(accuracy_score(y_test, rf_pred)),
                'auc': float(roc_auc_score(y_test, rf_proba)) if len(np.unique(y_test)) > 1 else 0.0
            }
        except Exception as e:
            results['random_forest'] = {'error': str(e)}
        
        return results
    
    def _analyze_contexts(self, features: np.ndarray, outcomes: np.ndarray,
                         feature_names: List[str]) -> Dict[str, Any]:
        """Analyze performance in different contexts."""
        # This would require game context data
        # For now, return placeholder
        return {
            'rivalry_games': {'note': 'Requires game context data'},
            'playoff_race': {'note': 'Requires game context data'},
            'historic_stadiums': {'note': 'Requires venue data'}
        }
    
    def generate_report(self, results: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """Generate text report from results."""
        report = []
        report.append("="*80)
        report.append("MLB STATISTICAL ANALYSIS REPORT")
        report.append("="*80)
        
        # Univariate correlations
        report.append("\nTOP CORRELATIONS:")
        report.append("-" * 80)
        if 'top_10' in results.get('univariate_correlations', {}):
            for name, corr in list(results['univariate_correlations']['top_10'].items())[:10]:
                report.append(f"  {name}: {corr:.4f}")
        
        # Feature importance
        report.append("\nFEATURE IMPORTANCE (Random Forest):")
        report.append("-" * 80)
        if 'top_features' in results.get('feature_importance', {}):
            for name, importance in list(results['feature_importance']['top_features'].items())[:10]:
                report.append(f"  {name}: {importance:.4f}")
        
        # Model performance
        report.append("\nMODEL PERFORMANCE:")
        report.append("-" * 80)
        for model_name, perf in results.get('model_performance', {}).items():
            if isinstance(perf, dict) and 'accuracy' in perf:
                report.append(f"  {model_name}:")
                report.append(f"    Accuracy: {perf['accuracy']:.4f}")
                if 'auc' in perf:
                    report.append(f"    AUC: {perf['auc']:.4f}")
        
        report.append("\n" + "="*80)
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"\n✓ Report saved to: {output_path}")
        
        return report_text


def main():
    """Test statistical analyzer."""
    import json
    
    # Load genome data
    genome_path = Path(__file__).parent / 'mlb_genome_data.npz'
    
    if not genome_path.exists():
        print(f"Genome data not found: {genome_path}")
        print("Run analyze_mlb_complete.py first")
        return
    
    data = np.load(genome_path, allow_pickle=True)
    features = data['genome']
    outcomes = data['outcomes']
    feature_names = data['feature_names'].tolist()
    
    print(f"Loaded features: {features.shape}")
    print(f"Outcomes: {outcomes.shape}")
    
    # Analyze
    analyzer = MLBStatisticalAnalyzer()
    results = analyzer.analyze_features(features, outcomes, feature_names)
    
    # Generate report
    report_path = Path(__file__).parent / 'mlb_statistical_report.txt'
    report = analyzer.generate_report(results, output_path=str(report_path))
    
    print("\n" + report)


if __name__ == '__main__':
    main()

