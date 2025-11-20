"""
Comparative Domain Analysis

Compares multiple domains to find patterns and relationships.

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import pandas as pd
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity


class ComparativeAnalyzer:
    """
    Compare multiple domains across dimensions.
    
    Analyses:
    - Performance comparison (R², Д)
    - Pattern similarity
    - Structural relationships
    - Universal vs domain-specific patterns
    """
    
    def __init__(self):
        self.domain_results = {}
        
    def add_domain(self, domain_name: str, results: Dict):
        """Add domain results for comparison."""
        self.domain_results[domain_name] = results
    
    def compare_performance(self) -> pd.DataFrame:
        """
        Compare performance metrics across domains.
        
        Returns
        -------
        DataFrame
            Comparison table
        """
        data = []
        
        for domain, results in self.domain_results.items():
            data.append({
                'domain': domain,
                'pi': results.get('narrativity', results.get('pi', 0)),
                'r_squared': results.get('r_squared', 0),
                'delta': results.get('delta', 0),
                'efficiency': results.get('efficiency', 0),
                'n_samples': len(results.get('texts', results.get('genomes', [])))
            })
        
        df = pd.DataFrame(data)
        
        # Sort by R²
        df = df.sort_values('r_squared', ascending=False)
        
        return df
    
    def find_performance_clusters(self, n_clusters: int = 3) -> Dict:
        """
        Cluster domains by performance characteristics.
        
        Parameters
        ----------
        n_clusters : int
            Number of clusters
        
        Returns
        -------
        dict
            Cluster assignments
        """
        from sklearn.cluster import KMeans
        
        # Create feature matrix
        features = []
        domain_names = []
        
        for domain, results in self.domain_results.items():
            features.append([
                results.get('narrativity', results.get('pi', 0)),
                results.get('r_squared', 0),
                results.get('delta', 0),
                results.get('efficiency', 0)
            ])
            domain_names.append(domain)
        
        features = np.array(features)
        
        # Normalize
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Cluster
        kmeans = KMeans(n_clusters=min(n_clusters, len(domain_names)), random_state=42)
        labels = kmeans.fit_predict(features_scaled)
        
        # Group by cluster
        clusters = {}
        for domain, label in zip(domain_names, labels):
            cluster_id = int(label)
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(domain)
        
        return clusters
    
    def identify_outliers(self, metric: str = 'r_squared', threshold: float = 2.0) -> List[str]:
        """
        Identify outlier domains.
        
        Parameters
        ----------
        metric : str
            Metric to check
        threshold : float
            Z-score threshold
        
        Returns
        -------
        list
            Outlier domain names
        """
        values = []
        domain_names = []
        
        for domain, results in self.domain_results.items():
            value = results.get(metric, 0)
            values.append(value)
            domain_names.append(domain)
        
        values = np.array(values)
        
        # Calculate z-scores
        z_scores = np.abs(stats.zscore(values))
        
        # Find outliers
        outliers = [
            domain_names[i]
            for i in range(len(domain_names))
            if z_scores[i] > threshold
        ]
        
        return outliers
    
    def correlation_matrix(self) -> pd.DataFrame:
        """
        Create correlation matrix of domain characteristics.
        
        Returns
        -------
        DataFrame
            Correlation matrix
        """
        # Create feature matrix
        features_dict = {}
        
        for domain, results in self.domain_results.items():
            features_dict[domain] = [
                results.get('narrativity', 0),
                results.get('r_squared', 0),
                results.get('delta', 0),
                results.get('efficiency', 0)
            ]
        
        df = pd.DataFrame(features_dict).T
        df.columns = ['pi', 'r_squared', 'delta', 'efficiency']
        
        return df.corr()
    
    def find_similar_by_performance(
        self,
        target_domain: str,
        n_similar: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Find domains with similar performance profile.
        
        Parameters
        ----------
        target_domain : str
            Target domain
        n_similar : int
            Number of similar domains
        
        Returns
        -------
        list of (domain, similarity)
        """
        if target_domain not in self.domain_results:
            return []
        
        target_features = np.array([
            self.domain_results[target_domain].get('narrativity', 0),
            self.domain_results[target_domain].get('r_squared', 0),
            self.domain_results[target_domain].get('delta', 0),
            self.domain_results[target_domain].get('efficiency', 0)
        ]).reshape(1, -1)
        
        similarities = []
        
        for domain, results in self.domain_results.items():
            if domain == target_domain:
                continue
            
            other_features = np.array([
                results.get('narrativity', 0),
                results.get('r_squared', 0),
                results.get('delta', 0),
                results.get('efficiency', 0)
            ]).reshape(1, -1)
            
            # Cosine similarity
            sim = cosine_similarity(target_features, other_features)[0, 0]
            similarities.append((domain, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n_similar]
    
    def generate_comparative_report(self) -> str:
        """Generate comparative analysis report."""
        report = "# Comparative Domain Analysis\n\n"
        
        # Performance comparison
        report += "## Performance Comparison\n\n"
        df = self.compare_performance()
        report += df.to_markdown() + "\n\n"
        
        # Clusters
        report += "## Performance Clusters\n\n"
        clusters = self.find_performance_clusters(n_clusters=3)
        for cluster_id, domains in clusters.items():
            report += f"**Cluster {cluster_id + 1}**: {', '.join(domains)}\n"
        report += "\n"
        
        # Outliers
        report += "## Outliers\n\n"
        outliers = self.identify_outliers('r_squared')
        if outliers:
            report += f"High/low R² domains: {', '.join(outliers)}\n"
        else:
            report += "No significant outliers detected.\n"
        report += "\n"
        
        # Correlations
        report += "## Characteristic Correlations\n\n"
        corr_matrix = self.correlation_matrix()
        report += corr_matrix.to_markdown() + "\n"
        
        return report


def batch_analyze_and_compare():
    """Batch analyze and generate comparison."""
    from src.registry import get_domain_registry
    from src.pipeline_config import get_config
    
    print("="*80)
    print("BATCH ANALYSIS WITH COMPARISON")
    print("="*80)
    
    registry = get_domain_registry()
    config = get_config()
    
    domains = [d.name for d in registry.get_all_domains()]
    
    print(f"\nAnalyzing {len(domains)} registered domains...\n")
    
    # Load and analyze
    comparator = ComparativeAnalyzer()
    loader = DataLoader()
    
    for domain in domains:
        print(f"  {domain}...", end=" ", flush=True)
        
        data_path = config.get_domain_data_path(domain)
        
        if data_path and data_path.exists():
            result = analyze_single_domain(domain, data_path)
            
            if result['status'] == 'success':
                comparator.add_domain(domain, result)
                print(f"✓ R²={result['r_squared']:.1%}")
            else:
                print(f"✗ {result.get('error', 'Failed')}")
        else:
            print("⊙ No data")
    
    # Generate comparison
    print(f"\n{'='*80}")
    print("GENERATING COMPARATIVE REPORT")
    print(f"{'='*80}\n")
    
    report = comparator.generate_comparative_report()
    
    # Save report
    report_path = Path(__file__).parent.parent / 'COMPARATIVE_ANALYSIS.md'
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"  ✓ Report saved: {report_path}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    df = comparator.compare_performance()
    print(df[['domain', 'pi', 'r_squared', 'delta']].head(10).to_string(index=False))


if __name__ == '__main__':
    batch_analyze_and_compare()

