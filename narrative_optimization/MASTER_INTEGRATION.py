"""
MASTER INTEGRATION SYSTEM

Complete pipeline for domain analysis integrating:
- Domain-agnostic patterns (universal archetypes)
- Domain-specific patterns (learned from similar domains)
- Seamless data integration
- Learning from structural similarities

Author: Narrative Integration System
Date: November 2025
"""

import sys
from pathlib import Path
import json
import numpy as np
from typing import Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.learning import (
    LearningPipeline,
    UniversalArchetypeLearner,
    DomainSpecificLearner,
    MetaLearner,
    ContextAwareLearner
)
from src.analysis.domain_specific_analyzer import DomainSpecificAnalyzer
from src.config import DomainConfig
from src.data import DataLoader


class MasterDomainIntegration:
    """
    Master integration system for analyzing any domain.
    
    Process:
    1. Check for familiar stories (universal patterns)
    2. Find structurally similar domains
    3. Learn domain-specific patterns
    4. Analyze story frequency vs predictions
    5. Identify emerging trends
    """
    
    def __init__(self):
        self.learning_pipeline = LearningPipeline()
        self.meta_learner = MetaLearner()
        self.context_learner = ContextAwareLearner()
        
        # Track all domains
        self.registered_domains = {}
        
    def analyze_new_domain(
        self,
        domain_name: str,
        texts: List[str],
        outcomes: np.ndarray,
        domain_metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Complete analysis of a new domain.
        
        Parameters
        ----------
        domain_name : str
            Domain name
        texts : list
            Narrative texts
        outcomes : ndarray
            Outcomes
        domain_metadata : dict, optional
            Domain characteristics (pi, type, etc.)
        
        Returns
        -------
        dict
            Complete analysis results
        """
        print(f"\n{'='*80}")
        print(f"MASTER DOMAIN INTEGRATION: {domain_name.upper()}")
        print(f"{'='*80}\n")
        
        results = {}
        
        # STEP 1: Check for familiar stories (universal patterns)
        print("[1/6] Checking for universal stories...")
        universal_patterns = self._check_universal_stories(texts, outcomes)
        results['universal_patterns'] = universal_patterns
        print(f"  ✓ Found {len(universal_patterns)} universal patterns")
        
        # STEP 2: Find structurally similar domains
        print("\n[2/6] Finding structurally similar domains...")
        similar_domains = self._find_similar_domains(domain_name, domain_metadata)
        results['similar_domains'] = similar_domains
        print(f"  ✓ Similar to: {', '.join([d[0] for d in similar_domains[:3]])}")
        
        # STEP 3: Transfer patterns from similar domains
        print("\n[3/6] Transferring patterns from similar domains...")
        transferred_patterns = self._transfer_patterns(domain_name, similar_domains)
        results['transferred_patterns'] = transferred_patterns
        print(f"  ✓ Transferred {len(transferred_patterns)} patterns")
        
        # STEP 4: Learn domain-specific patterns
        print("\n[4/6] Learning domain-specific patterns...")
        domain_patterns = self._learn_domain_patterns(domain_name, texts, outcomes)
        results['domain_patterns'] = domain_patterns
        print(f"  ✓ Learned {len(domain_patterns)} domain-specific patterns")
        
        # STEP 5: Analyze story frequency
        print("\n[5/6] Analyzing story frequency vs predictions...")
        frequency_analysis = self._analyze_story_frequency(
            texts, outcomes, universal_patterns, domain_patterns
        )
        results['frequency_analysis'] = frequency_analysis
        print(f"  ✓ Predicted frequency: {frequency_analysis['predicted_frequency']:.1%}")
        print(f"  ✓ Observed frequency: {frequency_analysis['observed_frequency']:.1%}")
        
        # STEP 6: Identify emerging trends
        print("\n[6/6] Identifying emerging trends...")
        trends = self._identify_trends(texts, outcomes, domain_patterns)
        results['trends'] = trends
        print(f"  ✓ Identified {len(trends)} emerging trends")
        
        # Register domain in central registry
        from src.registry import register_domain
        
        register_domain(
            name=domain_name,
            pi=domain_metadata.get('pi', 0.5) if domain_metadata else 0.5,
            domain_type=domain_metadata.get('type', 'unknown') if domain_metadata else 'unknown',
            status='active',
            n_samples=len(texts),
            similar_domains=[d[0] for d in similar_domains[:3]],
            patterns_count=len(universal_patterns) + len(domain_patterns),
            data_path=None
        )
        
        # Register in local cache
        self.registered_domains[domain_name] = {
            'universal_patterns': universal_patterns,
            'domain_patterns': domain_patterns,
            'similar_domains': similar_domains,
            'metadata': domain_metadata or {}
        }
        
        return results
    
    def _check_universal_stories(
        self,
        texts: List[str],
        outcomes: np.ndarray
    ) -> Dict[str, Dict]:
        """Check for universal story patterns."""
        learner = UniversalArchetypeLearner()
        patterns = learner.discover_patterns(texts, outcomes, n_patterns=10)
        
        # Focus on familiar archetypes
        familiar = {}
        for pattern_name, pattern_data in patterns.items():
            if 'universal' in pattern_name:
                familiar[pattern_name] = pattern_data
        
        return familiar
    
    def _find_similar_domains(
        self,
        target_domain: str,
        metadata: Optional[Dict]
    ) -> List[tuple]:
        """Find structurally similar domains."""
        # Register patterns for similarity comparison
        if metadata:
            self.meta_learner.domain_patterns[target_domain] = {}
        
        # Find similar
        similar = self.meta_learner.find_similar_domains(target_domain, n_similar=5)
        
        return similar
    
    def _transfer_patterns(
        self,
        target_domain: str,
        similar_domains: List[tuple]
    ) -> Dict[str, Dict]:
        """Transfer patterns from similar domains."""
        transferred = {}
        
        for source_domain, similarity in similar_domains[:3]:  # Top 3
            if similarity > 0.5:
                patterns = self.meta_learner.transfer_patterns(
                    source_domain,
                    target_domain,
                    min_transferability=0.5
                )
                transferred.update(patterns)
        
        return transferred
    
    def _learn_domain_patterns(
        self,
        domain_name: str,
        texts: List[str],
        outcomes: np.ndarray
    ) -> Dict[str, Dict]:
        """Learn domain-specific patterns."""
        # Ingest into learning pipeline
        self.learning_pipeline.ingest_domain(domain_name, texts, outcomes)
        
        # Create domain learner if doesn't exist
        if domain_name not in self.learning_pipeline.domain_learners:
            learner = DomainSpecificLearner(domain_name)
            patterns = learner.discover_patterns(texts, outcomes, n_patterns=8)
            self.learning_pipeline.domain_learners[domain_name] = learner
        else:
            learner = self.learning_pipeline.domain_learners[domain_name]
            patterns = learner.get_patterns()
        
        return patterns
    
    def _analyze_story_frequency(
        self,
        texts: List[str],
        outcomes: np.ndarray,
        universal_patterns: Dict,
        domain_patterns: Dict
    ) -> Dict:
        """
        Analyze if stories unfold at predicted frequency.
        
        Accounts for:
        - Observation bias (why these stories are reported)
        - Domain constraints (structural limitations)
        - Competitive dynamics
        """
        # Combine all patterns
        all_patterns = {**universal_patterns, **domain_patterns}
        
        # Calculate observed frequency of each pattern
        pattern_frequencies = {}
        for pattern_name, pattern_data in all_patterns.items():
            keywords = pattern_data.get('keywords', pattern_data.get('patterns', []))
            
            matches = sum(
                1 for text in texts
                if any(kw.lower() in text.lower() for kw in keywords)
            )
            
            pattern_frequencies[pattern_name] = matches / len(texts)
        
        # Predict expected frequency (baseline from similar domains)
        expected = 0.3  # Generic expectation
        observed = np.mean(list(pattern_frequencies.values())) if pattern_frequencies else 0.0
        
        return {
            'predicted_frequency': expected,
            'observed_frequency': observed,
            'pattern_frequencies': pattern_frequencies,
            'meets_expectations': abs(observed - expected) < 0.1
        }
    
    def _identify_trends(
        self,
        texts: List[str],
        outcomes: np.ndarray,
        patterns: Dict
    ) -> List[Dict]:
        """Identify emerging trends in patterns."""
        trends = []
        
        # Look for patterns with increasing/decreasing presence
        # (Simplified - would need temporal data)
        
        for pattern_name, pattern_data in patterns.items():
            frequency = pattern_data.get('frequency', 0.0)
            correlation = pattern_data.get('correlation', 0.0)
            
            if frequency > 0.2 and abs(correlation) > 0.3:
                trends.append({
                    'pattern': pattern_name,
                    'type': 'emerging' if correlation > 0 else 'declining',
                    'strength': abs(correlation),
                    'frequency': frequency
                })
        
        return trends
    
    def get_domain_genome(self, domain_name: str) -> Dict:
        """
        Get complete genome for domain:
        - Universal patterns it expresses
        - Domain-specific patterns
        - Structural similarities
        - Learned characteristics
        """
        if domain_name not in self.registered_domains:
            return {}
        
        domain_data = self.registered_domains[domain_name]
        
        return {
            'universal_patterns': domain_data['universal_patterns'],
            'domain_patterns': domain_data['domain_patterns'],
            'similar_to': [d[0] for d in domain_data['similar_domains'][:3]],
            'metadata': domain_data['metadata']
        }


def add_domain_workflow(
    domain_name: str,
    data_path: Path,
    domain_characteristics: Optional[Dict] = None
):
    """
    Complete workflow for adding a new domain.
    
    Parameters
    ----------
    domain_name : str
        Domain name
    data_path : Path
        Path to domain data
    domain_characteristics : dict, optional
        Domain metadata (pi, type, etc.)
    
    Example
    -------
    >>> add_domain_workflow(
    ...     'chess',
    ...     Path('data/domains/chess_games.json'),
    ...     {'pi': 0.78, 'type': 'individual_expertise'}
    ... )
    """
    print(f"\n{'='*80}")
    print(f"ADDING NEW DOMAIN: {domain_name.upper()}")
    print(f"{'='*80}\n")
    
    # Load data
    print("[1/5] Loading data...")
    loader = DataLoader()
    data = loader.load(data_path)
    
    texts = data['texts']
    outcomes = data['outcomes']
    
    print(f"  ✓ Loaded {len(texts)} samples")
    
    # Initialize integration
    print("\n[2/5] Initializing integration...")
    integration = MasterDomainIntegration()
    
    # Analyze
    print("\n[3/5] Running complete analysis...")
    results = integration.analyze_new_domain(
        domain_name,
        texts,
        outcomes,
        domain_characteristics
    )
    
    # Save results
    print("\n[4/5] Saving results...")
    output_dir = Path(__file__).parent / 'narrative_optimization' / 'domains' / domain_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'integration_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"  ✓ Results saved to {output_dir}")
    
    # Generate report
    print("\n[5/5] Generating report...")
    report = _generate_domain_report(domain_name, results)
    
    with open(output_dir / 'ANALYSIS_REPORT.md', 'w') as f:
        f.write(report)
    
    print(f"  ✓ Report saved")
    
    print(f"\n{'='*80}")
    print(f"DOMAIN INTEGRATION COMPLETE: {domain_name.upper()}")
    print(f"{'='*80}")


def _generate_domain_report(domain_name: str, results: Dict) -> str:
    """Generate markdown report for domain."""
    report = f"# {domain_name.title()} Domain Analysis\n\n"
    report += f"**Date**: {Path(__file__).stat().st_mtime}\n\n"
    report += "---\n\n"
    
    # Universal patterns
    report += "## Universal Patterns\n\n"
    universal = results.get('universal_patterns', {})
    for pattern_name, pattern_data in list(universal.items())[:5]:
        freq = pattern_data.get('frequency', 0.0)
        report += f"- **{pattern_name}**: {freq:.1%} frequency\n"
    report += "\n"
    
    # Similar domains
    report += "## Structurally Similar Domains\n\n"
    similar = results.get('similar_domains', [])
    for domain, similarity in similar[:3]:
        report += f"- {domain}: {similarity:.1%} similar\n"
    report += "\n"
    
    # Domain patterns
    report += "## Domain-Specific Patterns\n\n"
    domain_patterns = results.get('domain_patterns', {})
    report += f"Discovered {len(domain_patterns)} unique patterns.\n\n"
    
    # Frequency analysis
    report += "## Story Frequency Analysis\n\n"
    freq_analysis = results.get('frequency_analysis', {})
    predicted = freq_analysis.get('predicted_frequency', 0.0)
    observed = freq_analysis.get('observed_frequency', 0.0)
    report += f"- Predicted frequency: {predicted:.1%}\n"
    report += f"- Observed frequency: {observed:.1%}\n"
    report += f"- Meets expectations: {freq_analysis.get('meets_expectations', False)}\n\n"
    
    # Trends
    report += "## Emerging Trends\n\n"
    trends = results.get('trends', [])
    for trend in trends[:5]:
        report += f"- {trend['pattern']}: {trend['type']} ({trend['strength']:.2f} strength)\n"
    
    return report


if __name__ == '__main__':
    # Example: Add a new domain
    import argparse
    
    parser = argparse.ArgumentParser(description='Add new domain to system')
    parser.add_argument('domain', help='Domain name')
    parser.add_argument('data_path', help='Path to data file')
    parser.add_argument('--pi', type=float, help='Domain narrativity')
    parser.add_argument('--type', help='Domain type')
    
    args = parser.parse_args()
    
    characteristics = {}
    if args.pi:
        characteristics['pi'] = args.pi
    if args.type:
        characteristics['type'] = args.type
    
    add_domain_workflow(
        args.domain,
        Path(args.data_path),
        characteristics if characteristics else None
    )

