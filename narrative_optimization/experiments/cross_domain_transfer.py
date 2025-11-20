"""
Cross-Domain Transfer Experiment

Tests if archetype patterns learned from one domain transfer to another.

Key questions:
- Train on mythology → Predict literature success?
- Train on Hollywood → Predict indie film success?
- Which patterns are universal vs domain-specific?

Author: Narrative Optimization Framework
Date: November 13, 2025
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.archetype_cross_domain import ArchetypeCrossDomainAnalyzer


class CrossDomainTransferExperiment:
    """
    Test transfer learning across narrative domains.
    """
    
    def __init__(self):
        self.analyzer = ArchetypeCrossDomainAnalyzer()
        self.results = {}
    
    def load_domains(self) -> Dict:
        """Load all available domain datasets."""
        domains = {}
        
        # Try to load each domain
        data_dir = Path('data/domains')
        
        for domain_path in data_dir.iterdir():
            if domain_path.is_dir():
                dataset_file = domain_path / f'{domain_path.name}_complete_dataset.json'
                
                if dataset_file.exists():
                    with open(dataset_file) as f:
                        data = json.load(f)
                    
                    # Extract texts and outcomes (domain-specific)
                    texts, outcomes = self._extract_texts_outcomes(data, domain_path.name)
                    
                    if texts and outcomes is not None:
                        self.analyzer.load_domain_data(domain_path.name, texts, outcomes)
                        domains[domain_path.name] = {'texts': texts, 'outcomes': outcomes}
                        print(f"✅ Loaded {domain_path.name}: {len(texts)} samples")
        
        return domains
    
    def _extract_texts_outcomes(self, data: Dict, domain: str) -> Tuple[List[str], np.ndarray]:
        """Extract texts and outcomes from domain-specific format."""
        # Domain-specific extraction logic
        if domain == 'mythology':
            if 'myths' in data:
                texts = [m.get('full_narrative', '') for m in data['myths']]
                outcomes = np.array([m['outcome_measures']['cultural_persistence_score'] 
                                    for m in data['myths']])
                return texts, outcomes
        
        # Add other domains as needed
        return [], None
    
    def experiment_1_mythology_to_literature(self) -> Dict:
        """
        EXPERIMENT 1: Train on mythology, predict literature success.
        
        Tests: Are mythological patterns universal enough to predict literary success?
        """
        print("\n" + "="*70)
        print("EXPERIMENT 1: Mythology → Literature Transfer")
        print("="*70)
        
        result = self.analyzer.test_cross_domain_transfer('mythology', 'classical_literature')
        
        if result['overall']['transfer_success']:
            print(f"✅ Transfer successful (R²={result['overall']['mean_r2']:.3f})")
        else:
            print(f"❌ Transfer failed (R²={result['overall']['mean_r2']:.3f})")
        
        return {
            'source': 'mythology',
            'target': 'classical_literature',
            'r2_scores': result,
            'interpretation': 'Mythological patterns predict literature' if result['overall']['transfer_success']
                             else 'Mythology too different from literature'
        }
    
    def experiment_2_film_to_literature(self) -> Dict:
        """
        EXPERIMENT 2: Train on film, predict literature.
        
        Tests: Do cinematic patterns transfer to books?
        """
        print("\n" + "="*70)
        print("EXPERIMENT 2: Film → Literature Transfer")
        print("="*70)
        
        result = self.analyzer.test_cross_domain_transfer('film_extended', 'classical_literature')
        
        return {
            'source': 'film',
            'target': 'literature',
            'r2_scores': result,
            'expected': 'Moderate transfer (R² ≈ 0.40-0.50)',
            'interpretation': 'Different medium effects'
        }
    
    def experiment_3_identify_universal_patterns(self) -> Dict:
        """
        EXPERIMENT 3: Identify patterns that transfer across ALL domains.
        
        Universal patterns: Work everywhere
        Domain-specific: Work only in specific contexts
        """
        print("\n" + "="*70)
        print("EXPERIMENT 3: Universal Pattern Identification")
        print("="*70)
        
        universal = self.analyzer.identify_universal_patterns()
        
        print(f"\n✅ Found {len(universal['universal_patterns'])} universal patterns")
        print("Top 5 universal patterns:")
        for pattern in universal['universal_patterns'][:5]:
            print(f"  - {pattern['pattern']}: importance={pattern['mean_importance']:.3f}")
        
        print(f"\n✅ Found {len(universal['domain_specific_patterns'])} domain-specific patterns")
        
        return universal
    
    def run_all_experiments(self) -> Dict:
        """Run all cross-domain transfer experiments."""
        print("\n" + "="*70)
        print("CROSS-DOMAIN TRANSFER EXPERIMENTS")
        print("="*70)
        
        # Load domains
        domains = self.load_domains()
        
        if len(domains) < 2:
            print("\n⚠️  Need at least 2 domains for transfer experiments")
            return {'status': 'insufficient_data'}
        
        # Run experiments
        results = {}
        
        if 'mythology' in domains and 'classical_literature' in domains:
            results['exp1'] = self.experiment_1_mythology_to_literature()
        
        if 'film_extended' in domains and 'classical_literature' in domains:
            results['exp2'] = self.experiment_2_film_to_literature()
        
        if len(domains) >= 3:
            results['exp3'] = self.experiment_3_identify_universal_patterns()
        
        # Save results
        output_file = Path('narrative_optimization/results/cross_domain_transfer_results.json')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📊 Results saved: {output_file}")
        
        return results


def main():
    """Run cross-domain transfer experiments."""
    experiment = CrossDomainTransferExperiment()
    results = experiment.run_all_experiments()
    return results


if __name__ == '__main__':
    main()

