"""
Immigration Data Loader

Loads immigration studies and name adaptation data.

Author: Narrative Optimization Research
Date: November 2025
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


class ImmigrationDataLoader:
    """Load immigration identity transformation data."""
    
    def __init__(self, data_dir: Optional[str] = None):
        """Initialize immigration data loader."""
        if data_dir is None:
            data_dir = Path(__file__).parent / 'data'
        else:
            data_dir = Path(data_dir)
        
        self.data_dir = data_dir
        self.studies = None
    
    def load_studies(self) -> List[Dict]:
        """Load immigration studies."""
        if self.studies is None:
            # Load from JSON files
            complete_file = self.data_dir / 'complete_analysis.json'
            
            if complete_file.exists():
                with open(complete_file, 'r') as f:
                    data = json.load(f)
                    
                if isinstance(data, dict):
                    # Extract studies from nested structure
                    self.studies = data.get('studies', [])
                    if not self.studies and 'data' in data:
                        self.studies = data['data']
                else:
                    self.studies = data
            else:
                print("⚠️  No immigration data found, generating synthetic dataset")
                self.studies = self._generate_synthetic_studies()
        
        return self.studies
    
    def _generate_synthetic_studies(self, n: int = 200) -> List[Dict]:
        """Generate synthetic immigration studies."""
        import random
        
        origins = ['Italy', 'Ireland', 'Germany', 'Poland', 'China', 'Mexico', 
                  'India', 'Philippines', 'Vietnam', 'Korea']
        
        name_patterns = {
            'full_adaptation': 0.25,  # Giuseppe → Joe
            'partial_adaptation': 0.40,  # Giuseppe → Joseph
            'minimal_adaptation': 0.20,  # Giuseppe (kept)
            'hybrid': 0.15  # Giuseppe "Joe" Original
        }
        
        studies = []
        
        for i in range(n):
            origin = random.choice(origins)
            pattern = random.choices(
                list(name_patterns.keys()),
                weights=list(name_patterns.values())
            )[0]
            
            # Adaptation degree (0-1)
            adaptation_mapping = {
                'full_adaptation': random.uniform(0.8, 1.0),
                'partial_adaptation': random.uniform(0.4, 0.7),
                'minimal_adaptation': random.uniform(0.1, 0.3),
                'hybrid': random.uniform(0.5, 0.8)
            }
            adaptation_degree = adaptation_mapping[pattern]
            
            # Integration outcome (correlated with adaptation)
            base_integration = 0.60
            adaptation_effect = adaptation_degree * 0.20
            integration = base_integration + adaptation_effect + random.gauss(0, 0.15)
            integration = max(0.2, min(1.0, integration))
            
            study = {
                'id': i + 1,
                'origin_country': origin,
                'adaptation_pattern': pattern,
                'adaptation_degree': adaptation_degree,
                'integration_score': integration,
                'generation': random.choice([1, 2, 3]),
                'year': random.randint(1900, 2020)
            }
            
            studies.append(study)
        
        print(f"✅ Generated {len(studies)} synthetic immigration studies")
        return studies
    
    def get_dataset_statistics(self) -> Dict:
        """Calculate summary statistics."""
        studies = self.load_studies()
        
        if 'adaptation_degree' in studies[0]:
            adaptations = [s.get('adaptation_degree', 0.5) for s in studies]
            integrations = [s.get('integration_score', 0.6) for s in studies]
            
            return {
                'n_studies': len(studies),
                'adaptation': {
                    'mean': np.mean(adaptations),
                    'std': np.std(adaptations)
                },
                'integration': {
                    'mean': np.mean(integrations),
                    'std': np.std(integrations)
                }
            }
        else:
            return {'n_studies': len(studies)}
    
    def generate_data_report(self) -> str:
        """Generate data report."""
        studies = self.load_studies()
        stats = self.get_dataset_statistics()
        
        report = f"""
{'='*70}
IMMIGRATION IDENTITY TRANSFORMATION DATASET
{'='*70}

OVERVIEW
--------
Total Studies: {stats['n_studies']}
"""
        
        if 'adaptation' in stats:
            report += f"""
ADAPTATION PATTERNS
-------------------
Mean Adaptation Degree: {stats['adaptation']['mean']:.2f}
SD: {stats['adaptation']['std']:.2f}

INTEGRATION OUTCOMES
--------------------
Mean Integration Score: {stats['integration']['mean']:.2f}
SD: {stats['integration']['std']:.2f}
"""
        
        report += "\n" + "="*70 + "\n"
        
        return report


if __name__ == '__main__':
    # Demo
    loader = ImmigrationDataLoader()
    print(loader.generate_data_report())

