"""
Marriage Data Loader

Loads couples data with compatibility metrics:
- Partner names (phonetic and semantic features)
- Relationship duration
- Relative success (vs cohort baseline)

Author: Narrative Optimization Research
Date: November 2025
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


class MarriageDataLoader:
    """
    Load and manage marriage compatibility data.
    
    Data includes:
    - Partner names
    - Marriage duration
    - Relative success scores
    - Compatibility metrics
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize marriage data loader.
        
        Parameters
        ----------
        data_dir : str, optional
            Path to data directory
        """
        if data_dir is None:
            data_dir = Path(__file__).parent / 'data'
        else:
            data_dir = Path(data_dir)
        
        self.data_dir = data_dir
        self.couples = None
    
    def load_couples(self) -> List[Dict]:
        """
        Load couples dataset.
        
        Returns
        -------
        list of dict
            Couple records
        """
        if self.couples is None:
            # Try CSV first
            csv_files = list(self.data_dir.glob('*couples*.csv')) + \
                       list(self.data_dir.glob('*substantiated*.csv'))
            
            if csv_files:
                self.couples = self._load_from_csv(csv_files[0])
            else:
                # Try JSON
                json_files = list(self.data_dir.glob('*analysis*.json'))
                if json_files:
                    self.couples = self._load_from_json(json_files[0])
                else:
                    print("⚠️  No couples data found, generating synthetic dataset")
                    self.couples = self._generate_synthetic_couples()
        
        return self.couples
    
    def _load_from_csv(self, csv_file: Path) -> List[Dict]:
        """Load couples from CSV file."""
        couples = []
        
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                couples.append(dict(row))
        
        return couples
    
    def _load_from_json(self, json_file: Path) -> List[Dict]:
        """Load couples from JSON file."""
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract couples if nested
        if isinstance(data, dict) and 'couples' in data:
            return data['couples']
        elif isinstance(data, list):
            return data
        else:
            return []
    
    def _generate_synthetic_couples(self, n: int = 500) -> List[Dict]:
        """Generate synthetic couples data for demonstration."""
        import random
        
        first_names_m = ['James', 'John', 'Robert', 'Michael', 'David', 'William',
                        'Richard', 'Joseph', 'Thomas', 'Christopher']
        first_names_f = ['Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth',
                        'Barbara', 'Susan', 'Jessica', 'Sarah', 'Karen']
        
        couples = []
        
        for i in range(n):
            name1 = random.choice(first_names_m)
            name2 = random.choice(first_names_f)
            
            # Calculate simple compatibility metrics
            syllables1 = self._count_syllables(name1)
            syllables2 = self._count_syllables(name2)
            
            # Similarity theory: similar syllables → compatibility
            similarity = 1.0 - abs(syllables1 - syllables2) / max(syllables1, syllables2)
            
            # Golden ratio theory: ratio close to φ → harmony
            phi = 1.618
            ratio = max(syllables1, syllables2) / min(syllables1, syllables2)
            phi_distance = abs(ratio - phi) / phi
            phi_score = 1.0 - phi_distance
            
            # Base relative success
            base_success = 1.0  # 100% of baseline
            
            # Add small effect from compatibility
            similarity_effect = similarity * 0.10  # Up to 10% boost
            phi_effect = phi_score * 0.08  # Up to 8% boost
            
            relative_success = base_success + similarity_effect + phi_effect
            relative_success += random.gauss(0, 0.15)  # Add noise
            relative_success = max(0.5, min(1.8, relative_success))
            
            couple = {
                'id': i + 1,
                'name1': name1,
                'name2': name2,
                'syllables1': syllables1,
                'syllables2': syllables2,
                'similarity': similarity,
                'phi_score': phi_score,
                'relative_success': relative_success,
                'duration_years': int(relative_success * 25 * random.uniform(0.8, 1.2))
            }
            
            couples.append(couple)
        
        print(f"✅ Generated {len(couples)} synthetic couples")
        return couples
    
    def _count_syllables(self, name: str) -> int:
        """Simple syllable counter."""
        import re
        name = name.lower()
        name = re.sub(r'e$', '', name)  # Remove silent e
        vowels = 'aeiouy'
        syllables = 0
        previous_was_vowel = False
        
        for char in name:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllables += 1
            previous_was_vowel = is_vowel
        
        return max(1, syllables)
    
    def get_dataset_statistics(self) -> Dict:
        """Calculate summary statistics."""
        couples = self.load_couples()
        
        if 'relative_success' in couples[0]:
            successes = [c.get('relative_success', 1.0) for c in couples]
            
            return {
                'n_couples': len(couples),
                'relative_success': {
                    'mean': np.mean(successes),
                    'std': np.std(successes),
                    'range': (min(successes), max(successes))
                }
            }
        else:
            return {'n_couples': len(couples)}
    
    def generate_data_report(self) -> str:
        """Generate data report."""
        couples = self.load_couples()
        stats = self.get_dataset_statistics()
        
        report = f"""
{'='*70}
MARRIAGE COMPATIBILITY DATASET
{'='*70}

OVERVIEW
--------
Total Couples: {stats['n_couples']}
"""
        
        if 'relative_success' in stats:
            rs = stats['relative_success']
            report += f"""
RELATIVE SUCCESS
----------------
Mean: {rs['mean']:.2f} (vs 1.00 baseline)
SD: {rs['std']:.2f}
Range: {rs['range'][0]:.2f} - {rs['range'][1]:.2f}
"""
        
        report += "\n" + "="*70 + "\n"
        
        return report


if __name__ == '__main__':
    # Demo
    loader = MarriageDataLoader()
    print(loader.generate_data_report())

