"""
Mental Health Data Loader

Loads and structures mental health nomenclature data including:
- 510 disorder names with phonetic analysis
- Medication names
- Therapy modality names

Author: Narrative Optimization Research  
Date: November 2025
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


class MentalHealthDataLoader:
    """
    Load and manage mental health nomenclature datasets.
    
    Data includes:
    - Disorder names (DSM/ICD codes, phonetics, clinical outcomes)
    - Medication names
    - Therapy modality names
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize data loader.
        
        Parameters
        ----------
        data_dir : str, optional
            Path to data directory. If None, uses default location.
        """
        if data_dir is None:
            data_dir = Path(__file__).parent / 'data'
        else:
            data_dir = Path(data_dir)
        
        self.data_dir = data_dir
        self.disorders = None
        self.medications = None
        self.therapies = None
    
    def load_disorders(self) -> List[Dict]:
        """
        Load disorder names database.
        
        Returns
        -------
        list of dict
            Disorder records with full metadata
        """
        if self.disorders is None:
            disorder_file = self.data_dir / 'disorder_names_database.json'
            
            with open(disorder_file, 'r') as f:
                data = json.load(f)
            
            # Extract disorder lists
            if 'high_severity_disorders' in data:
                self.disorders = data['high_severity_disorders']
            elif isinstance(data, list):
                self.disorders = data
            else:
                raise ValueError("Unexpected data structure in disorder database")
        
        return self.disorders
    
    def load_medications(self) -> List[Dict]:
        """
        Load medication names database.
        
        Returns
        -------
        list of dict
            Medication records
        """
        if self.medications is None:
            med_file = self.data_dir / 'medication_names_database.json'
            
            if med_file.exists():
                with open(med_file, 'r') as f:
                    self.medications = json.load(f)
            else:
                self.medications = []
        
        return self.medications
    
    def load_therapies(self) -> List[Dict]:
        """
        Load therapy modality names.
        
        Returns
        -------
        list of dict
            Therapy modality records
        """
        if self.therapies is None:
            therapy_file = self.data_dir / 'therapy_names_database.json'
            
            if therapy_file.exists():
                with open(therapy_file, 'r') as f:
                    self.therapies = json.load(f)
            else:
                self.therapies = []
        
        return self.therapies
    
    def get_disorder_by_name(self, disorder_name: str) -> Optional[Dict]:
        """
        Get specific disorder by name.
        
        Parameters
        ----------
        disorder_name : str
            Name of disorder
        
        Returns
        -------
        dict or None
            Disorder record if found
        """
        disorders = self.load_disorders()
        
        for disorder in disorders:
            if disorder.get('disorder_name', '').lower() == disorder_name.lower():
                return disorder
        
        return None
    
    def get_disorders_by_severity(self, severity_category: str) -> List[Dict]:
        """
        Get disorders by severity category.
        
        Parameters
        ----------
        severity_category : str
            'high', 'medium', or 'low'
        
        Returns
        -------
        list of dict
            Matching disorder records
        """
        disorders = self.load_disorders()
        # This would need severity categorization in the data
        # For now, return all
        return disorders
    
    def get_disorders_by_stigma_range(self, min_stigma: float,
                                     max_stigma: float) -> List[Dict]:
        """
        Get disorders within stigma score range.
        
        Parameters
        ----------
        min_stigma : float
            Minimum stigma score (0-10)
        max_stigma : float
            Maximum stigma score (0-10)
        
        Returns
        -------
        list of dict
            Matching disorders
        """
        disorders = self.load_disorders()
        
        matching = []
        for disorder in disorders:
            stigma = disorder.get('social_impact', {}).get('stigma_score')
            if stigma is not None and min_stigma <= stigma <= max_stigma:
                matching.append(disorder)
        
        return matching
    
    def get_dataset_statistics(self) -> Dict:
        """
        Calculate summary statistics for dataset.
        
        Returns
        -------
        dict
            Summary statistics
        """
        disorders = self.load_disorders()
        
        # Extract phonetic features
        harshness_scores = []
        stigma_scores = []
        mortality_rates = []
        
        for disorder in disorders:
            phonetic = disorder.get('phonetic_analysis', {})
            social = disorder.get('social_impact', {})
            clinical = disorder.get('clinical_outcomes', {})
            
            if 'harshness_score' in phonetic:
                harshness_scores.append(phonetic['harshness_score'])
            
            if 'stigma_score' in social:
                stigma_scores.append(social['stigma_score'])
            
            if 'mortality_rate_per_100k' in clinical:
                mortality_rates.append(clinical['mortality_rate_per_100k'])
        
        stats = {
            'n_disorders': len(disorders),
            'phonetic_harshness': {
                'mean': np.mean(harshness_scores) if harshness_scores else None,
                'std': np.std(harshness_scores) if harshness_scores else None,
                'range': (min(harshness_scores), max(harshness_scores)) if harshness_scores else None
            },
            'stigma': {
                'mean': np.mean(stigma_scores) if stigma_scores else None,
                'std': np.std(stigma_scores) if stigma_scores else None,
                'range': (min(stigma_scores), max(stigma_scores)) if stigma_scores else None
            },
            'mortality': {
                'mean': np.mean(mortality_rates) if mortality_rates else None,
                'median': np.median(mortality_rates) if mortality_rates else None,
                'range': (min(mortality_rates), max(mortality_rates)) if mortality_rates else None
            }
        }
        
        return stats
    
    def export_for_analysis(self, output_file: str):
        """
        Export processed dataset for analysis.
        
        Parameters
        ----------
        output_file : str
            Output file path (JSON)
        """
        disorders = self.load_disorders()
        
        # Create simplified format for analysis
        simplified = []
        
        for disorder in disorders:
            simplified.append({
                'name': disorder.get('disorder_name'),
                'dsm_code': disorder.get('dsm_code'),
                'icd10': disorder.get('icd10'),
                'phonetic_harshness': disorder.get('phonetic_analysis', {}).get('harshness_score'),
                'syllables': disorder.get('phonetic_analysis', {}).get('syllables'),
                'stigma_score': disorder.get('social_impact', {}).get('stigma_score'),
                'treatment_seeking_rate': disorder.get('clinical_outcomes', {}).get('treatment_seeking_rate'),
                'mortality_rate': disorder.get('clinical_outcomes', {}).get('mortality_rate_per_100k')
            })
        
        with open(output_file, 'w') as f:
            json.dump(simplified, f, indent=2)
    
    def generate_data_report(self) -> str:
        """
        Generate comprehensive data report.
        
        Returns
        -------
        str
            Formatted report
        """
        disorders = self.load_disorders()
        stats = self.get_dataset_statistics()
        
        report = f"""
{'='*70}
MENTAL HEALTH NOMENCLATURE DATASET
{'='*70}

OVERVIEW
--------
Total Disorders: {stats['n_disorders']}

PHONETIC ANALYSIS
-----------------
Harshness Score:
  Mean: {stats['phonetic_harshness']['mean']:.1f}
  SD: {stats['phonetic_harshness']['std']:.1f}
  Range: {stats['phonetic_harshness']['range'][0]:.0f}-{stats['phonetic_harshness']['range'][1]:.0f}

STIGMA SCORES
-------------
Stigma (0-10 scale):
  Mean: {stats['stigma']['mean']:.2f}
  SD: {stats['stigma']['std']:.2f}
  Range: {stats['stigma']['range'][0]:.1f}-{stats['stigma']['range'][1]:.1f}

CLINICAL OUTCOMES
-----------------
Mortality Rate (per 100k):
  Mean: {stats['mortality']['mean']:.1f}
  Median: {stats['mortality']['median']:.1f}
  Range: {stats['mortality']['range'][0]:.0f}-{stats['mortality']['range'][1]:.0f}

SAMPLE DISORDERS
----------------
"""
        
        # Show first 5 disorders
        for disorder in disorders[:5]:
            name = disorder.get('disorder_name', 'Unknown')
            harshness = disorder.get('phonetic_analysis', {}).get('harshness_score', 'N/A')
            stigma = disorder.get('social_impact', {}).get('stigma_score', 'N/A')
            
            report += f"  {name}\n"
            report += f"    Harshness: {harshness}\n"
            report += f"    Stigma: {stigma}\n\n"
        
        report += "="*70 + "\n"
        
        return report


if __name__ == '__main__':
    # Demo usage
    loader = MentalHealthDataLoader()
    print(loader.generate_data_report())

