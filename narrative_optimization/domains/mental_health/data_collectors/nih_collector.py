"""
NIH Funding Data Collector

Collects research funding data from NIH RePORTER API for mental health disorders.

API Documentation: https://api.reporter.nih.gov/

Author: Narrative Optimization Research
Date: November 2025
"""

import requests
import time
from typing import Dict, List, Optional
import json


class NIHFundingCollector:
    """
    Collect NIH funding data for mental health disorders.
    
    Uses NIH RePORTER v2 API to search grants by disorder name.
    """
    
    API_BASE = "https://api.reporter.nih.gov/v2/projects/search"
    
    def __init__(self, rate_limit_delay: float = 0.5):
        """
        Initialize NIH funding collector.
        
        Parameters
        ----------
        rate_limit_delay : float
            Seconds to wait between API calls (respect rate limits)
        """
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
    
    def collect_funding_for_disorder(self, disorder_name: str,
                                    fiscal_years: List[int] = None) -> Dict:
        """
        Collect NIH funding for a specific disorder.
        
        Parameters
        ----------
        disorder_name : str
            Name of mental health disorder
        fiscal_years : list of int, optional
            Fiscal years to query (default: 2020-2024)
        
        Returns
        -------
        dict
            Funding statistics
        """
        if fiscal_years is None:
            fiscal_years = [2020, 2021, 2022, 2023, 2024]
        
        try:
            # Build search criteria
            criteria = {
                "advanced_text_search": {
                    "operator": "advanced",
                    "search_field": "all",
                    "search_text": disorder_name
                },
                "fiscal_years": fiscal_years
            }
            
            payload = {
                "criteria": criteria,
                "include_fields": [
                    "ProjectTitle",
                    "AwardAmount",
                    "FiscalYear",
                    "Organization"
                ],
                "limit": 500,
                "offset": 0
            }
            
            response = self.session.post(self.API_BASE, json=payload, timeout=30)
            
            # Respect rate limits
            time.sleep(self.rate_limit_delay)
            
            if response.status_code != 200:
                return {
                    'disorder_name': disorder_name,
                    'error': f'API returned status {response.status_code}',
                    'total_funding_millions': 0,
                    'n_grants': 0
                }
            
            data = response.json()
            results = data.get('results', [])
            
            # Calculate total funding (handle None values from API)
            total_funding = sum(
                r.get('award_amount', 0) or 0 for r in results
            )
            
            # Convert to millions
            total_funding_millions = total_funding / 1_000_000
            
            return {
                'disorder_name': disorder_name,
                'total_funding_millions': round(total_funding_millions, 2),
                'n_grants': len(results),
                'fiscal_years': fiscal_years,
                'mean_grant_size_millions': round(total_funding_millions / len(results), 3) if results else 0
            }
            
        except requests.exceptions.Timeout:
            return {
                'disorder_name': disorder_name,
                'error': 'API timeout',
                'total_funding_millions': 0,
                'n_grants': 0
            }
        except Exception as e:
            return {
                'disorder_name': disorder_name,
                'error': str(e),
                'total_funding_millions': 0,
                'n_grants': 0
            }
    
    def collect_funding_for_all_disorders(self, disorders: List[Dict],
                                         save_progress: bool = True,
                                         progress_file: str = 'nih_funding_progress.json') -> List[Dict]:
        """
        Collect funding for all disorders with progress saving.
        
        Parameters
        ----------
        disorders : list of dict
            Disorder records with names
        save_progress : bool
            Save progress after each disorder
        progress_file : str
            File to save progress to
        
        Returns
        -------
        list of dict
            Funding data for all disorders
        """
        results = []
        total = len(disorders)
        
        print(f"Collecting NIH funding data for {total} disorders...")
        print("This may take 5-10 minutes due to API rate limits.\n")
        
        for i, disorder in enumerate(disorders, 1):
            disorder_name = disorder.get('disorder_name', disorder.get('name', ''))
            
            if not disorder_name:
                continue
            
            print(f"[{i}/{total}] Querying: {disorder_name}...")
            
            funding_data = self.collect_funding_for_disorder(disorder_name)
            
            # Add to disorder record
            disorder_with_funding = disorder.copy()
            disorder_with_funding['nih_funding'] = funding_data
            results.append(disorder_with_funding)
            
            # Print result
            if 'error' in funding_data:
                print(f"  ⚠️  Error: {funding_data['error']}")
            else:
                print(f"  ✅ ${funding_data['total_funding_millions']:.1f}M from {funding_data['n_grants']} grants")
            
            # Save progress periodically
            if save_progress and i % 50 == 0:
                with open(progress_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\n💾 Progress saved: {i}/{total} complete\n")
        
        # Final save
        if save_progress:
            with open(progress_file, 'w') as f:
                json.dump(results, f, indent=2)
        
        print(f"\n✅ NIH funding collection complete: {total} disorders processed")
        
        return results
    
    def get_summary_statistics(self, results: List[Dict]) -> Dict:
        """Calculate summary statistics for funding data."""
        funding_amounts = []
        grant_counts = []
        
        for result in results:
            funding_data = result.get('nih_funding', {})
            if 'error' not in funding_data:
                funding_amounts.append(funding_data.get('total_funding_millions', 0))
                grant_counts.append(funding_data.get('n_grants', 0))
        
        if not funding_amounts:
            return {'error': 'No funding data collected'}
        
        import numpy as np
        
        return {
            'n_disorders': len(results),
            'n_with_funding': sum(1 for f in funding_amounts if f > 0),
            'total_funding_millions': sum(funding_amounts),
            'mean_funding': np.mean(funding_amounts),
            'median_funding': np.median(funding_amounts),
            'std_funding': np.std(funding_amounts),
            'max_funding': max(funding_amounts),
            'mean_grants_per_disorder': np.mean(grant_counts)
        }


if __name__ == '__main__':
    # Demo with a few disorders
    collector = NIHFundingCollector()
    
    test_disorders = [
        {'disorder_name': 'Schizophrenia'},
        {'disorder_name': 'Major Depressive Disorder'},
        {'disorder_name': 'Anxiety Disorder'}
    ]
    
    print("Testing NIH API with sample disorders...\n")
    
    for disorder in test_disorders:
        result = collector.collect_funding_for_disorder(disorder['disorder_name'])
        print(f"{result['disorder_name']}: ${result.get('total_funding_millions', 0):.1f}M")

