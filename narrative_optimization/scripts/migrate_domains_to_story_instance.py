"""
Domain Migration Script

Converts existing domain analyses to new StoryInstance framework.

For each domain:
1. Load existing analysis results
2. Convert to StoryInstance format
3. Calculate new features (π_effective, Β, θ_amp, ф_imperative)
4. Store in InstanceRepository
5. Preserve all existing features

Author: Narrative Optimization Framework
Date: November 2025
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import sys

# Add parent directories to path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root / 'narrative_optimization' / 'src'))

from core.story_instance import StoryInstance
from data.instance_repository import InstanceRepository
from config.domain_config import DomainConfig
from analysis.complexity_scorer import ComplexityScorer
from analysis.blind_narratio_calculator import BlindNarratioCalculator
from transformers.awareness_amplification import AwarenessAmplificationTransformer
from physics.imperative_gravity import ImperativeGravityCalculator


class DomainMigrator:
    """
    Migrate existing domain analyses to StoryInstance framework.
    """
    
    def __init__(
        self,
        domains_dir: Path,
        repository_path: Optional[str] = None
    ):
        """
        Initialize migrator.
        
        Parameters
        ----------
        domains_dir : Path
            Path to domains directory
        repository_path : str, optional
            Path for InstanceRepository
        """
        self.domains_dir = Path(domains_dir)
        self.repository = InstanceRepository(repository_path)
        
        # Initialize calculators
        self.blind_narratio_calc = BlindNarratioCalculator()
        self.awareness_transformer = AwarenessAmplificationTransformer()
        
        # Load all domain configs
        self.domain_configs = self._load_all_domain_configs()
        
        # Initialize imperative gravity with all configs
        self.imperative_gravity = ImperativeGravityCalculator(self.domain_configs)
        
        # Track migration progress
        self.migration_stats = {
            'domains_processed': 0,
            'instances_created': 0,
            'instances_failed': 0,
            'total_features_extracted': 0
        }
    
    def _load_all_domain_configs(self) -> Dict[str, DomainConfig]:
        """Load domain configs for all available domains."""
        configs = {}
        
        # Known domains with configs
        known_domains = [
            'golf', 'tennis', 'chess', 'boxing', 'wwe', 'oscars',
            'nba', 'nfl', 'nhl', 'mlb', 'supreme_court',
            'hurricanes', 'housing', 'startups', 'movies', 'novels',
            'music', 'poetry', 'aviation'
        ]
        
        for domain in known_domains:
            try:
                configs[domain] = DomainConfig(domain)
            except Exception as e:
                print(f"Warning: Could not load config for {domain}: {e}")
        
        return configs
    
    def migrate_domain(
        self,
        domain_name: str,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Migrate single domain to new framework.
        
        Parameters
        ----------
        domain_name : str
            Domain name (e.g., 'golf', 'supreme_court')
        verbose : bool
            Print progress messages
        
        Returns
        -------
        dict
            Migration results
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"MIGRATING DOMAIN: {domain_name.upper()}")
            print(f"{'='*60}\n")
        
        # Find domain directory
        domain_dir = self.domains_dir / domain_name
        if not domain_dir.exists():
            return {'error': f'Domain directory not found: {domain_dir}'}
        
        # Load existing domain data
        domain_data = self._load_domain_data(domain_dir, domain_name)
        if domain_data is None:
            return {'error': f'Could not load data for {domain_name}'}
        
        if verbose:
            print(f"  Loaded {len(domain_data)} instances from existing data")
        
        # Get domain config
        domain_config = self.domain_configs.get(domain_name)
        if domain_config is None:
            if verbose:
                print(f"  Warning: No domain config found, using defaults")
            domain_config = DomainConfig('generic')
        
        # Create complexity scorer
        complexity_scorer = ComplexityScorer(domain=domain_name)
        
        # Convert to StoryInstances
        instances = []
        
        for i, data_item in enumerate(domain_data):
            try:
                instance = self._convert_to_story_instance(
                    data_item,
                    domain_name,
                    i,
                    domain_config,
                    complexity_scorer
                )
                instances.append(instance)
                
                if verbose and (i + 1) % 10 == 0:
                    print(f"  Converted {i+1}/{len(domain_data)} instances...", end="\r")
            
            except Exception as e:
                self.migration_stats['instances_failed'] += 1
                if verbose:
                    print(f"  Warning: Failed to convert instance {i}: {e}")
        
        if verbose:
            print(f"  Converted {len(instances)}/{len(domain_data)} instances      ")
        
        # Calculate domain-level Blind Narratio
        if instances:
            if verbose:
                print(f"\n  Calculating domain Blind Narratio...")
            
            beta_result = self.blind_narratio_calc.calculate_domain_blind_narratio(
                instances,
                domain_name
            )
            
            if verbose:
                print(f"  Β = {beta_result['Β']:.3f} (stability: {beta_result['stability']:.3f})")
        
        # Calculate imperative gravity for each instance
        if instances and len(self.domain_configs) > 1:
            if verbose:
                print(f"\n  Calculating cross-domain imperative gravity...")
            
            all_domains = list(self.domain_configs.keys())
            
            for i, instance in enumerate(instances):
                neighbors = self.imperative_gravity.find_gravitational_neighbors(
                    instance,
                    all_domains,
                    n_neighbors=5,
                    exclude_same_domain=True
                )
                
                # Store on instance
                for domain, force in neighbors:
                    instance.add_imperative_gravity(
                        target_domain=domain,
                        target_instance_id="domain_aggregate",
                        force_magnitude=force
                    )
                
                if verbose and (i + 1) % 10 == 0:
                    print(f"  Calculated gravity for {i+1}/{len(instances)} instances...", end="\r")
            
            if verbose:
                print(f"  Calculated gravity for {len(instances)}/{len(instances)} instances      ")
        
        # Add to repository
        if instances:
            if verbose:
                print(f"\n  Adding instances to repository...")
            
            self.repository.add_instances_bulk(instances)
            
            if verbose:
                print(f"  ✓ Added {len(instances)} instances")
        
        # Update stats
        self.migration_stats['domains_processed'] += 1
        self.migration_stats['instances_created'] += len(instances)
        
        result = {
            'domain': domain_name,
            'instances_migrated': len(instances),
            'instances_failed': self.migration_stats['instances_failed'],
            'blind_narratio': beta_result['Β'] if instances else None,
            'pi_base': domain_config.get_pi() if domain_config else None,
            'pi_variance': beta_result.get('variance_by_context', {}) if instances else {}
        }
        
        if verbose:
            print(f"\n  ✓ Migration complete for {domain_name}\n")
        
        return result
    
    def migrate_all_domains(
        self,
        domain_list: Optional[List[str]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Migrate all domains or specified list.
        
        Parameters
        ----------
        domain_list : list of str, optional
            Domains to migrate (all if None)
        verbose : bool
            Print progress
        
        Returns
        -------
        dict
            Complete migration results
        """
        if domain_list is None:
            # Find all domain directories
            domain_list = [d.name for d in self.domains_dir.iterdir() 
                          if d.is_dir() and not d.name.startswith('_')]
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"MIGRATING {len(domain_list)} DOMAINS TO STORY INSTANCE FRAMEWORK")
            print(f"{'='*70}\n")
        
        results = {}
        
        for domain_name in domain_list:
            result = self.migrate_domain(domain_name, verbose=verbose)
            results[domain_name] = result
        
        # Save repository
        if verbose:
            print(f"\n{'='*70}")
            print(f"SAVING REPOSITORY")
            print(f"{'='*70}\n")
        
        self.repository.save_to_disk()
        
        # Generate summary
        if verbose:
            print(self._generate_migration_summary(results))
        
        return {
            'migration_stats': self.migration_stats,
            'domain_results': results,
            'repository_stats': self.repository.get_repository_statistics()
        }
    
    def _load_domain_data(
        self,
        domain_dir: Path,
        domain_name: str
    ) -> Optional[List[Dict]]:
        """
        Load existing domain data from various possible formats.
        
        Tries:
        1. data/*.json files
        2. *_results.json
        3. *_dataset.json
        """
        # Try data directory
        data_dir = domain_dir / 'data'
        if data_dir.exists():
            json_files = list(data_dir.glob('*.json'))
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    # Check if it's a list or dict
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict) and 'instances' in data:
                        return data['instances']
                    elif isinstance(data, dict) and 'data' in data:
                        return data['data']
                except:
                    continue
        
        # Try results files
        result_files = list(domain_dir.glob('*_results.json'))
        for result_file in result_files:
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    # Try to extract instances
                    for key in ['instances', 'data', 'samples', 'cases']:
                        if key in data and isinstance(data[key], list):
                            return data[key]
            except:
                continue
        
        return None
    
    def _convert_to_story_instance(
        self,
        data_item: Dict,
        domain_name: str,
        index: int,
        domain_config: DomainConfig,
        complexity_scorer: ComplexityScorer
    ) -> StoryInstance:
        """
        Convert a data item to StoryInstance.
        
        Parameters
        ----------
        data_item : dict
            Original data item
        domain_name : str
            Domain name
        index : int
            Item index
        domain_config : DomainConfig
            Domain configuration
        complexity_scorer : ComplexityScorer
            Complexity calculator
        
        Returns
        -------
        StoryInstance
            Converted instance
        """
        # Extract ID
        instance_id = data_item.get('id') or \
                     data_item.get('instance_id') or \
                     data_item.get('name') or \
                     f"{domain_name}_{index}"
        
        # Extract narrative text
        narrative_text = data_item.get('narrative') or \
                        data_item.get('text') or \
                        data_item.get('description') or \
                        ""
        
        # Extract outcome
        outcome = data_item.get('outcome') or \
                 data_item.get('result') or \
                 data_item.get('success') or \
                 data_item.get('label')
        
        # Create instance
        instance = StoryInstance(
            instance_id=str(instance_id),
            domain=domain_name,
            narrative_text=narrative_text,
            outcome=float(outcome) if outcome is not None else None,
            context=data_item.get('context', {})
        )
        
        # Calculate complexity
        complexity = complexity_scorer.calculate_complexity(instance, narrative_text)
        
        # Calculate π_effective
        pi_eff = domain_config.calculate_effective_pi(complexity)
        instance.pi_effective = pi_eff
        instance.pi_domain_base = domain_config.get_pi()
        
        # Extract existing features if available
        if 'features' in data_item:
            features = data_item['features']
            if isinstance(features, dict):
                instance.features_all = features
            elif isinstance(features, (list, np.ndarray)):
                instance.genome_full = np.array(features)
        
        # Extract genome components if available
        if 'genome' in data_item:
            genome_data = data_item['genome']
            if isinstance(genome_data, dict):
                instance.genome_nominative = np.array(genome_data.get('nominative', []))
                instance.genome_archetypal = np.array(genome_data.get('archetypal', []))
                instance.genome_historial = np.array(genome_data.get('historial', []))
                instance.genome_uniquity = np.array(genome_data.get('uniquity', []))
            elif isinstance(genome_data, (list, np.ndarray)):
                instance.genome_full = np.array(genome_data)
        
        # Extract story quality if available
        if 'story_quality' in data_item:
            instance.story_quality = float(data_item['story_quality'])
        elif 'prediction' in data_item:
            instance.story_quality = float(data_item['prediction'])
        
        # Extract timestamp if available
        if 'timestamp' in data_item or 'date' in data_item:
            timestamp_str = data_item.get('timestamp') or data_item.get('date')
            try:
                instance.timestamp = datetime.fromisoformat(str(timestamp_str))
            except:
                pass
        
        # Calculate mass (importance × stakes)
        importance = data_item.get('importance', 1.0)
        stakes_mult = data_item.get('stakes_multiplier', 1.0)
        instance.importance_score = float(importance)
        instance.stakes_multiplier = float(stakes_mult)
        instance.calculate_mass()
        
        return instance
    
    def _generate_migration_summary(self, results: Dict) -> str:
        """Generate migration summary report."""
        lines = []
        lines.append(f"\n{'='*70}")
        lines.append(f"MIGRATION COMPLETE")
        lines.append(f"{'='*70}\n")
        
        lines.append(f"Domains processed: {self.migration_stats['domains_processed']}")
        lines.append(f"Instances created: {self.migration_stats['instances_created']}")
        lines.append(f"Instances failed: {self.migration_stats['instances_failed']}")
        lines.append(f"Success rate: {self.migration_stats['instances_created']/(self.migration_stats['instances_created']+self.migration_stats['instances_failed']+1e-8)*100:.1f}%")
        lines.append(f"\nRepository statistics:")
        
        repo_stats = self.repository.get_repository_statistics()
        lines.append(f"  Total instances: {repo_stats['total_instances']}")
        lines.append(f"  Total domains: {repo_stats['total_domains']}")
        
        lines.append(f"\nDomain-level Blind Narratio (Β) values:")
        lines.append(f"{'-'*70}")
        
        # Sort by Β
        domain_betas = []
        for domain, result in results.items():
            if 'blind_narratio' in result and result['blind_narratio'] is not None:
                domain_betas.append((domain, result['blind_narratio']))
        
        domain_betas.sort(key=lambda x: x[1])
        
        for domain, beta in domain_betas:
            lines.append(f"  {domain:20s} Β = {beta:.3f}")
        
        lines.append(f"\n{'='*70}\n")
        
        return '\n'.join(lines)
    
    def export_migration_report(self, output_path: str):
        """Export detailed migration report to JSON."""
        report = {
            'migration_stats': self.migration_stats,
            'repository_stats': self.repository.get_repository_statistics(),
            'domain_betas': self.blind_narratio_calc.domain_betas,
            'migrated_at': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Exported migration report to {output_path}")


def main():
    """Run migration on test domains."""
    # Initialize migrator
    domains_dir = Path(__file__).parent.parent / 'domains'
    migrator = DomainMigrator(domains_dir)
    
    # Test domains first
    test_domains = ['golf', 'supreme_court', 'boxing', 'tennis', 'oscars']
    
    print(f"\nMigrating {len(test_domains)} test domains to StoryInstance framework...")
    
    results = migrator.migrate_all_domains(domain_list=test_domains, verbose=True)
    
    # Export report
    output_path = Path(__file__).parent.parent / 'results' / 'migration_report.json'
    migrator.export_migration_report(str(output_path))
    
    print(f"\n✓ Test migration complete!")
    print(f"\nTo migrate all domains, use:")
    print(f"  migrator.migrate_all_domains(verbose=True)")


if __name__ == '__main__':
    main()

