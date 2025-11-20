#!/usr/bin/env python3
"""
Generate Domain Configurations

Automatically create config.yaml files for all domains based on existing analysis.
"""

import sys
from pathlib import Path
import json
import yaml
from typing import Dict, List, Optional, Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipelines.domain_config import (
    DomainConfig, DomainType, OutcomeType,
    NarrativityComponents, DataSchema
)


class DomainConfigGenerator:
    """Generate config.yaml files for domains"""
    
    # Domain type mappings
    DOMAIN_TYPE_MAP = {
        'nba': DomainType.SPORTS_TEAM,
        'nfl': DomainType.SPORTS_TEAM,
        'tennis': DomainType.SPORTS_INDIVIDUAL,
        'golf': DomainType.SPORTS_INDIVIDUAL,
        'ufc': DomainType.SPORTS_INDIVIDUAL,
        'ncaa': DomainType.SPORTS_TEAM,
        'movies': DomainType.ENTERTAINMENT,
        'imdb': DomainType.ENTERTAINMENT,
        'oscars': DomainType.ENTERTAINMENT,
        'music': DomainType.ENTERTAINMENT,
        'wwe': DomainType.ENTERTAINMENT,
        'housing': DomainType.NOMINATIVE,
        'hurricanes': DomainType.NOMINATIVE,
        'startups': DomainType.BUSINESS,
        'crypto': DomainType.BUSINESS,
        'mental_health': DomainType.MEDICAL,
        'aviation': DomainType.SPORTS,
        'ships': DomainType.NOMINATIVE,
        'lottery': DomainType.HYBRID,
        'marriage': DomainType.HYBRID,
        'immigration': DomainType.HYBRID,
    }
    
    # Known narrativity values (from existing analyses)
    KNOWN_PI_VALUES = {
        'nba': 0.70,
        'tennis': 0.75,
        'golf': 0.65,
        'ufc': 0.722,
        'movies': 0.65,
        'imdb': 0.68,
        'oscars': 0.75,
        'music': 0.702,
        'wwe': 0.974,
        'housing': 0.45,
        'startups': 0.60,
        'crypto': 0.55,
        'mental_health': 0.70,
        'nfl': 0.65,
    }
    
    # Narrativity component estimates (when pi known, estimate components)
    def estimate_components_from_pi(self, pi: float, domain_type: DomainType) -> NarrativityComponents:
        """Estimate narrativity components from pi value"""
        # Use domain-type-specific heuristics
        if domain_type in [DomainType.SPORTS, DomainType.SPORTS_TEAM, DomainType.SPORTS_INDIVIDUAL]:
            # Sports: high temporal, medium agency, medium structural
            return NarrativityComponents(
                structural=pi * 0.9,
                temporal=min(0.95, pi * 1.2),
                agency=pi * 0.9,
                interpretive=pi * 0.7,
                format=pi * 0.6
            )
        elif domain_type == DomainType.ENTERTAINMENT:
            # Entertainment: high structural, high interpretive
            return NarrativityComponents(
                structural=pi * 1.1,
                temporal=pi * 0.9,
                agency=pi * 0.8,
                interpretive=pi * 1.2,
                format=pi * 0.9
            )
        elif domain_type == DomainType.NOMINATIVE:
            # Nominative: medium across board
            return NarrativityComponents(
                structural=pi * 0.8,
                temporal=pi * 0.7,
                agency=pi * 0.6,
                interpretive=pi * 0.8,
                format=pi * 0.7
            )
        else:
            # Default: balanced
            return NarrativityComponents(
                structural=pi,
                temporal=pi,
                agency=pi,
                interpretive=pi,
                format=pi
            )
    
    def detect_data_schema(self, domain_dir: Path) -> Optional[DataSchema]:
        """Detect data schema from existing files"""
        # Look for data files
        data_files = list(domain_dir.glob('data/*.json')) + list(domain_dir.glob('data/*.csv'))
        if not data_files:
            # Check for results files that might indicate schema
            results_files = list(domain_dir.glob('*_results.json'))
            if results_files:
                try:
                    with open(results_files[0], 'r') as f:
                        results = json.load(f)
                        # Try to infer from results structure
                        if 'text_fields' in results:
                            return DataSchema(
                                text_fields=results.get('text_fields', ['text']),
                                outcome_field=results.get('outcome_field', 'outcome'),
                                context_fields=results.get('context_fields'),
                                name_field=results.get('name_field')
                            )
                except:
                    pass
        
        # Default schema
        return DataSchema(
            text_fields=['text'],
            outcome_field='outcome',
            context_fields=None,
            name_field=None
        )
    
    def detect_outcome_type(self, domain_dir: Path) -> OutcomeType:
        """Detect outcome type from domain"""
        # Check results files for clues
        results_files = list(domain_dir.glob('*_results.json'))
        for results_file in results_files:
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    if 'outcome_type' in results:
                        outcome_str = results['outcome_type'].lower()
                        if 'binary' in outcome_str:
                            return OutcomeType.BINARY
                        elif 'continuous' in outcome_str:
                            return OutcomeType.CONTINUOUS
                        elif 'ranked' in outcome_str:
                            return OutcomeType.RANKED
            except:
                continue
        
        # Default based on domain type
        domain_name = domain_dir.name
        if domain_name in ['nba', 'nfl', 'tennis', 'golf', 'ufc', 'wwe']:
            return OutcomeType.BINARY  # Win/loss
        else:
            return OutcomeType.CONTINUOUS  # Ratings, scores, etc.
    
    def generate_config(self, domain_name: str, domain_dir: Path) -> Optional[DomainConfig]:
        """Generate config for a domain"""
        print(f"\nGenerating config for: {domain_name}")
        
        # Determine domain type
        domain_type = self.DOMAIN_TYPE_MAP.get(domain_name, DomainType.HYBRID)
        
        # Get pi value
        pi = self.KNOWN_PI_VALUES.get(domain_name)
        if pi is None:
            # Try to extract from existing analysis files
            pi = self.extract_pi_from_files(domain_dir)
            if pi is None:
                print(f"  ⚠ No pi value found, using default 0.5")
                pi = 0.5
        
        # Estimate components
        narrativity = self.estimate_components_from_pi(pi, domain_type)
        
        # Detect data schema
        data_schema = self.detect_data_schema(domain_dir)
        
        # Detect outcome type
        outcome_type = self.detect_outcome_type(domain_dir)
        
        # Create config
        config = DomainConfig(
            domain=domain_name,
            type=domain_type,
            narrativity=narrativity,
            data=data_schema,
            outcome_type=outcome_type
        )
        
        print(f"  ✓ Created config: п={config.pi:.3f}, type={domain_type.value}")
        return config
    
    def extract_pi_from_files(self, domain_dir: Path) -> Optional[float]:
        """Try to extract pi from existing analysis files"""
        # Look for narrativity calculation files
        narrativity_files = list(domain_dir.glob('*narrativity*.py')) + \
                          list(domain_dir.glob('*narrativity*.json'))
        
        for file_path in narrativity_files:
            if file_path.suffix == '.json':
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if 'pi' in data:
                            return float(data['pi'])
                        if 'narrativity' in data and 'pi' in data['narrativity']:
                            return float(data['narrativity']['pi'])
                except:
                    continue
        
        # Look in results files
        results_files = list(domain_dir.glob('*_results.json'))
        for results_file in results_files:
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    if 'narrativity' in results:
                        narr = results['narrativity']
                        if isinstance(narr, dict) and 'pi' in narr:
                            return float(narr['pi'])
                        if isinstance(narr, dict) and 'π' in narr:
                            return float(narr['π'])
            except:
                continue
        
        return None
    
    def generate_all_configs(self, domains_dir: Path) -> Dict[str, DomainConfig]:
        """Generate configs for all domains"""
        configs = {}
        
        for domain_dir in domains_dir.iterdir():
            if not domain_dir.is_dir():
                continue
            
            domain_name = domain_dir.name
            if domain_name.startswith('_') or domain_name == 'benchmarks':
                continue
            
            try:
                config = self.generate_config(domain_name, domain_dir)
                if config:
                    configs[domain_name] = config
                    # Save config
                    config_path = domain_dir / 'config.yaml'
                    config.to_yaml(config_path)
                    print(f"  ✓ Saved: {config_path}")
            except Exception as e:
                print(f"  ✗ Error generating config for {domain_name}: {e}")
        
        return configs


def main():
    """Main entry point"""
    domains_dir = project_root / 'domains'
    
    print("=" * 80)
    print("DOMAIN CONFIG GENERATION")
    print("=" * 80)
    
    generator = DomainConfigGenerator()
    configs = generator.generate_all_configs(domains_dir)
    
    print("\n" + "=" * 80)
    print(f"Generated {len(configs)} domain configurations")
    print("=" * 80)
    
    return configs


if __name__ == '__main__':
    configs = main()

