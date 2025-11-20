"""
Domain Analysis Migration Tool

Migrates existing domain analyses to use new DomainSpecificAnalyzer
while maintaining backward compatibility.

Author: Narrative Integration System
Date: November 2025
"""

import sys
from pathlib import Path
import json
import numpy as np
from typing import Dict, List, Optional, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.domain_specific_analyzer import DomainSpecificAnalyzer


class DomainAnalysisMigrator:
    """
    Migrates existing domain analyses to new architecture.
    
    Maintains backward compatibility by:
    - Loading existing data formats
    - Converting to new analyzer format
    - Preserving existing results structure
    - Adding new features (historial, uniquity)
    """
    
    def __init__(self, domain_name: str):
        self.domain_name = domain_name
        self.analyzer = DomainSpecificAnalyzer(domain_name)
        
    def load_existing_data(self, data_path: Path) -> Dict[str, Any]:
        """
        Load data from existing domain analysis format.
        
        Handles multiple formats:
        - JSON with narratives/outcomes
        - JSON with tournament/match data
        - CSV files
        """
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        if data_path.suffix == '.json':
            with open(data_path) as f:
                data = json.load(f)
            
            # Try different formats
            if 'narratives' in data:
                texts = data['narratives']
                outcomes = np.array(data.get('outcomes', data.get('results', [])))
            elif 'texts' in data:
                texts = data['texts']
                outcomes = np.array(data.get('outcomes', data.get('results', [])))
            elif isinstance(data, list):
                # List of records
                texts = [item.get('narrative', item.get('text', str(item))) for item in data]
                outcomes = np.array([item.get('outcome', item.get('result', 0)) for item in data])
            else:
                raise ValueError(f"Unknown JSON format in {data_path}")
            
            return {
                'texts': texts,
                'outcomes': outcomes,
                'names': data.get('names', None),
                'timestamps': data.get('timestamps', data.get('dates', None))
            }
        
        elif data_path.suffix == '.csv':
            import pandas as pd
            df = pd.read_csv(data_path)
            
            # Try to find narrative/text column
            text_col = None
            for col in ['narrative', 'text', 'description', 'story']:
                if col in df.columns:
                    text_col = col
                    break
            
            if text_col is None:
                raise ValueError(f"No narrative column found in {data_path}")
            
            texts = df[text_col].tolist()
            outcomes = df.get('outcome', df.get('result', df.get('y', np.zeros(len(df))))).values
            
            return {
                'texts': texts,
                'outcomes': outcomes,
                'names': df.get('name', None),
                'timestamps': df.get('timestamp', df.get('date', None))
            }
        
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
    
    def migrate_analysis(
        self,
        data_path: Path,
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Migrate existing analysis to new architecture.
        
        Parameters
        ----------
        data_path : Path
            Path to existing data file
        output_path : Path, optional
            Path to save migrated results
        
        Returns
        -------
        dict
            Migrated analysis results
        """
        print(f"\n{'='*80}")
        print(f"MIGRATING: {self.domain_name.upper()}")
        print(f"{'='*80}")
        
        # Load existing data
        print(f"\n[1/4] Loading existing data...")
        data = self.load_existing_data(data_path)
        print(f"  ✓ Loaded {len(data['texts'])} narratives")
        
        # Run new analysis
        print(f"\n[2/4] Running domain-specific analysis...")
        results = self.analyzer.analyze_complete(
            texts=data['texts'],
            outcomes=data['outcomes'],
            names=data.get('names'),
            timestamps=data.get('timestamps')
        )
        
        # Preserve backward compatibility
        print(f"\n[3/4] Preserving backward compatibility...")
        migrated_results = {
            # New results
            'domain': results['domain'],
            'narrativity': results['narrativity'],
            'r_squared': results['r_squared'],
            'delta': results['delta'],
            'efficiency': results['efficiency'],
            'passes_threshold': results['passes_threshold'],
            
            # Backward compatible fields
            'r': results['r'],
            'story_quality': results['story_quality'].tolist(),
            'outcomes': results['outcomes'].tolist(),
            
            # New features
            'has_historial': True,
            'has_uniquity': True,
            'historial_features': results['historial_features'].tolist(),
            'uniquity_features': results['uniquity_features'].tolist(),
            
            # Metadata
            'migration_date': str(Path(__file__).stat().st_mtime),
            'architecture_version': 'domain_specific_xi'
        }
        
        # Save if output path provided
        if output_path:
            print(f"\n[4/4] Saving migrated results...")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(migrated_results, f, indent=2)
            print(f"  ✓ Saved to: {output_path}")
        
        return migrated_results


def migrate_golf_analysis():
    """Migrate Golf domain analysis."""
    migrator = DomainAnalysisMigrator('golf')
    
    # Find golf data
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data' / 'domains' / 'golf_tournaments.json'
    
    if not data_path.exists():
        # Try alternative paths
        alt_paths = [
            project_root / 'narrative_optimization' / 'domains' / 'golf' / 'golf_results.json',
            project_root / 'data' / 'domains' / 'golf_enhanced_player_tournaments.json'
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                data_path = alt_path
                break
    
    output_path = project_root / 'narrative_optimization' / 'domains' / 'golf' / 'migrated_results.json'
    
    return migrator.migrate_analysis(data_path, output_path)


def migrate_tennis_analysis():
    """Migrate Tennis domain analysis."""
    migrator = DomainAnalysisMigrator('tennis')
    
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data' / 'domains' / 'tennis_complete_dataset.json'
    
    if not data_path.exists():
        alt_paths = [
            project_root / 'narrative_optimization' / 'domains' / 'tennis' / 'tennis_results.json',
            project_root / 'data' / 'domains' / 'tennis_matches.json'
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                data_path = alt_path
                break
    
    output_path = project_root / 'narrative_optimization' / 'domains' / 'tennis' / 'migrated_results.json'
    
    return migrator.migrate_analysis(data_path, output_path)


def migrate_boxing_analysis():
    """Migrate Boxing domain analysis."""
    migrator = DomainAnalysisMigrator('boxing')
    
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'narrative_optimization' / 'domains' / 'boxing' / 'boxing_complete_analysis.json'
    
    output_path = project_root / 'narrative_optimization' / 'domains' / 'boxing' / 'migrated_results.json'
    
    return migrator.migrate_analysis(data_path, output_path)


def migrate_nba_analysis():
    """Migrate NBA domain analysis."""
    migrator = DomainAnalysisMigrator('nba')
    
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'narrative_optimization' / 'domains' / 'nba' / 'nba_proper_results.json'
    
    output_path = project_root / 'narrative_optimization' / 'domains' / 'nba' / 'migrated_results.json'
    
    return migrator.migrate_analysis(data_path, output_path)


def migrate_wwe_analysis():
    """Migrate WWE domain analysis."""
    migrator = DomainAnalysisMigrator('wwe')
    
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'narrative_optimization' / 'domains' / 'wwe' / 'wwe_framework_results.json'
    
    output_path = project_root / 'narrative_optimization' / 'domains' / 'wwe' / 'migrated_results.json'
    
    return migrator.migrate_analysis(data_path, output_path)


if __name__ == '__main__':
    print("="*80)
    print("DOMAIN ANALYSIS MIGRATION")
    print("="*80)
    
    domains = {
        'golf': migrate_golf_analysis,
        'tennis': migrate_tennis_analysis,
        'boxing': migrate_boxing_analysis,
        'nba': migrate_nba_analysis,
        'wwe': migrate_wwe_analysis
    }
    
    results = {}
    
    for domain, migrate_func in domains.items():
        try:
            print(f"\n\n{'='*80}")
            print(f"MIGRATING {domain.upper()}")
            print(f"{'='*80}")
            result = migrate_func()
            results[domain] = {'status': 'success', 'r_squared': result['r_squared']}
        except Exception as e:
            print(f"\n✗ Migration failed: {e}")
            results[domain] = {'status': 'failed', 'error': str(e)}
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n\n{'='*80}")
    print("MIGRATION SUMMARY")
    print(f"{'='*80}\n")
    
    for domain, result in results.items():
        if result['status'] == 'success':
            print(f"  {domain.upper():15s}: ✓ Migrated (R²: {result['r_squared']:.1%})")
        else:
            print(f"  {domain.upper():15s}: ✗ Failed ({result.get('error', 'Unknown error')})")
    
    successful = sum(1 for r in results.values() if r['status'] == 'success')
    print(f"\nTotal: {successful}/{len(results)} migrated successfully")

