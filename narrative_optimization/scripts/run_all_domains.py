#!/usr/bin/env python3
"""
Batch Pipeline Runner

Run unified pipeline system on all domains and generate comprehensive results.
"""

import sys
from pathlib import Path
import json
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import traceback

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipelines.domain_config import DomainConfig
from src.pipelines.pipeline_composer import PipelineComposer


class BatchPipelineRunner:
    """Run pipelines on all domains"""
    
    def __init__(self, project_root: Path = None):
        """Initialize runner"""
        if project_root is None:
            project_root = Path(__file__).parent.parent
        
        self.project_root = Path(project_root)
        self.domains_dir = self.project_root / 'domains'
        # Data can be in parent's data/domains or narrative_optimization/data/domains
        self.data_dirs = [
            self.project_root.parent / 'data' / 'domains',
            self.project_root / 'data' / 'domains'
        ]
        self.composer = PipelineComposer(self.project_root)
        self.results_summary = []
    
    def find_domain_configs(self) -> List[Path]:
        """Find all domain config files"""
        configs = []
        for domain_dir in self.domains_dir.iterdir():
            if not domain_dir.is_dir():
                continue
            
            config_path = domain_dir / 'config.yaml'
            if config_path.exists():
                configs.append(config_path)
        
        return sorted(configs)
    
    def find_data_file(self, domain_dir: Path, domain_name: str) -> Optional[Path]:
        """Find data file for domain"""
        # Look in data subdirectory
        data_dir = domain_dir / 'data'
        if data_dir.exists():
            # Try JSON first
            json_files = list(data_dir.glob('*.json'))
            if json_files:
                return json_files[0]
            # Then CSV
            csv_files = list(data_dir.glob('*.csv'))
            if csv_files:
                return csv_files[0]
        
        # Look for domain-specific data files in domain directory
        patterns = [
            f'{domain_name}_data.json',
            f'{domain_name}_data.csv',
            f'{domain_name}_dataset.json',
            f'{domain_name}_dataset.csv',
            'data.json',
            'data.csv',
            'dataset.json',
            'dataset.csv'
        ]
        
        for pattern in patterns:
            data_file = domain_dir / pattern
            if data_file.exists():
                return data_file
        
        # Look in parent data/domains directory
        parent_data_dir = self.project_root / 'data' / 'domains' / domain_name
        if parent_data_dir.exists():
            json_files = list(parent_data_dir.glob('*.json'))
            if json_files:
                return json_files[0]
            csv_files = list(parent_data_dir.glob('*.csv'))
            if csv_files:
                return csv_files[0]
        
        # Look for complete dataset files (tennis, etc.) in multiple locations
        complete_patterns = [
            f'{domain_name}_complete_dataset.json',
            f'{domain_name}_dataset.json',
            f'complete_{domain_name}_dataset.json',
            f'{domain_name}_complete.json',
            f'{domain_name}_enhanced_narratives.json',  # Golf enhanced format (preferred)
            f'{domain_name}_with_narratives.json',  # Golf format
            f'{domain_name}_narratives.json',  # Alternative format
            f'{domain_name}_tournaments.json'  # Golf tournaments format
        ]
        for data_dir in self.data_dirs:
            for pattern in complete_patterns:
                data_file = data_dir / pattern
                if data_file.exists():
                    return data_file
        
        # Look for any JSON/CSV files in domain directory (last resort)
        all_json = list(domain_dir.glob('*.json'))
        all_csv = list(domain_dir.glob('*.csv'))
        
        # Filter out results/config files - be more aggressive
        exclude_keywords = ['results', 'config', 'analysis', 'discoveries', 'context', 'betting', 
                          'optimized', 'formula', 'ensemble', 'models', 'genome', 'features',
                          'narrativity', 'summary', 'findings', 'comparison']
        data_json = [f for f in all_json if not any(kw in f.name.lower() for kw in exclude_keywords)]
        data_csv = [f for f in all_csv if not any(kw in f.name.lower() for kw in exclude_keywords)]
        
        if data_json:
            return data_json[0]
        if data_csv:
            return data_csv[0]
        
        return None
    
    def run_domain(self, config_path: Path, skip_if_exists: bool = True) -> Dict[str, Any]:
        """Run pipeline for a single domain"""
        domain_name = config_path.parent.name
        
        print("\n" + "=" * 80)
        print(f"PROCESSING DOMAIN: {domain_name.upper()}")
        print("=" * 80)
        
        # Check if results already exist
        results_path = config_path.parent / f"{domain_name}_results.json"
        if skip_if_exists and results_path.exists():
            print(f"⚠ Results already exist, skipping: {results_path}")
            try:
                with open(results_path, 'r') as f:
                    existing_results = json.load(f)
                    return {
                        'domain': domain_name,
                        'status': 'skipped',
                        'config_path': str(config_path),
                        'results_path': str(results_path),
                        'pi': existing_results.get('pi', 0),
                        'r_narrative': existing_results.get('analysis', {}).get('r_narrative', 0)
                    }
            except:
                pass
        
        try:
            # Load config
            config = DomainConfig.from_yaml(config_path)
            print(f"✓ Loaded config: п={config.pi:.3f}, type={config.type.value}")
            
            # Find data file
            data_path = self.find_data_file(config_path.parent, domain_name)
            if data_path is None:
                print(f"⚠ No data file found for {domain_name}, skipping")
                return {
                    'domain': domain_name,
                    'status': 'no_data',
                    'error': 'No data file found'
                }
            
            print(f"✓ Found data file: {data_path}")
            
            # Run pipeline
            print(f"\nRunning pipeline...")
            results = self.composer.run_pipeline(
                config,
                data_path=data_path,
                target_feature_count=300,
                use_cache=True
            )
            
            # Format results for saving
            formatted_results = self.format_results(results, config, domain_name)
            
            # Save results
            results_path = config_path.parent / f"{domain_name}_results.json"
            with open(results_path, 'w') as f:
                json.dump(formatted_results, f, indent=2, default=self._json_serializer)
            
            print(f"✓ Results saved: {results_path}")
            
            # Extract summary metrics
            analysis = results.get('analysis', {})
            comprehensive = results.get('comprehensive_ю', {})
            
            summary = {
                'domain': domain_name,
                'status': 'success',
                'config_path': str(config_path),
                'results_path': str(results_path),
                'pi': config.pi,
                'r_narrative': analysis.get('r_narrative', 0),
                'Д': analysis.get('Д', 0),
                'efficiency': analysis.get('efficiency', 0),
                'n_perspectives': len(comprehensive.get('ю_perspectives', {})),
                'n_methods': len(comprehensive.get('ю_methods', {})),
                'n_scales': len(comprehensive.get('ю_scales', {}))
            }
            
            self.results_summary.append(summary)
            return summary
            
        except Exception as e:
            error_msg = str(e)
            print(f"✗ Error processing {domain_name}: {error_msg}")
            traceback.print_exc()
            
            return {
                'domain': domain_name,
                'status': 'error',
                'error': error_msg,
                'traceback': traceback.format_exc()
            }
    
    def format_results(self, results: Dict, config: DomainConfig, domain_name: str) -> Dict:
        """Format results for JSON serialization"""
        # Convert numpy arrays to lists
        def make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            return obj
        
        formatted = {
            'domain': domain_name,
            'pi': config.pi,
            'domain_type': config.type.value,
            'timestamp': datetime.now().isoformat(),
            'analysis': make_serializable(results.get('analysis', {})),
            'comprehensive_ю': make_serializable(results.get('comprehensive_ю', {})),
            'top_features': make_serializable(results.get('top_features', [])),
            'metadata': {
                'config': config.to_dict(),
                'pipeline_info': {
                    'transformers': results.get('pipeline_info', {}).get('transformers', []),
                    'feature_count': results.get('pipeline_info', {}).get('feature_count', 0)
                }
            }
        }
        
        return formatted
    
    def _json_serializer(self, obj):
        """JSON serializer for numpy types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Type {type(obj)} not serializable")
    
    def run_all(self, skip_if_exists: bool = True) -> Dict[str, Any]:
        """Run pipelines on all domains"""
        print("=" * 80)
        print("BATCH PIPELINE EXECUTION")
        print("=" * 80)
        
        configs = self.find_domain_configs()
        print(f"\nFound {len(configs)} domain configurations")
        
        results = []
        for config_path in configs:
            result = self.run_domain(config_path, skip_if_exists=skip_if_exists)
            results.append(result)
        
        # Generate summary
        summary = self.generate_summary(results)
        
        # Save summary
        summary_path = self.project_root / 'batch_results_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=self._json_serializer)
        
        print("\n" + "=" * 80)
        print("BATCH EXECUTION COMPLETE")
        print("=" * 80)
        print(f"\nSummary saved: {summary_path}")
        print(f"\nProcessed: {len([r for r in results if r['status'] == 'success'])}/{len(results)}")
        print(f"Skipped: {len([r for r in results if r['status'] == 'skipped'])}")
        print(f"Errors: {len([r for r in results if r['status'] == 'error'])}")
        
        return summary
    
    def generate_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate execution summary"""
        successful = [r for r in results if r['status'] == 'success']
        errors = [r for r in results if r['status'] == 'error']
        skipped = [r for r in results if r['status'] == 'skipped']
        no_data = [r for r in results if r['status'] == 'no_data']
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_domains': len(results),
            'successful': len(successful),
            'errors': len(errors),
            'skipped': len(skipped),
            'no_data': len(no_data),
            'results': results,
            'statistics': {
                'avg_pi': np.mean([r.get('pi', 0) for r in successful]) if successful else 0,
                'avg_r_narrative': np.mean([r.get('r_narrative', 0) for r in successful]) if successful else 0,
                'avg_efficiency': np.mean([r.get('efficiency', 0) for r in successful]) if successful else 0,
            }
        }
        
        return summary


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run unified pipeline on all domains')
    parser.add_argument('--force', action='store_true', help='Force regeneration even if results exist')
    parser.add_argument('--domain', type=str, help='Run only specific domain')
    
    args = parser.parse_args()
    
    runner = BatchPipelineRunner()
    
    if args.domain:
        # Run single domain
        config_path = runner.domains_dir / args.domain / 'config.yaml'
        if config_path.exists():
            runner.run_domain(config_path, skip_if_exists=not args.force)
        else:
            print(f"Error: Config not found for domain {args.domain}")
    else:
        # Run all domains
        runner.run_all(skip_if_exists=not args.force)


if __name__ == '__main__':
    main()

