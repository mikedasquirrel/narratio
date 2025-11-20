"""
Continuous Learning Workflow

Automated workflow for continuous pattern learning and system improvement.

Author: Narrative Integration System
Date: November 2025
"""

import sys
from pathlib import Path
import json
from datetime import datetime
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.learning import LearningPipeline
from src.data import DataLoader
from src.registry import get_domain_registry, register_domain
from src.optimization import get_global_cache


class ContinuousLearningWorkflow:
    """
    Continuous learning workflow that runs periodically.
    
    Workflow:
    1. Check for new data
    2. Ingest new data
    3. Run learning cycle
    4. Validate improvements
    5. Update registry
    6. Generate reports
    """
    
    def __init__(
        self,
        check_interval: int = 3600,
        min_new_samples: int = 10,
        auto_commit: bool = False
    ):
        self.check_interval = check_interval
        self.min_new_samples = min_new_samples
        self.auto_commit = auto_commit
        
        self.pipeline = LearningPipeline(incremental=True, auto_prune=True)
        self.loader = DataLoader()
        self.last_check = {}
        
    def check_for_new_data(self) -> Dict[str, List]:
        """
        Check for new data files.
        
        Returns
        -------
        dict
            domain -> new_data_paths
        """
        from src.pipeline_config import get_config
        config = get_config()
        
        new_data = {}
        
        # Check data directory
        data_files = list(config.data_dir.glob('**/*.json'))
        
        for data_file in data_files:
            # Check modification time
            mtime = data_file.stat().st_mtime
            
            file_key = str(data_file)
            last_check_time = self.last_check.get(file_key, 0)
            
            if mtime > last_check_time:
                # New or updated file
                # Extract domain name from filename
                domain_name = data_file.stem.split('_')[0]
                
                if domain_name not in new_data:
                    new_data[domain_name] = []
                
                new_data[domain_name].append(data_file)
                self.last_check[file_key] = mtime
        
        return new_data
    
    def run_cycle(self):
        """Run one complete learning cycle."""
        print(f"\n{'='*80}")
        print(f"CONTINUOUS LEARNING CYCLE: {datetime.now().isoformat()}")
        print(f"{'='*80}\n")
        
        # Check for new data
        print("[1/6] Checking for new data...")
        new_data = self.check_for_new_data()
        
        if len(new_data) == 0:
            print("  No new data found")
            return
        
        print(f"  ✓ Found new data for {len(new_data)} domains")
        
        # Ingest new data
        print("\n[2/6] Ingesting new data...")
        ingested_domains = []
        
        for domain, data_files in new_data.items():
            for data_file in data_files:
                try:
                    data = self.loader.load(data_file)
                    
                    if self.loader.validate_data(data):
                        self.pipeline.ingest_domain(
                            domain,
                            data['texts'],
                            data['outcomes']
                        )
                        ingested_domains.append(domain)
                        print(f"  ✓ {domain}: {len(data['texts'])} samples")
                except Exception as e:
                    print(f"  ✗ {domain}: {e}")
        
        if len(ingested_domains) == 0:
            print("  No valid data ingested")
            return
        
        # Run learning
        print("\n[3/6] Running learning cycle...")
        metrics = self.pipeline.learn_cycle(
            domains=list(set(ingested_domains)),
            learn_universal=True,
            learn_domain_specific=True
        )
        
        print(f"  ✓ Learning complete")
        print(f"    Discovered: {metrics.patterns_discovered}")
        print(f"    Validated: {metrics.patterns_validated}")
        print(f"    Improvement: {metrics.improvement:+.3f}")
        
        # Validate improvements
        print("\n[4/6] Validating improvements...")
        if metrics.improvement > 0.01:
            print(f"  ✓ Significant improvement: {metrics.improvement:+.3f}")
        else:
            print(f"  ⊙ Minor improvement: {metrics.improvement:+.3f}")
        
        # Update registry
        print("\n[5/6] Updating registry...")
        registry = get_domain_registry()
        
        for domain in set(ingested_domains):
            # Get domain info
            domain_entry = registry.get_domain(domain)
            
            if domain_entry:
                # Update existing
                domain_entry.patterns_count = len(
                    self.pipeline.get_archetypes(domain).get('domain_specific', {})
                )
                domain_entry.last_updated = datetime.now().isoformat()
        
        registry.save()
        print(f"  ✓ Registry updated")
        
        # Save state
        print("\n[6/6] Saving pipeline state...")
        state_path = Path(__file__).parent.parent / 'pipeline_state.json'
        self.pipeline.save_state(state_path)
        print(f"  ✓ State saved to {state_path}")
        
        print(f"\n{'='*80}")
        print("CYCLE COMPLETE")
        print(f"{'='*80}")
    
    def run_continuous(self, max_cycles: Optional[int] = None):
        """
        Run continuous learning loop.
        
        Parameters
        ----------
        max_cycles : int, optional
            Maximum cycles (None = infinite)
        """
        print("="*80)
        print("STARTING CONTINUOUS LEARNING")
        print("="*80)
        print(f"Check interval: {self.check_interval}s")
        print(f"Min new samples: {self.min_new_samples}")
        
        cycle = 0
        
        try:
            while max_cycles is None or cycle < max_cycles:
                self.run_cycle()
                
                cycle += 1
                
                if max_cycles is None or cycle < max_cycles:
                    print(f"\nSleeping {self.check_interval}s until next cycle...")
                    time.sleep(self.check_interval)
        
        except KeyboardInterrupt:
            print("\n\nContinuous learning interrupted by user")
            print(f"Completed {cycle} cycles")


def run_workflow(
    continuous: bool = False,
    interval: int = 3600,
    max_cycles: Optional[int] = None
):
    """
    Run continuous learning workflow.
    
    Parameters
    ----------
    continuous : bool
        Run continuously
    interval : int
        Check interval in seconds
    max_cycles : int, optional
        Maximum cycles
    """
    workflow = ContinuousLearningWorkflow(
        check_interval=interval,
        min_new_samples=10,
        auto_commit=False
    )
    
    if continuous:
        workflow.run_continuous(max_cycles=max_cycles)
    else:
        workflow.run_cycle()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Continuous learning workflow')
    parser.add_argument('--continuous', action='store_true', help='Run continuously')
    parser.add_argument('--interval', type=int, default=3600, help='Check interval (seconds)')
    parser.add_argument('--max-cycles', type=int, help='Maximum cycles')
    
    args = parser.parse_args()
    
    run_workflow(
        continuous=args.continuous,
        interval=args.interval,
        max_cycles=args.max_cycles
    )

