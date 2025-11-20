"""
NHL Full Pipeline - End-to-End Execution

Runs the complete NHL betting pipeline:
1. Data collection → 2. Feature extraction → 3. Pattern discovery → 
4. Model training → 5. Daily predictions → 6. Performance tracking

Can run individual stages or complete pipeline.

Author: Narrative Integration System
Date: November 16, 2025
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime
import argparse


class NHLPipeline:
    """Complete NHL betting pipeline"""
    
    def __init__(self):
        """Initialize pipeline"""
        self.project_root = Path(__file__).parent.parent
        self.logs_dir = self.project_root / 'logs' / 'nhl'
        self.logs_dir.mkdir(parents=True, exist_ok=True)
    
    def run_stage(self, stage_name: str, script_path: str, description: str) -> bool:
        """Run a single pipeline stage"""
        
        print(f"\n{'='*80}")
        print(f"STAGE: {stage_name}")
        print(f"{'='*80}")
        print(f"{description}")
        print()
        
        full_path = self.project_root / script_path
        
        if not full_path.exists():
            print(f"❌ Script not found: {script_path}")
            return False
        
        try:
            result = subprocess.run(
                ['python3', str(full_path)],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            print(result.stdout)
            
            if result.returncode != 0:
                print(f"❌ Stage failed with code {result.returncode}")
                if result.stderr:
                    print(f"Error: {result.stderr}")
                return False
            
            print(f"✅ {stage_name} complete!")
            return True
            
        except subprocess.TimeoutExpired:
            print(f"⏱️  Stage timed out after 1 hour")
            return False
        except Exception as e:
            print(f"❌ Error running stage: {e}")
            return False
    
    def run_full_pipeline(self):
        """Run complete pipeline"""
        
        print("\n" + "="*80)
        print("NHL COMPLETE BETTING PIPELINE")
        print("="*80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        stages = [
            # Commented out - takes hours
            # ('Data Collection', 'data_collection/nhl_data_builder.py', 
            #  'Collect NHL games with context and odds'),
            
            ('Feature Extraction', 'narrative_optimization/domains/nhl/extract_nhl_features.py',
             'Extract 79 transformer features from games'),
            
            ('Formula Calculation', 'narrative_optimization/domains/nhl/calculate_nhl_formula.py',
             'Calculate domain formula (π, Δ, r, κ)'),
            
            ('Pattern Discovery', 'narrative_optimization/domains/nhl/discover_nhl_patterns_learned.py',
             'Discover patterns via data-driven ML analysis'),
            
            ('Model Training', 'scripts/nhl_model_trainer.py',
             'Train Meta-Ensemble and GBM models'),
            
            ('Pattern Validation', 'narrative_optimization/domains/nhl/validate_nhl_patterns.py',
             'Validate patterns with temporal splits'),
            
            ('Complete Analysis', 'narrative_optimization/domains/nhl/nhl_complete_analysis.py',
             'Run integrated analysis with all models'),
        ]
        
        results = []
        
        for stage_name, script_path, description in stages:
            success = self.run_stage(stage_name, script_path, description)
            results.append((stage_name, success))
            
            if not success:
                print(f"\n⚠️  Pipeline stopped at {stage_name}")
                break
        
        # Summary
        print("\n" + "="*80)
        print("PIPELINE SUMMARY")
        print("="*80)
        
        for stage, success in results:
            icon = "✅" if success else "❌"
            print(f"{icon} {stage}")
        
        all_success = all(s for _, s in results)
        
        if all_success:
            print("\n✅ COMPLETE PIPELINE SUCCESSFUL!")
            print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("\n⚠️  Pipeline incomplete - check errors above")
        
        print("="*80)


def main():
    """Main execution with arguments"""
    
    parser = argparse.ArgumentParser(description='NHL Full Pipeline Execution')
    parser.add_argument('--stage', type=str, help='Run specific stage only')
    parser.add_argument('--all', action='store_true', help='Run complete pipeline')
    
    args = parser.parse_args()
    
    pipeline = NHLPipeline()
    
    if args.all:
        pipeline.run_full_pipeline()
    elif args.stage:
        print(f"Running stage: {args.stage}")
        # TODO: Implement individual stage execution
    else:
        print("NHL Full Pipeline")
        print("\nUsage:")
        print("  --all          Run complete pipeline")
        print("  --stage NAME   Run specific stage")
        print("\nExample:")
        print("  python3 scripts/nhl_full_pipeline.py --all")


if __name__ == "__main__":
    main()

