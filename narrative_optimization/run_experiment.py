#!/usr/bin/env python
"""
Command-line interface for running narrative optimization experiments.

Usage:
    python run_experiment.py --experiment 01_baseline_comparison
    python run_experiment.py --list
"""

import argparse
import sys
from pathlib import Path
import importlib.util


def list_experiments():
    """List all available experiments."""
    experiments_dir = Path(__file__).parent / "experiments"
    
    if not experiments_dir.exists():
        print("No experiments directory found.")
        return
    
    print("Available Experiments:")
    print("=" * 60)
    
    for exp_dir in sorted(experiments_dir.iterdir()):
        if exp_dir.is_dir() and not exp_dir.name.startswith('_'):
            run_script = exp_dir / "run_experiment.py"
            if run_script.exists():
                print(f"  {exp_dir.name}")
                
                # Try to extract description from the script
                try:
                    with open(run_script, 'r') as f:
                        lines = f.readlines()
                        for line in lines[:20]:  # Check first 20 lines
                            if line.strip().startswith('"""') or line.strip().startswith("'''"):
                                continue
                            if 'description' in line.lower() or 'test' in line.lower():
                                desc = line.strip().replace('#', '').strip()
                                if desc:
                                    print(f"    {desc}")
                                    break
                except:
                    pass
                print()


def run_experiment(experiment_name: str):
    """
    Run a specific experiment by name.
    
    Parameters
    ----------
    experiment_name : str
        Name of the experiment directory (e.g., '01_baseline_comparison')
    """
    experiments_dir = Path(__file__).parent / "experiments"
    experiment_dir = experiments_dir / experiment_name
    
    if not experiment_dir.exists():
        print(f"Error: Experiment '{experiment_name}' not found.")
        print(f"Looking in: {experiment_dir}")
        print("\nUse --list to see available experiments.")
        sys.exit(1)
    
    run_script = experiment_dir / "run_experiment.py"
    
    if not run_script.exists():
        print(f"Error: No run_experiment.py found in {experiment_dir}")
        sys.exit(1)
    
    print(f"Running experiment: {experiment_name}")
    print(f"Script: {run_script}")
    print()
    
    # Load and execute the experiment module
    spec = importlib.util.spec_from_file_location(
        f"experiment_{experiment_name}",
        run_script
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    
    # The experiment script should execute when loaded
    # If it has a main function, it will run via if __name__ == '__main__'


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run narrative optimization experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py --list
  python run_experiment.py --experiment 01_baseline_comparison
  python run_experiment.py -e 01_baseline_comparison

For more information, see the project README.
        """
    )
    
    parser.add_argument(
        '-e', '--experiment',
        type=str,
        help='Name of the experiment to run'
    )
    
    parser.add_argument(
        '-l', '--list',
        action='store_true',
        help='List all available experiments'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (optional)'
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_experiments()
    elif args.experiment:
        run_experiment(args.experiment)
    else:
        parser.print_help()
        print("\nUse --list to see available experiments")


if __name__ == '__main__':
    main()

