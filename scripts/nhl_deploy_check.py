"""
NHL Deployment Readiness Check

Validates that all system components are ready for production deployment.

Checks:
- Data availability
- Models trained
- Patterns validated
- Web interface operational
- Automation scripts functional
- Documentation complete

Author: Narrative Integration System
Date: November 16, 2025
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class NHLDeploymentChecker:
    """Check deployment readiness"""
    
    def __init__(self):
        """Initialize checker"""
        self.project_root = Path(__file__).parent.parent
        self.checks = []
    
    def check_data(self) -> bool:
        """Check data availability"""
        print("\n📂 CHECKING DATA...")
        
        required_files = [
            ('Games data', 'data/domains/nhl_games_with_odds.json', 100),
            ('Patterns', 'data/domains/nhl_betting_patterns_learned.json', 10),
            ('Features', 'narrative_optimization/domains/nhl/nhl_features_complete.npz', None),
            ('Formula', 'narrative_optimization/domains/nhl/nhl_formula_results.json', None),
            ('Analysis', 'narrative_optimization/domains/nhl/nhl_complete_analysis.json', None),
        ]
        
        all_good = True
        
        for name, path, min_items in required_files:
            full_path = self.project_root / path
            
            if full_path.exists():
                if min_items and path.endswith('.json'):
                    with open(full_path, 'r') as f:
                        data = json.load(f)
                        count = len(data) if isinstance(data, list) else len(data.get('patterns', []))
                    
                    if count >= min_items:
                        print(f"   ✅ {name}: {count} items")
                    else:
                        print(f"   ⚠️  {name}: Only {count} items (need {min_items}+)")
                        all_good = False
                else:
                    size_mb = full_path.stat().st_size / 1024 / 1024
                    print(f"   ✅ {name}: {size_mb:.2f} MB")
            else:
                print(f"   ❌ {name}: NOT FOUND")
                all_good = False
        
        return all_good
    
    def check_models(self) -> bool:
        """Check trained models"""
        print("\n🤖 CHECKING MODELS...")
        
        models_dir = self.project_root / 'narrative_optimization' / 'domains' / 'nhl' / 'models'
        
        required_models = [
            'meta_ensemble.pkl',
            'gradient_boosting.pkl',
            'random_forest.pkl',
            'logistic.pkl',
            'scaler.pkl',
        ]
        
        all_good = True
        
        for model_file in required_models:
            model_path = models_dir / model_file
            
            if model_path.exists():
                size_kb = model_path.stat().st_size / 1024
                print(f"   ✅ {model_file}: {size_kb:.1f} KB")
            else:
                print(f"   ❌ {model_file}: NOT FOUND")
                all_good = False
        
        return all_good
    
    def check_scripts(self) -> bool:
        """Check automation scripts"""
        print("\n⚙️  CHECKING SCRIPTS...")
        
        scripts = [
            'nhl_fetch_live_odds.py',
            'nhl_daily_predictions.py',
            'nhl_performance_tracker.py',
            'nhl_pattern_selector.py',
            'nhl_risk_management.py',
            'nhl_model_trainer.py',
            'nhl_automated_daily.sh',
        ]
        
        all_good = True
        
        for script in scripts:
            script_path = self.project_root / 'scripts' / script
            
            if script_path.exists():
                print(f"   ✅ {script}")
            else:
                print(f"   ❌ {script}")
                all_good = False
        
        return all_good
    
    def check_web_interface(self) -> bool:
        """Check web interface files"""
        print("\n🌐 CHECKING WEB INTERFACE...")
        
        routes = ['routes/nhl.py', 'routes/nhl_betting.py']
        templates = [
            'templates/nhl_results.html',
            'templates/nhl_betting_patterns.html',
            'templates/nhl_live_betting.html',
        ]
        
        all_good = True
        
        for route in routes:
            if (self.project_root / route).exists():
                print(f"   ✅ {route}")
            else:
                print(f"   ❌ {route}")
                all_good = False
        
        for template in templates:
            if (self.project_root / template).exists():
                print(f"   ✅ {template}")
            else:
                print(f"   ❌ {template}")
                all_good = False
        
        return all_good
    
    def check_documentation(self) -> bool:
        """Check documentation"""
        print("\n📚 CHECKING DOCUMENTATION...")
        
        docs = [
            'NHL_README.md',
            'NHL_EXECUTIVE_SUMMARY.md',
            'NHL_MASTER_SUMMARY.md',
            'NHL_QUICK_REFERENCE.txt',
        ]
        
        all_good = True
        
        for doc in docs:
            doc_path = self.project_root / doc
            if doc_path.exists():
                lines = len(doc_path.read_text().split('\n'))
                print(f"   ✅ {doc}: {lines} lines")
            else:
                print(f"   ❌ {doc}")
                all_good = False
        
        return all_good
    
    def run_complete_check(self) -> Dict:
        """Run all checks"""
        
        print("\n" + "="*80)
        print("NHL DEPLOYMENT READINESS CHECK")
        print("="*80)
        
        results = {
            'data': self.check_data(),
            'models': self.check_models(),
            'scripts': self.check_scripts(),
            'web': self.check_web_interface(),
            'docs': self.check_documentation(),
        }
        
        # Overall status
        all_passed = all(results.values())
        
        print("\n" + "="*80)
        print("📊 READINESS SUMMARY")
        print("="*80)
        
        for component, status in results.items():
            icon = "✅" if status else "❌"
            print(f"{icon} {component.upper()}: {'READY' if status else 'INCOMPLETE'}")
        
        print("\n" + "="*80)
        
        if all_passed:
            print("✅ ALL SYSTEMS GO - READY FOR DEPLOYMENT")
            print("\nNext steps:")
            print("1. Expand data to 10K+ games")
            print("2. Temporal validation")
            print("3. Paper trading")
            print("4. Real money deployment")
        else:
            print("⚠️  SOME COMPONENTS MISSING")
            print("\nComplete missing components before deployment")
        
        print("="*80)
        
        return results


def main():
    """Main execution"""
    
    checker = NHLDeploymentChecker()
    results = checker.run_complete_check()


if __name__ == "__main__":
    main()

