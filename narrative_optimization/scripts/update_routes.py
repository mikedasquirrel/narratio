#!/usr/bin/env python3
"""
Update Routes Script

Helper script to update Flask routes to use unified results format.
This script adds the necessary imports and functions to routes.
"""

import sys
from pathlib import Path
import re

project_root = Path(__file__).parent.parent.parent
routes_dir = project_root.parent / 'routes'


def update_route_file(route_path: Path, domain_name: str) -> bool:
    """Update a route file to use unified results"""
    try:
        content = route_path.read_text()
        
        # Check if already updated
        if 'load_unified_results' in content:
            print(f"  ⚠ {domain_name}: Already updated")
            return False
        
        # Add import if not present
        if 'from utils.result_loader import' not in content:
            # Find where to insert import
            import_pattern = r'(from flask import.*?\n)'
            match = re.search(import_pattern, content)
            if match:
                insert_pos = match.end()
                import_line = f"import sys\nfrom pathlib import Path\n\nsys.path.insert(0, str(Path(__file__).parent.parent / 'narrative_optimization'))\nfrom utils.result_loader import load_unified_results, extract_stats_from_results, get_chart_data\n\n"
                content = content[:insert_pos] + import_line + content[insert_pos:]
        
        # Update load_results function
        load_func_pattern = r'def load_\w+_results\(\):.*?(?=\n@|\ndef |\Z)'
        match = re.search(load_func_pattern, content, re.DOTALL)
        if match:
            old_func = match.group(0)
            # Create new function that tries unified format first
            new_func = f'''def load_{domain_name}_results():
    """Load {domain_name} analysis results (unified format preferred)"""
    if 'results' not in _cache:
        # Try unified format first
        unified_results = load_unified_results('{domain_name}')
        if unified_results:
            _cache['results'] = unified_results
        else:
            # Fallback to legacy format
            try:
                path = Path(__file__).parent.parent / 'narrative_optimization' / 'domains' / '{domain_name}' / '{domain_name}_analysis_results.json'
                with open(path) as f:
                    _cache['results'] = json.load(f)
            except:
                _cache['results'] = None
    return _cache['results']
'''
            content = content.replace(old_func, new_func)
        
        # Write updated content
        route_path.write_text(content)
        print(f"  ✓ {domain_name}: Updated")
        return True
        
    except Exception as e:
        print(f"  ✗ {domain_name}: Error - {e}")
        return False


def main():
    """Update all route files"""
    print("=" * 80)
    print("UPDATING ROUTES FOR UNIFIED RESULTS")
    print("=" * 80)
    
    route_files = {
        'tennis': 'tennis.py',
        'golf': 'golf.py',
        'nfl': 'nfl.py',
        'ufc': 'ufc.py',
        'movies': 'movies.py',
        'imdb': 'imdb.py',
        'oscars': 'oscars.py',
        'music': 'music.py',
        'startups': 'startups.py',
        'crypto': 'crypto.py',
        'mental_health': 'mental_health.py',
        'housing': 'housing.py',
        'wwe_domain': 'wwe_domain.py',
    }
    
    updated = 0
    for domain_name, filename in route_files.items():
        route_path = routes_dir / filename
        if route_path.exists():
            if update_route_file(route_path, domain_name):
                updated += 1
        else:
            print(f"  ⚠ {domain_name}: File not found ({filename})")
    
    print(f"\n✓ Updated {updated} route files")


if __name__ == '__main__':
    main()

