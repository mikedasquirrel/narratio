#!/usr/bin/env python3
"""
Phase 1: Data Validation & Preparation
Validates enriched NFL data and creates validation report
"""

import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def main():
    print("="*60)
    print(f"PHASE 1: DATA VALIDATION - {datetime.now().strftime('%H:%M:%S')}")
    print("="*60)
    
    # Load enriched data
    data_path = Path(__file__).parent.parent.parent / "data" / "domains" / "nfl_enriched_with_rosters.json"
    
    print(f"\n📂 Loading: {data_path.name}")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    games = data['games']
    print(f"✓ Loaded {len(games):,} games")
    
    # Validation checks
    validation = {
        'timestamp': datetime.now().isoformat(),
        'total_games': len(games),
        'seasons': {'min': min(g['season'] for g in games), 
                   'max': max(g['season'] for g in games)},
        'coverage': {},
        'quality_checks': {},
        'sample_game': None
    }
    
    # Check field coverage
    print("\n🔍 Validating field coverage...")
    validation['coverage']['qb_data'] = sum(1 for g in games if g.get('home_qb') and g.get('away_qb'))
    validation['coverage']['key_players'] = sum(1 for g in games if g.get('home_key_players'))
    validation['coverage']['team_records'] = sum(1 for g in games if g.get('home_record_before'))
    validation['coverage']['matchup_history'] = sum(1 for g in games if g.get('matchup_history'))
    validation['coverage']['spread_data'] = sum(1 for g in games if g.get('spread_line') is not None)
    validation['coverage']['playoff_games'] = sum(1 for g in games if g.get('playoff'))
    validation['coverage']['overtime_games'] = sum(1 for g in games if g.get('overtime'))
    
    for field, count in validation['coverage'].items():
        pct = 100 * count / len(games)
        print(f"  {field}: {count:,} ({pct:.1f}%)")
    
    # Quality checks
    print("\n✅ Running quality checks...")
    validation['quality_checks']['missing_scores'] = sum(1 for g in games if g.get('home_score') is None)
    validation['quality_checks']['missing_teams'] = sum(1 for g in games if not g.get('home_team'))
    validation['quality_checks']['invalid_weeks'] = sum(1 for g in games if g.get('week') and g['week'] > 20)
    
    all_passed = all(v == 0 for v in validation['quality_checks'].values())
    status = "✓ PASSED" if all_passed else "⚠ WARNINGS"
    print(f"  {status}")
    
    # Get sample enriched game
    sample = next((g for g in games if g.get('home_qb') and g.get('matchup_history') and g['season'] == 2024), None)
    if sample:
        validation['sample_game'] = {
            'matchup': f"{sample['away_team']} @ {sample['home_team']}",
            'week': sample.get('week'),
            'season': sample['season'],
            'qbs': f"{sample.get('away_qb', {}).get('qb_name', 'N/A')} vs {sample.get('home_qb', {}).get('qb_name', 'N/A')}",
            'records': f"{sample.get('away_record_before', '?')} @ {sample.get('home_record_before', '?')}",
            'has_matchup_history': bool(sample.get('matchup_history'))
        }
    
    # Save validation report
    output_path = Path(__file__).parent.parent.parent / "data" / "domains" / "nfl_data_validation.json"
    with open(output_path, 'w') as f:
        json.dump(validation, f, indent=2)
    
    print(f"\n✓ Validation report saved: {output_path.name}")
    print(f"\n{'='*60}")
    print("PHASE 1 COMPLETE ✓")
    print(f"{'='*60}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

