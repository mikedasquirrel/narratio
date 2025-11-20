"""
Script to fetch and prepare REAL NBA data for narrative analysis.

This script:
1. Fetches actual NBA games from official API
2. Collects real team descriptions
3. Implements proper temporal split (exclude every 10th season)
4. Prepares data for narrative feature extraction

Run this first to populate the dataset with real data.
"""

import sys
from pathlib import Path
import json
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from domains.nba.real_data_collector import RealNBADataCollector, HybridNBACollector


def main():
    """Fetch real NBA data and prepare for analysis."""
    
    parser = argparse.ArgumentParser(description='Fetch real NBA data')
    parser.add_argument('--seasons', nargs='+', default=['2022-23', '2023-24'],
                       help='Seasons to fetch (e.g., 2022-23 2023-24)')
    parser.add_argument('--output', default='data/nba_real_games.json',
                       help='Output file path')
    parser.add_argument('--odds-api-key', default=None,
                       help='The Odds API key for betting lines (optional)')
    parser.add_argument('--hybrid', action='store_true',
                       help='Use hybrid mode (real games + generated narratives)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("REAL NBA DATA COLLECTION")
    print("="*70 + "\n")
    
    try:
        if args.hybrid:
            print("Mode: HYBRID (real games + generated narratives)")
            collector = HybridNBACollector()
            
            all_games = []
            for season in args.seasons:
                games = collector.fetch_games_with_narratives(season)
                all_games.extend(games)
            
            dataset = all_games
        
        else:
            print("Mode: FULL REAL DATA (real games + scraped narratives)")
            collector = RealNBADataCollector(odds_api_key=args.odds_api_key)
            dataset = collector.create_complete_dataset(
                seasons=args.seasons,
                save_path=args.output
            )
        
        # Implement temporal split
        print("\nImplementing temporal train/test split...")
        print("Rule: Exclude every 10th season for testing\n")
        
        # For the seasons we have, mark which are test
        all_seasons = sorted(set(g['season'] for g in dataset))
        test_seasons = []
        train_seasons = []
        
        for idx, season in enumerate(all_seasons):
            if (idx + 1) % 10 == 0:
                test_seasons.append(season)
            else:
                train_seasons.append(season)
        
        print(f"Training seasons: {train_seasons}")
        print(f"Test seasons: {test_seasons}")
        
        train_data = [g for g in dataset if g['season'] in train_seasons]
        test_data = [g for g in dataset if g['season'] in test_seasons]
        
        print(f"\nTraining games: {len(train_data)}")
        print(f"Test games: {len(test_data)}")
        
        # Save split data
        output_dir = Path(args.output).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'train_games.json', 'w') as f:
            json.dump(train_data, f, indent=2)
        
        with open(output_dir / 'test_games.json', 'w') as f:
            json.dump(test_data, f, indent=2)
        
        print(f"\n✅ Data saved:")
        print(f"   All games: {args.output}")
        print(f"   Training: {output_dir / 'train_games.json'}")
        print(f"   Testing: {output_dir / 'test_games.json'}")
        
        print("\n" + "="*70)
        print("DATA COLLECTION COMPLETE")
        print("="*70 + "\n")
        
        print("NEXT STEPS:")
        print("1. Run narrative feature extraction")
        print("2. Train prediction models")
        print("3. Backtest betting strategies")
        print("4. View results in web interface")
        
        return dataset
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    main()

