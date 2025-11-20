"""
Tennis Data Collection Module

Parses Tennis Abstract dataset (Jeff Sackmann) for comprehensive match analysis.
Includes player names, rankings, surfaces, tournaments for nominative-rich analysis.
"""

import pandas as pd
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import glob
import random
import warnings
warnings.filterwarnings('ignore')


class TennisDataCollector:
    """
    Collects tennis match data from Tennis Abstract CSV files.
    
    For each match, collects:
    - Player names, rankings, ages, countries
    - Surface type (clay/grass/hard)
    - Tournament level (Grand Slam/Masters/ATP)
    - Match scores and statistics
    - Head-to-head context
    - Career achievements
    """
    
    def __init__(self, years: Optional[List[int]] = None):
        """
        Initialize tennis data collector.
        
        Parameters
        ----------
        years : list of int, optional
            Years to collect (e.g., [2000, 2001, ..., 2024])
            If None, defaults to 2000-2024
        """
        self.years = years or list(range(2000, 2025))
        self.data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'tennis_atp'
        print(f"Initializing Tennis Data Collector for years: {min(self.years)}-{max(self.years)}")
        print(f"Data directory: {self.data_dir}")
        
    def collect_all_matches(self, output_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Collect complete tennis dataset for configured years.
        
        Returns
        -------
        matches : list of dict
            Complete match data with nominative information
        """
        print("\n" + "="*80)
        print("TENNIS DATA COLLECTION - COMPREHENSIVE DATASET")
        print("="*80)
        
        # Step 1: Load all CSV files
        print("\n[1/4] Loading CSV files...")
        all_matches = self._load_csv_files()
        print(f"✓ Loaded {len(all_matches)} matches")
        
        # Step 2: Parse and structure matches
        print("\n[2/4] Parsing and structuring matches...")
        structured_matches = self._structure_matches(all_matches)
        print(f"✓ Structured {len(structured_matches)} matches")
        
        # Step 3: Add head-to-head records
        print("\n[3/4] Calculating head-to-head records...")
        enriched_matches = self._add_h2h_records(structured_matches)
        print(f"✓ Enriched {len(enriched_matches)} matches")
        
        # Step 4: Add context metadata
        print("\n[4/4] Adding context metadata...")
        complete_matches = self._add_context(enriched_matches)
        print(f"✓ Added context to {len(complete_matches)} matches")
        
        # Save if output path provided
        if output_path:
            self._save_dataset(complete_matches, output_path)
        
        return complete_matches
    
    def _load_csv_files(self) -> pd.DataFrame:
        """Load all CSV files for specified years."""
        dfs = []
        
        for year in self.years:
            csv_path = self.data_dir / f'atp_matches_{year}.csv'
            
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path, encoding='utf-8', low_memory=False)
                    df['year'] = year
                    dfs.append(df)
                    print(f"  {year}: {len(df)} matches")
                except Exception as e:
                    print(f"  {year}: Error - {e}")
            else:
                print(f"  {year}: File not found")
        
        combined = pd.concat(dfs, ignore_index=True)
        return combined
    
    def _structure_matches(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Structure matches into standardized format with randomized player assignment."""
        matches = []
        
        for idx, row in df.iterrows():
            # Randomize which player is player1 vs player2 (50/50 split)
            # This removes positional bias and ensures ~50% player1 win rate
            swap_players = random.random() < 0.5
            
            if swap_players:
                # Loser is player1, winner is player2
                player1_data = {
                    'name': row.get('loser_name', 'Unknown'),
                    'id': row.get('loser_id', ''),
                    'seed': int(row['loser_seed']) if pd.notna(row.get('loser_seed')) else None,
                    'ranking': int(row['loser_rank']) if pd.notna(row.get('loser_rank')) else None,
                    'rank_points': int(row['loser_rank_points']) if pd.notna(row.get('loser_rank_points')) else None,
                    'age': float(row['loser_age']) if pd.notna(row.get('loser_age')) else None,
                    'country': row.get('loser_ioc', 'Unknown'),
                    'hand': row.get('loser_hand', 'Unknown'),
                    'height_cm': int(row['loser_ht']) if pd.notna(row.get('loser_ht')) else None
                }
                player2_data = {
                    'name': row.get('winner_name', 'Unknown'),
                    'id': row.get('winner_id', ''),
                    'seed': int(row['winner_seed']) if pd.notna(row.get('winner_seed')) else None,
                    'ranking': int(row['winner_rank']) if pd.notna(row.get('winner_rank')) else None,
                    'rank_points': int(row['winner_rank_points']) if pd.notna(row.get('winner_rank_points')) else None,
                    'age': float(row['winner_age']) if pd.notna(row.get('winner_age')) else None,
                    'country': row.get('winner_ioc', 'Unknown'),
                    'hand': row.get('winner_hand', 'Unknown'),
                    'height_cm': int(row['winner_ht']) if pd.notna(row.get('winner_ht')) else None
                }
                player1_won = False
                # Swap stats too
                match_stats = {
                    'w_ace': int(row['l_ace']) if pd.notna(row.get('l_ace')) else None,
                    'w_df': int(row['l_df']) if pd.notna(row.get('l_df')) else None,
                    'l_ace': int(row['w_ace']) if pd.notna(row.get('w_ace')) else None,
                    'l_df': int(row['w_df']) if pd.notna(row.get('w_df')) else None,
                }
            else:
                # Winner is player1, loser is player2 (original)
                player1_data = {
                    'name': row.get('winner_name', 'Unknown'),
                    'id': row.get('winner_id', ''),
                    'seed': int(row['winner_seed']) if pd.notna(row.get('winner_seed')) else None,
                    'ranking': int(row['winner_rank']) if pd.notna(row.get('winner_rank')) else None,
                    'rank_points': int(row['winner_rank_points']) if pd.notna(row.get('winner_rank_points')) else None,
                    'age': float(row['winner_age']) if pd.notna(row.get('winner_age')) else None,
                    'country': row.get('winner_ioc', 'Unknown'),
                    'hand': row.get('winner_hand', 'Unknown'),
                    'height_cm': int(row['winner_ht']) if pd.notna(row.get('winner_ht')) else None
                }
                player2_data = {
                    'name': row.get('loser_name', 'Unknown'),
                    'id': row.get('loser_id', ''),
                    'seed': int(row['loser_seed']) if pd.notna(row.get('loser_seed')) else None,
                    'ranking': int(row['loser_rank']) if pd.notna(row.get('loser_rank')) else None,
                    'rank_points': int(row['loser_rank_points']) if pd.notna(row.get('loser_rank_points')) else None,
                    'age': float(row['loser_age']) if pd.notna(row.get('loser_age')) else None,
                    'country': row.get('loser_ioc', 'Unknown'),
                    'hand': row.get('loser_hand', 'Unknown'),
                    'height_cm': int(row['loser_ht']) if pd.notna(row.get('loser_ht')) else None
                }
                player1_won = True
                match_stats = {
                    'w_ace': int(row['w_ace']) if pd.notna(row.get('w_ace')) else None,
                    'w_df': int(row['w_df']) if pd.notna(row.get('w_df')) else None,
                    'l_ace': int(row['l_ace']) if pd.notna(row.get('l_ace')) else None,
                    'l_df': int(row['l_df']) if pd.notna(row.get('l_df')) else None,
                }
            
            match = {
                'match_id': f"{row['year']}_{row.get('tourney_id', idx)}_{row.get('match_num', idx)}",
                'tournament': row.get('tourney_name', 'Unknown'),
                'date': str(row.get('tourney_date', '')),
                'year': int(row['year']),
                'surface': self._normalize_surface(row.get('surface')),
                'level': self._categorize_level(row.get('tourney_level')),
                'round': row.get('round', 'Unknown'),
                'draw_size': int(row['draw_size']) if pd.notna(row.get('draw_size')) else None,
                'best_of': int(row['best_of']) if pd.notna(row.get('best_of')) else 3,
                
                'player1': player1_data,
                'player2': player2_data,
                
                # Match outcome - now properly randomized
                'player1_won': player1_won,
                'score': row.get('score', 'Unknown'),
                'minutes': int(row['minutes']) if pd.notna(row.get('minutes')) else None,
                
                # Match statistics
                'match_stats': match_stats,
                
                # Placeholder for h2h and context (added later)
                'head_to_head': {},
                'context': {}
            }
            
            matches.append(match)
            
            if (idx + 1) % 5000 == 0:
                print(f"  Processed {idx + 1} matches...")
        
        return matches
    
    def _normalize_surface(self, surface: str) -> str:
        """Normalize surface types."""
        if pd.isna(surface):
            return 'unknown'
        
        surface = str(surface).lower().strip()
        
        if 'clay' in surface:
            return 'clay'
        elif 'grass' in surface:
            return 'grass'
        elif 'hard' in surface:
            return 'hard'
        elif 'carpet' in surface:
            return 'carpet'
        else:
            return 'unknown'
    
    def _categorize_level(self, level: str) -> str:
        """Categorize tournament level."""
        if pd.isna(level):
            return 'unknown'
        
        level = str(level).upper().strip()
        
        if level == 'G':
            return 'grand_slam'
        elif level == 'M':
            return 'masters_1000'
        elif level == 'A':
            return 'atp_500'
        elif level == 'D':
            return 'davis_cup'
        elif level == 'F':
            return 'atp_finals'
        else:
            return 'atp_250'
    
    def _add_h2h_records(self, matches: List[Dict]) -> List[Dict]:
        """Calculate head-to-head records for each match."""
        # Build player matchup history
        h2h_records = {}
        
        for match in matches:
            p1_id = match['player1']['id']
            p2_id = match['player2']['id']
            
            # Create sorted key for matchup
            if p1_id and p2_id:
                matchup_key = tuple(sorted([p1_id, p2_id]))
                
                # Get current h2h before this match
                if matchup_key not in h2h_records:
                    h2h_records[matchup_key] = {
                        'total': 0,
                        'p1_wins': 0,
                        'p2_wins': 0,
                        'by_surface': {}
                    }
                
                h2h = h2h_records[matchup_key]
                
                match['head_to_head'] = {
                    'total_matches': h2h['total'],
                    'player1_wins': h2h['p1_wins'] if p1_id == matchup_key[0] else h2h['p2_wins'],
                    'player2_wins': h2h['p2_wins'] if p1_id == matchup_key[0] else h2h['p1_wins'],
                    'surface_record': h2h['by_surface'].get(match['surface'], {'p1': 0, 'p2': 0})
                }
                
                # Update h2h for next matches
                h2h['total'] += 1
                if p1_id == matchup_key[0]:
                    h2h['p1_wins'] += 1
                else:
                    h2h['p2_wins'] += 1
                
                # Update surface record
                surf = match['surface']
                if surf not in h2h['by_surface']:
                    h2h['by_surface'][surf] = {'p1': 0, 'p2': 0}
                
                if p1_id == matchup_key[0]:
                    h2h['by_surface'][surf]['p1'] += 1
                else:
                    h2h['by_surface'][surf]['p2'] += 1
        
        return matches
    
    def _add_context(self, matches: List[Dict]) -> List[Dict]:
        """Add match context metadata."""
        for match in matches:
            p1_rank = match['player1']['ranking']
            p2_rank = match['player2']['ranking']
            
            context = {
                'grand_slam': match['level'] == 'grand_slam',
                'masters': match['level'] == 'masters_1000',
                'surface': match['surface'],
                'year': match['year'],
                'ranking_upset': False,
                'top_10_match': False,
                'rivalry': False,
                'surface_specialist_advantage': None
            }
            
            # Ranking upset (lower ranked won)
            if p1_rank and p2_rank:
                context['ranking_upset'] = p1_rank > p2_rank
                context['top_10_match'] = p1_rank <= 10 and p2_rank <= 10
            
            # H2H rivalry (played 10+ times)
            h2h_total = match['head_to_head'].get('total_matches', 0)
            context['rivalry'] = h2h_total >= 10
            
            match['context'] = context
        
        return matches
    
    def _save_dataset(self, matches: List[Dict], output_path: str):
        """Save dataset to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(matches, f, indent=2)
        
        print(f"\n✓ Dataset saved to: {output_file}")
        print(f"  Total matches: {len(matches)}")


def main():
    """Main execution: collect complete tennis dataset."""
    collector = TennisDataCollector(years=list(range(2000, 2025)))
    
    output_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'tennis_matches_raw.json'
    
    matches = collector.collect_all_matches(output_path=str(output_path))
    
    # Statistics
    print("\n" + "="*80)
    print("DATA COLLECTION COMPLETE")
    print("="*80)
    print(f"Total matches: {len(matches)}")
    print(f"Years: {min([m['year'] for m in matches])}-{max([m['year'] for m in matches])}")
    
    # Surface breakdown
    surfaces = {}
    levels = {}
    for m in matches:
        surf = m['surface']
        level = m['level']
        surfaces[surf] = surfaces.get(surf, 0) + 1
        levels[level] = levels.get(level, 0) + 1
    
    print(f"\nSurface breakdown:")
    for surf, count in sorted(surfaces.items(), key=lambda x: x[1], reverse=True):
        print(f"  {surf}: {count} ({100*count/len(matches):.1f}%)")
    
    print(f"\nTournament levels:")
    for level, count in sorted(levels.items(), key=lambda x: x[1], reverse=True):
        print(f"  {level}: {count} ({100*count/len(matches):.1f}%)")
    
    print(f"\nOutput: {output_path}")


if __name__ == '__main__':
    main()

