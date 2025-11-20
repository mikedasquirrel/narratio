"""
NBA Raw Player Data Collector - NO HARD-CODED CATEGORIES

Philosophy:
-----------
Collect RAW player data WITHOUT pre-defining:
- "Star" vs "role player"
- "Veteran" vs "rookie"  
- "Clutch" vs "choker"
- Any other categories

Let the Context Pattern Transformer DISCOVER:
- Which player stats create narrative contexts
- Which combinations matter (e.g., PPG>25 + experience>5 → outcome)
- What hierarchies emerge from the data
- Which player types the algorithm identifies

Author: Narrative Optimization Framework
Date: November 16, 2025
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

try:
    from nba_api.stats.endpoints import (
        leaguegamefinder, 
        boxscoretraditionalv2,
        playergamelog,
        commonplayerinfo
    )
    from nba_api.stats.static import players, teams
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False
    print("⚠️  nba_api not installed")
    print("   Install: pip install nba_api")

print("="*80)
print("NBA RAW PLAYER DATA COLLECTOR")
print("="*80)


class NBAPlayerDataCollector:
    """
    Collect raw player stats WITHOUT categorization.
    
    The transformer will discover:
    - Which stats matter
    - What thresholds create contexts
    - How players cluster naturally
    - What combinations predict outcomes
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _parse_minutes(self, minutes_str) -> float:
        """Convert minutes from MM:SS or MM format to float"""
        if not minutes_str:
            return 0.0
        try:
            if isinstance(minutes_str, (int, float)):
                if pd.isna(minutes_str):
                    return 0.0
                return float(minutes_str)
            minutes_str = str(minutes_str)
            if ':' in minutes_str:
                parts = minutes_str.split(':')
                return float(parts[0]) + float(parts[1]) / 60.0
            return float(minutes_str)
        except:
            return 0.0
    
    def _safe_int(self, value) -> int:
        """Safely convert to int, handling NaN"""
        try:
            if pd.isna(value):
                return 0
            return int(value)
        except:
            return 0
    
    def _safe_float(self, value) -> float:
        """Safely convert to float, handling NaN"""
        try:
            if pd.isna(value):
                return 0.0
            return float(value)
        except:
            return 0.0
    
    def collect_player_box_scores(self, game_id: str, max_retries: int = 3) -> List[Dict]:
        """
        Collect RAW box score stats for all players in a game.
        
        Returns raw numbers - NO interpretation, NO categories
        """
        if not NBA_API_AVAILABLE:
            return []
        
        import time
        
        for attempt in range(max_retries):
            try:
                boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id, timeout=60)
                player_stats = boxscore.get_data_frames()[0]
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 2, 4, 6 seconds
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"   Error fetching box score {game_id} after {max_retries} attempts: {e}")
                    return []
        
        try:
            
            # Return RAW stats - let transformer discover what matters
            raw_stats = []
            for _, player in player_stats.iterrows():
                raw_stats.append({
                    # Identity (for nominative analysis)
                    'player_name': player['PLAYER_NAME'],
                    'player_id': player['PLAYER_ID'],
                    'team_id': player['TEAM_ID'],
                    
                    # Raw performance numbers (let transformer find patterns)
                    'minutes': self._parse_minutes(player['MIN']),
                    'points': self._safe_int(player['PTS']),
                    'rebounds': self._safe_int(player['REB']),
                    'assists': self._safe_int(player['AST']),
                    'steals': self._safe_int(player['STL']),
                    'blocks': self._safe_int(player['BLK']),
                    'turnovers': self._safe_int(player['TO']),
                    'fouls': self._safe_int(player['PF']),
                    
                    # Shooting (raw percentages)
                    'fg_made': self._safe_int(player['FGM']),
                    'fg_attempted': self._safe_int(player['FGA']),
                    'fg_pct': self._safe_float(player['FG_PCT']),
                    'three_made': self._safe_int(player['FG3M']),
                    'three_attempted': self._safe_int(player['FG3A']),
                    'three_pct': self._safe_float(player['FG3_PCT']),
                    'ft_made': self._safe_int(player['FTM']),
                    'ft_attempted': self._safe_int(player['FTA']),
                    'ft_pct': self._safe_float(player['FT_PCT']),
                    
                    # Efficiency (let transformer decide if it matters)
                    'plus_minus': self._safe_int(player['PLUS_MINUS']),
                    
                    # Position (raw label, not hierarchy)
                    'position': player.get('START_POSITION', ''),
                    'started': player.get('START_POSITION', '') != ''
                })
            
            return raw_stats
            
        except Exception as e:
            print(f"   Error processing box score {game_id}: {e}")
            return []
    
    def collect_player_season_stats(self, player_id: int, season: str) -> Dict:
        """
        Collect player's season-to-date stats.
        
        Returns raw cumulative stats - transformer discovers what matters
        """
        if not NBA_API_AVAILABLE:
            return {}
        
        try:
            # Get player info
            info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
            info_df = info.get_data_frames()[0]
            
            if len(info_df) == 0:
                return {}
            
            player_info = info_df.iloc[0]
            
            # Get game log for season
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season
            )
            games = gamelog.get_data_frames()[0]
            
            if len(games) == 0:
                return {}
            
            # Calculate raw season averages (no interpretation)
            return {
                'player_name': player_info['DISPLAY_FIRST_LAST'],
                'player_id': player_id,
                'season': season,
                
                # Raw career info
                'years_experience': int(player_info.get('SEASON_EXP', 0)),
                'height_inches': self._parse_height(player_info.get('HEIGHT', '')),
                'weight_lbs': int(player_info.get('WEIGHT', 0)),
                'position': player_info.get('POSITION', ''),
                'draft_year': player_info.get('DRAFT_YEAR', ''),
                'draft_round': player_info.get('DRAFT_ROUND', ''),
                'draft_number': player_info.get('DRAFT_NUMBER', ''),
                
                # Raw season averages (transformer finds patterns)
                'games_played': len(games),
                'ppg': float(games['PTS'].mean()),
                'rpg': float(games['REB'].mean()),
                'apg': float(games['AST'].mean()),
                'spg': float(games['STL'].mean()),
                'bpg': float(games['BLK'].mean()),
                'topg': float(games['TO'].mean()),
                'mpg': float(games['MIN'].mean()),
                'fg_pct': float(games['FG_PCT'].mean()),
                'three_pct': float(games['FG3_PCT'].mean()),
                'ft_pct': float(games['FT_PCT'].mean()),
                
                # Usage/role indicators (raw numbers)
                'shots_per_game': float(games['FGA'].mean()),
                'three_attempts_per_game': float(games['FG3A'].mean()),
                'ft_attempts_per_game': float(games['FTA'].mean()),
                
                # Recent form (last 10 games - let transformer find patterns)
                'ppg_l10': float(games.head(10)['PTS'].mean()) if len(games) >= 10 else float(games['PTS'].mean()),
                'fg_pct_l10': float(games.head(10)['FG_PCT'].mean()) if len(games) >= 10 else float(games['FG_PCT'].mean()),
            }
            
        except Exception as e:
            print(f"   Error fetching season stats for player {player_id}: {e}")
            return {}
    
    def _parse_height(self, height_str: str) -> int:
        """Convert height string (6-7) to inches"""
        try:
            if '-' in height_str:
                feet, inches = height_str.split('-')
                return int(feet) * 12 + int(inches)
        except:
            pass
        return 0
    
    def aggregate_team_player_stats(self, player_stats: List[Dict]) -> Dict:
        """
        Aggregate raw player stats to team level.
        
        Returns ONLY distributions and raw numbers - ZERO interpretation:
        - Count of players who scored N points
        - Distribution of minutes
        - Numerical thresholds
        
        Transformer discovers what matters: maybe "3+ players 20pts" predicts, 
        maybe "top1_minutes>38" predicts, maybe neither - WE DON'T KNOW YET.
        """
        if not player_stats:
            return {}
        
        # Sort by minutes played (proxy for importance, but don't label it)
        sorted_by_minutes = sorted(player_stats, key=lambda x: x['minutes'], reverse=True)
        
        # Raw distributions - let transformer find patterns
        points_list = [p['points'] for p in player_stats]
        assists_list = [p['assists'] for p in player_stats]
        minutes_list = [p['minutes'] for p in player_stats]
        
        return {
            # Player count thresholds (transformer discovers which matter)
            'players_used': len([p for p in player_stats if p['minutes'] > 0]),
            'players_10plus_min': len([p for p in player_stats if p['minutes'] >= 10]),
            'players_20plus_min': len([p for p in player_stats if p['minutes'] >= 20]),
            
            # Scoring distribution (transformer finds patterns)
            'players_20plus_pts': len([p for p in points_list if p >= 20]),
            'players_15plus_pts': len([p for p in points_list if p >= 15]),
            'players_10plus_pts': len([p for p in points_list if p >= 10]),
            
            # Assists distribution
            'players_5plus_ast': len([p for p in assists_list if p >= 5]),
            
            # Top player stats (by minutes, no "star" label)
            'top1_minutes': sorted_by_minutes[0]['minutes'] if len(sorted_by_minutes) > 0 else 0,
            'top1_points': sorted_by_minutes[0]['points'] if len(sorted_by_minutes) > 0 else 0,
            'top1_assists': sorted_by_minutes[0]['assists'] if len(sorted_by_minutes) > 0 else 0,
            'top1_plus_minus': sorted_by_minutes[0]['plus_minus'] if len(sorted_by_minutes) > 0 else 0,
            'top1_name': sorted_by_minutes[0]['player_name'] if len(sorted_by_minutes) > 0 else '',
            
            'top2_minutes': sorted_by_minutes[1]['minutes'] if len(sorted_by_minutes) > 1 else 0,
            'top2_points': sorted_by_minutes[1]['points'] if len(sorted_by_minutes) > 1 else 0,
            'top2_name': sorted_by_minutes[1]['player_name'] if len(sorted_by_minutes) > 1 else '',
            
            'top3_minutes': sorted_by_minutes[2]['minutes'] if len(sorted_by_minutes) > 2 else 0,
            'top3_points': sorted_by_minutes[2]['points'] if len(sorted_by_minutes) > 2 else 0,
            'top3_name': sorted_by_minutes[2]['player_name'] if len(sorted_by_minutes) > 2 else '',
            
            # Minutes distribution (concentration)
            'top3_minutes_share': sum(p['minutes'] for p in sorted_by_minutes[:3]) / sum(minutes_list) if sum(minutes_list) > 0 else 0,
            'top5_minutes_share': sum(p['minutes'] for p in sorted_by_minutes[:5]) / sum(minutes_list) if sum(minutes_list) > 0 else 0,
            
            # Scoring concentration (transformer discovers if it matters)
            'top1_scoring_share': sorted_by_minutes[0]['points'] / sum(points_list) if sum(points_list) > 0 else 0,
            'top3_scoring_share': sum(p['points'] for p in sorted_by_minutes[:3]) / sum(points_list) if sum(points_list) > 0 else 0,
            
            # Bench contribution (raw numbers)
            'bench_points': sum(p['points'] for p in player_stats if not p['started']),
            'bench_minutes': sum(p['minutes'] for p in player_stats if not p['started']),
            
            # Team composition (let transformer cluster)
            'avg_experience': sum(p.get('years_experience', 0) for p in player_stats) / len(player_stats),
            'experienced_players': len([p for p in player_stats if p.get('years_experience', 0) >= 5]),
        }
    
    def enhance_games_with_player_data(
        self, 
        games: List[Dict],
        sample_size: Optional[int] = None
    ) -> List[Dict]:
        """
        Add raw player data to games.
        
        NO CATEGORIES. Just raw stats.
        Transformer discovers:
        - "top1_points>30 & top2_points>20 & home=1" → 75% win rate
        - "players_20plus_pts>=3 & experienced_players>=7" → 68% win rate
        - "top1_scoring_share>0.4 & l10_win_pct<0.3" → 45% win rate (over-reliance)
        """
        if not NBA_API_AVAILABLE:
            print("⚠️  nba_api not available")
            return games
        
        print("\n📊 Collecting RAW player data...")
        print(f"   No categories, no labels - pure discovery")
        print(f"   Transformer will find what matters\n")
        
        if sample_size:
            games = games[:sample_size]
        
        enhanced = []
        success_count = 0
        fail_count = 0
        
        for i, game in enumerate(games):
            if i % 100 == 0:
                print(f"   Processing {i}/{len(games)} (Success: {success_count}, Failed: {fail_count})...")
            
            game_id = game.get('game_id', '')
            if not game_id:
                enhanced.append(game)
                continue
            
            # Collect raw player box scores with retry logic
            player_stats = self.collect_player_box_scores(game_id, max_retries=3)
            
            if not player_stats:
                game['player_data'] = {
                    'available': False,
                    'note': 'Could not fetch player data'
                }
                enhanced.append(game)
                fail_count += 1
                continue
            
            # Aggregate to team level (distributions, not categories)
            team_aggregates = self.aggregate_team_player_stats(player_stats)
            
            game['player_data'] = {
                'available': True,
                'team_aggregates': team_aggregates,
                'individual_players': player_stats,  # Keep full detail
                'note': 'Raw stats - no pre-defined categories'
            }
            
            enhanced.append(game)
            success_count += 1
            
            # Rate limiting - be conservative to avoid bans
            if i % 10 == 0 and i > 0:
                import time
                time.sleep(1.0)  # NBA API rate limit
        
        print(f"\n✓ Added player data to {len(enhanced)} games")
        print(f"   Success: {success_count} ({success_count/len(games)*100:.1f}%)")
        print(f"   Failed: {fail_count} ({fail_count/len(games)*100:.1f}%)")
        return enhanced


def main():
    """Main execution"""
    print("\n💡 PLAYER DATA COLLECTION PHILOSOPHY")
    print("-"*80)
    print("This collector does NOT pre-define:")
    print("  ✗ 'Star' vs 'role player'")
    print("  ✗ 'Veteran' vs 'rookie'")
    print("  ✗ 'Clutch' vs 'choker'")
    print("  ✗ Any other categories")
    print()
    print("Instead, it collects RAW data:")
    print("  ✓ PPG, APG, RPG (let transformer find thresholds)")
    print("  ✓ Years experience (transformer finds veteran patterns)")
    print("  ✓ Minutes played (transformer finds usage patterns)")
    print("  ✓ Scoring distribution (transformer finds balance patterns)")
    print()
    print("The Context Pattern Transformer will discover:")
    print("  → Which player stats create contexts")
    print("  → What thresholds matter (e.g., PPG>25)")
    print("  → How players cluster naturally")
    print("  → What combinations predict outcomes")
    print()
    print("Example patterns transformer might find:")
    print("  - 'top1_points>35 & top2_points<15' → 58% (over-reliance)")
    print("  - 'players_20plus_pts>=3 & home=1' → 72% (balanced attack)")
    print("  - 'experienced_players>=7 & games_played>60' → 68% (veteran late)")
    print()
    print("⚠️  WARNING: This will make ~12,000 NBA API calls")
    print("   Rate limit: 600 req/min → will take ~20-30 minutes")
    print("\n🚀 Starting collection automatically...")
    
    # Load existing enhanced data
    data_path = Path(__file__).parent.parent / 'data' / 'domains' / 'nba_enhanced_betting_data.json'
    
    if not data_path.exists():
        print(f"\n✗ {data_path} not found")
        print("   Run nba_data_builder.py first")
        return
    
    with open(data_path) as f:
        games = json.load(f)
    
    print(f"\n✓ Loaded {len(games):,} games")
    
    print("\n🚀 FULL COLLECTION MODE")
    print("="*80)
    print(f"   Processing ALL {len(games):,} games")
    print("   Estimated time: 30-45 minutes")
    print("   With retry logic for API timeouts")
    print("="*80)
    print()
    
    import time
    start_time = time.time()
    
    collector = NBAPlayerDataCollector(data_path.parent)
    enhanced_all = collector.enhance_games_with_player_data(games, sample_size=None)
    
    elapsed = time.time() - start_time
    
    # Save complete dataset
    output_path = data_path.parent / 'nba_complete_with_players.json'
    
    print(f"\n💾 Saving complete dataset...")
    with open(output_path, 'w') as f:
        json.dump(enhanced_all, f, indent=2)
    
    print(f"\n✓ Complete dataset saved: {output_path}")
    print(f"   Total time: {elapsed/60:.1f} minutes")
    print(f"   Games: {len(enhanced_all):,}")
    
    # Count successful games
    success = sum(1 for g in enhanced_all if g.get('player_data', {}).get('available'))
    print(f"   With player data: {success:,} ({success/len(enhanced_all)*100:.1f}%)")
    
    print("\n" + "="*80)
    print("READY FOR PATTERN DISCOVERY")
    print("="*80)
    print("\nNext step: Run Context Pattern Transformer on complete dataset")
    print("  Command: python3 discover_player_patterns.py")
    print()
    print("This will discover:")
    print("  - Player hierarchy patterns")
    print("  - Scoring distribution effects")
    print("  - Experience dynamics")
    print("  - Load management impacts")
    print("  - All WITHOUT pre-defining categories!")
    print("="*80)


if __name__ == '__main__':
    main()

