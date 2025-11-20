"""
Collect Easy Nominative Data for NBA

Starting with what we can compute/collect programmatically:
1. Team name analysis (syllables, power, memorability)
2. Historical championships and legacy metrics
3. Quantified momentum from game history
4. Rivalry intensity from historical matchups
5. Stakes computed from standings context

Then add manual data for what we can't compute.
"""

import json
import re
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path


class NBANominativeCollector:
    """Collects comprehensive nominative data for NBA teams."""
    
    def __init__(self):
        # EASY DATA: Manual entry (one-time, 5 minutes)
        self.team_colors = {
            'Lakers': {'primary': 'purple', 'secondary': 'gold', 'intimidation': 'royal'},
            'Celtics': {'primary': 'green', 'secondary': 'white', 'intimidation': 'classic'},
            'Warriors': {'primary': 'blue', 'secondary': 'gold', 'intimidation': 'electric'},
            'Heat': {'primary': 'red', 'secondary': 'black', 'intimidation': 'aggressive'},
            'Bulls': {'primary': 'red', 'secondary': 'black', 'intimidation': 'fierce'},
            'Knicks': {'primary': 'blue', 'secondary': 'orange', 'intimidation': 'bold'},
            'Nets': {'primary': 'black', 'secondary': 'white', 'intimidation': 'sleek'},
            'Spurs': {'primary': 'black', 'secondary': 'silver', 'intimidation': 'professional'},
            'Mavericks': {'primary': 'blue', 'secondary': 'silver', 'intimidation': 'modern'},
            'Rockets': {'primary': 'red', 'secondary': 'white', 'intimidation': 'bold'},
            'Clippers': {'primary': 'red', 'secondary': 'blue', 'intimidation': 'aggressive'},
            'Suns': {'primary': 'orange', 'secondary': 'purple', 'intimidation': 'vibrant'},
            'Trail Blazers': {'primary': 'red', 'secondary': 'black', 'intimidation': 'fierce'},
            'Jazz': {'primary': 'navy', 'secondary': 'gold', 'intimidation': 'traditional'},
            'Nuggets': {'primary': 'navy', 'secondary': 'gold', 'intimidation': 'mountain'},
            'Timberwolves': {'primary': 'blue', 'secondary': 'green', 'intimidation': 'wild'},
            'Thunder': {'primary': 'blue', 'secondary': 'orange', 'intimidation': 'electric'},
            'Grizzlies': {'primary': 'navy', 'secondary': 'gold', 'intimidation': 'gritty'},
            'Pelicans': {'primary': 'navy', 'secondary': 'red', 'intimidation': 'regional'},
            'Hornets': {'primary': 'teal', 'secondary': 'purple', 'intimidation': 'unique'},
            '76ers': {'primary': 'blue', 'secondary': 'red', 'intimidation': 'patriotic'},
            'Raptors': {'primary': 'red', 'secondary': 'black', 'intimidation': 'predator'},
            'Bucks': {'primary': 'green', 'secondary': 'cream', 'intimidation': 'natural'},
            'Cavaliers': {'primary': 'wine', 'secondary': 'gold', 'intimidation': 'regal'},
            'Pistons': {'primary': 'blue', 'secondary': 'red', 'intimidation': 'industrial'},
            'Pacers': {'primary': 'navy', 'secondary': 'gold', 'intimidation': 'speed'},
            'Magic': {'primary': 'blue', 'secondary': 'black', 'intimidation': 'mystical'},
            'Hawks': {'primary': 'red', 'secondary': 'black', 'intimidation': 'predator'},
            'Wizards': {'primary': 'navy', 'secondary': 'red', 'intimidation': 'magical'},
            'Kings': {'primary': 'purple', 'secondary': 'silver', 'intimidation': 'royal'}
        }
        
        self.championships_alltime = {
            'Lakers': 17, 'Celtics': 17, 'Warriors': 7, 'Bulls': 6, 'Spurs': 5,
            'Heat': 3, 'Pistons': 3, '76ers': 3, 'Knicks': 2, 'Rockets': 2,
            'Bucks': 2, 'Cavaliers': 1, 'Mavericks': 1, 'Raptors': 1, 'Thunder': 1,
            'Hawks': 1, 'Wizards': 1, 'Trail Blazers': 1, 'Kings': 1,
            # Rest have 0
        }
        
        self.team_archetypes = {
            'Lakers': 'historic_powerhouse',
            'Celtics': 'historic_powerhouse',
            'Warriors': 'modern_dynasty',
            'Spurs': 'sustained_excellence',
            'Heat': 'championship_culture',
            'Bulls': 'legendary_past',
            'Knicks': 'sleeping_giant',
            'Nets': 'rebuilding_ambition',
            # etc
        }
    
    def analyze_team_name(self, team_name: str) -> dict:
        """Analyze team name for nominative power."""
        # Syllable count
        vowels = 'aeiou'
        syllables = sum(1 for c in team_name.lower() if c in vowels)
        
        # Power words
        power_words = ['warriors', 'thunder', 'heat', 'bulls', 'raptors', 'grizzlies', 
                       'hawks', 'hornets', 'pistons', 'rockets', 'magic', 'wizards']
        is_power_name = any(pw in team_name.lower() for pw in power_words)
        
        # Animal/aggressive names
        animal_names = ['bulls', 'raptors', 'grizzlies', 'hawks', 'hornets', 'pelicans',
                       'timberwolves', 'bucks']
        is_animal = any(an in team_name.lower() for an in animal_names)
        
        # Royal/prestigious
        prestige_words = ['kings', 'cavaliers', 'knicks', 'lakers', 'celtics']
        is_prestige = any(pw in team_name.lower() for pw in prestige_words)
        
        # Memorability (simple heuristic: shorter + unique)
        memorability = (10 - len(team_name)) / 10.0
        
        return {
            'name': team_name,
            'syllables': syllables,
            'is_power_name': is_power_name,
            'is_animal': is_animal,
            'is_prestige': is_prestige,
            'memorability': memorability,
            'name_power_score': (
                syllables * 0.3 +
                (5 if is_power_name else 0) +
                (4 if is_animal else 0) +
                (3 if is_prestige else 0) +
                memorability * 2
            )
        }
    
    def calculate_momentum(self, recent_games: list) -> dict:
        """Calculate rich momentum metrics."""
        if len(recent_games) < 5:
            return {'streak': 0, 'trend': 'neutral', 'confidence': 0.5}
        
        last_5 = recent_games[-5:]
        last_10 = recent_games[-10:] if len(recent_games) >= 10 else recent_games
        
        # Win streak
        wins_5 = sum(1 for g in last_5 if g.get('won', False))
        wins_10 = sum(1 for g in last_10 if g.get('won', False))
        
        # Streak length
        streak = 0
        streak_type = 'none'
        for g in reversed(recent_games[-20:]):
            if len(recent_games) == 0:
                break
            won = g.get('won', False)
            if streak == 0:
                streak = 1
                streak_type = 'winning' if won else 'losing'
            elif (streak_type == 'winning' and won) or (streak_type == 'losing' and not won):
                streak += 1
            else:
                break
        
        # Trend
        if wins_5 >= 4:
            trend = 'hot'
        elif wins_5 <= 1:
            trend = 'cold'
        else:
            trend = 'neutral'
        
        # Confidence score
        confidence = wins_5 / 5.0
        
        return {
            'streak_length': streak,
            'streak_type': streak_type,
            'wins_last_5': wins_5,
            'wins_last_10': wins_10,
            'trend': trend,
            'confidence_score': confidence,
            'momentum_score': (wins_5 / 5.0) * (1 + (streak / 10.0))
        }
    
    def calculate_rivalry_intensity(self, team1: str, team2: str, historical_games: list) -> dict:
        """Calculate rivalry intensity from historical matchups."""
        # Historic rivalries (documented)
        historic_pairs = {
            frozenset(['Lakers', 'Celtics']): 10,
            frozenset(['Lakers', 'Clippers']): 7,
            frozenset(['Knicks', 'Nets']): 6,
            frozenset(['Warriors', 'Cavaliers']): 8,
            frozenset(['Heat', 'Pacers']): 6,
        }
        
        pair = frozenset([team1, team2])
        base_intensity = historic_pairs.get(pair, 0)
        
        # Count historical matchups
        matchup_count = len([
            g for g in historical_games
            if (g['team_name'] == team1 or g['team_name'] == team2)
        ])
        
        # Recent playoff meetings
        playoff_meetings = 0  # Would need playoff data
        
        rivalry_score = base_intensity + (matchup_count / 100.0)
        
        return {
            'base_intensity': base_intensity,
            'historical_matchups': matchup_count,
            'playoff_history': playoff_meetings,
            'rivalry_score': min(10, rivalry_score),
            'classification': 'historic' if base_intensity >= 8 else 'division' if base_intensity >= 5 else 'standard'
        }
    
    def calculate_stakes(self, game: dict, team_record: dict, season_context: dict) -> dict:
        """Calculate game stakes nominatively."""
        games_played = team_record['wins'] + team_record['losses']
        games_remaining = 82 - games_played
        win_pct = team_record['wins'] / max(games_played, 1)
        
        # Playoff implications
        in_playoff_race = (40 < team_record['wins'] < 52) and games_remaining < 15
        
        # Must-win classification
        must_win = games_remaining < 5 and abs(win_pct - 0.500) < 0.05
        
        # Stakes level
        if must_win:
            stakes = 'critical'
            weight = 2.5
        elif in_playoff_race:
            stakes = 'high'
            weight = 1.8
        elif games_remaining < 10:
            stakes = 'elevated'
            weight = 1.3
        else:
            stakes = 'standard'
            weight = 1.0
        
        return {
            'stakes_level': stakes,
            'narrative_weight': weight,
            'playoff_implications': in_playoff_race,
            'must_win': must_win,
            'pressure_score': weight
        }
    
    def collect_all_nominative_data(self, all_games: list) -> dict:
        """Collect all nominative data from game history."""
        print("\n" + "="*70)
        print("COLLECTING NOMINATIVE DATA")
        print("="*70)
        
        nominative_data = {
            'teams': {},
            'games_enhanced': []
        }
        
        # Analyze all team names
        print("\n1. Analyzing team names...")
        unique_teams = set(g['team_name'] for g in all_games)
        for team in unique_teams:
            # Extract short name for lookup
            short_name = team.split()[-1] if ' ' in team else team
            
            nominative_data['teams'][team] = {
                'name_analysis': self.analyze_team_name(team),
                'colors': self.team_colors.get(short_name, {'primary': 'unknown', 'secondary': 'unknown', 'intimidation': 'standard'}),
                'championships': self.championships_alltime.get(short_name, 0),
                'archetype': self.team_archetypes.get(short_name, 'standard'),
                'short_name': short_name
            }
        
        print(f"   ✅ {len(unique_teams)} teams analyzed")
        
        # Enhanced game data
        print("\n2. Enhancing games with nominative context...")
        
        for i, game in enumerate(all_games):
            if (i + 1) % 2000 == 0:
                print(f"   Progress: {i+1}/{len(all_games)}...")
            
            team = game['team_name']
            
            # Get opponent
            matchup = game.get('matchup', '')
            if ' vs. ' in matchup:
                opponent = matchup.split(' vs. ')[1].strip()
            elif ' @ ' in matchup:
                opponent = matchup.split(' @ ')[1].strip()
            else:
                opponent = 'Unknown'
            
            # Prior games for this team
            prior_games = [
                g for g in all_games
                if g['team_name'] == team and g['date'] < game['date']
            ]
            
            # Calculate metrics
            wins_before = sum(1 for g in prior_games if g.get('won', False))
            losses_before = len(prior_games) - wins_before
            
            momentum = self.calculate_momentum(prior_games)
            rivalry = self.calculate_rivalry_intensity(team, opponent, prior_games)
            stakes = self.calculate_stakes(
                game,
                {'wins': wins_before, 'losses': losses_before},
                {'games_played': len(prior_games)}
            )
            
            # Enhanced game
            enhanced = {
                **game,  # Original data
                'nominative_context': {
                    'team_name_power': nominative_data['teams'][team]['name_analysis']['name_power_score'],
                    'team_colors': nominative_data['teams'][team]['colors']['primary'],
                    'team_legacy': nominative_data['teams'][team]['championships'],
                    'team_archetype': nominative_data['teams'][team]['archetype'],
                    'momentum': momentum,
                    'rivalry': rivalry,
                    'stakes': stakes,
                    'wins_before': wins_before,
                    'losses_before': losses_before,
                    'win_pct_before': wins_before / max(len(prior_games), 1)
                }
            }
            
            nominative_data['games_enhanced'].append(enhanced)
        
        print(f"   ✅ All games enhanced with nominative context")
        
        return nominative_data


def main():
    """Collect and save nominative data."""
    print("\n" + "="*70)
    print("NBA NOMINATIVE DATA COLLECTION")
    print("="*70)
    
    # Load games
    print("\nLoading NBA games...")
    with open('../../../data/domains/nba_all_seasons_real.json', 'r') as f:
        all_games = json.load(f)
    
    # Sort by date
    all_games = sorted(all_games, key=lambda x: x['date'])
    print(f"✅ Loaded {len(all_games)} games")
    
    # Collect nominative data
    collector = NBANominativeCollector()
    nominative_data = collector.collect_all_nominative_data(all_games)
    
    # Save
    output_dir = Path('../../../data/domains')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save team nominative data
    with open(output_dir / 'nba_nominative_teams.json', 'w') as f:
        json.dump(nominative_data['teams'], f, indent=2)
    
    # Save enhanced games
    with open(output_dir / 'nba_games_with_nominative.json', 'w') as f:
        json.dump(nominative_data['games_enhanced'], f, indent=2)
    
    print("\n" + "="*70)
    print("COLLECTION COMPLETE")
    print("="*70)
    
    print(f"\n✅ Team nominative data: {output_dir / 'nba_nominative_teams.json'}")
    print(f"✅ Enhanced games: {output_dir / 'nba_games_with_nominative.json'}")
    
    # Sample output
    print(f"\n📊 Sample team (Lakers):")
    lakers_data = nominative_data['teams'].get('Los Angeles Lakers', {})
    print(json.dumps(lakers_data, indent=2))
    
    print(f"\n🎯 Next: Generate rich narratives using this data")
    print(f"   python domains/nba/generate_rich_narratives.py")


if __name__ == '__main__':
    main()

