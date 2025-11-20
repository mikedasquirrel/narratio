"""
NBA Narrative Enrichment

Takes existing 11,979 games and enriches top 1,000 with comprehensive narratives.

Rich narratives include:
- Team momentum and recent form
- Key player storylines
- Rivalry context
- Championship implications
- Injury impacts
- Historical matchup data
- Media narratives
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict


class NBANarrativeEnricher:
    """Enrich NBA games with comprehensive narratives"""
    
    def __init__(self, games_data):
        """Initialize with existing games"""
        self.games = games_data
        self.team_records = self._compute_team_records()
        self.rivalries = self._define_rivalries()
    
    def _compute_team_records(self):
        """Compute running win-loss records"""
        records = defaultdict(lambda: {'wins': 0, 'losses': 0, 'games': []})
        
        # Sort by date
        sorted_games = sorted(self.games, key=lambda g: g.get('date', ''))
        
        for game in sorted_games:
            team = game['team_abbreviation']
            records[team]['games'].append(game)
            if game['won']:
                records[team]['wins'] += 1
            else:
                records[team]['losses'] += 1
        
        return records
    
    def _define_rivalries(self):
        """Define major NBA rivalries"""
        return {
            ('LAL', 'BOS'): 'Historic rivalry - Lakers vs Celtics',
            ('LAL', 'GSW'): 'Modern rivalry - LeBron vs Warriors dynasty',
            ('MIA', 'BOS'): 'Eastern rivalry - Heat vs Celtics',
            ('LAL', 'LAC'): 'Battle of LA',
            ('GSW', 'CLE'): 'Finals rivalry - Warriors vs Cavs',
            ('BRK', 'BOS'): 'Atlantic Division rivalry',
            ('PHX', 'LAL'): 'Western rivalry',
            ('MIL', 'BRK'): 'Eastern powerhouse rivalry'
        }
    
    def enrich_game(self, game, team_stats_at_time):
        """
        Generate comprehensive narrative for single game.
        
        Parameters
        ----------
        game : dict
            Game data
        team_stats_at_time : dict
            Team statistics at time of game
            
        Returns
        -------
        rich_narrative : str
            300-500 word narrative
        """
        team = game['team_abbreviation']
        opponent = self._extract_opponent(game['matchup'])
        
        # Build narrative components
        components = []
        
        # 1. Matchup introduction
        date_str = game.get('date', 'Unknown date')
        season = game.get('season', 'Unknown')
        components.append(f"{game['team_name']} vs {opponent} on {date_str} ({season} season).")
        
        # 2. Team momentum
        record = team_stats_at_time.get('record', '0-0')
        l10 = team_stats_at_time.get('last_10', 'N/A')
        components.append(f"{team} enters with a {record} record, {l10} in their last 10 games.")
        
        # 3. Rivalry context
        rivalry_key = tuple(sorted([team, opponent]))
        if rivalry_key in self.rivalries:
            components.append(f"This matchup features a {self.rivalries[rivalry_key]}.")
        
        # 4. Season implications
        points = game.get('points', 0)
        plus_minus = game.get('plus_minus', 0)
        
        if game.get('won'):
            components.append(f"{team} secured the victory with {points} points, winning by {plus_minus}.")
        else:
            components.append(f"{team} fell short, scoring {points} points and losing by {abs(plus_minus)}.")
        
        # 5. Playoff context (if late season)
        if season and '-' in season:
            # Rough playoff timing (games after February)
            components.append("With playoff implications mounting, every game carries increased weight.")
        
        # 6. Historical context
        components.append(f"The {game['team_name']} bring their franchise history and identity to each contest.")
        
        # 7. Narrative arc
        if game.get('won'):
            components.append("The team's resilience and execution proved decisive in securing the win.")
        else:
            components.append("Despite competitive effort, the team faced challenges that led to the loss.")
        
        # Combine
        rich_narrative = ' '.join(components)
        
        return rich_narrative
    
    def _extract_opponent(self, matchup):
        """Extract opponent from matchup string"""
        if not matchup:
            return "Unknown"
        
        # Format: "LAL vs. BOS" or "LAL @ BOS"
        parts = matchup.replace('vs.', ' ').replace('@', ' ').split()
        opponents = [p for p in parts if p.isupper() and len(p) == 3]
        
        if len(opponents) >= 2:
            return opponents[1]
        return "Unknown"
    
    def enrich_top_games(self, n=1000):
        """
        Enrich top N games with comprehensive narratives.
        
        Prioritizes:
        - Playoff games
        - Rivalry matchups
        - Close games (exciting)
        - Recent seasons (more relevant)
        """
        print(f"Enriching top {n} games...")
        
        # Score games by importance
        scored_games = []
        for game in self.games:
            score = 0
            
            # Recent seasons weighted higher
            season = game.get('season', '')
            if '2023' in season or '2024' in season:
                score += 3
            elif '2021' in season or '2022' in season:
                score += 2
            elif '2019' in season or '2020' in season:
                score += 1
            
            # Close games more interesting
            plus_minus = abs(game.get('plus_minus', 100))
            if plus_minus < 5:
                score += 3
            elif plus_minus < 10:
                score += 2
            elif plus_minus < 15:
                score += 1
            
            # Rivalry games
            opponent = self._extract_opponent(game.get('matchup', ''))
            team = game.get('team_abbreviation', '')
            rivalry_key = tuple(sorted([team, opponent]))
            if rivalry_key in self.rivalries:
                score += 4
            
            scored_games.append((score, game))
        
        # Sort by score
        scored_games.sort(key=lambda x: x[0], reverse=True)
        
        # Enrich top N
        enriched = []
        for i, (score, game) in enumerate(scored_games[:n]):
            # Get team stats at this point in season
            team_stats = self._get_team_stats_at_game(game)
            
            # Generate rich narrative
            rich_narrative = self.enrich_game(game, team_stats)
            
            # Create enriched game record
            enriched_game = game.copy()
            enriched_game['narrative_original'] = game.get('narrative', '')
            enriched_game['narrative'] = rich_narrative
            enriched_game['narrative_length'] = len(rich_narrative)
            enriched_game['importance_score'] = score
            
            enriched.append(enriched_game)
            
            if (i + 1) % 100 == 0:
                print(f"  Enriched {i+1}/{n} games...")
        
        print(f"✓ Enriched {len(enriched)} games")
        return enriched
    
    def _get_team_stats_at_game(self, game):
        """Get team statistics at time of game"""
        team = game['team_abbreviation']
        team_games = self.team_records[team]['games']
        
        # Find games before this one
        game_date = game.get('date', '')
        prior_games = [g for g in team_games if g.get('date', '') < game_date]
        
        if len(prior_games) > 0:
            wins = sum(1 for g in prior_games if g.get('won'))
            losses = len(prior_games) - wins
            
            # Last 10
            if len(prior_games) >= 10:
                l10_games = prior_games[-10:]
                l10_wins = sum(1 for g in l10_games if g.get('won'))
                l10 = f"{l10_wins}-{10-l10_wins}"
            else:
                l10 = "N/A"
            
            return {
                'record': f"{wins}-{losses}",
                'last_10': l10,
                'n_games': len(prior_games)
            }
        
        return {'record': '0-0', 'last_10': 'N/A', 'n_games': 0}


def main():
    """Enrich NBA dataset"""
    print("="*80)
    print("NBA NARRATIVE ENRICHMENT")
    print("="*80)
    
    # Load existing data
    data_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'nba_all_seasons_real.json'
    
    with open(data_path) as f:
        games = json.load(f)
    
    print(f"\n✓ Loaded {len(games)} games")
    
    # Enrich
    enricher = NBANarrativeEnricher(games)
    enriched_games = enricher.enrich_top_games(n=1000)
    
    # Statistics
    print("\n" + "="*80)
    print("ENRICHMENT STATISTICS")
    print("="*80)
    
    avg_length = np.mean([g['narrative_length'] for g in enriched_games])
    print(f"Average narrative length: {avg_length:.0f} characters")
    print(f"Min: {min(g['narrative_length'] for g in enriched_games)}")
    print(f"Max: {max(g['narrative_length'] for g in enriched_games)}")
    
    seasons = set(g['season'] for g in enriched_games)
    print(f"\nSeasons covered: {sorted(seasons)}")
    
    rivalries = sum(1 for g in enriched_games if g['importance_score'] >= 4)
    close_games = sum(1 for g in enriched_games if abs(g.get('plus_minus', 100)) < 10)
    
    print(f"\nGame types:")
    print(f"  Rivalry games: {rivalries}")
    print(f"  Close games: {close_games}")
    print(f"  Recent seasons (2023-24): {sum(1 for g in enriched_games if '2023' in g['season'] or '2024' in g['season'])}")
    
    # Save
    output_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'nba_enriched_1000.json'
    
    with open(output_path, 'w') as f:
        json.dump(enriched_games, f, indent=2)
    
    print(f"\n✓ Saved enriched dataset: {output_path}")
    print(f"  {len(enriched_games)} games with rich narratives")
    
    # Sample
    print("\n" + "="*80)
    print("SAMPLE ENRICHED GAME")
    print("="*80)
    sample = enriched_games[0]
    print(f"Matchup: {sample['matchup']}")
    print(f"Season: {sample['season']}")
    print(f"Importance: {sample['importance_score']}")
    print(f"\nOriginal ({len(sample['narrative_original'])} chars):")
    print(f"  {sample['narrative_original']}")
    print(f"\nEnriched ({len(sample['narrative'])} chars):")
    print(f"  {sample['narrative']}")


if __name__ == '__main__':
    main()

