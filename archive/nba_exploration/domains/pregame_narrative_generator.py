"""
Pre-Game Narrative Generator

Generates narratives BEFORE games using only historical context.
No outcome information included - pure predictive setup.

Uses:
- Team name and identity
- Historical performance (prior games only)
- League standings (before this game)
- Rivalry context
- Recent momentum
- Season context

Does NOT use:
- Game outcome
- Game score
- Post-game information
"""

from typing import Dict, List, Any
import random


class PreGameNarrativeGenerator:
    """
    Generates narratives before games for true predictive testing.
    """
    
    def __init__(self):
        # Team identity templates (inherent, not outcome-dependent)
        self.team_identities = {
            'Lakers': 'storied franchise with championship legacy',
            'Celtics': 'historic dynasty with winning tradition',
            'Warriors': 'modern powerhouse known for three-point excellence',
            'Heat': 'culture of toughness and defensive intensity',
            'Spurs': 'model of sustained success and team basketball',
            'Bulls': 'legacy of greatness from Jordan era',
            'Knicks': 'iconic franchise in basketball capital',
            # Add more as needed
        }
        
        # Performance descriptors (based on record, not single game)
        self.record_descriptors = {
            'elite': ['championship contender', 'powerhouse', 'dominant force'],
            'good': ['playoff team', 'competitive squad', 'solid contender'],
            'average': ['middle-of-pack team', 'fighting for position'],
            'struggling': ['rebuilding squad', 'developmental team'],
            'poor': ['lottery-bound team', 'rebuilding project']
        }
        
        # Momentum descriptors (from recent games)
        self.momentum_types = {
            'hot': ['riding momentum', 'on a winning streak', 'surging'],
            'cold': ['struggling lately', 'searching for rhythm'],
            'neutral': ['steady performance', 'consistent play']
        }
    
    def generate_pregame_narrative(
        self,
        team_name: str,
        opponent_name: str,
        team_record_before: Dict[str, int],  # wins, losses BEFORE this game
        recent_games: List[Dict],  # Last 5 games BEFORE this one
        season_context: Dict[str, Any],
        rivalry_level: str = 'none'
    ) -> str:
        """
        Generate pre-game narrative using only information available BEFORE game.
        
        Parameters
        ----------
        team_name : str
            Team name
        opponent_name : str
            Opponent
        team_record_before : dict
            {'wins': X, 'losses': Y} BEFORE this game
        recent_games : list
            Last 5 games (outcomes only, not descriptions)
        season_context : dict
            {'games_played': N, 'games_remaining': M, 'playoff_race': bool}
        rivalry_level : str
            'historic', 'division', 'none'
        
        Returns
        -------
        narrative : str
            Pre-game narrative with NO outcome information
        """
        # Team identity
        identity = self.team_identities.get(
            team_name,
            f"competitive {team_name} squad"
        )
        
        # Record-based classification
        total_games = team_record_before['wins'] + team_record_before['losses']
        win_pct = team_record_before['wins'] / max(total_games, 1)
        
        if win_pct >= 0.65:
            tier = 'elite'
        elif win_pct >= 0.55:
            tier = 'good'
        elif win_pct >= 0.45:
            tier = 'average'
        elif win_pct >= 0.35:
            tier = 'struggling'
        else:
            tier = 'poor'
        
        descriptor = random.choice(self.record_descriptors[tier])
        
        # Recent momentum (from last 5 games)
        if len(recent_games) >= 3:
            recent_wins = sum(1 for g in recent_games[-5:] if g.get('won', False))
            if recent_wins >= 4:
                momentum = random.choice(self.momentum_types['hot'])
            elif recent_wins <= 1:
                momentum = random.choice(self.momentum_types['cold'])
            else:
                momentum = random.choice(self.momentum_types['neutral'])
        else:
            momentum = "entering the matchup"
        
        # Rivalry context
        if rivalry_level == 'historic':
            rivalry_text = f"in a historic rivalry matchup against the {opponent_name}"
        elif rivalry_level == 'division':
            rivalry_text = f"facing division rival {opponent_name}"
        else:
            rivalry_text = f"taking on the {opponent_name}"
        
        # Season context
        if season_context.get('playoff_race', False):
            stakes = "with playoff implications on the line"
        elif season_context.get('games_remaining', 82) < 15:
            stakes = "in the crucial late-season stretch"
        else:
            stakes = "continuing their season campaign"
        
        # Construct narrative
        narrative = f"The {team_name}, a {identity}, comes in as a {descriptor} {momentum}. They face this matchup {rivalry_text} {stakes}."
        
        return narrative
    
    def generate_narrative_from_game_data(
        self,
        game: Dict,
        all_games: List[Dict]
    ) -> str:
        """
        Generate pre-game narrative using only information available before game.
        
        Parameters
        ----------
        game : dict
            Current game (will extract team, opponent, date)
        all_games : list
            All games (to calculate prior record and recent games)
        
        Returns
        -------
        narrative : str
            Pre-game narrative
        """
        team = game['team_name']
        opponent = game.get('matchup', '').split(' vs. ')[-1].split(' @ ')[-1].strip()
        game_date = game.get('date')
        
        # Find all games for this team BEFORE this game
        prior_games = [
            g for g in all_games
            if g['team_name'] == team and g['date'] < game_date
        ]
        
        # Calculate record before this game
        wins_before = sum(1 for g in prior_games if g.get('won', False))
        losses_before = len(prior_games) - wins_before
        
        record_before = {'wins': wins_before, 'losses': losses_before}
        
        # Get recent games (last 5 before this one)
        recent = sorted(prior_games, key=lambda x: x['date'])[-5:]
        
        # Season context
        games_played = len(prior_games)
        games_remaining = 82 - games_played
        playoff_race = games_remaining < 20 and 0.45 < (wins_before / max(games_played, 1)) < 0.55
        
        season_context = {
            'games_played': games_played,
            'games_remaining': games_remaining,
            'playoff_race': playoff_race
        }
        
        # Detect rivalry (placeholder - would use historical data)
        rivalry_pairs = {
            ('Lakers', 'Celtics'), ('Celtics', 'Lakers'),
            ('Lakers', 'Clippers'), ('Clippers', 'Lakers'),
            ('Knicks', 'Nets'), ('Nets', 'Knicks')
        }
        
        rivalry = 'historic' if (team, opponent) in rivalry_pairs else 'none'
        
        # Generate narrative
        return self.generate_pregame_narrative(
            team,
            opponent,
            record_before,
            recent,
            season_context,
            rivalry
        )


def create_pregame_generator():
    """Factory function."""
    return PreGameNarrativeGenerator()

