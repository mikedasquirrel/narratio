"""
NFL Narrative Generation Module

Generates nominative-rich narratives (150-250 words) for each game.
Heavy emphasis on player names, coach names, position groups, and ensembles.
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import random


class NFLNarrativeGenerator:
    """
    Generates comprehensive, nominative-rich narratives for NFL games.
    
    Each narrative includes:
    - Team names and records (estimated)
    - QB names prominently
    - Star player names by position
    - Coach names
    - Position group narratives (O-line, receiving corps, defensive front)
    - Individual matchups (QB vs QB, etc.)
    - Ensemble dynamics
    - Game context (rivalry, playoff implications, weather)
    """
    
    def __init__(self):
        """Initialize narrative generator."""
        self.narrative_templates = self._load_templates()
    
    def generate_narratives(
        self,
        games: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate narratives for all games.
        
        Parameters
        ----------
        games : list of dict
            Games with roster and context data
            
        Returns
        -------
        games_with_narratives : list of dict
            Games enriched with narratives
        """
        print("\n" + "="*80)
        print("GENERATING NOMINATIVE-RICH NARRATIVES")
        print("="*80)
        
        enriched_games = []
        
        for idx, game in enumerate(games):
            narrative = self._generate_game_narrative(game)
            game['narrative'] = narrative
            enriched_games.append(game)
            
            if (idx + 1) % 500 == 0:
                print(f"  Generated {idx + 1}/{len(games)} narratives...")
        
        print(f"\n✓ Generated {len(enriched_games)} narratives")
        
        return enriched_games
    
    def _generate_game_narrative(self, game: Dict[str, Any]) -> str:
        """
        Generate single game narrative with heavy nominative content.
        
        Following INSTRUCTIONS example (lines 146-159):
        - Team names and context
        - QB names prominently
        - Star players by position
        - Coach names
        - Position groups
        - Individual matchups
        - Ensemble dynamics
        """
        # Extract data
        home_team = game['home_team']
        away_team = game['away_team']
        season = game['season']
        week = game['week']
        
        home_roster = game['home_roster']
        away_roster = game['away_roster']
        home_coaches = game['home_coaches']
        away_coaches = game['away_coaches']
        context = game['context']
        
        # Build narrative components
        
        # 1. Opening: Teams and context
        opening = self._generate_opening(
            home_team, away_team, season, week, context
        )
        
        # 2. Home team narrative (QB, stars, coach, position groups)
        home_narrative = self._generate_team_narrative(
            home_team, home_roster, home_coaches, "home"
        )
        
        # 3. Away team narrative
        away_narrative = self._generate_team_narrative(
            away_team, away_roster, away_coaches, "away"
        )
        
        # 4. Matchup analysis
        matchup = self._generate_matchup_narrative(
            game['position_matchups'], home_roster, away_roster
        )
        
        # 5. Context and stakes
        stakes = self._generate_stakes(context, home_team, away_team)
        
        # Combine into full narrative (150-250 words)
        narrative = f"{opening} {home_narrative} {away_narrative} {matchup} {stakes}"
        
        return narrative.strip()
    
    def _generate_opening(
        self,
        home_team: str,
        away_team: str,
        season: int,
        week: int,
        context: Dict
    ) -> str:
        """Generate opening sentence with teams and context."""
        week_str = f"Week {week}" if week else "playoff"
        
        # Estimate records (heuristic for narrative purposes)
        home_record = self._estimate_record(week)
        away_record = self._estimate_record(week)
        
        templates = [
            f"The {home_team} ({home_record}) host the {away_team} ({away_record}) in {week_str} of the {season} season.",
            f"In a {week_str} matchup, the {away_team} ({away_record}) travel to face the {home_team} ({home_record}).",
            f"Week {week} brings a showdown between the {away_team} ({away_record}) and {home_team} ({home_record})."
        ]
        
        opening = random.choice(templates)
        
        # Add context modifiers
        if context.get('rivalry'):
            opening += " This heated rivalry matchup carries extra intensity."
        if context.get('playoff_game'):
            opening += " Playoff implications hang in the balance."
        if context.get('primetime'):
            opening += " The nation watches in primetime."
        if context.get('division_game'):
            opening += " This crucial division game could determine playoff seeding."
        
        return opening
    
    def _generate_team_narrative(
        self,
        team: str,
        roster: Dict,
        coaches: Dict,
        side: str
    ) -> str:
        """
        Generate team narrative with QB, stars, coach, position groups.
        
        Heavy nominative content.
        """
        # QB (most important)
        qb_name = roster.get('starting_qb', {}).get('name', f'{team} QB')
        
        # Star players
        stars = roster.get('key_players', [])
        star_names = [s['name'] for s in stars if s['name'] != qb_name][:3]
        
        # Coach
        hc_name = coaches.get('head_coach', f'{team} HC')
        
        # Position groups
        offensive_unit = roster.get('offense', [])[:5]  # Sample 5
        defensive_unit = roster.get('defense', [])[:3]  # Sample 3
        
        # Build narrative
        parts = []
        
        # QB + coach introduction
        parts.append(
            f"Led by quarterback {qb_name}, head coach {hc_name}'s {team} squad"
        )
        
        # Star players
        if star_names:
            if len(star_names) == 1:
                parts.append(f"features star {star_names[0]}")
            elif len(star_names) == 2:
                parts.append(f"features stars {star_names[0]} and {star_names[1]}")
            else:
                parts.append(
                    f"features stars {star_names[0]}, {star_names[1]}, and {star_names[2]}"
                )
        
        # Position group
        if offensive_unit:
            names = [p['name'] for p in offensive_unit[:3]]
            if len(names) >= 2:
                parts.append(
                    f"The offensive unit, anchored by {names[0]} and {names[1]}, provides {qb_name} with solid protection."
                )
        
        # Defensive unit
        if defensive_unit:
            names = [p['name'] for p in defensive_unit[:2]]
            if len(names) >= 1:
                parts.append(
                    f"Defensively, {names[0]} leads a unit looking to disrupt the opponent."
                )
        
        return " ".join(parts) + "."
    
    def _generate_matchup_narrative(
        self,
        matchups: Dict,
        home_roster: Dict,
        away_roster: Dict
    ) -> str:
        """Generate individual matchup narratives."""
        qb_matchup = matchups.get('starting_qb', {})
        home_qb = qb_matchup.get('home', 'Unknown')
        away_qb = qb_matchup.get('away', 'Unknown')
        
        rb_matchup = matchups.get('starting_rb', {})
        home_rb = rb_matchup.get('home', 'Unknown')
        away_rb = rb_matchup.get('away', 'Unknown')
        
        wr_matchup = matchups.get('starting_wr1', {})
        home_wr = wr_matchup.get('home', 'Unknown')
        away_wr = wr_matchup.get('away', 'Unknown')
        
        matchup_text = (
            f"The quarterback duel between {home_qb} and {away_qb} will be pivotal. "
            f"In the ground game, {home_rb} faces off against {away_rb}. "
            f"Top receivers {home_wr} and {away_wr} will look to make explosive plays."
        )
        
        return matchup_text
    
    def _generate_stakes(
        self,
        context: Dict,
        home_team: str,
        away_team: str
    ) -> str:
        """Generate stakes and context narrative."""
        stakes = []
        
        if context.get('playoff_game'):
            stakes.append("Playoff survival is at stake.")
        elif context.get('division_game'):
            stakes.append("Division supremacy hangs in the balance.")
        
        if context.get('rivalry'):
            stakes.append(f"The {home_team}-{away_team} rivalry adds edge to this clash.")
        
        if context.get('primetime'):
            stakes.append("The spotlight shines bright on this primetime showcase.")
        
        if not stakes:
            stakes.append("Both teams look to build momentum.")
        
        return " ".join(stakes)
    
    def _estimate_record(self, week: Optional[int]) -> str:
        """Estimate team record for narrative (heuristic)."""
        if week is None or week > 18:
            return "playoff team"
        
        # Random but realistic record
        wins = random.randint(max(0, week - 10), week)
        losses = week - wins
        return f"{wins}-{losses}"
    
    def _load_templates(self) -> Dict[str, List[str]]:
        """Load narrative templates for variation."""
        return {
            'qb_intro': [
                "Led by quarterback {qb}",
                "Behind the arm of {qb}",
                "Quarterback {qb} directs",
                "{qb} pilots"
            ],
            'coach_intro': [
                "head coach {coach}'s",
                "under {coach},",
                "{coach}'s squad",
                "coached by {coach},"
            ]
        }


def main():
    """Main execution: generate narratives for all games."""
    print("="*80)
    print("NFL NARRATIVE GENERATION")
    print("="*80)
    
    # Load complete dataset
    dataset_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'nfl_complete_dataset.json'
    
    print(f"\nLoading dataset from: {dataset_path}")
    with open(dataset_path) as f:
        games = json.load(f)
    
    print(f"✓ Loaded {len(games)} games")
    
    # Generate narratives
    generator = NFLNarrativeGenerator()
    games_with_narratives = generator.generate_narratives(games)
    
    # Save enriched dataset
    with open(dataset_path, 'w') as f:
        json.dump(games_with_narratives, f, indent=2)
    
    print(f"\n✓ Saved games with narratives to: {dataset_path}")
    
    # Display sample
    print("\n" + "="*80)
    print("SAMPLE NARRATIVE")
    print("="*80)
    sample = games_with_narratives[0]
    print(f"\nGame: {sample['away_team']} @ {sample['home_team']}")
    print(f"Season: {sample['season']}, Week: {sample['week']}")
    print(f"\nNarrative ({len(sample['narrative'].split())} words):")
    print(sample['narrative'])


if __name__ == '__main__':
    main()

