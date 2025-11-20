"""
Multi-Scale Multi-Perspective NBA Analysis

Captures narratives at ALL levels:
- SEASON arc (team's journey over 82 games)
- SERIES context (playoff matchup, rivalry history)
- GAME narrative (single contest story)
- QUARTER momentum (within-game shifts)

From ALL perspectives:
- TEAM collective (organization narrative)
- COACH strategy (tactical narrative)
- STAR players (individual narratives)
- ROLE players (ensemble narrative)

Each level has:
- Nominative features (names at that scale)
- Narrative features (stories at that scale)

This is the fractal, gravitational, quantum-ish analysis.
Stories within stories within stories.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict


class MultiScaleNBAAnalyzer:
    """
    Analyze NBA at multiple nested scales.
    
    Levels (fractal structure):
    1. Season (macro) - 82 games, playoff journey
    2. Series (meso) - playoff matchup or rivalry stretch
    3. Game (micro) - single contest
    4. Quarter (nano) - momentum within game
    
    Perspectives (parallel):
    - Team collective
    - Coach leadership
    - Star players (top 3)
    - Role players (bench)
    - Opponent (mirror narratives)
    """
    
    def __init__(self, games_data):
        """Initialize with full game dataset"""
        self.games = games_data
        self.teams = set(g['team_abbreviation'] for g in games_data)
        
        # Group by scales
        self.by_season = self._group_by_season()
        self.by_team_season = self._group_by_team_season()
        self.by_matchup = self._group_by_matchup()
    
    def _group_by_season(self):
        """Group games by season"""
        seasons = defaultdict(list)
        for game in self.games:
            seasons[game['season']].append(game)
        return dict(seasons)
    
    def _group_by_team_season(self):
        """Group by team-season (a team's journey)"""
        team_seasons = defaultdict(list)
        for game in self.games:
            key = (game['team_abbreviation'], game['season'])
            team_seasons[key].append(game)
        return dict(team_seasons)
    
    def _group_by_matchup(self):
        """Group by recurring matchups (series/rivalries)"""
        matchups = defaultdict(list)
        for game in self.games:
            # Normalize matchup (LAL vs BOS = BOS vs LAL)
            teams = game['matchup'].replace('vs.', ' ').replace('@', ' ').split()
            teams = [t for t in teams if len(t) == 3 and t.isupper()]
            if len(teams) >= 2:
                key = tuple(sorted(teams))
                matchups[key].append(game)
        return dict(matchups)
    
    def extract_season_narrative(self, team, season):
        """
        SEASON-LEVEL narrative (macro).
        
        A team's 82-game journey:
        - Early struggles vs. hot start
        - Mid-season slump or surge
        - Playoff push or tank
        - Championship aspirations
        - Injury narratives
        - Coaching changes
        - Roster evolution
        """
        key = (team, season)
        if key not in self.by_team_season:
            return {}
        
        games = sorted(self.by_team_season[key], key=lambda g: g.get('date', ''))
        
        # Season arc
        wins = [int(g['won']) for g in games]
        cumulative_record = []
        w, l = 0, 0
        for won in wins:
            if won:
                w += 1
            else:
                l += 1
            cumulative_record.append((w, l))
        
        # Narrative components
        narrative = {
            'team': team,
            'season': season,
            'n_games': len(games),
            'final_record': f"{w}-{l}",
            'win_pct': w / (w + l) if (w + l) > 0 else 0.5,
            
            # Arc shape
            'started_strong': wins[:10].count(1) / 10 if len(wins) >= 10 else 0.5,
            'finished_strong': wins[-10:].count(1) / 10 if len(wins) >= 10 else 0.5,
            'arc_trajectory': (wins[-10:].count(1) - wins[:10].count(1)) / 10 if len(wins) >= 20 else 0,
            
            # Volatility
            'streakiness': self._calculate_streakiness(wins),
            'consistency': 1.0 - self._calculate_streakiness(wins),
            
            # Momentum phases
            'best_10_game_stretch': max(sum(wins[i:i+10]) for i in range(len(wins)-10)) / 10 if len(wins) >= 10 else 0.5,
            'worst_10_game_stretch': min(sum(wins[i:i+10]) for i in range(len(wins)-10)) / 10 if len(wins) >= 10 else 0.5,
            
            # Narrative
            'season_narrative': self._generate_season_narrative(w, l, wins)
        }
        
        return narrative
    
    def extract_series_narrative(self, team1, team2, season):
        """
        SERIES-LEVEL narrative (meso).
        
        Matchup between two teams:
        - Historical rivalry
        - Season series (4 meetings)
        - Playoff implications
        - Competitive intensity
        - Momentum swings
        """
        matchup_key = tuple(sorted([team1, team2]))
        
        if matchup_key not in self.by_matchup:
            return {}
        
        # Get all games in this matchup for season
        games = [g for g in self.by_matchup[matchup_key] if g['season'] == season]
        
        # Series narrative
        team1_wins = sum(1 for g in games if g['team_abbreviation'] == team1 and g['won'])
        team2_wins = len([g for g in games if g['team_abbreviation'] == team2 and g['won']])
        
        narrative = {
            'matchup': f"{team1} vs {team2}",
            'season': season,
            'n_meetings': len(games),
            'series_record': f"{team1_wins}-{team2_wins}",
            'dominance': abs(team1_wins - team2_wins) / max(len(games), 1),
            'series_narrative': f"{team1} and {team2} met {len(games)} times, with {team1} winning {team1_wins}."
        }
        
        return narrative
    
    def extract_game_narrative(self, game):
        """
        GAME-LEVEL narrative (micro).
        
        Single contest:
        - Pre-game context (records, momentum)
        - Key players active/injured
        - Score trajectory
        - Turning points
        - Final outcome
        """
        # Already have game narrative
        # Enhance with context
        
        return {
            'matchup': game['matchup'],
            'date': game.get('date', ''),
            'outcome': int(game['won']),
            'score': game.get('points', 0),
            'margin': game.get('plus_minus', 0),
            'game_narrative': game.get('narrative', '')
        }
    
    def extract_quarter_narratives(self, game):
        """
        QUARTER-LEVEL narratives (nano).
        
        Within-game momentum:
        - Q1: Opening strategy
        - Q2: First half adjustments
        - Q3: Coming out of halftime
        - Q4: Closing execution
        """
        # Would need quarter-by-quarter data
        # For now, simulate from final score
        
        return {
            'has_quarter_data': False,
            'final_margin': game.get('plus_minus', 0),
            'was_close': abs(game.get('plus_minus', 100)) < 10
        }
    
    def extract_perspective_narratives(self, game):
        """
        PERSPECTIVE-LEVEL narratives.
        
        From different viewpoints:
        - Team organization (franchise story)
        - Coach (strategic narrative)
        - Star players (hero narratives)
        - Role players (supporting narratives)
        - Fans (community narrative)
        """
        team = game['team_abbreviation']
        
        return {
            # Team perspective (organization)
            'team_name': game['team_name'],
            'team_narrative': f"The {game['team_name']} organization",
            
            # Coach perspective (would need coach names)
            'coach_narrative': "Coaching staff strategic approach",
            
            # Player perspectives (would need roster)
            'star_narrative': "Star players lead the charge",
            'ensemble_narrative': "Role players provide depth",
            
            # Opponent perspective (mirror)
            'opponent_perspective': f"Facing {game['matchup'].split()[-1]}"
        }
    
    def create_complete_feature_vector(self, game):
        """
        Create feature vector capturing ALL scales and perspectives.
        
        Returns:
        - Season features (team's arc)
        - Series features (matchup history)
        - Game features (contest itself)
        - Quarter features (momentum)
        - Perspective features (team/coach/players)
        - Nominative features (names at all levels)
        - Gravitational features (clustering effects)
        """
        team = game['team_abbreviation']
        season = game['season']
        
        features = {}
        
        # Scale 1: Season narrative
        season_narr = self.extract_season_narrative(team, season)
        features['season_win_pct'] = season_narr.get('win_pct', 0.5)
        features['season_trajectory'] = season_narr.get('arc_trajectory', 0)
        features['season_consistency'] = season_narr.get('consistency', 0.5)
        
        # Scale 2: Series narrative
        opponent = self._extract_opponent(game['matchup'])
        if opponent:
            series_narr = self.extract_series_narrative(team, opponent, season)
            features['series_dominance'] = series_narr.get('dominance', 0)
        else:
            features['series_dominance'] = 0
        
        # Scale 3: Game narrative
        game_narr = self.extract_game_narrative(game)
        features['game_margin'] = abs(game_narr['margin']) / 100
        
        # Scale 4: Quarter (when available)
        quarter_narr = self.extract_quarter_narratives(game)
        features['was_close_game'] = float(quarter_narr['was_close'])
        
        # Scale 5: Perspectives
        persp_narr = self.extract_perspective_narratives(game)
        features['has_star_narrative'] = 1.0  # Placeholder
        
        # Nominative at each scale
        features['team_name_length'] = len(team)
        features['season_length'] = len(season)
        
        return features
    
    def _extract_opponent(self, matchup):
        """Extract opponent from matchup string"""
        teams = matchup.replace('vs.', ' ').replace('@', ' ').split()
        teams = [t for t in teams if len(t) == 3 and t.isupper()]
        return teams[1] if len(teams) >= 2 else None
    
    def _calculate_streakiness(self, wins):
        """Calculate streakiness (volatility in win/loss)"""
        if len(wins) < 2:
            return 0.0
        
        # Count streak changes
        changes = sum(1 for i in range(1, len(wins)) if wins[i] != wins[i-1])
        max_possible = len(wins) - 1
        
        return changes / max_possible if max_possible > 0 else 0.0
    
    def _generate_season_narrative(self, wins, losses, win_sequence):
        """Generate season narrative text"""
        record = f"{wins}-{losses}"
        pct = wins / (wins + losses) if (wins + losses) > 0 else 0.5
        
        if pct > 0.65:
            quality = "dominant"
        elif pct > 0.55:
            quality = "competitive"
        elif pct > 0.45:
            quality = "mediocre"
        else:
            quality = "struggling"
        
        # Arc
        if len(win_sequence) >= 20:
            early = sum(win_sequence[:10])
            late = sum(win_sequence[-10:])
            
            if late > early + 3:
                arc = "improving throughout the season"
            elif early > late + 3:
                arc = "declining from early success"
            else:
                arc = "consistent performance"
        else:
            arc = "season in progress"
        
        return f"A {quality} team ({record}), {arc}."


def main():
    """Demo multi-scale analysis"""
    print("="*80)
    print("MULTI-SCALE MULTI-PERSPECTIVE NBA ANALYSIS")
    print("="*80)
    
    # Load data
    data_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'nba_enriched_1000.json'
    
    with open(data_path) as f:
        games = json.load(f)
    
    print(f"\n✓ Loaded {len(games)} games")
    
    # Initialize analyzer
    analyzer = MultiScaleNBAAnalyzer(games)
    
    print(f"\n✓ Identified:")
    print(f"  {len(analyzer.by_season)} seasons")
    print(f"  {len(analyzer.by_team_season)} team-seasons")
    print(f"  {len(analyzer.by_matchup)} unique matchups")
    
    # Extract features for all games
    print(f"\nExtracting multi-scale features...")
    
    multi_scale_features = []
    for game in games[:100]:  # Sample for demo
        features = analyzer.create_complete_feature_vector(game)
        multi_scale_features.append(list(features.values()))
    
    multi_scale_features = np.array(multi_scale_features)
    
    print(f"\n✓ Multi-scale features: {multi_scale_features.shape}")
    print(f"  {multi_scale_features.shape[1]} features per game")
    print(f"  Captures: season + series + game + quarter + perspectives")
    
    # Example season narrative
    print(f"\n{'='*80}")
    print(f"EXAMPLE: SEASON-LEVEL NARRATIVE")
    print(f"{'='*80}")
    
    sample_team = list(analyzer.teams)[0]
    sample_season = list(analyzer.by_season.keys())[0]
    
    season_narrative = analyzer.extract_season_narrative(sample_team, sample_season)
    
    print(f"\nTeam: {season_narrative.get('team')}")
    print(f"Season: {season_narrative.get('season')}")
    print(f"Record: {season_narrative.get('final_record')}")
    print(f"Win %: {season_narrative.get('win_pct', 0):.1%}")
    print(f"Trajectory: {season_narrative.get('arc_trajectory', 0):+.2f}")
    print(f"\nNarrative: {season_narrative.get('season_narrative')}")
    
    # Example series narrative
    print(f"\n{'='*80}")
    print(f"EXAMPLE: SERIES-LEVEL NARRATIVE")
    print(f"{'='*80}")
    
    if analyzer.by_matchup:
        sample_matchup = list(analyzer.by_matchup.keys())[0]
        team1, team2 = sample_matchup
        
        series_narrative = analyzer.extract_series_narrative(team1, team2, sample_season)
        
        print(f"\nMatchup: {series_narrative.get('matchup')}")
        print(f"Meetings: {series_narrative.get('n_meetings')}")
        print(f"Series: {series_narrative.get('series_record')}")
        print(f"\nNarrative: {series_narrative.get('series_narrative')}")
    
    print(f"\n{'='*80}")
    print(f"💡 MULTI-SCALE CONCEPT")
    print(f"{'='*80}")
    print("""
Like characters in a story, NBA games exist at multiple scales:

SEASON = Novel arc (team's journey)
SERIES = Chapter (rivalry/matchup)
GAME = Scene (single contest)
QUARTER = Moment (momentum shift)

PERSPECTIVES = POV (team/coach/players)

Each scale has:
- Names (nominative features)
- Stories (narrative features)
- Gravitational effects (ф, ة)

This is fractal, quantum-ish, gravitational narrative analysis.
    """)
    
    print(f"\n✅ Multi-scale framework demonstrated")
    print(f"   Ready for full extraction + transformer pipeline")


if __name__ == '__main__':
    main()

