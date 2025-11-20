"""
Nested Narrative Tracker

Tracks narratives across 5 hierarchical levels:
1. Moment (quarters, plays)
2. Game (single game)
3. Series (playoff series, rivalry sequences)
4. Season (82-game arc)
5. Era (multi-year dynasty/rebuild)

Each level has optimal α parameter and accumulation rules.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict


class NarrativeLevel:
    """Represents a single level in narrative hierarchy."""
    
    def __init__(
        self,
        name: str,
        level: int,
        typical_duration: int,
        alpha_expected: float,
        description: str
    ):
        self.name = name
        self.level = level
        self.typical_duration = typical_duration  # Number of lower-level units
        self.alpha_expected = alpha_expected  # Expected α for this level
        self.description = description
        self.alpha_discovered = None  # To be optimized
        self.narratives = {}  # narrative_id -> narrative data
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'level': self.level,
            'typical_duration': self.typical_duration,
            'alpha_expected': self.alpha_expected,
            'alpha_discovered': self.alpha_discovered,
            'description': self.description,
            'narrative_count': len(self.narratives)
        }


class NestedNarrativeTracker:
    """
    Tracks narratives across hierarchical temporal scales.
    
    Implements: "Stories within stories" - games within series within seasons.
    
    Discovers:
    - Optimal α at each level
    - How narratives accumulate upward
    - When higher-level narratives emerge
    - Cross-level influence patterns
    """
    
    def __init__(self):
        # Define 5 narrative levels
        self.levels = {
            'moment': NarrativeLevel(
                name='moment',
                level=0,
                typical_duration=4,  # 4 quarters
                alpha_expected=0.15,
                description='Within-game moments, emotional flow, runs'
            ),
            'game': NarrativeLevel(
                name='game',
                level=1,
                typical_duration=1,  # Single game
                alpha_expected=0.28,
                description='Single game narrative, matchup story'
            ),
            'series': NarrativeLevel(
                name='series',
                level=2,
                typical_duration=7,  # Up to 7 games in series
                alpha_expected=0.22,
                description='Series narrative, playoff arc, rivalry sequence'
            ),
            'season': NarrativeLevel(
                name='season',
                level=3,
                typical_duration=82,  # Regular season games
                alpha_expected=0.25,
                description='Full season arc, championship quest, rebuild year'
            ),
            'era': NarrativeLevel(
                name='era',
                level=4,
                typical_duration=5,  # Multiple seasons
                alpha_expected=0.18,
                description='Dynasty, rebuilding era, franchise legacy period'
            )
        }
        
        # Story threads that span levels
        self.threads = {}  # thread_id -> {games, weight, archetype, status}
        
        # Narrative accumulation tracking
        self.accumulations = defaultdict(list)  # level -> accumulated narratives
    
    def add_game_narrative(
        self,
        game_id: str,
        team: str,
        opponent: str,
        narrative_text: str,
        features: Dict[str, float],
        context: Dict[str, Any],
        timestamp: datetime,
        outcome: Optional[int] = None
    ):
        """
        Add a game-level narrative and propagate upward.
        
        Parameters
        ----------
        game_id : str
            Unique game identifier
        team : str
            Team name
        opponent : str
            Opponent name
        narrative_text : str
            Generated narrative text
        features : dict
            Extracted narrative features
        context : dict
            Game context (stakes, rivalry, etc.)
        timestamp : datetime
            When game occurred
        outcome : int, optional
            Game result (1=win, 0=loss)
        """
        # Store at game level
        game_narrative = {
            'id': game_id,
            'team': team,
            'opponent': opponent,
            'text': narrative_text,
            'features': features,
            'context': context,
            'timestamp': timestamp,
            'outcome': outcome,
            'level': 'game'
        }
        
        self.levels['game'].narratives[game_id] = game_narrative
        
        # Propagate upward
        self._propagate_upward(game_narrative)
    
    def _propagate_upward(self, lower_narrative: Dict):
        """Propagate narrative from lower level to higher levels."""
        # Check if this game is part of a series
        series_id = self._detect_series(lower_narrative)
        if series_id:
            self._accumulate_to_level('series', series_id, lower_narrative)
        
        # Check if this game is part of season narrative
        season_id = self._detect_season(lower_narrative)
        if season_id:
            self._accumulate_to_level('season', season_id, lower_narrative)
        
        # Check if this is part of era
        era_id = self._detect_era(lower_narrative)
        if era_id:
            self._accumulate_to_level('era', era_id, lower_narrative)
    
    def _detect_series(self, game_narrative: Dict) -> Optional[str]:
        """Detect if game is part of playoff series or rivalry sequence."""
        context = game_narrative['context']
        
        # Playoff series
        if context.get('playoff', False):
            return f"playoff_{game_narrative['team']}_vs_{game_narrative['opponent']}_{context.get('playoff_round', 'round1')}"
        
        # Rivalry sequence (multiple games in short time)
        matchup = f"{game_narrative['team']}_vs_{game_narrative['opponent']}"
        if context.get('rivalry', False):
            season = game_narrative['timestamp'].year
            return f"rivalry_{matchup}_{season}"
        
        return None
    
    def _detect_season(self, game_narrative: Dict) -> str:
        """Detect season identifier."""
        season_year = self._get_season_year(game_narrative['timestamp'])
        team = game_narrative['team']
        return f"season_{team}_{season_year}"
    
    def _detect_era(self, game_narrative: Dict) -> str:
        """Detect era (5-year periods)."""
        year = game_narrative['timestamp'].year
        era_start = (year // 5) * 5
        team = game_narrative['team']
        return f"era_{team}_{era_start}s"
    
    def _get_season_year(self, timestamp: datetime) -> str:
        """Convert datetime to NBA season year."""
        # NBA season spans two calendar years
        if timestamp.month >= 10:  # Oct-Dec
            return f"{timestamp.year}-{timestamp.year+1}"
        else:  # Jan-June
            return f"{timestamp.year-1}-{timestamp.year}"
    
    def _accumulate_to_level(self, level: str, narrative_id: str, lower_narrative: Dict):
        """Accumulate lower-level narrative into higher level."""
        if narrative_id not in self.levels[level].narratives:
            self.levels[level].narratives[narrative_id] = {
                'id': narrative_id,
                'level': level,
                'component_narratives': [],
                'accumulated_features': None,
                'emergence_score': 0.0,
                'archetype': None
            }
        
        # Add to components
        self.levels[level].narratives[narrative_id]['component_narratives'].append(lower_narrative)
        
        # Update accumulation
        self._update_accumulated_features(level, narrative_id)
    
    def _update_accumulated_features(self, level: str, narrative_id: str):
        """Update accumulated features for higher-level narrative."""
        higher_narrative = self.levels[level].narratives[narrative_id]
        component_narratives = higher_narrative['component_narratives']
        
        if not component_narratives:
            return
        
        # Accumulate with temporal decay
        accumulated = {}
        total_weight = 0
        
        # Sort by timestamp (oldest first)
        sorted_components = sorted(component_narratives, key=lambda x: x['timestamp'])
        
        for i, comp in enumerate(sorted_components):
            # Recency weight (more recent = higher weight)
            recency_weight = 0.95 ** (len(sorted_components) - i - 1)
            
            # Context weight (high-stakes games persist)
            context_weight = comp['context'].get('weight', 1.0)
            
            # Combined weight
            weight = recency_weight * context_weight
            
            # Accumulate features
            for feature, value in comp['features'].items():
                if feature not in accumulated:
                    accumulated[feature] = 0
                accumulated[feature] += value * weight
            
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            for feature in accumulated:
                accumulated[feature] /= total_weight
        
        higher_narrative['accumulated_features'] = accumulated
        
        # Calculate emergence score
        higher_narrative['emergence_score'] = self._calculate_emergence(component_narratives)
    
    def _calculate_emergence(self, component_narratives: List[Dict]) -> float:
        """
        Calculate emergence score: When does higher narrative crystallize?
        
        Returns score [0, 1]:
        - 0.0: No higher narrative yet (too few components)
        - 0.5: Emerging (pattern visible but uncertain)
        - 1.0: Fully emerged (clear higher-level story)
        """
        n_components = len(component_narratives)
        
        if n_components < 2:
            return 0.0
        elif n_components < 3:
            return 0.3
        elif n_components < 5:
            return 0.6
        else:
            return min(1.0, 0.6 + (n_components - 5) * 0.08)
    
    def get_narrative_at_level(
        self,
        level: str,
        narrative_id: str
    ) -> Optional[Dict]:
        """Get narrative at specified level."""
        return self.levels[level].narratives.get(narrative_id)
    
    def get_all_narratives_at_level(self, level: str) -> Dict[str, Dict]:
        """Get all narratives at specified level."""
        return self.levels[level].narratives
    
    def get_hierarchy_for_game(self, game_id: str) -> Dict[str, Any]:
        """
        Get complete narrative hierarchy for a game.
        
        Returns narratives at all levels that include this game.
        """
        game_narrative = self.levels['game'].narratives.get(game_id)
        if not game_narrative:
            return {}
        
        hierarchy = {
            'game': game_narrative
        }
        
        # Find series
        series_id = self._detect_series(game_narrative)
        if series_id and series_id in self.levels['series'].narratives:
            hierarchy['series'] = self.levels['series'].narratives[series_id]
        
        # Find season
        season_id = self._detect_season(game_narrative)
        if season_id and season_id in self.levels['season'].narratives:
            hierarchy['season'] = self.levels['season'].narratives[season_id]
        
        # Find era
        era_id = self._detect_era(game_narrative)
        if era_id and era_id in self.levels['era'].narratives:
            hierarchy['era'] = self.levels['era'].narratives[era_id]
        
        return hierarchy
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about narrative hierarchy."""
        stats = {}
        
        for level_name, level in self.levels.items():
            stats[level_name] = {
                'count': len(level.narratives),
                'alpha_expected': level.alpha_expected,
                'alpha_discovered': level.alpha_discovered,
                'description': level.description
            }
        
        # Calculate emergence statistics
        total_emerged = 0
        total_narratives = 0
        
        for level_name in ['series', 'season', 'era']:
            for narrative in self.levels[level_name].narratives.values():
                total_narratives += 1
                if narrative.get('emergence_score', 0) > 0.7:
                    total_emerged += 1
        
        stats['emergence'] = {
            'total_higher_narratives': total_narratives,
            'fully_emerged': total_emerged,
            'emergence_rate': total_emerged / total_narratives if total_narratives > 0 else 0
        }
        
        return stats
    
    def detect_story_archetypes(self, level: str) -> Dict[str, List[str]]:
        """
        Detect story archetypes at given level.
        
        Returns archetype classifications for narratives.
        """
        archetypes = defaultdict(list)
        
        for narrative_id, narrative in self.levels[level].narratives.items():
            archetype = self._classify_archetype(narrative, level)
            archetypes[archetype].append(narrative_id)
        
        return dict(archetypes)
    
    def _classify_archetype(self, narrative: Dict, level: str) -> str:
        """Classify narrative into archetype based on features and context."""
        if level == 'game':
            # Game archetypes
            context = narrative.get('context', {})
            if context.get('championship', False):
                return 'Epic Clash'
            elif context.get('blowout', False):
                return 'Dominant Victory'
            elif context.get('close', False):
                return 'Nail Biter'
            elif context.get('upset', False):
                return 'Underdog Triumph'
            else:
                return 'Standard Match'
        
        elif level == 'series':
            # Series archetypes
            components = narrative.get('component_narratives', [])
            n_games = len(components)
            
            if n_games >= 7:
                return 'Epic Seven-Game Battle'
            elif n_games <= 4:
                return 'Decisive Sweep'
            else:
                return 'Competitive Series'
        
        elif level == 'season':
            # Season archetypes
            return 'Championship Quest'  # Placeholder - refine based on features
        
        elif level == 'era':
            # Era archetypes
            return 'Dynasty Era'  # Placeholder
        
        return 'Unknown'


class NestedNarrativeTracker:
    """Main tracker for hierarchical narratives."""
    
    def __init__(self):
        self.level_manager = self._initialize_levels()
        self.game_narratives = {}
        self.hierarchy_index = {}  # game_id -> {series_id, season_id, era_id}
    
    def _initialize_levels(self) -> Dict[str, NarrativeLevel]:
        """Initialize the 5 narrative levels."""
        return {
            'moment': NarrativeLevel('moment', 0, 4, 0.15, 'In-game moments and runs'),
            'game': NarrativeLevel('game', 1, 1, 0.28, 'Single game narrative'),
            'series': NarrativeLevel('series', 2, 7, 0.22, 'Playoff series or rivalry sequence'),
            'season': NarrativeLevel('season', 3, 82, 0.25, 'Full season arc'),
            'era': NarrativeLevel('era', 4, 5, 0.18, 'Multi-year dynasty/rebuild era')
        }
    
    def add_game(
        self,
        game_id: str,
        team: str,
        opponent: str,
        narrative_features: Dict[str, float],
        context: Dict[str, Any],
        date: datetime,
        outcome: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Add game and propagate through hierarchy.
        
        Returns hierarchy info including series, season, era IDs.
        """
        # Store game
        game_narrative = {
            'id': game_id,
            'team': team,
            'opponent': opponent,
            'features': narrative_features,
            'context': context,
            'date': date,
            'outcome': outcome
        }
        
        self.game_narratives[game_id] = game_narrative
        self.level_manager['game'].narratives[game_id] = game_narrative
        
        # Detect higher-level memberships
        series_id = self._detect_series_membership(game_narrative)
        season_id = self._detect_season_membership(game_narrative)
        era_id = self._detect_era_membership(game_narrative)
        
        # Store hierarchy index
        self.hierarchy_index[game_id] = {
            'game_id': game_id,
            'series_id': series_id,
            'season_id': season_id,
            'era_id': era_id
        }
        
        # Accumulate upward
        if series_id:
            self._accumulate_into_series(series_id, game_narrative)
        
        if season_id:
            self._accumulate_into_season(season_id, game_narrative)
        
        if era_id:
            self._accumulate_into_era(era_id, game_narrative)
        
        return self.hierarchy_index[game_id]
    
    def _detect_series_membership(self, game: Dict) -> Optional[str]:
        """Detect if game belongs to a series (playoff or rivalry)."""
        context = game['context']
        team = game['team']
        opponent = game['opponent']
        
        # Playoff series
        if context.get('is_playoff', False):
            round_name = context.get('playoff_round', 'round1')
            season = game['date'].year
            return f"playoff_{team}_{opponent}_{round_name}_{season}"
        
        # Rivalry series (games within 30 days)
        if context.get('rivalry', False):
            season = game['date'].year
            return f"rivalry_{team}_{opponent}_{season}"
        
        return None
    
    def _detect_season_membership(self, game: Dict) -> str:
        """Detect season identifier."""
        team = game['team']
        date = game['date']
        
        # NBA season spans Oct-June
        if date.month >= 10:
            season_year = f"{date.year}-{date.year+1}"
        else:
            season_year = f"{date.year-1}-{date.year}"
        
        return f"season_{team}_{season_year}"
    
    def _detect_era_membership(self, game: Dict) -> str:
        """Detect era (5-year periods)."""
        team = game['team']
        year = game['date'].year
        era_decade = (year // 5) * 5
        return f"era_{team}_{era_decade}s"
    
    def _accumulate_into_series(self, series_id: str, game: Dict):
        """Accumulate game into series narrative."""
        if series_id not in self.level_manager['series'].narratives:
            self.level_manager['series'].narratives[series_id] = {
                'id': series_id,
                'level': 'series',
                'games': [],
                'accumulated_features': {},
                'emergence_score': 0.0
            }
        
        series = self.level_manager['series'].narratives[series_id]
        series['games'].append(game)
        
        # Recalculate accumulated features
        self._recalculate_accumulated_features(series)
    
    def _accumulate_into_season(self, season_id: str, game: Dict):
        """Accumulate game into season narrative."""
        if season_id not in self.level_manager['season'].narratives:
            self.level_manager['season'].narratives[season_id] = {
                'id': season_id,
                'level': 'season',
                'games': [],
                'accumulated_features': {},
                'emergence_score': 0.0
            }
        
        season = self.level_manager['season'].narratives[season_id]
        season['games'].append(game)
        self._recalculate_accumulated_features(season)
    
    def _accumulate_into_era(self, era_id: str, game: Dict):
        """Accumulate game into era narrative."""
        if era_id not in self.level_manager['era'].narratives:
            self.level_manager['era'].narratives[era_id] = {
                'id': era_id,
                'level': 'era',
                'games': [],
                'accumulated_features': {},
                'emergence_score': 0.0
            }
        
        era = self.level_manager['era'].narratives[era_id]
        era['games'].append(game)
        self._recalculate_accumulated_features(era)
    
    def _recalculate_accumulated_features(self, higher_narrative: Dict):
        """Recalculate accumulated features with decay."""
        games = higher_narrative.get('games', [])
        
        if not games:
            return
        
        # Sort by date
        sorted_games = sorted(games, key=lambda x: x['date'])
        
        accumulated = {}
        total_weight = 0
        
        # Temporal decay rate
        gamma = 0.95
        
        for i, game in enumerate(sorted_games):
            # Recency weight
            recency = gamma ** (len(sorted_games) - i - 1)
            
            # Context weight
            context_weight = game['context'].get('weight', 1.0)
            
            # Win/loss amplification
            outcome_mult = 1.2 if game.get('outcome') == 1 else 0.8
            
            # Total weight
            weight = recency * context_weight * outcome_mult
            
            # Accumulate
            for feature, value in game['features'].items():
                if feature not in accumulated:
                    accumulated[feature] = 0
                accumulated[feature] += value * weight
            
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            for feature in accumulated:
                accumulated[feature] /= total_weight
        
        higher_narrative['accumulated_features'] = accumulated
        
        # Update emergence score
        n_games = len(games)
        if higher_narrative['level'] == 'series':
            # Series emerges after 2-3 games
            higher_narrative['emergence_score'] = min(1.0, n_games / 3.0)
        elif higher_narrative['level'] == 'season':
            # Season emerges after ~20 games
            higher_narrative['emergence_score'] = min(1.0, n_games / 20.0)
        elif higher_narrative['level'] == 'era':
            # Era emerges after ~200 games
            higher_narrative['emergence_score'] = min(1.0, n_games / 200.0)
    
    def get_multi_scale_prediction(
        self,
        game_id: str,
        predictor_function
    ) -> Dict[str, Any]:
        """
        Get predictions at all narrative levels for a game.
        
        Parameters
        ----------
        game_id : str
            Game to predict
        predictor_function : callable
            Function(features, alpha) -> prediction
        
        Returns
        -------
        predictions : dict
            Predictions at game, series, season, era levels
        """
        hierarchy = self.hierarchy_index.get(game_id, {})
        
        if not hierarchy:
            return {}
        
        predictions = {}
        
        # Game-level prediction
        game = self.game_narratives.get(game_id)
        if game:
            alpha_game = self.level_manager['game'].alpha_discovered or 0.28
            predictions['game'] = {
                'level': 'game',
                'alpha': alpha_game,
                'prediction': predictor_function(game['features'], alpha_game),
                'narrative_type': 'immediate'
            }
        
        # Series-level prediction (if part of series)
        series_id = hierarchy.get('series_id')
        if series_id and series_id in self.level_manager['series'].narratives:
            series = self.level_manager['series'].narratives[series_id]
            if series['accumulated_features']:
                alpha_series = self.level_manager['series'].alpha_discovered or 0.22
                predictions['series'] = {
                    'level': 'series',
                    'alpha': alpha_series,
                    'prediction': predictor_function(series['accumulated_features'], alpha_series),
                    'emergence': series['emergence_score'],
                    'games_in_series': len(series['games']),
                    'narrative_type': 'series_arc'
                }
        
        # Season-level prediction
        season_id = hierarchy.get('season_id')
        if season_id and season_id in self.level_manager['season'].narratives:
            season = self.level_manager['season'].narratives[season_id]
            if season['accumulated_features']:
                alpha_season = self.level_manager['season'].alpha_discovered or 0.25
                predictions['season'] = {
                    'level': 'season',
                    'alpha': alpha_season,
                    'prediction': predictor_function(season['accumulated_features'], alpha_season),
                    'emergence': season['emergence_score'],
                    'games_in_season': len(season['games']),
                    'narrative_type': 'season_arc'
                }
        
        # Era-level prediction
        era_id = hierarchy.get('era_id')
        if era_id and era_id in self.level_manager['era'].narratives:
            era = self.level_manager['era'].narratives[era_id]
            if era['accumulated_features']:
                alpha_era = self.level_manager['era'].alpha_discovered or 0.18
                predictions['era'] = {
                    'level': 'era',
                    'alpha': alpha_era,
                    'prediction': predictor_function(era['accumulated_features'], alpha_era),
                    'emergence': era['emergence_score'],
                    'games_in_era': len(era['games']),
                    'narrative_type': 'legacy_arc'
                }
        
        return predictions
    
    def export_hierarchy(self) -> Dict[str, Any]:
        """Export complete hierarchy for analysis."""
        export = {
            'levels': {},
            'hierarchy_index': self.hierarchy_index,
            'statistics': self.get_statistics()
        }
        
        for level_name, level in self.level_manager.items():
            export['levels'][level_name] = level.to_dict()
        
        return export

