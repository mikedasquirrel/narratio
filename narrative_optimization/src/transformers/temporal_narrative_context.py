"""
Temporal Narrative Context Transformer

FUNDAMENTAL ADDITION: Captures the SERIAL nature of narratives.
Events aren't isolated - they're chapters in season-long, career-long, 
and franchise-long stories.

This is a CORE framework requirement, not an optimization.

Features Extracted (50 total):
- Recent performance (L3, L5, L10, streaks)
- Season position (standings, playoff race)
- Historical context (H2H, rivalries)
- Legacy/arc position (dynasty, rebuild, redemption)
- Narrative timing (act structure, journey stage)

Author: Narrative Optimization Framework
Date: November 13, 2025
"""

from typing import List, Dict, Any, Optional
import numpy as np
import re
from collections import defaultdict
from .base import NarrativeTransformer


class TemporalNarrativeContextTransformer(NarrativeTransformer):
    """
    Extracts temporal and seasonal narrative context.
    
    Recognizes that narratives are SERIAL:
    - Each game/match is a chapter in multiple overlapping stories
    - Season arcs, career trajectories, franchise legacies
    - Historical rivalries, momentum cascades, legacy implications
    
    This is essential for proper narrative analysis, especially in team sports
    where seasonal context dominates.
    """
    
    def __init__(self):
        super().__init__(
            narrative_id="temporal_narrative_context",
            description="Serial narrative: seasons, careers, legacies, rivalries, momentum"
        )
        
        # Will store reference data during fit
        self.season_patterns_ = None
        self.rivalry_database_ = None
        self.momentum_indicators_ = None
    
    def fit(self, X, y=None, metadata=None):
        """
        Learn temporal patterns from corpus.
        
        Parameters
        ----------
        X : list of str
            Narrative texts
        y : ignored
        metadata : dict, optional
            Temporal metadata:
            - season_data: records, standings by date
            - historical_h2h: head-to-head databases
            - career_data: player/coach trajectories
        
        Returns
        -------
        self
        """
        # Extract temporal patterns from narratives
        # (even without metadata, can extract from text)
        
        season_patterns = defaultdict(list)
        rivalry_patterns = defaultdict(int)
        momentum_patterns = []
        
        for narrative in X:
            # Extract season indicators from text
            # L10, L5, win streaks, etc. mentioned in narrative
            
            # Recent form patterns
            if 'L10' in narrative or 'last 10' in narrative.lower():
                season_patterns['has_recent_form'].append(1)
            
            # Streak indicators
            if 'streak' in narrative.lower():
                season_patterns['has_streak'].append(1)
            
            # Playoff mentions
            if 'playoff' in narrative.lower():
                season_patterns['has_playoff_context'].append(1)
            
            # Rivalry indicators
            if 'rival' in narrative.lower() or 'historic' in narrative.lower():
                rivalry_patterns['rivalry_games'] += 1
            
            # Momentum indicators
            if 'surge' in narrative.lower() or 'hot' in narrative.lower():
                momentum_patterns.append('positive')
            elif 'struggle' in narrative.lower() or 'cold' in narrative.lower():
                momentum_patterns.append('negative')
        
        self.season_patterns_ = dict(season_patterns)
        self.rivalry_database_ = dict(rivalry_patterns)
        self.momentum_indicators_ = momentum_patterns
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X, metadata=None):
        """
        Extract temporal context features from narratives.
        
        Parameters
        ----------
        X : list of str
            Narratives to transform
        metadata : dict, optional
            Game-specific temporal metadata
        
        Returns
        -------
        features : ndarray, shape (n_samples, 50)
            Temporal context features
        """
        self._validate_fitted()
        
        features_list = []
        
        for idx, narrative in enumerate(X):
            doc_features = []
            
            # ============================================================
            # RECENT PERFORMANCE (10 features)
            # ============================================================
            
            # L3, L5, L10 win percentage (extract from text or metadata)
            l10_match = re.search(r'(\d+)-(\d+)\s+(?:in\s+)?(?:last|L)\s*10', narrative, re.I)
            if l10_match:
                wins = int(l10_match.group(1))
                l10_pct = wins / 10.0
            else:
                l10_pct = 0.5  # Neutral if not mentioned
            
            doc_features.append(l10_pct)  # L10_win_pct
            doc_features.append(l10_pct)  # L5_win_pct (simplified)
            doc_features.append(l10_pct)  # L3_win_pct (simplified)
            
            # Streak detection
            streak_pattern = re.search(r'(\d+)[\s-]game\s+(win|loss|winning|losing)\s+streak', narrative, re.I)
            if streak_pattern:
                streak_length = int(streak_pattern.group(1))
                streak_type = 1.0 if 'win' in streak_pattern.group(2).lower() else -1.0
            else:
                streak_length = 0
                streak_type = 0.0
            
            doc_features.append(streak_type)  # Current_streak
            doc_features.append(min(streak_length / 10.0, 1.0))  # Streak_length (normalized)
            
            # Form trend (surge/struggle indicators)
            surge_score = len(re.findall(r'surg|hot|momentum|dominat', narrative, re.I)) / 10.0
            struggle_score = len(re.findall(r'struggl|cold|slump|woes', narrative, re.I)) / 10.0
            doc_features.append(min(surge_score, 1.0))  # Form_trend_positive
            doc_features.append(min(struggle_score, 1.0))  # Form_trend_negative
            
            # Recent margins (from narrative if mentioned)
            doc_features.append(0.5)  # Recent_margin_avg (placeholder)
            doc_features.append(0.5)  # Recent_scoring_avg (placeholder)
            doc_features.append(0.5)  # Home_road_split (placeholder)
            
            # ============================================================
            # SEASON POSITION (8 features)
            # ============================================================
            
            # Extract record (W-L format)
            record_match = re.search(r'\((\d+)-(\d+)\)', narrative)
            if record_match:
                wins = int(record_match.group(1))
                losses = int(record_match.group(2))
                total = wins + losses
                season_pct = wins / total if total > 0 else 0.5
                games_played_pct = total / 82.0  # Assume ~82 game season
            else:
                season_pct = 0.5
                games_played_pct = 0.5
            
            doc_features.append(games_played_pct)  # Games_played_pct
            doc_features.append(season_pct)  # Season_win_pct
            
            # Division/playoff context
            division_score = len(re.findall(r'division|conference', narrative, re.I)) / 5.0
            playoff_score = len(re.findall(r'playoff|postseason|contend', narrative, re.I)) / 5.0
            
            doc_features.append(min(division_score, 1.0))  # Division_importance
            doc_features.append(min(playoff_score, 1.0))  # Playoff_implications
            
            # Must-win indicators
            must_win = 1.0 if any(x in narrative.lower() for x in ['must win', 'elimination', 'crucial', 'decisive']) else 0.0
            doc_features.append(must_win)
            
            # Season phase
            if 'week 1' in narrative.lower() or 'opening' in narrative.lower():
                phase = 0.0  # Early
            elif 'playoff' in narrative.lower() or 'championship' in narrative.lower():
                phase = 1.0  # Late/playoffs
            else:
                phase = 0.5  # Midseason
            
            doc_features.append(phase)  # Season_phase
            doc_features.append(playoff_score)  # Playoff_probability
            doc_features.append(0.5)  # Games_back (placeholder)
            
            # ============================================================
            # HISTORICAL CONTEXT (12 features)
            # ============================================================
            
            # Rivalry indicators
            rivalry_score = len(re.findall(r'rival|historic|tradition|classic', narrative, re.I)) / 5.0
            doc_features.append(min(rivalry_score, 1.0))  # Rivalry_intensity
            
            # Head-to-head mentions
            h2h_score = len(re.findall(r'head.to.head|previous|last time|earlier this', narrative, re.I)) / 3.0
            doc_features.append(min(h2h_score, 1.0))  # H2H_context
            
            # Revenge narrative
            revenge = 1.0 if any(x in narrative.lower() for x in ['revenge', 'payback', 'redemption']) else 0.0
            doc_features.append(revenge)
            
            # Historical dominance
            dominance = len(re.findall(r'dominat|dynasty|powerhouse', narrative, re.I)) / 3.0
            doc_features.append(min(dominance, 1.0))
            
            # Placeholders for features that need full metadata
            doc_features.extend([0.5] * 8)  # H2H_win_pct, H2H_total, etc.
            
            # ============================================================
            # LEGACY/ARC CONTEXT (10 features)
            # ============================================================
            
            # Dynasty indicators
            dynasty = len(re.findall(r'dynasty|champion|legacy|golden age', narrative, re.I)) / 3.0
            doc_features.append(min(dynasty, 1.0))
            
            # Rebuild indicators
            rebuild = len(re.findall(r'rebuild|young|development|future', narrative, re.I)) / 3.0
            doc_features.append(min(rebuild, 1.0))
            
            # Contender indicators
            contender = len(re.findall(r'contend|elite|top tier|championship', narrative, re.I)) / 3.0
            doc_features.append(min(contender, 1.0))
            
            # Breakthrough narrative
            breakthrough = len(re.findall(r'breakthrough|emerge|rise|ascend', narrative, re.I)) / 3.0
            doc_features.append(min(breakthrough, 1.0))
            
            # Curse/drought
            curse = len(re.findall(r'curse|drought|futility|struggle', narrative, re.I)) / 3.0
            doc_features.append(min(curse, 1.0))
            
            # Placeholders
            doc_features.extend([0.5] * 5)  # Coach_tenure, Star_phase, etc.
            
            # ============================================================
            # NARRATIVE TIMING (10 features)
            # ============================================================
            
            # Act structure (where in season narrative?)
            if games_played_pct < 0.33:
                act = 0.33  # Act 1 (setup)
            elif games_played_pct < 0.67:
                act = 0.67  # Act 2 (conflict)
            else:
                act = 1.0  # Act 3 (resolution)
            
            doc_features.append(act)  # Act_structure_season
            
            # Journey stage
            if games_played_pct < 0.25:
                journey = 0.25  # Departure
            elif games_played_pct < 0.75:
                journey = 0.50  # Trials
            else:
                journey = 1.0  # Return
            
            doc_features.append(journey)  # Journey_stage_season
            
            # Urgency/stakes
            urgency = playoff_score + must_win + rivalry_score
            doc_features.append(min(urgency, 1.0))  # Story_urgency
            
            # Momentum direction
            momentum_score = surge_score - struggle_score
            doc_features.append(max(-1, min(1, momentum_score)))  # Narrative_momentum
            
            # Placeholders
            doc_features.extend([0.5] * 6)  # Climax_proximity, Resolution_phase, etc.
            
            features_list.append(doc_features)
        
        return np.array(features_list)
    
    def get_feature_names(self):
        """Return names of extracted features."""
        return [
            # Recent Performance (10)
            'L10_win_pct', 'L5_win_pct', 'L3_win_pct',
            'current_streak_direction', 'streak_length_norm',
            'form_trend_positive', 'form_trend_negative',
            'recent_margin_avg', 'recent_scoring_avg', 'home_road_split',
            
            # Season Position (8)
            'games_played_pct', 'season_win_pct',
            'division_importance', 'playoff_implications',
            'must_win_game', 'season_phase', 'playoff_probability', 'games_back',
            
            # Historical Context (12)
            'rivalry_intensity', 'h2h_context', 'revenge_narrative', 'historical_dominance',
            'h2h_win_pct', 'h2h_total_games', 'recent_h2h_l3', 'playoff_h2h',
            'venue_h2h', 'season_series', 'rivalry_age', 'momentum_shift',
            
            # Legacy/Arc (10)
            'dynasty_indicator', 'rebuild_indicator', 'contender_indicator',
            'breakthrough_narrative', 'curse_drought',
            'coach_tenure', 'star_career_phase', 'franchise_era', 
            'championship_window', 'legacy_game',
            
            # Narrative Timing (10)
            'act_structure_season', 'journey_stage_season', 
            'story_urgency', 'narrative_momentum',
            'climax_proximity', 'resolution_phase', 'plot_twist_potential',
            'culmination_indicator', 'redemption_arc', 'vindication_opportunity'
        ]

