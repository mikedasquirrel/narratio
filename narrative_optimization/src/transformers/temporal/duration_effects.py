"""
Duration Effects Transformer

Analyzes how narrative duration affects accessibility, constraints, and possibilities.

Core Insights:
- Short-form (< 5 min): Forces pure plot, no character depth possible
- Medium-form (5-120 min): Balanced narrative possible
- Long-form (> 2 hours): Requires exceptional pacing to maintain engagement
- Serial vs episodic vs continuous: Different duration effects

Key Relationships:
- π_accessible = π_theoretical × (1 - |τ - 1|/3)  # Extreme durations reduce narrativity
- Attention span compatibility: Duration must match audience capacity
- Character development requires minimum duration threshold

Author: Narrative Optimization Framework - Phase 1 Implementation
Date: November 2025
"""

from typing import List, Dict, Any, Optional
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from ..base import NarrativeTransformer
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from base import NarrativeTransformer


class DurationEffectsTransformer(NarrativeTransformer):
    """
    Extracts duration effects features: accessibility, constraints, form requirements.
    
    Features (45 total):
    - Form classification (10): Short/medium/long, serial/episodic
    - Accessibility effects (10): How duration affects narrativity
    - Content requirements (10): What's possible at this duration
    - Attention span matching (10): Cognitive capacity alignment
    - Duration quality (5): Overall duration appropriateness
    
    Applications:
    - Film: Does 180min epic justify extended duration?
    - TV: Is this better as series or movie?
    - Music: Does prog epic warrant 20min vs 3min pop?
    - Literature: Novel-worthy or short story?
    """
    
    def __init__(self, domain: str = 'general'):
        """Initialize with domain-specific duration thresholds."""
        super().__init__(
            narrative_id="duration_effects",
            description="Duration accessibility, constraints, and form requirements"
        )
        
        self.domain = domain
        
        # Duration thresholds (in minutes)
        self.duration_categories = {
            'micro': (0, 1),  # Flash fiction, tweets
            'mini': (1, 5),  # Short video, haiku
            'short': (5, 30),  # Short story, pop song
            'medium': (30, 120),  # Film, novella
            'long': (120, 360),  # Epic film, novel
            'extended': (360, 1440),  # Long novel, TV season
            'marathon': (1440, float('inf'))  # Epic series, saga
        }
        
        # Minimum durations for narrative elements (empirical)
        self.minimum_durations = {
            'character_introduction': 2,  # minutes
            'character_development': 10,
            'character_transformation': 30,
            'relationship_development': 15,
            'subplot': 20,
            'world_building': 20,
            'theme_exploration': 30,
            'emotional_arc': 15,
            'mystery_setup_payoff': 40,
            'complex_plot': 60,
        }
        
        # Attention span thresholds by audience
        self.attention_thresholds = {
            'young_children': 15,  # minutes
            'children': 30,
            'teens': 60,
            'adults_casual': 120,
            'adults_engaged': 180,
            'enthusiasts': 360,
            'scholars': 600,
        }
        
        self.duration_stats_ = None
        
    def fit(self, X, y=None, metadata=None):
        """Learn duration distribution from corpus."""
        durations = []
        
        for idx, narrative in enumerate(X):
            duration = self._estimate_duration(narrative, metadata, idx)
            if duration:
                durations.append(duration)
        
        if durations:
            self.duration_stats_ = {
                'mean': np.mean(durations),
                'std': np.std(durations),
                'percentiles': np.percentile(durations, [10, 25, 50, 75, 90]),
                'min': np.min(durations),
                'max': np.max(durations)
            }
        else:
            self.duration_stats_ = {
                'mean': 60, 'std': 30, 'percentiles': [15, 30, 60, 120, 240],
                'min': 1, 'max': 500
            }
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X, metadata=None):
        """Extract duration effects features."""
        self._validate_fitted()
        
        features_list = []
        
        for idx, narrative in enumerate(X):
            doc_features = []
            
            # Estimate duration
            duration = self._estimate_duration(narrative, metadata, idx)
            
            # Count narrative elements present
            elements = self._count_narrative_elements(narrative)
            
            # ============================================================
            # Form Classification (10 features)
            # ============================================================
            
            # Duration category indicators
            for category, (min_dur, max_dur) in self.duration_categories.items():
                in_category = 1.0 if min_dur <= duration < max_dur else 0.0
                doc_features.append(in_category)
            
            # Serial structure indicators
            has_chapters = len(re.findall(r'Chapter \d+', narrative)) > 1
            has_episodes = len(re.findall(r'Episode \d+', narrative)) > 1
            has_acts = len(re.findall(r'Act [IVX\d]+', narrative, re.I)) > 1
            
            doc_features.append(1.0 if has_chapters else 0.0)
            doc_features.append(1.0 if has_episodes else 0.0)
            doc_features.append(1.0 if has_acts else 0.0)
            
            # ============================================================
            # Accessibility Effects (10 features)
            # ============================================================
            
            # Calculate π_accessible (duration reduces effective narrativity)
            tau = duration / 60.0  # Normalize to ~60min baseline
            accessibility_penalty = abs(tau - 1.0) / 3.0
            accessibility = max(0, 1.0 - accessibility_penalty)
            doc_features.append(accessibility)
            
            # Duration appropriateness for content
            content_complexity = len(elements) / 10.0  # Normalize
            duration_content_ratio = duration / max(content_complexity * 60, 1)
            doc_features.append(min(duration_content_ratio, 5.0) / 5.0)
            
            # Overly short for content (rushed)
            min_needed = sum(self.minimum_durations.get(elem, 5) for elem in elements.keys())
            rushed = 1.0 if duration < min_needed * 0.7 else 0.0
            doc_features.append(rushed)
            
            # Overly long for content (padded)
            padded = 1.0 if duration > min_needed * 2.0 else 0.0
            doc_features.append(padded)
            
            # Attention span compatibility scores (multiple audiences)
            for audience, threshold in list(self.attention_thresholds.items())[:4]:
                compatible = 1.0 if duration <= threshold else max(0, 1.0 - (duration - threshold) / threshold)
                doc_features.append(compatible)
            
            # Universal attention compatibility (adults engaged)
            universal_compatible = 1.0 if duration <= 180 else max(0, 1.0 - (duration - 180) / 180)
            doc_features.append(universal_compatible)
            
            # Marathon indicator (requires exceptional commitment)
            marathon = 1.0 if duration > 360 else 0.0
            doc_features.append(marathon)
            
            # ============================================================
            # Content Requirements (10 features)
            # ============================================================
            
            # Check if duration supports desired elements
            for element, min_duration in list(self.minimum_durations.items())[:8]:
                has_element = elements.get(element, 0) > 0
                has_duration = duration >= min_duration
                supported = 1.0 if (not has_element) or (has_element and has_duration) else 0.0
                doc_features.append(supported)
            
            # Character depth possible
            char_depth_possible = 1.0 if duration >= 10 else duration / 10.0
            doc_features.append(char_depth_possible)
            
            # Multiple subplots possible
            subplots_possible = min(duration / 60.0, 1.0)  # Need ~60 min for subplots
            doc_features.append(subplots_possible)
            
            # ============================================================
            # Attention Span Matching (10 features)
            # ============================================================
            
            # Cognitive load per unit time
            words = len(narrative.split())
            words_per_minute = words / max(duration, 1)
            cognitive_load = words_per_minute / 150  # Normalize to ~150 wpm
            doc_features.append(min(cognitive_load, 2.0) / 2.0)
            
            # Requires sustained attention
            sustained_attention_needed = 1.0 if duration > 90 else duration / 90.0
            doc_features.append(sustained_attention_needed)
            
            # Requires episodic memory (for long works)
            episodic_memory_load = min(duration / 180.0, 1.0)
            doc_features.append(episodic_memory_load)
            
            # Can be consumed in single sitting
            single_sitting = 1.0 if duration <= 180 else 0.0
            doc_features.append(single_sitting)
            
            # Requires multiple sessions
            multiple_sessions = 1.0 if duration > 300 else 0.0
            doc_features.append(multiple_sessions)
            
            # Break points present (for long works)
            break_points = len(re.findall(r'\n\n\n+', narrative))  # Triple line breaks
            break_point_density = break_points / max(duration / 60, 1)
            doc_features.append(min(break_point_density, 1.0))
            
            # Pacing requirements (longer works need varied pacing)
            pacing_variation_needed = min(duration / 120.0, 1.0)
            doc_features.append(pacing_variation_needed)
            
            # Investment payoff ratio (longer works need more payoff)
            # Measured by climax intensity markers
            climax_markers = len(re.findall(r'\b(climax|revelation|turning point|breakthrough)\b', narrative, re.I))
            payoff_ratio = climax_markers / max(duration / 60.0, 1)
            doc_features.append(min(payoff_ratio, 1.0))
            
            # Rewatch/reread potential (longer = less rewatchable)
            rewatch_potential = max(0, 1.0 - duration / 240.0)
            doc_features.append(rewatch_potential)
            
            # Commitment barrier (longer = higher barrier)
            commitment_barrier = min(duration / 180.0, 1.0)
            doc_features.append(commitment_barrier)
            
            # ============================================================
            # Duration Quality (5 features)
            # ============================================================
            
            # Overall duration appropriateness
            appropriateness = (accessibility + (1.0 - rushed) + (1.0 - padded)) / 3.0
            doc_features.append(appropriateness)
            
            # Form-content fit
            form_fit = 1.0 - abs(duration_content_ratio - 1.0) / 2.0
            form_fit = max(0, form_fit)
            doc_features.append(form_fit)
            
            # Audience accessibility (weighted average across audiences)
            audience_scores = []
            for audience, threshold in self.attention_thresholds.items():
                score = 1.0 if duration <= threshold else max(0, 1.0 - (duration - threshold) / threshold)
                audience_scores.append(score)
            avg_audience_accessibility = np.mean(audience_scores)
            doc_features.append(avg_audience_accessibility)
            
            # Duration efficiency (content per minute)
            efficiency = content_complexity / max(duration / 60.0, 1)
            doc_features.append(min(efficiency, 1.0))
            
            # Overall duration quality score
            duration_quality = (appropriateness + form_fit + avg_audience_accessibility + efficiency) / 4.0
            doc_features.append(duration_quality)
            
            features_list.append(doc_features)
        
        return np.array(features_list)
    
    def _estimate_duration(self, text: str, metadata: Optional[Dict] = None, idx: int = 0) -> float:
        """Estimate duration in minutes."""
        if metadata and 'durations' in metadata and idx < len(metadata['durations']):
            return metadata['durations'][idx]
        
        words = len(text.split())
        
        # Domain-specific reading/viewing rates
        if self.domain in ['novel', 'short_story']:
            return words / 250  # Reading speed
        elif self.domain in ['film', 'tv']:
            return words / 100  # Screenplay approximation
        elif self.domain in ['music']:
            return min(words / 50, 20)
        else:
            return words / 150  # General
    
    def _count_narrative_elements(self, text: str) -> Dict[str, int]:
        """Count presence of key narrative elements."""
        elements = {}
        
        # Character indicators
        character_names = len(set(re.findall(r'\b[A-Z][a-z]+\b', text)))
        elements['character_introduction'] = min(character_names, 10)
        
        # Character development markers
        dev_markers = len(re.findall(r'\b(grew|learned|realized|understood|changed|became)\b', text, re.I))
        elements['character_development'] = dev_markers
        
        # Transformation markers
        trans_markers = len(re.findall(r'\b(transform|evolution|journey|arc|growth)\b', text, re.I))
        elements['character_transformation'] = trans_markers
        
        # Relationship markers
        rel_markers = len(re.findall(r'\b(relationship|friend|love|family|partner|bond)\b', text, re.I))
        elements['relationship_development'] = rel_markers
        
        # Subplot indicators
        subplot_markers = len(re.findall(r'\b(meanwhile|elsewhere|subplot|secondary)\b', text, re.I))
        elements['subplot'] = subplot_markers
        
        # World-building markers
        world_markers = len(re.findall(r'\b(world|setting|place|location|environment)\b', text, re.I))
        elements['world_building'] = world_markers
        
        # Theme exploration
        theme_markers = len(re.findall(r'\b(theme|meaning|symbol|represent|signif)\b', text, re.I))
        elements['theme_exploration'] = theme_markers
        
        # Emotional arc
        emotion_markers = len(re.findall(r'\b(feel|emotion|heart|soul|passion)\b', text, re.I))
        elements['emotional_arc'] = emotion_markers
        
        # Mystery setup
        mystery_markers = len(re.findall(r'\b(mystery|secret|hidden|reveal|discover)\b', text, re.I))
        elements['mystery_setup_payoff'] = mystery_markers
        
        # Plot complexity
        plot_markers = len(re.findall(r'\b(plot|event|happen|occur|action|conflict)\b', text, re.I))
        elements['complex_plot'] = plot_markers
        
        return elements
    
    def get_feature_names(self) -> List[str]:
        """Return feature names."""
        return [
            # Form Classification (10)
            'form_micro', 'form_mini', 'form_short', 'form_medium',
            'form_long', 'form_extended', 'form_marathon',
            'has_chapters', 'has_episodes', 'has_acts',
            
            # Accessibility Effects (10)
            'accessibility_score', 'duration_content_ratio', 'rushed_indicator', 'padded_indicator',
            'compatible_children', 'compatible_teens', 'compatible_adults_casual', 'compatible_adults_engaged',
            'universal_compatibility', 'marathon_indicator',
            
            # Content Requirements (10)
            'supports_char_intro', 'supports_char_dev', 'supports_transformation',
            'supports_relationships', 'supports_subplot', 'supports_worldbuilding',
            'supports_theme', 'supports_emotional_arc',
            'char_depth_possible', 'subplots_possible',
            
            # Attention Span Matching (10)
            'cognitive_load_per_minute', 'sustained_attention_needed',
            'episodic_memory_load', 'single_sitting_possible',
            'requires_multiple_sessions', 'break_point_density',
            'pacing_variation_needed', 'investment_payoff_ratio',
            'rewatch_potential', 'commitment_barrier',
            
            # Duration Quality (5)
            'duration_appropriateness', 'form_content_fit',
            'audience_accessibility_avg', 'duration_efficiency',
            'overall_duration_quality'
        ]

