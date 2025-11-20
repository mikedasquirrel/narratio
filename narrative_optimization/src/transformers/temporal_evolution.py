"""
Temporal Evolution Transformer

Captures how narratives unfold and evolve over time - the dynamic dimension
of narrative force. Names and stories age, trend, and cycle through cultural zeitgeist.

Research Foundation:
- "Better stories win over time, better ones over longer periods"
- Baby names follow ~30-year vintage cycles
- Crypto: "CyberCoin" (1990s) feels dated vs "Web3Protocol" (2020s)
- Brand names age curves affect perception
- Temporal momentum creates narrative force

Core Insight:
Narrative force isn't static - it's a dynamic process that unfolds temporally.
What works in one era may fail in another. Trend momentum creates opportunities.

Universal across domains:
- Names have life cycles (rising, peak, declining, dormant, revival)
- Era-appropriateness affects success
- Temporal momentum matters (rising vs. falling)
- Vintage revival patterns are predictable
"""

from typing import List, Dict, Any
import numpy as np
import re
from collections import Counter
from datetime import datetime
from .base import NarrativeTransformer


class TemporalEvolutionTransformer(NarrativeTransformer):
    """
    Analyzes temporal evolution and aging of narratives.
    
    Tests hypothesis that narrative force unfolds over time - what's fresh
    becomes stale, what's dated becomes vintage, trends create momentum.
    
    Features extracted (30):
    - Era markers (vintage, modern, futuristic language)
    - Temporal momentum indicators (rising, trending, fading)
    - Age-appropriateness (matches current zeitgeist)
    - Novelty vs. staleness markers
    - Recency bias indicators
    - Historical reference density
    - Future-past balance
    - Trend velocity language
    - Revival potential markers
    - Saturation phase indicators
    
    Parameters
    ----------
    current_year : int
        Reference year for temporal calculations (default: current year)
    """
    
    def __init__(self, current_year: int = None):
        super().__init__(
            narrative_id="temporal_evolution",
            description="Temporal evolution: how narratives unfold and age over time"
        )
        
        self.current_year = current_year or datetime.now().year
        
        # Era markers
        self.vintage_markers = [
            'classic', 'vintage', 'retro', 'old-school', 'traditional', 'heritage',
            'nostalgic', 'throwback', 'timeless', 'historic', 'legendary', 'iconic'
        ]
        
        self.modern_markers = [
            'modern', 'contemporary', 'current', 'today', 'now', 'present',
            'recent', 'latest', 'new', 'fresh', 'innovative', 'cutting-edge'
        ]
        
        self.futuristic_markers = [
            'future', 'next-gen', 'tomorrow', 'upcoming', 'emerging', 'revolutionary',
            'breakthrough', 'pioneering', 'futuristic', 'advanced', 'next', 'quantum',
            'cyber', 'digital', 'ai', 'tech', 'web3', 'meta', 'ultra', 'neo'
        ]
        
        # Trend momentum indicators
        self.rising_trend_markers = [
            'rising', 'growing', 'trending', 'gaining', 'ascending', 'climbing',
            'increasing', 'surging', 'booming', 'emerging', 'taking off', 'momentum'
        ]
        
        self.falling_trend_markers = [
            'falling', 'declining', 'fading', 'dying', 'descending', 'dropping',
            'decreasing', 'waning', 'collapsing', 'disappearing', 'losing', 'stale'
        ]
        
        # Novelty vs. staleness
        self.novelty_markers = [
            'new', 'novel', 'original', 'unprecedented', 'unique', 'fresh',
            'innovative', 'groundbreaking', 'first-ever', 'never-before', 'debut'
        ]
        
        self.staleness_markers = [
            'old', 'stale', 'dated', 'outdated', 'obsolete', 'ancient', 'archaic',
            'tired', 'worn', 'past', 'former', 'previous', 'ex', 'defunct'
        ]
        
        # Recency indicators
        self.recency_markers = [
            'recently', 'lately', 'just', 'this week', 'this month', 'this year',
            'today', 'yesterday', 'last week', 'last month', 'now', 'currently'
        ]
        
        # Historical depth
        self.historical_markers = [
            'history', 'historically', 'tradition', 'legacy', 'heritage', 'roots',
            'origins', 'founded', 'established', 'since', 'years', 'decades', 'centuries'
        ]
        
        # Temporal references (explicit dates/times)
        self.year_pattern = r'\b(19|20)\d{2}\b'
        self.decade_pattern = r'\b\d{2}s\b'
        self.month_pattern = r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b'
        
        # Zeitgeist tech morphemes by era
        self.tech_era_markers = {
            '1990s': ['cyber', 'digital', 'dot-com', 'e-', 'i-', 'virtual', 'web'],
            '2000s': ['social', 'mobile', 'app', 'cloud', 'smart', 'i-'],
            '2010s': ['sharing', 'platform', 'on-demand', 'uber', 'disrupt'],
            '2020s': ['ai', 'ml', 'web3', 'crypto', 'blockchain', 'metaverse', 'nft', 'defi']
        }
    
    def fit(self, X, y=None):
        """
        Learn temporal patterns from corpus.
        
        Parameters
        ----------
        X : list of str
            Text documents
        y : ignored
        
        Returns
        -------
        self
        """
        # Analyze corpus temporal characteristics
        era_distribution = {'vintage': 0, 'modern': 0, 'futuristic': 0}
        trend_distribution = {'rising': 0, 'falling': 0}
        novelty_distribution = {'novel': 0, 'stale': 0}
        
        year_mentions = []
        
        for text in X:
            text_lower = text.lower()
            
            # Era markers
            era_distribution['vintage'] += sum(1 for m in self.vintage_markers if m in text_lower)
            era_distribution['modern'] += sum(1 for m in self.modern_markers if m in text_lower)
            era_distribution['futuristic'] += sum(1 for m in self.futuristic_markers if m in text_lower)
            
            # Trend markers
            trend_distribution['rising'] += sum(1 for m in self.rising_trend_markers if m in text_lower)
            trend_distribution['falling'] += sum(1 for m in self.falling_trend_markers if m in text_lower)
            
            # Novelty
            novelty_distribution['novel'] += sum(1 for m in self.novelty_markers if m in text_lower)
            novelty_distribution['stale'] += sum(1 for m in self.staleness_markers if m in text_lower)
            
            # Year mentions
            years = re.findall(self.year_pattern, text)
            year_mentions.extend([int(y) for y in years if y.isdigit()])
        
        # Metadata
        self.metadata['era_distribution'] = era_distribution
        self.metadata['dominant_era'] = max(era_distribution.items(), key=lambda x: x[1])[0]
        self.metadata['trend_distribution'] = trend_distribution
        self.metadata['novelty_distribution'] = novelty_distribution
        
        if year_mentions:
            self.metadata['year_range'] = (min(year_mentions), max(year_mentions))
            self.metadata['temporal_breadth'] = max(year_mentions) - min(year_mentions)
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Transform documents to temporal evolution features.
        
        Parameters
        ----------
        X : list of str
            Documents to transform
        
        Returns
        -------
        features : array, shape (n_samples, 30)
            Temporal evolution feature matrix
        """
        self._validate_fitted()
        
        features_list = []
        
        for text in X:
            doc_features = self._extract_temporal_features(text)
            features_list.append(doc_features)
        
        return np.array(features_list)
    
    def _extract_temporal_features(self, text: str) -> np.ndarray:
        """Extract all 30 temporal evolution features."""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        word_count = max(1, len(words))
        
        features = []
        
        # === ERA POSITIONING (10) ===
        
        # 1-3. Era marker densities
        vintage_count = sum(1 for m in self.vintage_markers if m in text_lower)
        modern_count = sum(1 for m in self.modern_markers if m in text_lower)
        futuristic_count = sum(1 for m in self.futuristic_markers if m in text_lower)
        
        features.append(vintage_count / word_count * 100)
        features.append(modern_count / word_count * 100)
        features.append(futuristic_count / word_count * 100)
        
        # 4. Era orientation (vintage=0, modern=0.5, futuristic=1)
        total_era = vintage_count + modern_count + futuristic_count
        if total_era > 0:
            era_score = (modern_count * 0.5 + futuristic_count * 1.0) / total_era
        else:
            era_score = 0.5  # Neutral
        features.append(era_score)
        
        # 5. Era clarity (how clearly positioned)
        era_clarity = max(vintage_count, modern_count, futuristic_count) / max(1, total_era)
        features.append(era_clarity)
        
        # 6-9. Tech era morpheme matches (1990s, 2000s, 2010s, 2020s)
        for era, morphemes in self.tech_era_markers.items():
            era_match = sum(1 for m in morphemes if m in text_lower)
            features.append(era_match / word_count * 100)
        
        # 10. Dominant tech era
        era_scores = []
        for morphemes in self.tech_era_markers.values():
            era_scores.append(sum(1 for m in morphemes if m in text_lower))
        dominant_era_idx = np.argmax(era_scores) if max(era_scores) > 0 else 1  # Default modern
        features.append(dominant_era_idx / 3.0)  # Normalize 0-1
        
        # === TREND MOMENTUM (10) ===
        
        # 11. Rising trend markers
        rising_count = sum(1 for m in self.rising_trend_markers if m in text_lower)
        features.append(rising_count / word_count * 100)
        
        # 12. Falling trend markers
        falling_count = sum(1 for m in self.falling_trend_markers if m in text_lower)
        features.append(falling_count / word_count * 100)
        
        # 13. Trend direction (rising=1, falling=0, neutral=0.5)
        total_trend = rising_count + falling_count
        if total_trend > 0:
            trend_direction = rising_count / total_trend
        else:
            trend_direction = 0.5
        features.append(trend_direction)
        
        # 14. Trend intensity (how strong is momentum language)
        features.append(total_trend / word_count * 100)
        
        # 15. Novelty markers
        novelty_count = sum(1 for m in self.novelty_markers if m in text_lower)
        features.append(novelty_count / word_count * 100)
        
        # 16. Staleness markers
        stale_count = sum(1 for m in self.staleness_markers if m in text_lower)
        features.append(stale_count / word_count * 100)
        
        # 17. Novelty-staleness balance
        total_ns = novelty_count + stale_count
        novelty_score = novelty_count / total_ns if total_ns > 0 else 0.5
        features.append(novelty_score)
        
        # 18. Recency indicator density
        recency_count = sum(text_lower.count(m) for m in self.recency_markers)
        features.append(recency_count / word_count * 100)
        
        # 19. Historical depth indicator
        historical_count = sum(1 for m in self.historical_markers if m in text_lower)
        features.append(historical_count / word_count * 100)
        
        # 20. Recency-history balance (present vs past focus)
        total_temporal = recency_count + historical_count
        recency_ratio = recency_count / total_temporal if total_temporal > 0 else 0.5
        features.append(recency_ratio)
        
        # === TEMPORAL SPECIFICITY (10) ===
        
        # 21. Year mentions count
        years = re.findall(self.year_pattern, text)
        features.append(len(years))
        
        # 22. Decade references
        decades = re.findall(self.decade_pattern, text_lower)
        features.append(len(decades))
        
        # 23. Month mentions
        months = re.findall(self.month_pattern, text_lower)
        features.append(len(months))
        
        # 24. Specific date density (years + decades + months)
        date_references = len(years) + len(decades) + len(months)
        features.append(date_references / word_count * 100)
        
        # 25. Temporal range (if years mentioned)
        if years:
            year_values = [int(y) for y in years if y.isdigit()]
            if year_values:
                temporal_range = max(year_values) - min(year_values)
                features.append(temporal_range)
            else:
                features.append(0)
        else:
            features.append(0)
        
        # 26. Years since most recent mention
        if years:
            year_values = [int(y) for y in years if y.isdigit()]
            if year_values:
                most_recent = max(year_values)
                years_since = self.current_year - most_recent
                features.append(years_since)
            else:
                features.append(0)
        else:
            features.append(0)
        
        # 27. Age appropriateness (distance from current zeitgeist)
        # Approximated by futuristic + modern vs vintage ratio
        current_orientation = (modern_count + futuristic_count) / max(1, vintage_count + modern_count + futuristic_count)
        features.append(current_orientation)
        
        # 28. Temporal breadth (past + present + future markers)
        past_markers = ['was', 'were', 'had', 'ago', 'before', 'previously', 'historically']
        present_markers = ['is', 'are', 'am', 'now', 'currently', 'today']
        future_markers = ['will', 'shall', 'going to', 'future', 'next', 'upcoming']
        
        past_count = sum(1 for m in past_markers if m in text_lower)
        present_count = sum(1 for m in present_markers if m in text_lower)
        future_count = sum(1 for m in future_markers if m in text_lower)
        
        time_periods_covered = sum([past_count > 0, present_count > 0, future_count > 0])
        features.append(time_periods_covered / 3.0)
        
        # 29. Temporal momentum (future - past focus)
        temporal_momentum = (future_count - past_count) / max(1, past_count + future_count)
        features.append(temporal_momentum)
        
        # 30. Overall temporal dynamism (how much temporal language)
        total_temporal_markers = (
            vintage_count + modern_count + futuristic_count +
            rising_count + falling_count + novelty_count + stale_count +
            recency_count + historical_count
        )
        features.append(total_temporal_markers / word_count * 100)
        
        return np.array(features)
    
    def get_feature_names(self) -> List[str]:
        """Return names of all 30 features."""
        return [
            # Era positioning (10)
            'vintage_density', 'modern_density', 'futuristic_density', 'era_orientation',
            'era_clarity', 'tech_1990s', 'tech_2000s', 'tech_2010s', 'tech_2020s',
            'dominant_tech_era',
            
            # Trend momentum (10)
            'rising_trend', 'falling_trend', 'trend_direction', 'trend_intensity',
            'novelty_markers', 'staleness_markers', 'novelty_staleness_balance',
            'recency_indicators', 'historical_depth', 'recency_history_balance',
            
            # Temporal specificity (10)
            'year_mentions', 'decade_references', 'month_mentions', 'date_density',
            'temporal_range', 'years_since_recent', 'age_appropriateness',
            'temporal_breadth', 'temporal_momentum', 'overall_temporal_dynamism'
        ]
    
    def interpret_features(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Interpret temporal evolution features in plain English.
        
        Parameters
        ----------
        features : array, shape (30,)
            Feature vector for one document
        
        Returns
        -------
        interpretation : dict
            Plain English interpretation
        """
        names = self.get_feature_names()
        
        interpretation = {
            'summary': self._generate_summary(features),
            'features': {},
            'insights': []
        }
        
        # Era positioning
        era_orientation = features[3]
        if era_orientation > 0.7:
            interpretation['insights'].append("Futuristic orientation - positioned for tomorrow")
        elif era_orientation < 0.3:
            interpretation['insights'].append("Vintage orientation - drawing on tradition")
        else:
            interpretation['insights'].append("Modern/contemporary positioning")
        
        # Trend momentum
        trend_direction = features[12]
        trend_intensity = features[13]
        if trend_intensity > 2.0:
            if trend_direction > 0.6:
                interpretation['insights'].append("RISING trend momentum - gaining traction")
            elif trend_direction < 0.4:
                interpretation['insights'].append("FALLING trend momentum - losing relevance")
        
        # Novelty
        novelty_balance = features[16]
        if novelty_balance > 0.7:
            interpretation['insights'].append("High novelty emphasis - positioned as new/fresh")
        elif novelty_balance < 0.3:
            interpretation['insights'].append("Established/mature framing - not emphasizing newness")
        
        # Temporal dynamism
        dynamism = features[29]
        if dynamism > 5.0:
            interpretation['insights'].append("Highly temporal narrative - time is central theme")
        elif dynamism < 1.0:
            interpretation['insights'].append("Minimal temporal markers - timeless framing")
        
        return interpretation
    
    def _generate_summary(self, features: np.ndarray) -> str:
        """Generate plain English summary of temporal evolution."""
        era_orientation = features[3]
        trend_direction = features[12]
        novelty_balance = features[16]
        temporal_momentum = features[28]
        
        summary_parts = []
        
        # Era
        if era_orientation > 0.7:
            summary_parts.append("Futuristic narrative")
        elif era_orientation < 0.3:
            summary_parts.append("Vintage/classic narrative")
        else:
            summary_parts.append("Contemporary narrative")
        
        # Trend
        if trend_direction > 0.6:
            summary_parts.append("with rising momentum")
        elif trend_direction < 0.4:
            summary_parts.append("with fading momentum")
        
        # Novelty
        if novelty_balance > 0.7:
            summary_parts.append("emphasizing novelty")
        elif novelty_balance < 0.3:
            summary_parts.append("emphasizing tradition")
        
        # Temporal focus
        if temporal_momentum > 0.3:
            summary_parts.append("forward-looking")
        elif temporal_momentum < -0.3:
            summary_parts.append("backward-looking")
        
        return ", ".join(summary_parts) + "."

