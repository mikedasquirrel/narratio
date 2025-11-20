"""
Linguistic Resonance Pattern Analyzer

Detects and quantifies the narrative power of language through
phonetic patterns, etymology, and linguistic harmony in sports matchups.

This transformer identifies when team/player names create resonant
linguistic patterns that amplify narrative momentum.

Author: Narrative Enhancement System  
Date: November 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
import re
from collections import Counter
import unicodedata


class LinguisticResonanceTransformer(BaseEstimator, TransformerMixin):
    """
    Extract linguistic resonance features from sports matchups.
    
    Philosophy:
    - Language patterns create subconscious narrative expectations
    - Alliterative matchups have inherent dramatic appeal
    - Name etymology carries historical power dynamics
    - Phonetic harmony/discord affects perception
    - Commentary patterns can crystallize narratives
    
    Features (35 total):
    - Alliterative matchup strength (5)
    - Phonetic harmony/discord metrics (8)
    - Name etymology power dynamics (6)
    - Syllabic rhythm patterns (6)
    - Linguistic momentum indicators (5)
    - Commentary pattern crystallization (5)
    """
    
    def __init__(
        self,
        include_commentary: bool = True,
        etymology_depth: str = 'basic',
        phonetic_weight: float = 0.7
    ):
        """
        Initialize linguistic resonance analyzer.
        
        Parameters
        ----------
        include_commentary : bool
            Analyze commentary/media language patterns
        etymology_depth : str
            'basic' or 'deep' etymology analysis
        phonetic_weight : float
            Weight for phonetic vs semantic features
        """
        self.include_commentary = include_commentary
        self.etymology_depth = etymology_depth
        self.phonetic_weight = phonetic_weight
        
        # Phonetic patterns
        self.consonant_clusters = {
            'explosive': ['p', 'b', 't', 'd', 'k', 'g'],
            'fricative': ['f', 'v', 's', 'z', 'sh', 'zh'],
            'liquid': ['l', 'r'],
            'nasal': ['m', 'n', 'ng']
        }
        
        # Etymology power mappings
        self.etymology_power = {
            # Powerful/dominant origins
            'king': 1.0, 'royal': 0.9, 'eagle': 0.8, 'lion': 0.9,
            'thunder': 0.8, 'lightning': 0.8, 'storm': 0.7,
            'warrior': 0.8, 'knight': 0.7, 'ranger': 0.7,
            
            # Neutral origins  
            'color': 0.5, 'geographic': 0.5, 'occupation': 0.5,
            
            # Underdog origins
            'small': 0.3, 'young': 0.3, 'new': 0.3
        }
        
        # Syllable rhythm patterns
        self.rhythm_patterns = {
            'iambic': [0, 1],           # da-DUM (e.g., "Kings")
            'trochaic': [1, 0],         # DUM-da (e.g., "Rangers")  
            'dactylic': [1, 0, 0],      # DUM-da-da (e.g., "Canadiens")
            'anapestic': [0, 0, 1],     # da-da-DUM (e.g., "Avalanche")
            'spondaic': [1, 1],         # DUM-DUM (e.g., "Red Sox")
        }
        
    def fit(self, X, y=None):
        """
        Learn linguistic patterns from training data.
        
        Parameters
        ----------
        X : list of dicts or DataFrame
            Sports data with team/player names
        y : ignored
        
        Returns
        -------
        self
        """
        # Build vocabulary of team/player names
        self.vocabulary_ = set()
        
        if isinstance(X, pd.DataFrame):
            for col in ['home_team', 'away_team', 'team_name', 'player_name']:
                if col in X.columns:
                    self.vocabulary_.update(X[col].dropna().unique())
        elif isinstance(X, list):
            for item in X:
                for key in ['home_team', 'away_team', 'team_name', 'player_name']:
                    if key in item and item[key]:
                        self.vocabulary_.add(item[key])
                        
        # Build phonetic mappings
        self._build_phonetic_mappings()
        
        return self
        
    def transform(self, X):
        """
        Extract linguistic resonance features.
        
        Parameters
        ----------
        X : list of dicts or DataFrame
            Sports matchup data
            
        Returns
        -------
        np.ndarray
            Linguistic features (n_samples, 35)
        """
        # Handle single sample
        if isinstance(X, dict):
            X = [X]
            
        features = []
        for item in X:
            feature_vec = []
            
            # Get team/player names
            home_name = item.get('home_team', item.get('team_name', ''))
            away_name = item.get('away_team', item.get('opponent_name', ''))
            
            # Alliterative matchup features (5)
            alliteration_features = self._extract_alliteration(home_name, away_name)
            feature_vec.extend(alliteration_features)
            
            # Phonetic harmony/discord (8)
            phonetic_features = self._extract_phonetic_patterns(home_name, away_name)
            feature_vec.extend(phonetic_features)
            
            # Etymology power dynamics (6)
            etymology_features = self._extract_etymology_power(home_name, away_name)
            feature_vec.extend(etymology_features)
            
            # Syllabic rhythm (6)
            rhythm_features = self._extract_rhythm_patterns(home_name, away_name)
            feature_vec.extend(rhythm_features)
            
            # Linguistic momentum (5)
            momentum_features = self._extract_linguistic_momentum(item)
            feature_vec.extend(momentum_features)
            
            # Commentary crystallization (5)
            if self.include_commentary:
                commentary_features = self._extract_commentary_patterns(item)
                feature_vec.extend(commentary_features)
            else:
                feature_vec.extend([0.0] * 5)
                
            features.append(feature_vec)
            
        return np.array(features, dtype=np.float32)
        
    def _build_phonetic_mappings(self):
        """Build phonetic analysis mappings for known vocabulary."""
        self.phonetic_map_ = {}
        
        for name in self.vocabulary_:
            # Basic phonetic features
            self.phonetic_map_[name] = {
                'syllables': self._count_syllables(name),
                'consonants': self._extract_consonants(name),
                'vowels': self._extract_vowels(name),
                'stress_pattern': self._estimate_stress_pattern(name)
            }
            
    def _extract_alliteration(self, name1: str, name2: str) -> List[float]:
        """
        Extract alliterative matchup strength features.
        
        Returns 5 features measuring alliteration power.
        """
        features = []
        
        # Clean names
        n1 = name1.lower().strip()
        n2 = name2.lower().strip()
        
        # Initial letter match
        if n1 and n2 and n1[0] == n2[0]:
            features.append(1.0)
            
            # Strong alliteration (same first 2 letters)
            if len(n1) > 1 and len(n2) > 1 and n1[:2] == n2[:2]:
                features.append(1.0)
            else:
                features.append(0.5)
        else:
            features.extend([0.0, 0.0])
            
        # Consonant cluster alliteration
        c1 = self._extract_initial_consonant_cluster(n1)
        c2 = self._extract_initial_consonant_cluster(n2)
        
        if c1 and c2 and c1 == c2:
            features.append(1.0)
        else:
            features.append(0.0)
            
        # End-rhyme (not alliteration but resonant)
        if len(n1) > 2 and len(n2) > 2 and n1[-3:] == n2[-3:]:
            features.append(1.0)
        elif len(n1) > 1 and len(n2) > 1 and n1[-2:] == n2[-2:]:
            features.append(0.5)
        else:
            features.append(0.0)
            
        # Alliterative strength score (combined)
        alliteration_score = sum(features[:3]) / 3.0
        if alliteration_score > 0.5:
            # Boost for strong alliteration
            features.append(min(1.0, alliteration_score * 1.5))
        else:
            features.append(alliteration_score)
            
        return features
        
    def _extract_phonetic_patterns(self, name1: str, name2: str) -> List[float]:
        """
        Extract phonetic harmony/discord metrics.
        
        Returns 8 features measuring phonetic relationships.
        """
        features = []
        
        # Get consonant types for each name
        consonants1 = self._classify_consonants(name1)
        consonants2 = self._classify_consonants(name2)
        
        # Consonant type harmony (matching types)
        harmony_score = 0.0
        for ctype in self.consonant_clusters:
            if consonants1.get(ctype, 0) > 0 and consonants2.get(ctype, 0) > 0:
                harmony_score += 0.25
                
        features.append(min(1.0, harmony_score))
        
        # Consonant type discord (opposing types)
        if consonants1.get('explosive', 0) > 0 and consonants2.get('fricative', 0) > 0:
            features.append(1.0)  # Explosive vs smooth conflict
        else:
            features.append(0.0)
            
        # Vowel harmony
        vowels1 = self._extract_vowels(name1)
        vowels2 = self._extract_vowels(name2)
        
        vowel_similarity = self._calculate_vowel_similarity(vowels1, vowels2)
        features.append(vowel_similarity)
        
        # Length harmony/discord
        len_diff = abs(len(name1) - len(name2))
        if len_diff == 0:
            features.append(1.0)  # Perfect length match
        elif len_diff <= 2:
            features.append(0.5)  # Close length
        else:
            features.append(0.0)  # Length discord
            
        # Syllable count harmony
        syl1 = self._count_syllables(name1)
        syl2 = self._count_syllables(name2)
        
        if syl1 == syl2:
            features.append(1.0)
        elif abs(syl1 - syl2) == 1:
            features.append(0.5)
        else:
            features.append(0.0)
            
        # Phonetic complexity differential
        complexity1 = self._calculate_phonetic_complexity(name1)
        complexity2 = self._calculate_phonetic_complexity(name2)
        
        features.append(np.tanh(complexity1 - complexity2))  # -1 to 1
        
        # Sound symbolism alignment (harsh vs soft)
        harsh1 = self._calculate_harshness(name1)
        harsh2 = self._calculate_harshness(name2)
        
        if abs(harsh1 - harsh2) < 0.2:
            features.append(1.0)  # Similar harshness
        else:
            features.append(0.0)  # Contrasting harshness
            
        # Overall phonetic resonance
        resonance = sum(features[:7]) / 7.0
        features.append(resonance)
        
        return features
        
    def _extract_etymology_power(self, name1: str, name2: str) -> List[float]:
        """
        Extract etymology-based power dynamics.
        
        Returns 6 features measuring etymological narrative power.
        """
        features = []
        
        # Get etymology scores
        power1 = self._get_etymology_power(name1)
        power2 = self._get_etymology_power(name2)
        
        # Individual power scores
        features.append(power1)
        features.append(power2)
        
        # Power differential
        features.append(np.tanh((power1 - power2) * 2))  # -1 to 1
        
        # David vs Goliath indicator
        if power1 < 0.4 and power2 > 0.7:
            features.append(1.0)  # Clear underdog story
        elif power2 < 0.4 and power1 > 0.7:
            features.append(-1.0)  # Reverse underdog
        else:
            features.append(0.0)
            
        # Etymology clash (different origins)
        origin1 = self._get_etymology_category(name1)
        origin2 = self._get_etymology_category(name2)
        
        if origin1 != origin2 and origin1 and origin2:
            features.append(1.0)  # Cultural clash
        else:
            features.append(0.0)
            
        # Historical weight (older names have more power)
        if self.etymology_depth == 'deep':
            age1 = self._estimate_name_age(name1)
            age2 = self._estimate_name_age(name2)
            features.append(np.tanh((age1 - age2) / 50))  # Normalize by decades
        else:
            features.append(0.0)
            
        return features
        
    def _extract_rhythm_patterns(self, name1: str, name2: str) -> List[float]:
        """
        Extract syllabic rhythm pattern features.
        
        Returns 6 features measuring rhythmic relationships.
        """
        features = []
        
        # Get stress patterns
        pattern1 = self._estimate_stress_pattern(name1)
        pattern2 = self._estimate_stress_pattern(name2)
        
        # Identify rhythm types
        rhythm1 = self._classify_rhythm(pattern1)
        rhythm2 = self._classify_rhythm(pattern2)
        
        # Rhythm harmony (matching patterns)
        if rhythm1 == rhythm2 and rhythm1 is not None:
            features.append(1.0)
        else:
            features.append(0.0)
            
        # Rhythm contrast (opposing patterns)
        contrasting_pairs = [
            ('iambic', 'trochaic'),
            ('dactylic', 'anapestic')
        ]
        
        if (rhythm1, rhythm2) in contrasting_pairs or (rhythm2, rhythm1) in contrasting_pairs:
            features.append(1.0)  # Dramatic contrast
        else:
            features.append(0.0)
            
        # Syllable train rhythm (combined names)
        combined_syllables = self._count_syllables(name1) + self._count_syllables(name2)
        
        # Check for special numbers
        if combined_syllables in [5, 7]:  # Haiku-like
            features.append(1.0)
        elif combined_syllables in [10, 14]:  # Sonnet-like
            features.append(0.8)
        else:
            features.append(0.0)
            
        # Stress pattern momentum
        if pattern1 and pattern2:
            # Rising action (unstressed to stressed)
            if pattern1[-1] == 0 and pattern2[0] == 1:
                features.append(1.0)
            # Falling action  
            elif pattern1[-1] == 1 and pattern2[0] == 0:
                features.append(-1.0)
            else:
                features.append(0.0)
        else:
            features.append(0.0)
            
        # Rhythmic complexity differential
        complexity1 = len(set(pattern1)) if pattern1 else 1
        complexity2 = len(set(pattern2)) if pattern2 else 1
        
        features.append(np.tanh(complexity1 - complexity2))
        
        # Overall rhythmic flow score
        flow_score = 0.0
        if rhythm1 and rhythm2:
            # Some rhythms flow better together
            if rhythm1 in ['iambic', 'anapestic'] and rhythm2 in ['iambic', 'anapestic']:
                flow_score = 0.8
            elif rhythm1 == rhythm2:
                flow_score = 0.6
            else:
                flow_score = 0.3
                
        features.append(flow_score)
        
        return features
        
    def _extract_linguistic_momentum(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract linguistic momentum indicators.
        
        Returns 5 features measuring narrative linguistic crystallization.
        """
        features = []
        
        # Nickname momentum (when nicknames take over)
        if item.get('nickname_usage_rate', 0) > 0.7:
            features.append(1.0)
        else:
            features.append(item.get('nickname_usage_rate', 0))
            
        # Catchphrase crystallization
        catchphrase = item.get('team_catchphrase', '')
        if catchphrase and item.get('catchphrase_mentions', 0) > 100:
            features.append(1.0)
        else:
            features.append(0.0)
            
        # Name chant potential (based on syllables)
        team_name = item.get('team_name', '')
        chantability = self._calculate_chantability(team_name)
        features.append(chantability)
        
        # Linguistic viral moment indicator
        recent_viral = item.get('recent_viral_phrase', False)
        if recent_viral:
            features.append(1.0)
        else:
            features.append(0.0)
            
        # Commentary convergence (when all commentators use same terms)
        commentary_diversity = item.get('commentary_term_diversity', 1.0)
        convergence = 1.0 - min(1.0, commentary_diversity)
        features.append(convergence)
        
        return features
        
    def _extract_commentary_patterns(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract commentary language pattern features.
        
        Returns 5 features measuring media narrative crystallization.
        """
        features = []
        
        # Get commentary text if available
        commentary = item.get('recent_commentary', '')
        headlines = item.get('recent_headlines', [])
        
        # Narrative keyword saturation
        narrative_keywords = ['destiny', 'fate', 'meant to be', 'story', 
                            'cinderella', 'david', 'goliath', 'dynasty']
        
        keyword_count = sum(1 for keyword in narrative_keywords 
                          if keyword in commentary.lower())
        
        features.append(min(1.0, keyword_count / 5.0))
        
        # Superlative usage (extreme language)
        superlatives = ['greatest', 'worst', 'best', 'most', 'least',
                       'never', 'always', 'historic', 'unprecedented']
        
        superlative_count = sum(1 for word in superlatives
                              if word in commentary.lower())
        
        features.append(min(1.0, superlative_count / 4.0))
        
        # Question mark intensity (building drama)
        question_density = commentary.count('?') / max(1, len(commentary.split()))
        features.append(min(1.0, question_density * 20))
        
        # Consensus building (similar language across sources)
        if len(headlines) > 3:
            # Check for repeated phrases
            all_text = ' '.join(headlines).lower()
            words = all_text.split()
            word_counts = Counter(words)
            
            # High repetition indicates consensus
            max_count = max(word_counts.values()) if word_counts else 0
            consensus_score = min(1.0, max_count / 10.0)
            features.append(consensus_score)
        else:
            features.append(0.0)
            
        # Narrative momentum language
        momentum_words = ['building', 'growing', 'surging', 'rolling',
                         'unstoppable', 'inevitable', 'mounting']
        
        momentum_count = sum(1 for word in momentum_words
                           if word in commentary.lower())
        
        features.append(min(1.0, momentum_count / 3.0))
        
        return features
        
    # Helper methods
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word."""
        word = word.lower()
        vowels = 'aeiouy'
        count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                count += 1
            previous_was_vowel = is_vowel
            
        # Adjust for silent e
        if word.endswith('e'):
            count -= 1
            
        # Ensure at least 1 syllable
        return max(1, count)
        
    def _extract_consonants(self, word: str) -> List[str]:
        """Extract consonants from word."""
        vowels = set('aeiouAEIOU')
        return [c.lower() for c in word if c.isalpha() and c not in vowels]
        
    def _extract_vowels(self, word: str) -> List[str]:
        """Extract vowels from word."""
        vowels = set('aeiouAEIOU')
        return [c.lower() for c in word if c in vowels]
        
    def _extract_initial_consonant_cluster(self, word: str) -> str:
        """Extract initial consonant cluster."""
        word = word.lower()
        vowels = set('aeiouy')
        cluster = ''
        
        for char in word:
            if char not in vowels and char.isalpha():
                cluster += char
            else:
                break
                
        return cluster
        
    def _classify_consonants(self, word: str) -> Dict[str, int]:
        """Classify consonants by type."""
        consonants = self._extract_consonants(word)
        classification = {ctype: 0 for ctype in self.consonant_clusters}
        
        for consonant in consonants:
            for ctype, clist in self.consonant_clusters.items():
                if consonant in clist:
                    classification[ctype] += 1
                    
        return classification
        
    def _calculate_vowel_similarity(self, vowels1: List[str], vowels2: List[str]) -> float:
        """Calculate similarity between vowel patterns."""
        if not vowels1 or not vowels2:
            return 0.0
            
        # Compare vowel sets
        set1 = set(vowels1)
        set2 = set(vowels2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
        
    def _calculate_phonetic_complexity(self, word: str) -> float:
        """Calculate phonetic complexity score."""
        # Factors: consonant clusters, unusual letters, length
        consonants = self._extract_consonants(word)
        
        # Consonant cluster complexity
        clusters = 0
        for i in range(len(consonants) - 1):
            if i + 1 < len(consonants):
                clusters += 1
                
        # Unusual letters
        unusual = set('xzqj')
        unusual_count = sum(1 for c in word.lower() if c in unusual)
        
        complexity = (clusters * 0.3 + unusual_count * 0.5 + len(word) * 0.1)
        return min(1.0, complexity / 5.0)
        
    def _calculate_harshness(self, word: str) -> float:
        """Calculate phonetic harshness score."""
        harsh_sounds = set('kgtdzx')
        soft_sounds = set('lmnwy')
        
        harsh_count = sum(1 for c in word.lower() if c in harsh_sounds)
        soft_count = sum(1 for c in word.lower() if c in soft_sounds)
        
        if harsh_count + soft_count == 0:
            return 0.5
            
        return harsh_count / (harsh_count + soft_count)
        
    def _get_etymology_power(self, name: str) -> float:
        """Get etymology-based power score."""
        name_lower = name.lower()
        
        # Check for known powerful terms
        for term, power in self.etymology_power.items():
            if term in name_lower:
                return power
                
        # Check for specific patterns
        if any(word in name_lower for word in ['new', 'fc', 'united']):
            return 0.3  # Modern/new teams
        elif any(word in name_lower for word in ['original', 'classic']):
            return 0.7  # Historic teams
            
        return 0.5  # Default neutral
        
    def _get_etymology_category(self, name: str) -> Optional[str]:
        """Categorize name etymology."""
        name_lower = name.lower()
        
        # Animal names
        animals = ['bear', 'wolf', 'eagle', 'hawk', 'panther', 'tiger']
        if any(animal in name_lower for animal in animals):
            return 'animal'
            
        # Geographic names
        if any(geo in name_lower for geo in ['north', 'south', 'east', 'west', 'city']):
            return 'geographic'
            
        # Color names
        colors = ['red', 'blue', 'white', 'black', 'gold', 'green']
        if any(color in name_lower for color in colors):
            return 'color'
            
        # Military/warrior names
        if any(war in name_lower for war in ['ranger', 'warrior', 'knight', 'general']):
            return 'military'
            
        return None
        
    def _estimate_name_age(self, name: str) -> int:
        """Estimate historical age of team name in years."""
        # This would ideally use actual founding dates
        # For now, use heuristics
        
        if 'original' in name.lower() or 'classic' in name.lower():
            return 75
        elif any(old in name.lower() for old in ['metropolitan', 'athletic']):
            return 100
        elif 'fc' in name.lower() or 'united' in name.lower():
            return 25
        else:
            return 50  # Default
            
    def _estimate_stress_pattern(self, word: str) -> List[int]:
        """Estimate stress pattern (1=stressed, 0=unstressed)."""
        syllables = self._count_syllables(word)
        
        if syllables == 1:
            return [1]
        elif syllables == 2:
            # Most two-syllable words are trochaic in English
            return [1, 0]
        elif syllables == 3:
            # Common pattern for 3 syllables
            return [1, 0, 0]
        else:
            # Alternate stress for longer words
            pattern = []
            for i in range(syllables):
                pattern.append(1 if i % 2 == 0 else 0)
            return pattern
            
    def _classify_rhythm(self, pattern: List[int]) -> Optional[str]:
        """Classify rhythm pattern."""
        if not pattern:
            return None
            
        pattern_str = ''.join(map(str, pattern))
        
        for rhythm_name, rhythm_pattern in self.rhythm_patterns.items():
            rhythm_str = ''.join(map(str, rhythm_pattern))
            if pattern_str == rhythm_str or pattern_str.startswith(rhythm_str):
                return rhythm_name
                
        return None
        
    def _calculate_chantability(self, name: str) -> float:
        """Calculate how chantable a name is."""
        syllables = self._count_syllables(name)
        
        # 2-3 syllables are most chantable
        if syllables in [2, 3]:
            score = 1.0
        elif syllables == 1:
            score = 0.7  # Can work but harder
        elif syllables == 4:
            score = 0.5
        else:
            score = 0.2
            
        # Boost for strong consonants
        if name and name[0].lower() in 'bdgkpt':
            score = min(1.0, score * 1.2)
            
        # Boost for repeated sounds
        if len(set(name.lower())) < len(name) * 0.6:
            score = min(1.0, score * 1.1)
            
        return score
        
    def get_feature_names(self) -> List[str]:
        """Return feature names for interpretability."""
        names = []
        
        # Alliteration features
        names.extend([
            'alliteration_initial_match',
            'alliteration_strong_match',
            'alliteration_consonant_cluster',
            'alliteration_end_rhyme',
            'alliteration_combined_strength'
        ])
        
        # Phonetic features
        names.extend([
            'phonetic_consonant_harmony',
            'phonetic_consonant_discord',
            'phonetic_vowel_harmony',
            'phonetic_length_harmony',
            'phonetic_syllable_harmony',
            'phonetic_complexity_differential',
            'phonetic_harshness_alignment',
            'phonetic_overall_resonance'
        ])
        
        # Etymology features
        names.extend([
            'etymology_power_home',
            'etymology_power_away',
            'etymology_power_differential',
            'etymology_david_goliath',
            'etymology_category_clash',
            'etymology_historical_weight'
        ])
        
        # Rhythm features
        names.extend([
            'rhythm_pattern_harmony',
            'rhythm_pattern_contrast',
            'rhythm_syllable_train',
            'rhythm_stress_momentum',
            'rhythm_complexity_differential',
            'rhythm_flow_score'
        ])
        
        # Linguistic momentum features
        names.extend([
            'momentum_nickname_usage',
            'momentum_catchphrase_crystallization',
            'momentum_chantability',
            'momentum_viral_indicator',
            'momentum_commentary_convergence'
        ])
        
        # Commentary features
        names.extend([
            'commentary_narrative_keywords',
            'commentary_superlative_usage',
            'commentary_question_intensity',
            'commentary_consensus_building',
            'commentary_momentum_language'
        ])
        
        return names
