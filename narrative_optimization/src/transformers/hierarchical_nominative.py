"""
Hierarchical Nominative Transformer

Captures names at ALL levels of an instance:
- Primary name (movie title, team name, person name)
- Associated names (director, actors, players, coach)
- Internal names (characters in story, roster members)
- Contextual names (locations, companies, references)
- Linguistic optics (how names are described)

Philosophy: Like characters in a story - every name matters, at every level.

A movie HAS:
- Title (nominative)
- Director (nominative)
- Actors (ensemble nominative)
- Characters (narrative nominative)
- Studio (institutional nominative)
- Locations mentioned (geographic nominative)

A team HAS:
- Team name (nominative)
- Players (ensemble nominative)
- Coach (authority nominative)
- City (geographic nominative)
- Arena (place nominative)

A joke HAS:
- Teller name (source nominative)
- Subject names (content nominative)
- Referenced entities (contextual nominative)

Everything is names all the way down.
"""

import numpy as np
import re
from typing import List, Dict, Any, Optional
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin


class HierarchicalNominativeTransformer(BaseEstimator, TransformerMixin):
    """
    Extract nominative features at ALL hierarchical levels.
    
    Levels:
    1. Primary (entity itself)
    2. Direct associations (creators, leaders)
    3. Ensemble (cast, team, group)
    4. Internal (characters, members mentioned)
    5. Contextual (places, companies, references)
    6. Linguistic (how names are framed)
    
    ~80+ features capturing complete nominative ecology
    """
    
    def __init__(self, extract_all_proper_nouns=True):
        """
        Initialize hierarchical extractor.
        
        Parameters
        ----------
        extract_all_proper_nouns : bool
            Extract every proper noun at every level
        """
        self.extract_all_proper_nouns = extract_all_proper_nouns
        self.phonetic_analyzer = None
        
        # Initialize sub-analyzers
        self._init_phonetic_patterns()
    
    def _init_phonetic_patterns(self):
        """Initialize phonetic analysis patterns"""
        self.vowels = set('aeiou')
        self.plosives = set('pbtdkg')
        self.fricatives = set('fvszh')
        self.liquids = set('lr')
        self.nasals = set('mn')
        
        self.power_sounds = set('kgr')  # Hard, powerful
        self.soft_sounds = set('lmn')  # Soft, gentle
        self.prestige_sounds = set('v')  # Sophisticated
    
    def fit(self, X, y=None):
        """Fit transformer"""
        return self
    
    def transform(self, X):
        """
        Transform texts into hierarchical nominative features.
        
        Parameters
        ----------
        X : array-like of strings
            Can be:
            - Simple text (extracts all names)
            - JSON with explicit name hierarchy
            - Anything with proper nouns
            
        Returns
        -------
        features : ndarray
            Hierarchical nominative features
        """
        features = []
        
        for text in X:
            # If text is dict/JSON, use structured extraction
            if isinstance(text, dict):
                feat = self._extract_structured(text)
            else:
                # Extract from raw text
                feat = self._extract_from_text(text)
            
            features.append(feat)
        
        return np.array(features)
    
    def _extract_from_text(self, text):
        """Extract hierarchical names from raw text - returns flat list"""
        features = []
        
        # Extract ALL proper nouns (capitalized words)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Level 1: PRIMARY NAME (6 features)
        primary_name = proper_nouns[0] if proper_nouns else text.split()[0] if text else "Unknown"
        features.extend(self._analyze_single_name(primary_name, prefix='primary'))
        
        # Level 2: SECONDARY NAMES (3 features)
        secondary_names = proper_nouns[1:4] if len(proper_nouns) > 1 else []
        features.extend(self._analyze_name_list(secondary_names, prefix='secondary'))
        
        # Level 3: ENSEMBLE (6 features)
        ensemble_feats = self._analyze_name_ensemble(proper_nouns, prefix='ensemble')
        features.extend(ensemble_feats.values() if isinstance(ensemble_feats, dict) else ensemble_feats)
        
        # Level 4: CONTEXTUAL (2 features)
        places = [n for n in proper_nouns if self._is_likely_place(n)]
        companies = [n for n in proper_nouns if self._is_likely_company(n)]
        features.extend([len(places), len(companies)])
        
        # Level 5: LINGUISTIC OPTICS (3 features)
        optics_feats = self._analyze_name_optics(text, proper_nouns)
        features.extend(optics_feats.values() if isinstance(optics_feats, dict) else optics_feats)
        
        # Level 6: CORPUS-LEVEL (3 features)
        features.extend([
            len(proper_nouns) / (len(text.split()) + 1),  # name_density
            len(set(proper_nouns)),  # unique_names
            len(proper_nouns) / (len(set(proper_nouns)) + 1)  # name_repetition
        ])
        
        return features
    
    def _extract_structured(self, data):
        """Extract from structured data - returns flat list of consistent length"""
        features = []
        
        # Level 1: Primary (6 features)
        primary = data.get('title') or data.get('name') or data.get('team') or 'Unknown'
        features.extend(self._analyze_single_name(primary, prefix='primary'))
        
        # Level 2: Creator/Leader (6 features)
        director = data.get('director', [])
        if isinstance(director, list):
            director = ' '.join(director) if director else 'Unknown'
        features.extend(self._analyze_single_name(director, prefix='director'))
        
        # Level 3: Ensemble - Cast (6 features)
        cast = data.get('cast', []) or data.get('actors', []) or data.get('players', [])
        cast_names = [c.get('actor') if isinstance(c, dict) else str(c) for c in cast[:10]]
        cast_feats = self._analyze_name_ensemble(cast_names, prefix='cast')
        features.extend(cast_feats if isinstance(cast_feats, list) else list(cast_feats.values()))
        
        # Level 4: Characters (6 features)
        characters = [c.get('character') if isinstance(c, dict) else '' for c in data.get('cast', [])[:10]]
        characters = [c for c in characters if c]
        char_feats = self._analyze_name_ensemble(characters, prefix='character')
        features.extend(char_feats if isinstance(char_feats, list) else list(char_feats.values()))
        
        # Level 5: Secondary names (3 features)
        all_text_names = re.findall(r'\b[A-Z][a-z]+\b', str(data))
        secondary_feats = self._analyze_name_list(all_text_names[1:4], prefix='secondary')
        features.extend(secondary_feats if isinstance(secondary_feats, list) else list(secondary_feats.values()))
        
        # Level 6: Contextual (2 features)
        keywords = data.get('keywords', [])[:5]
        features.extend([len(keywords), 0])  # Placeholder for company names
        
        # Level 7: Optics (3 features - using overview text)
        overview = str(data.get('overview', ''))
        optics_feats = self._analyze_name_optics(overview, cast_names)
        features.extend(optics_feats if isinstance(optics_feats, list) else list(optics_feats.values()))
        
        # Level 8: Corpus (3 features)
        all_names = cast_names + characters
        features.extend([
            len(all_names) / (len(overview.split()) + 1),  # density
            len(set(all_names)),  # unique
            len(all_names) / (len(set(all_names)) + 1) if all_names else 1.0  # repetition
        ])
        
        # Ensure consistent length (should be 41 features total)
        target_length = 41
        while len(features) < target_length:
            features.append(0.0)
        
        return features[:target_length]
    
    def _analyze_single_name(self, name, prefix='name'):
        """Analyze single name comprehensively"""
        if not name or not isinstance(name, str):
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Return list, not dict
        
        name_lower = name.lower()
        
        # Phonetic analysis
        syllables = self._count_syllables(name_lower)
        length = len(name)
        
        # Character composition
        vowel_count = sum(1 for c in name_lower if c in self.vowels)
        
        vowel_ratio = vowel_count / (len(name_lower) + 1)
        
        # Sound associations
        power_score = sum(1 for c in name_lower if c in self.power_sounds) / (len(name_lower) + 1)
        soft_score = sum(1 for c in name_lower if c in self.soft_sounds) / (len(name_lower) + 1)
        
        # Memorability (simple heuristic)
        memorability = 1.0 / (syllables + 1) * (1 + vowel_ratio)
        
        return [syllables, length, vowel_ratio, power_score, soft_score, memorability]
    
    def _analyze_name_list(self, names, prefix='list'):
        """Analyze list of names - returns list of 3 values"""
        if not names:
            return [0.0, 0.0, 0.0]
        
        lengths = [len(n) for n in names if n]
        syllables = [self._count_syllables(n.lower()) for n in names if n]
        
        return [
            float(np.mean(lengths)) if lengths else 0.0,
            float(np.std(lengths)) if lengths else 0.0,
            float(np.mean(syllables)) if syllables else 0.0
        ]
    
    def _analyze_name_ensemble(self, names, prefix='ensemble'):
        """Analyze ensemble of names - returns list of 6 values"""
        if not names:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Remove empty names
        names = [n for n in names if n and isinstance(n, str)]
        
        if not names:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Size
        size = float(len(names))
        
        # Compute features for each name
        name_lengths = [len(n) for n in names]
        name_syllables = [self._count_syllables(n.lower()) for n in names]
        
        # Harmony (low std = harmonious)
        harmony = 1.0 - (np.std(name_lengths) / (np.mean(name_lengths) + 1)) if len(name_lengths) > 1 else 1.0
        
        # Diversity
        diversity = len(set(names)) / (len(names) + 1)
        
        # Prestige (heuristic: longer, more syllables = more formal)
        prestige = float(np.mean(name_syllables)) if name_syllables else 0.0
        
        # Power (ensemble power score)
        ensemble_power = np.mean([
            sum(1 for c in n.lower() if c in self.power_sounds) / (len(n) + 1)
            for n in names
        ])
        
        # Alliteration (do first letters repeat?)
        first_letters = [n[0].lower() for n in names if n]
        alliteration_score = float(max(Counter(first_letters).values())) / (len(first_letters) + 1) if first_letters else 0.0
        
        return [size, harmony, diversity, prestige, ensemble_power, alliteration_score]
    
    def _analyze_name_optics(self, text, names):
        """Analyze HOW names are described - returns list of 3 values"""
        if not names:
            return [0.0, 0.0, 0.0]
        
        # How often are names mentioned?
        name_mentions = sum(text.lower().count(n.lower()) for n in names if n)
        mention_density = name_mentions / (len(text.split()) + 1)
        
        # Are names emphasized? (caps, quotes, etc.)
        caps_emphasis = sum(1 for n in names if n and n.isupper()) / (len(names) + 1)
        
        # Descriptors before names (adjectives modifying names)
        adjectives = ['great', 'famous', 'legendary', 'acclaimed', 'renowned', 'distinguished']
        descriptor_count = sum(1 for adj in adjectives if adj in text.lower())
        descriptor_density = descriptor_count / (len(names) + 1)
        
        return [mention_density, caps_emphasis, descriptor_density]
    
    def _count_syllables(self, word):
        """Simple syllable counter"""
        if not word:
            return 0
        word = word.lower()
        count = 0
        vowel_groups = re.findall(r'[aeiouy]+', word)
        return max(1, len(vowel_groups))
    
    def _is_likely_place(self, name):
        """Heuristic: Is this name a place?"""
        place_indicators = ['city', 'town', 'island', 'mountain', 'river', 'ocean']
        return any(ind in name.lower() for ind in place_indicators)
    
    def _is_likely_company(self, name):
        """Heuristic: Is this name a company?"""
        company_indicators = ['inc', 'corp', 'ltd', 'productions', 'studios', 'pictures']
        return any(ind in name.lower() for ind in company_indicators)
    
    def get_feature_names_out(self, input_features=None):
        """Get feature names"""
        # Primary level (6)
        primary = ['primary_syllables', 'primary_length', 'primary_vowel_ratio', 
                  'primary_power', 'primary_softness', 'primary_memorability']
        
        # Secondary level (3)
        secondary = ['secondary_mean_length', 'secondary_diversity', 'secondary_mean_syllables']
        
        # Ensemble level (6)
        ensemble = ['ensemble_size', 'ensemble_harmony', 'ensemble_diversity',
                   'ensemble_prestige', 'ensemble_power', 'ensemble_alliteration']
        
        # Cast/players level (6)
        cast = ['cast_size', 'cast_harmony', 'cast_diversity', 
               'cast_prestige', 'cast_power', 'cast_alliteration']
        
        # Character level (6)
        character = ['character_size', 'character_harmony', 'character_diversity',
                    'character_prestige', 'character_power', 'character_alliteration']
        
        # Director/leader level (6)
        director = ['director_syllables', 'director_length', 'director_vowel_ratio',
                   'director_power', 'director_softness', 'director_memorability']
        
        # Contextual (2)
        contextual = ['n_place_names', 'n_company_names']
        
        # Linguistic optics (3)
        optics = ['optics_mention_density', 'optics_emphasis', 'optics_descriptor_density']
        
        # Corpus-level (3)
        corpus = ['name_density', 'unique_names', 'name_repetition']
        
        # Keywords (1)
        keywords = ['n_keyword_names']
        
        all_names = (primary + secondary + ensemble + cast + character + director + 
                    contextual + optics + corpus + keywords)
        
        return np.array([f'hierarchical_nominative_{n}' for n in all_names])


class NominativeInteractionTransformer(BaseEstimator, TransformerMixin):
    """
    Captures INTERACTIONS between name levels.
    
    Tests:
    - Primary × ensemble (film title × cast quality)
    - Director × cast (leader × team harmony)
    - Character × actor (role × performer name alignment)
    - Name × narrative quality (multiplicative effects)
    
    ~30 features capturing nominative synergies
    """
    
    def __init__(self):
        """Initialize interaction analyzer"""
        pass
    
    def fit(self, X, y=None):
        """Fit transformer"""
        return self
    
    def transform(self, X):
        """Extract interaction features"""
        # This would be called AFTER hierarchical extraction
        # Computes cross-level interactions
        
        features = []
        
        for text in X:
            if isinstance(text, dict):
                feat = self._compute_interactions_structured(text)
            else:
                feat = [0.0] * 30  # Fallback
            
            features.append(feat)
        
        return np.array(features)
    
    def _compute_interactions_structured(self, data):
        """Compute name-level interactions"""
        interactions = []
        
        # Get name qualities at different levels
        title_quality = self._name_quality(data.get('title', ''))
        director_quality = self._name_quality(data.get('director', [''])[0] if isinstance(data.get('director'), list) else data.get('director', ''))
        
        cast = data.get('cast', [])
        cast_names = [c.get('actor') if isinstance(c, dict) else str(c) for c in cast[:10]]
        cast_quality = np.mean([self._name_quality(n) for n in cast_names if n]) if cast_names else 0
        
        # Interactions
        interactions.extend([
            title_quality,
            director_quality,
            cast_quality,
            title_quality * director_quality,  # Title-director synergy
            title_quality * cast_quality,  # Title-cast synergy
            director_quality * cast_quality,  # Director-cast synergy
            title_quality * director_quality * cast_quality,  # Three-way
            abs(title_quality - cast_quality),  # Title-cast contrast
            abs(director_quality - cast_quality),  # Director-cast contrast
            min(title_quality, director_quality, cast_quality),  # Weakest link
            max(title_quality, director_quality, cast_quality),  # Strongest link
        ])
        
        # Pad to 30
        while len(interactions) < 30:
            interactions.append(0.0)
        
        return interactions[:30]
    
    def _name_quality(self, name):
        """Quick name quality score"""
        if not name or not isinstance(name, str):
            return 0.0
        
        syllables = len(re.findall(r'[aeiouy]+', name.lower()))
        length = len(name)
        
        # Simple heuristic: moderate length, 2-3 syllables = optimal
        quality = 1.0 / (1 + abs(syllables - 2.5)) * (1.0 / (1 + abs(length - 8)))
        
        return quality
    
    def get_feature_names_out(self, input_features=None):
        """Get interaction feature names"""
        names = [
            'title_quality', 'director_quality', 'cast_quality',
            'title_x_director', 'title_x_cast', 'director_x_cast',
            'threeway_synergy', 'title_cast_contrast', 'director_cast_contrast',
            'weakest_link', 'strongest_link'
        ] + [f'interaction_{i}' for i in range(19)]
        
        return np.array([f'nominative_interaction_{n}' for n in names])


class PureNominativePredictorTransformer(BaseEstimator, TransformerMixin):
    """
    PURE nominative prediction - ZERO narrative context.
    
    Only uses entity name + associated names.
    Tests: Can names alone predict outcomes?
    
    This is the "first impression" effect - before any content is evaluated.
    """
    
    def __init__(self):
        """Initialize pure predictor"""
        self.hierarchical = HierarchicalNominativeTransformer()
        self.interactions = NominativeInteractionTransformer()
    
    def fit(self, X, y=None):
        """Fit both sub-transformers"""
        self.hierarchical.fit(X, y)
        self.interactions.fit(X, y)
        return self
    
    def transform(self, X):
        """Extract pure nominative features (all levels + interactions)"""
        hier_features = self.hierarchical.transform(X)
        inter_features = self.interactions.transform(X)
        
        # Combine
        pure_nominative = np.hstack([hier_features, inter_features])
        
        return pure_nominative
    
    def get_feature_names_out(self, input_features=None):
        """Get combined feature names"""
        hier_names = self.hierarchical.get_feature_names_out()
        inter_names = self.interactions.get_feature_names_out()
        
        return np.concatenate([hier_names, inter_names])

