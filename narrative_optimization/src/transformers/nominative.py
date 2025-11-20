"""
Nominative Analysis Transformer

Analyzes how things are named, labeled, and categorized in narratives.
Tests whether naming choices reveal deeper narrative structure.
"""

from typing import List, Dict, Any, Set
import numpy as np
import re
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from .base import NarrativeTransformer
from .utils.input_validation import ensure_string_list


class NominativeAnalysisTransformer(NarrativeTransformer):
    """
    Analyzes naming and categorization patterns in narratives.
    
    Tests the hypothesis that how entities are named and categorized reveals
    narrative structure and identity construction.
    
    Features extracted (51 total):
    - Semantic fields (10): motion, cognition, emotion, perception, communication, etc.
    - Label choices and patterns (what names are used)
    - Category memberships (how things are grouped)
    - Naming consistency (same entity, same name?)
    - Identity markers in language
    - **NEW: Sensory language (12)**: tactile, olfactory, gustatory, auditory, temperature, crossmodal
    - **NEW: Power semantics (15)**: power, speed, darkness, prestige, innovation, tradition, aggression
    
    Parameters
    ----------
    n_semantic_fields : int
        Number of semantic fields to track
    track_proper_nouns : bool
        Whether to extract proper noun patterns
    track_categories : bool
        Whether to track categorical language
    """
    
    def __init__(
        self,
        n_semantic_fields: int = 10,
        track_proper_nouns: bool = True,
        track_categories: bool = True
    ):
        super().__init__(
            narrative_id="nominative_analysis",
            description="Naming analysis: how categorization choices reveal narrative structure"
        )
        
        self.n_semantic_fields = n_semantic_fields
        self.track_proper_nouns = track_proper_nouns
        self.track_categories = track_categories
        
        # Semantic field dictionaries (simplified for demonstration)
        self.semantic_fields = {
            'motion': ['go', 'come', 'move', 'walk', 'run', 'travel', 'journey', 'arrive', 'leave', 'return'],
            'cognition': ['think', 'know', 'believe', 'understand', 'realize', 'remember', 'forget', 'learn', 'consider'],
            'emotion': ['feel', 'love', 'hate', 'fear', 'hope', 'worry', 'enjoy', 'suffer', 'care', 'trust'],
            'perception': ['see', 'look', 'watch', 'hear', 'listen', 'touch', 'taste', 'smell', 'notice', 'observe'],
            'communication': ['say', 'tell', 'speak', 'talk', 'ask', 'answer', 'explain', 'describe', 'discuss', 'argue'],
            'creation': ['make', 'create', 'build', 'design', 'produce', 'generate', 'construct', 'form', 'develop'],
            'change': ['become', 'change', 'transform', 'grow', 'develop', 'evolve', 'shift', 'turn', 'convert'],
            'possession': ['have', 'own', 'possess', 'get', 'obtain', 'acquire', 'lose', 'give', 'take', 'receive'],
            'existence': ['be', 'exist', 'live', 'die', 'survive', 'remain', 'stay', 'continue', 'end', 'begin'],
            'social': ['meet', 'join', 'help', 'work', 'play', 'fight', 'cooperate', 'compete', 'share', 'connect']
        }
        
        # Category markers
        self.category_markers = {
            'kind': [r'\bkind of\b', r'\btype of\b', r'\bsort of\b'],
            'example': [r'\bsuch as\b', r'\blike\b', r'\bincluding\b', r'\bfor example\b', r'\bfor instance\b'],
            'class': [r'\bcategory\b', r'\bclass\b', r'\bgroup\b', r'\bset\b'],
            'identity': [r'\bi am\b', r'\bi\'m\b', r'\bwe are\b', r'\bthey are\b'],
            'comparison': [r'\blike\b', r'\bas\b', r'\bsimilar to\b', r'\bdifferent from\b', r'\bcompared to\b']
        }
        
        self.vectorizer_ = None
        self.semantic_field_vocab_ = {}
    
    def fit(self, X, y=None):
        """
        Learn naming patterns from corpus.
        
        Parameters
        ----------
        X : list of str
            Text documents
        y : ignored
        
        Returns
        -------
        self
        """
        # Build vocabulary for each semantic field
        for field_name, field_words in self.semantic_fields.items():
            self.semantic_field_vocab_[field_name] = set(field_words)
        
        # Analyze corpus-level naming patterns
        all_proper_nouns = []
        all_category_usage = []
        
        # Ensure X is list of strings (not bytes)
        X = ensure_string_list(X)
        
        for text in X:
            # Extract proper nouns (capitalized words not at sentence start)
            if self.track_proper_nouns:
                proper_nouns = self._extract_proper_nouns(text)
                all_proper_nouns.extend(proper_nouns)
            
            # Track category usage
            if self.track_categories:
                category_count = self._count_category_markers(text)
                all_category_usage.append(category_count)
        
        # Metadata
        if self.track_proper_nouns and all_proper_nouns:
            self.metadata['top_proper_nouns'] = Counter(all_proper_nouns).most_common(20)
            self.metadata['proper_noun_diversity'] = len(set(all_proper_nouns)) / (len(all_proper_nouns) + 1)
        
        if self.track_categories:
            self.metadata['avg_category_usage'] = np.mean(all_category_usage) if all_category_usage else 0
        
        self.metadata['semantic_fields'] = list(self.semantic_fields.keys())
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Transform documents to nominative features.
        
        Parameters
        ----------
        X : list of str
            Documents to transform
        
        Returns
        -------
        features : array
            Nominative feature matrix
        """
        self._validate_fitted()
        
        # Ensure X is list of strings (not bytes or numpy arrays)
        X = ensure_string_list(X)
        
        features_list = []
        
        for text in X:
            doc_features = self._extract_nominative_features(text)
            features_list.append(doc_features)
        
        return np.array(features_list)
    
    def _extract_nominative_features(self, text: str) -> List[float]:
        """Extract nominative features from a single document."""
        features = []
        text_lower = text.lower()
        words = text_lower.split()
        n_words = len(words) + 1
        
        # 1. Semantic Field Distribution
        field_counts = {}
        for field_name, field_vocab in self.semantic_field_vocab_.items():
            count = sum(1 for word in words if word in field_vocab)
            field_counts[field_name] = count
            features.append(count / n_words)  # Density for each field
        
        # Dominant semantic field
        if field_counts:
            max_field_count = max(field_counts.values())
            features.append(max_field_count / n_words)
        else:
            features.append(0.0)
        
        # Semantic diversity (entropy across fields)
        field_values = np.array(list(field_counts.values())) + 1
        field_dist = field_values / field_values.sum()
        semantic_entropy = -np.sum(field_dist * np.log(field_dist + 1e-10))
        features.append(semantic_entropy)
        
        # 2. Proper Noun Patterns
        if self.track_proper_nouns:
            proper_nouns = self._extract_proper_nouns(text)
            proper_noun_density = len(proper_nouns) / n_words
            features.append(proper_noun_density)
            
            # Proper noun diversity (unique / total)
            if proper_nouns:
                proper_noun_diversity = len(set(proper_nouns)) / len(proper_nouns)
                features.append(proper_noun_diversity)
            else:
                features.append(0.0)
            
            # Proper noun repetition (1 - diversity)
            features.append(1 - features[-1])
        
        # 3. Category Usage
        if self.track_categories:
            category_count = self._count_category_markers(text)
            features.append(category_count / n_words)
            
            # Category marker diversity
            category_types = self._count_category_types(text)
            features.append(len(category_types))
        
        # 4. Identity Markers
        identity_count = sum(len(re.findall(pattern, text_lower)) for pattern in self.category_markers['identity'])
        features.append(identity_count / n_words)
        
        # 5. Comparison Usage
        comparison_count = sum(len(re.findall(pattern, text_lower)) for pattern in self.category_markers['comparison'])
        features.append(comparison_count / n_words)
        
        # 6. Naming Consistency
        # Measure through capitalized word repetition patterns
        if self.track_proper_nouns and proper_nouns:
            noun_counts = Counter(proper_nouns)
            # Consistency = how often repeated nouns appear
            repeated = sum(1 for count in noun_counts.values() if count > 1)
            naming_consistency = repeated / len(set(proper_nouns)) if proper_nouns else 0
            features.append(naming_consistency)
        else:
            features.append(0.0)
        
        # 7. Specificity vs Generality
        # Specific: proper nouns, specific category markers
        # General: abstract semantic fields, broad categories
        specificity_score = proper_noun_density if self.track_proper_nouns else 0
        features.append(specificity_score)
        
        # 8. Categorical Thinking
        # High category usage = more categorical/taxonomic thinking
        categorical_thinking = category_count / n_words if self.track_categories else 0
        features.append(categorical_thinking)
        
        # 9. Semantic Field Balance
        # Are semantic fields balanced or dominated by one?
        if field_counts:
            field_values = list(field_counts.values())
            field_std = np.std(field_values)
            field_mean = np.mean(field_values) + 1e-10
            field_balance = 1 / (1 + (field_std / field_mean))
            features.append(field_balance)
        else:
            features.append(0.0)
        
        # 10. Identity Construction
        # Combination of identity markers + self-categorization
        identity_construction = identity_count + category_count
        features.append(identity_construction / n_words)
        
        # === NEW: SENSORY LANGUAGE (12 features) ===
        
        # Tactile language
        tactile_words = ['rough', 'smooth', 'soft', 'hard', 'silky', 'coarse', 'grainy', 'textured', 'bumpy', 'slick']
        tactile_count = sum(1 for w in tactile_words if w in text_lower)
        features.append(tactile_count / n_words)
        
        # Olfactory language
        olfactory_words = ['smell', 'scent', 'fragrance', 'aroma', 'odor', 'perfume', 'stink', 'whiff', 'bouquet']
        olfactory_count = sum(1 for w in olfactory_words if w in text_lower)
        features.append(olfactory_count / n_words)
        
        # Gustatory language
        gustatory_words = ['taste', 'flavor', 'sweet', 'bitter', 'sour', 'savory', 'salty', 'tangy', 'delicious', 'bland']
        gustatory_count = sum(1 for w in gustatory_words if w in text_lower)
        features.append(gustatory_count / n_words)
        
        # Auditory language (beyond basic perception)
        auditory_words = ['loud', 'quiet', 'musical', 'harsh', 'melodic', 'noisy', 'silent', 'ringing', 'humming', 'cacophony']
        auditory_count = sum(1 for w in auditory_words if w in text_lower)
        features.append(auditory_count / n_words)
        
        # Temperature language
        temperature_words = ['hot', 'cold', 'warm', 'cool', 'freezing', 'burning', 'icy', 'tepid', 'sizzling', 'chilly']
        temperature_count = sum(1 for w in temperature_words if w in text_lower)
        features.append(temperature_count / n_words)
        
        # Texture associations (smooth vs rough)
        smooth_texture = ['smooth', 'silky', 'sleek', 'polished', 'glossy', 'slick']
        rough_texture = ['rough', 'coarse', 'grainy', 'textured', 'rugged', 'gritty']
        smooth_count = sum(1 for w in smooth_texture if w in text_lower)
        rough_count = sum(1 for w in rough_texture if w in text_lower)
        texture_ratio = smooth_count / max(1, smooth_count + rough_count)
        features.append(texture_ratio)
        
        # Overall sensory language density
        total_sensory = tactile_count + olfactory_count + gustatory_count + auditory_count + temperature_count
        features.append(total_sensory / n_words)
        
        # Multisensory richness (how many senses engaged)
        senses_engaged = sum([
            tactile_count > 0,
            olfactory_count > 0,
            gustatory_count > 0,
            auditory_count > 0,
            temperature_count > 0
        ])
        features.append(senses_engaged / 5.0)
        
        # Sensory vividness (density × diversity)
        sensory_vividness = (total_sensory / n_words) * (senses_engaged / 5.0)
        features.append(sensory_vividness)
        
        # Dominant sense (which sensory modality is most present)
        sensory_counts = [tactile_count, olfactory_count, gustatory_count, auditory_count, temperature_count]
        if max(sensory_counts) > 0:
            dominant_sense_idx = sensory_counts.index(max(sensory_counts))
            features.append(dominant_sense_idx / 4.0)  # Normalize 0-1
        else:
            features.append(0.5)  # Neutral
        
        # Sensory-emotional coupling (sensory + emotion words together)
        emotion_words_basic = ['feel', 'love', 'hate', 'fear', 'hope', 'enjoy']
        emotion_count = sum(1 for w in emotion_words_basic if w in text_lower)
        sensory_emotion_coupling = (total_sensory * emotion_count) / (n_words ** 2)
        features.append(sensory_emotion_coupling * 1000)  # Scale up
        
        # Synesthetic language (cross-sensory metaphors)
        synesthetic = ['loud color', 'warm sound', 'sweet smell', 'sharp taste', 'bright sound']
        synesthetic_count = sum(1 for phrase in synesthetic if phrase in text_lower)
        features.append(synesthetic_count)
        
        # === NEW: POWER SEMANTICS (15 features) ===
        
        # Power/strength semantics
        power_words = ['power', 'strong', 'strength', 'dominant', 'force', 'mighty', 'powerful', 'potent']
        power_count = sum(1 for w in power_words if w in text_lower)
        features.append(power_count / n_words)
        
        # Speed/agility semantics
        speed_words = ['fast', 'quick', 'swift', 'rapid', 'agile', 'nimble', 'speedy', 'fleet']
        speed_count = sum(1 for w in speed_words if w in text_lower)
        features.append(speed_count / n_words)
        
        # Darkness/ominous semantics
        darkness_words = ['dark', 'shadow', 'gloomy', 'ominous', 'sinister', 'menacing', 'threatening', 'forbidding']
        darkness_count = sum(1 for w in darkness_words if w in text_lower)
        features.append(darkness_count / n_words)
        
        # Prestige/elite semantics
        prestige_words = ['prestige', 'elite', 'premium', 'exclusive', 'distinguished', 'esteemed', 'renowned']
        prestige_count = sum(1 for w in prestige_words if w in text_lower)
        features.append(prestige_count / n_words)
        
        # Innovation semantics
        innovation_words = ['innovative', 'revolutionary', 'novel', 'groundbreaking', 'pioneering', 'cutting-edge', 'advanced']
        innovation_count = sum(1 for w in innovation_words if w in text_lower)
        features.append(innovation_count / n_words)
        
        # Tradition semantics
        tradition_words = ['traditional', 'classic', 'heritage', 'legacy', 'timeless', 'established', 'venerable']
        tradition_count = sum(1 for w in tradition_words if w in text_lower)
        features.append(tradition_count / n_words)
        
        # Innovation-tradition balance
        total_it = innovation_count + tradition_count
        innovation_ratio = innovation_count / total_it if total_it > 0 else 0.5
        features.append(innovation_ratio)
        
        # Aggression semantics
        aggression_words = ['aggressive', 'fierce', 'intense', 'bold', 'assertive', 'forceful', 'vigorous']
        aggression_count = sum(1 for w in aggression_words if w in text_lower)
        features.append(aggression_count / n_words)
        
        # Gentleness semantics
        gentleness_words = ['gentle', 'soft', 'mild', 'calm', 'peaceful', 'serene', 'tranquil', 'tender']
        gentleness_count = sum(1 for w in gentleness_words if w in text_lower)
        features.append(gentleness_count / n_words)
        
        # Aggression-gentleness balance
        total_ag = aggression_count + gentleness_count
        aggression_ratio = aggression_count / total_ag if total_ag > 0 else 0.5
        features.append(aggression_ratio)
        
        # Reliability semantics
        reliability_words = ['reliable', 'dependable', 'trustworthy', 'consistent', 'steady', 'stable', 'solid']
        reliability_count = sum(1 for w in reliability_words if w in text_lower)
        features.append(reliability_count / n_words)
        
        # Excitement semantics
        excitement_words = ['exciting', 'thrilling', 'exhilarating', 'dynamic', 'energetic', 'vibrant', 'electric']
        excitement_count = sum(1 for w in excitement_words if w in text_lower)
        features.append(excitement_count / n_words)
        
        # Reliability-excitement balance
        total_re = reliability_count + excitement_count
        reliability_ratio = reliability_count / total_re if total_re > 0 else 0.5
        features.append(reliability_ratio)
        
        # Composite power semantics score
        power_composite = (
            power_count + speed_count + aggression_count + excitement_count
        ) / (n_words * 4)
        features.append(power_composite)
        
        # Overall semantic richness (all power dimensions)
        semantic_richness = (
            power_count + speed_count + darkness_count + prestige_count +
            innovation_count + tradition_count + aggression_count + gentleness_count +
            reliability_count + excitement_count
        ) / n_words
        features.append(semantic_richness)
        
        return features
    
    def _ensure_strings(self, X):
        """Convert input to list of strings, handling bytes and numpy arrays"""
        if isinstance(X, str):
            return [X]
        elif isinstance(X, bytes):
            return [X.decode('utf-8', errors='ignore')]
        elif isinstance(X, np.ndarray):
            # Handle numpy arrays - flatten if needed and convert to strings
            if X.ndim == 0:
                # Scalar array
                return [str(X.item())]
            elif X.ndim == 1:
                # 1D array - convert each element
                result = []
                for item in X:
                    if isinstance(item, bytes):
                        result.append(item.decode('utf-8', errors='ignore'))
                    elif isinstance(item, np.str_) or isinstance(item, str):
                        result.append(str(item))
                    else:
                        result.append(str(item))
                return result
            else:
                # Multi-dimensional - flatten first
                return self._ensure_strings(X.flatten())
        elif isinstance(X, (list, tuple)):
            result = []
            for item in X:
                if isinstance(item, bytes):
                    result.append(item.decode('utf-8', errors='ignore'))
                elif isinstance(item, np.ndarray):
                    # Handle nested arrays
                    if item.ndim == 0:
                        result.append(str(item.item()))
                    else:
                        result.append(str(item))
                else:
                    result.append(str(item))
            return result
        else:
            return [str(X)]
    
    def _extract_proper_nouns(self, text: str) -> List[str]:
        """Extract proper nouns (simplified: capitalized words not at sentence start)."""
        # Ensure text is a string, not bytes
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='ignore')
        text = str(text)
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        proper_nouns = []
        
        for sentence in sentences:
            words = sentence.strip().split()
            # Skip first word (sentence start)
            for word in words[1:]:
                # Check if capitalized and alphabetic
                if word and word[0].isupper() and word.isalpha() and len(word) > 1:
                    proper_nouns.append(word)
        
        return proper_nouns
    
    def _count_category_markers(self, text: str) -> int:
        """Count all category markers in text."""
        text_lower = text.lower()
        total_count = 0
        
        for marker_type, patterns in self.category_markers.items():
            for pattern in patterns:
                total_count += len(re.findall(pattern, text_lower))
        
        return total_count
    
    def _count_category_types(self, text: str) -> Set[str]:
        """Count which types of category markers are used."""
        text_lower = text.lower()
        used_types = set()
        
        for marker_type, patterns in self.category_markers.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    used_types.add(marker_type)
                    break
        
        return used_types
    
    def get_semantic_field_profile(self, text: str) -> Dict[str, float]:
        """
        Get semantic field profile for a document.
        
        Parameters
        ----------
        text : str
            Document to analyze
        
        Returns
        -------
        profile : dict
            Semantic field densities
        """
        self._validate_fitted()
        
        text_lower = text.lower()
        words = text_lower.split()
        n_words = len(words) + 1
        
        profile = {}
        for field_name, field_vocab in self.semantic_field_vocab_.items():
            count = sum(1 for word in words if word in field_vocab)
            profile[field_name] = count / n_words
        
        return profile
    
    def _generate_interpretation(self):
        """Generate human-readable interpretation."""
        fields = self.metadata.get('semantic_fields', [])
        
        interpretation = (
            f"Nominative Analysis: Analyzes how entities are named and categorized. "
            f"Tracking {len(fields)} semantic fields: {', '.join(fields[:5])}... "
        )
        
        if 'proper_noun_diversity' in self.metadata:
            diversity = self.metadata['proper_noun_diversity']
            interpretation += f"Corpus proper noun diversity: {diversity:.3f}. "
        
        if 'avg_category_usage' in self.metadata:
            avg_cat = self.metadata['avg_category_usage']
            interpretation += f"Average category marker usage: {avg_cat:.3f}. "
        
        interpretation += (
            "Features capture semantic field distribution (what domains of meaning), "
            "proper noun patterns (naming choices and consistency), category usage "
            "(taxonomic thinking), identity markers (self-definition), comparison "
            "(relational positioning), naming consistency, specificity vs generality, "
            "and identity construction. This tests whether how we name and categorize "
            "reveals deeper narrative structure and identity."
        )
        
        return interpretation

