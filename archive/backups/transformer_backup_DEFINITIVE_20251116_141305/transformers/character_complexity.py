"""
Character Complexity Transformer

Uses advanced NLP to analyze character depth and development.
Dependency parsing, entity analysis, and semantic embeddings.

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
import re
from typing import List, Dict, Any, Optional, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter, defaultdict
from .utils.input_validation import ensure_string_list

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class CharacterComplexityTransformer(BaseEstimator, TransformerMixin):
    """
    Analyzes character depth, development, and complexity using NLP.
    
    Uses:
    - Named Entity Recognition for character tracking
    - Dependency parsing for character actions
    - Semantic embeddings for trait analysis
    - Sentiment analysis for character valence
    - Co-reference resolution patterns
    
    Features (15 total):
    1. Round vs flat character detection
    2. Dynamic vs static character detection
    3. Character arc type (positive/negative/flat)
    4. Transformation magnitude
    5. Trait diversity (how many different traits)
    6. Moral complexity (grey vs black-and-white)
    7. Motivation clarity
    8. Action-to-description ratio
    9. Dialogue presence
    10. Relationship network density
    11. Agency score (active vs passive)
    12. Consistency score
    13. Depth indicators
    14. Character focus (prominence in narrative)
    15. Complexity index (composite)
    """
    
    def __init__(self, use_spacy: bool = True, use_embeddings: bool = True):
        """Initialize character analyzer"""
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        self.use_embeddings = use_embeddings and SENTENCE_TRANSFORMERS_AVAILABLE
        
        # Load models
        if self.use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                try:
                    self.nlp = spacy.load("en_core_web_md")
                except:
                    self.use_spacy = False
        
        if self.use_embeddings:
            try:
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            except:
                self.use_embeddings = False
    
    def fit(self, X, y=None):
        """Fit transformer"""
        X = ensure_string_list(X)
        return self
    
    def transform(self, X):
        """
        Transform texts to character complexity features.
        
        Parameters
        ----------
        X : array-like of strings
            Narrative texts
            
        Returns
        -------
        features : ndarray of shape (n_samples, 15)
            Character complexity features
        """
        X = ensure_string_list(X)
        features = []
        
        for text in X:
            feat = self._extract_character_features(text)
            features.append(feat)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_character_features(self, text: str) -> List[float]:
        """Extract all character complexity features"""
        features = []
        
        if self.use_spacy:
            doc = self.nlp(text)
            
            # Extract character entities
            characters = self._extract_characters(doc)
            
            if characters:
                # Analyze primary character (most mentioned)
                primary_char = max(characters.items(), key=lambda x: x[1])[0]
                
                # 1. Round vs flat (trait diversity)
                roundness = self._compute_character_roundness(doc, primary_char)
                features.append(roundness)
                
                # 2. Dynamic vs static (change over time)
                dynamism = self._compute_character_dynamism(doc, primary_char)
                features.append(dynamism)
                
                # 3. Character arc type (-1 negative, 0 flat, +1 positive)
                arc_type = self._compute_arc_type(doc, primary_char)
                features.append(arc_type)
                
                # 4. Transformation magnitude
                transformation = self._compute_transformation_magnitude(doc, primary_char)
                features.append(transformation)
                
                # 5. Trait diversity
                trait_diversity = self._compute_trait_diversity(doc, primary_char)
                features.append(trait_diversity)
                
                # 6. Moral complexity
                moral_complexity = self._compute_moral_complexity(doc, primary_char)
                features.append(moral_complexity)
                
                # 7. Motivation clarity
                motivation_clarity = self._compute_motivation_clarity(doc, primary_char)
                features.append(motivation_clarity)
                
                # 8. Action-to-description ratio
                action_ratio = self._compute_action_ratio(doc, primary_char)
                features.append(action_ratio)
                
                # 9. Dialogue presence
                dialogue_presence = self._compute_dialogue_presence(text, primary_char)
                features.append(dialogue_presence)
                
                # 10. Relationship network density
                network_density = self._compute_network_density(doc, characters)
                features.append(network_density)
                
                # 11. Agency score
                agency = self._compute_agency_score(doc, primary_char)
                features.append(agency)
                
                # 12. Consistency score
                consistency = self._compute_consistency(doc, primary_char)
                features.append(consistency)
                
                # 13. Depth indicators
                depth = self._compute_depth_indicators(doc, primary_char)
                features.append(depth)
                
                # 14. Character focus
                focus = characters[primary_char] / len(doc)
                features.append(min(1.0, focus * 20))
                
                # 15. Complexity index (composite)
                complexity_index = np.mean([
                    roundness, trait_diversity, moral_complexity, 
                    depth, dynamism
                ])
                features.append(complexity_index)
            else:
                # No characters detected
                features = [0.0] * 15
        else:
            # Fallback without spaCy
            features = self._extract_simple_features(text)
        
        return features
    
    def _extract_characters(self, doc) -> Dict[str, int]:
        """Extract character entities and count mentions"""
        characters = Counter()
        
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                characters[ent.text] += 1
        
        return dict(characters)
    
    def _compute_character_roundness(self, doc, character: str) -> float:
        """
        Round characters have multiple traits.
        Analyze adjectives and nouns associated with character.
        """
        traits = set()
        
        for sent in doc.sents:
            if character.lower() in sent.text.lower():
                # Find adjectives modifying character
                for token in sent:
                    if token.pos_ == 'ADJ':
                        traits.add(token.lemma_)
                    # Find nouns in apposition
                    if token.pos_ == 'NOUN' and token.dep_ == 'appos':
                        traits.add(token.lemma_)
        
        # Normalize by document length
        roundness = len(traits) / 10.0  # Scale to 0-1
        return min(1.0, roundness)
    
    def _compute_character_dynamism(self, doc, character: str) -> float:
        """
        Dynamic characters change over time.
        Look for transformation/change verbs associated with character.
        """
        change_count = 0
        total_mentions = 0
        
        # Transformation verbs
        transform_lemmas = {'become', 'change', 'evolve', 'grow', 'transform', 
                          'turn', 'develop', 'learn', 'realize'}
        
        for sent in doc.sents:
            if character.lower() in sent.text.lower():
                total_mentions += 1
                for token in sent:
                    if token.pos_ == 'VERB' and token.lemma_ in transform_lemmas:
                        # Check if character is subject
                        subjects = [child for child in token.children if child.dep_ in ['nsubj', 'nsubjpass']]
                        if any(character.lower() in subj.text.lower() for subj in subjects):
                            change_count += 1
        
        if total_mentions > 0:
            return min(1.0, change_count / total_mentions * 5)
        return 0.0
    
    def _compute_arc_type(self, doc, character: str) -> float:
        """
        Determine if arc is positive, negative, or flat.
        Analyze sentiment trajectory across document.
        """
        sentiments = []
        
        for i, sent in enumerate(doc.sents):
            if character.lower() in sent.text.lower():
                # Compute sentiment of sentence
                positive_count = sum(1 for token in sent if token.pos_ == 'ADJ' and 
                                   token.lemma_ in {'good', 'happy', 'great', 'wonderful', 'success'})
                negative_count = sum(1 for token in sent if token.pos_ == 'ADJ' and
                                   token.lemma_ in {'bad', 'sad', 'terrible', 'fail', 'wrong'})
                
                sent_sentiment = positive_count - negative_count
                sentiments.append((i / len(list(doc.sents)), sent_sentiment))
        
        if len(sentiments) >= 2:
            # Compute trend
            early_sentiment = np.mean([s[1] for s in sentiments[:len(sentiments)//2]])
            late_sentiment = np.mean([s[1] for s in sentiments[len(sentiments)//2:]])
            
            arc = late_sentiment - early_sentiment
            # Normalize to [-1, 1]
            return np.clip(arc / 3.0, -1.0, 1.0)
        
        return 0.0
    
    def _compute_transformation_magnitude(self, doc, character: str) -> float:
        """Measure magnitude of character change"""
        # Look for explicit transformation language
        transform_score = 0.0
        
        for sent in doc.sents:
            if character.lower() in sent.text.lower():
                for token in sent:
                    # Strong transformation indicators
                    if token.lemma_ in {'completely', 'totally', 'fundamentally', 'radically'}:
                        transform_score += 0.3
                    if token.lemma_ in {'never', 'always'} and token.head.pos_ == 'VERB':
                        # "never again" or "always will" indicate permanent change
                        transform_score += 0.2
        
        return min(1.0, transform_score)
    
    def _compute_trait_diversity(self, doc, character: str) -> float:
        """Count unique traits/descriptors"""
        descriptors = set()
        
        for sent in doc.sents:
            if character.lower() in sent.text.lower():
                for token in sent:
                    if token.pos_ in ['ADJ', 'ADV'] and token.dep_ in ['amod', 'advmod']:
                        descriptors.add(token.lemma_)
        
        return min(1.0, len(descriptors) / 8.0)
    
    def _compute_moral_complexity(self, doc, character: str) -> float:
        """
        Measure moral ambiguity vs. clear good/evil.
        Look for contradictory traits or moral qualifiers.
        """
        positive_traits = 0
        negative_traits = 0
        qualifiers = 0
        
        pos_lemmas = {'good', 'kind', 'honest', 'noble', 'virtuous', 'righteous'}
        neg_lemmas = {'bad', 'evil', 'cruel', 'dishonest', 'corrupt', 'wicked'}
        qualifier_lemmas = {'but', 'although', 'however', 'despite', 'yet', 'still'}
        
        for sent in doc.sents:
            if character.lower() in sent.text.lower():
                for token in sent:
                    if token.lemma_ in pos_lemmas:
                        positive_traits += 1
                    if token.lemma_ in neg_lemmas:
                        negative_traits += 1
                    if token.lemma_ in qualifier_lemmas:
                        qualifiers += 1
        
        # High complexity if both positive and negative traits, or many qualifiers
        if positive_traits > 0 and negative_traits > 0:
            return min(1.0, (qualifiers + 1) / 3.0)
        
        return min(1.0, qualifiers / 5.0)
    
    def _compute_motivation_clarity(self, doc, character: str) -> float:
        """Measure how clear character's motivations are"""
        motivation_count = 0
        
        # Motivation indicators
        motivation_lemmas = {'want', 'need', 'desire', 'hope', 'goal', 'aim', 'seek', 'pursue'}
        
        for sent in doc.sents:
            if character.lower() in sent.text.lower():
                for token in sent:
                    if token.lemma_ in motivation_lemmas:
                        # Check if character is subject
                        if token.pos_ == 'VERB':
                            subjects = [c for c in token.children if c.dep_ in ['nsubj']]
                            if any(character.lower() in s.text.lower() for s in subjects):
                                motivation_count += 1
                        else:
                            motivation_count += 0.5
        
        return min(1.0, motivation_count / 3.0)
    
    def _compute_action_ratio(self, doc, character: str) -> float:
        """Ratio of actions to descriptions"""
        action_verbs = 0
        descriptive_tokens = 0
        
        for sent in doc.sents:
            if character.lower() in sent.text.lower():
                for token in sent:
                    if token.pos_ == 'VERB' and token.dep_ != 'aux':
                        # Check if character is agent
                        subjects = [c for c in token.children if c.dep_ == 'nsubj']
                        if any(character.lower() in s.text.lower() for s in subjects):
                            action_verbs += 1
                    if token.pos_ in ['ADJ', 'ADV']:
                        descriptive_tokens += 1
        
        total = action_verbs + descriptive_tokens
        if total > 0:
            return action_verbs / total
        return 0.5
    
    def _compute_dialogue_presence(self, text: str, character: str) -> float:
        """Detect dialogue associated with character"""
        # Simple heuristic: quoted text near character name
        import re
        
        quotes = re.findall(r'"[^"]+"|\'[^\']+\'|"[^"]+"|\'[^\']+\'', text)
        char_dialogue = 0
        
        for i, quote in enumerate(quotes):
            # Check if character name appears near quote
            quote_pos = text.find(quote)
            context = text[max(0, quote_pos-100):min(len(text), quote_pos+len(quote)+100)]
            if character.lower() in context.lower():
                char_dialogue += 1
        
        if quotes:
            return char_dialogue / len(quotes)
        return 0.0
    
    def _compute_network_density(self, doc, characters: Dict[str, int]) -> float:
        """Measure interconnection between characters"""
        if len(characters) < 2:
            return 0.0
        
        connections = set()
        char_list = list(characters.keys())
        
        for sent in doc.sents:
            # Find which characters appear in same sentence
            chars_in_sent = [c for c in char_list if c.lower() in sent.text.lower()]
            if len(chars_in_sent) >= 2:
                # Create connection pairs
                for i, c1 in enumerate(chars_in_sent):
                    for c2 in chars_in_sent[i+1:]:
                        connections.add(tuple(sorted([c1, c2])))
        
        # Density = actual connections / possible connections
        max_connections = len(char_list) * (len(char_list) - 1) / 2
        if max_connections > 0:
            return len(connections) / max_connections
        return 0.0
    
    def _compute_agency_score(self, doc, character: str) -> float:
        """Measure how much agency character has"""
        active_count = 0
        passive_count = 0
        
        for sent in doc.sents:
            if character.lower() in sent.text.lower():
                for token in sent:
                    if token.pos_ == 'VERB':
                        # Active voice
                        subjects = [c for c in token.children if c.dep_ == 'nsubj']
                        if any(character.lower() in s.text.lower() for s in subjects):
                            active_count += 1
                        
                        # Passive voice
                        pass_subjects = [c for c in token.children if c.dep_ == 'nsubjpass']
                        if any(character.lower() in s.text.lower() for s in pass_subjects):
                            passive_count += 1
        
        total = active_count + passive_count
        if total > 0:
            return active_count / total
        return 0.5
    
    def _compute_consistency(self, doc, character: str) -> float:
        """Measure behavioral consistency"""
        # Track actions in different parts of document
        actions_by_section = [[], [], []]
        
        sents = list(doc.sents)
        section_size = len(sents) // 3
        
        for i, sent in enumerate(sents):
            section = min(i // max(1, section_size), 2)
            if character.lower() in sent.text.lower():
                for token in sent:
                    if token.pos_ == 'VERB':
                        actions_by_section[section].append(token.lemma_)
        
        # Compute overlap between sections
        if all(actions_by_section):
            overlap = len(set(actions_by_section[0]) & set(actions_by_section[1]) & set(actions_by_section[2]))
            total_unique = len(set(sum(actions_by_section, [])))
            if total_unique > 0:
                return overlap / total_unique
        
        return 0.5
    
    def _compute_depth_indicators(self, doc, character: str) -> float:
        """Composite measure of character depth"""
        depth_score = 0.0
        
        for sent in doc.sents:
            if character.lower() in sent.text.lower():
                # Internal thoughts/emotions
                for token in sent:
                    if token.lemma_ in {'think', 'feel', 'believe', 'remember', 'realize', 'understand'}:
                        depth_score += 0.2
                    # Subordinate clauses indicate complexity
                    if token.dep_ in ['ccomp', 'xcomp', 'advcl']:
                        depth_score += 0.1
        
        return min(1.0, depth_score / 3.0)
    
    def _extract_simple_features(self, text: str) -> List[float]:
        """Fallback feature extraction without spaCy"""
        # Very basic analysis
        features = []
        
        # Count capitals words (character names)
        words = text.split()
        proper_nouns = sum(1 for w in words if w and w[0].isupper() and len(w) > 1)
        
        # Estimate features based on text statistics
        features.append(min(1.0, proper_nouns / 20.0))  # roundness proxy
        features.append(0.3)  # dynamism unknown
        features.append(0.0)  # arc type unknown
        features.append(0.2)  # transformation unknown
        features.append(min(1.0, proper_nouns / 15.0))  # trait diversity proxy
        features.append(0.5)  # moral complexity unknown
        features.append(0.3)  # motivation clarity unknown
        features.append(0.5)  # action ratio unknown
        
        # Count quotes
        quote_count = text.count('"') + text.count("'")
        features.append(min(1.0, quote_count / 20.0))  # dialogue presence
        
        features.append(0.3)  # network density unknown
        features.append(0.5)  # agency unknown
        features.append(0.5)  # consistency unknown
        features.append(0.3)  # depth unknown
        features.append(min(1.0, proper_nouns / 10.0))  # focus proxy
        features.append(0.4)  # complexity index
        
        return features
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names"""
        return np.array([
            'char_roundness',
            'char_dynamism',
            'char_arc_type',
            'char_transformation_magnitude',
            'char_trait_diversity',
            'char_moral_complexity',
            'char_motivation_clarity',
            'char_action_ratio',
            'char_dialogue_presence',
            'char_network_density',
            'char_agency_score',
            'char_consistency',
            'char_depth',
            'char_focus',
            'char_complexity_index'
        ])

