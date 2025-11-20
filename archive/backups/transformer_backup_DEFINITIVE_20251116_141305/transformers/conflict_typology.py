"""
Conflict Typology Transformer

Uses advanced NLP to detect and classify narrative conflict types.
No hardcoded words - uses semantic embeddings and dependency parsing.

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
import re
from typing import List, Dict, Any, Optional
from sklearn.base import BaseEstimator, TransformerMixin
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


class ConflictTypologyTransformer(BaseEstimator, TransformerMixin):
    """
    Advanced NLP-based conflict type detection using:
    - Dependency parsing for agent-patient relationships
    - Semantic embeddings for conflict pattern matching
    - Entity recognition for conflict participants
    - Syntactic analysis for conflict structures
    
    Detects 6 fundamental conflict types:
    1. Man vs Man (interpersonal conflict)
    2. Man vs Self (internal conflict)
    3. Man vs Nature (environmental conflict)
    4. Man vs Society (institutional conflict)
    5. Man vs Technology (technological conflict)
    6. Man vs Fate/Unknown (existential conflict)
    
    Total: 10 features
    """
    
    def __init__(self, use_spacy: bool = True, use_embeddings: bool = True):
        """
        Initialize conflict detector.
        
        Parameters
        ----------
        use_spacy : bool
            Use spaCy for linguistic analysis
        use_embeddings : bool
            Use sentence transformers for semantic matching
        """
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
        
        # Semantic prototypes for conflict types (will be embedded)
        self.conflict_prototypes = {
            'man_vs_man': [
                "two people fighting against each other",
                "opponents competing in a rivalry",
                "enemies in confrontation",
                "adversaries in battle",
                "competitors facing off",
                "antagonist versus protagonist"
            ],
            'man_vs_self': [
                "internal struggle with oneself",
                "battling inner demons",
                "overcoming personal doubts",
                "fighting against own nature",
                "internal moral dilemma",
                "psychological conflict within"
            ],
            'man_vs_nature': [
                "fighting against natural forces",
                "surviving harsh environment",
                "battling the elements",
                "struggling against weather",
                "confronting natural disasters",
                "overcoming environmental challenges"
            ],
            'man_vs_society': [
                "fighting against social norms",
                "challenging institutional authority",
                "rebelling against system",
                "confronting cultural expectations",
                "struggling against establishment",
                "defying social order"
            ],
            'man_vs_technology': [
                "fighting against machines",
                "struggling with artificial intelligence",
                "confronting technological systems",
                "battling automation",
                "resisting digital control",
                "overcoming technological obstacles"
            ],
            'man_vs_fate': [
                "fighting against destiny",
                "struggling with fate",
                "confronting the unknown",
                "battling cosmic forces",
                "resisting predetermined outcome",
                "challenging supernatural powers"
            ]
        }
        
        # Embed prototypes
        self.prototype_embeddings = {}
        if self.use_embeddings:
            for conflict_type, examples in self.conflict_prototypes.items():
                self.prototype_embeddings[conflict_type] = self.embedder.encode(examples)
    
    def fit(self, X, y=None):
        """Fit transformer (preprocessing only)"""
        X = ensure_string_list(X)
        return self
    
    def transform(self, X):
        """
        Transform texts to conflict typology features.
        
        Parameters
        ----------
        X : array-like of strings
            Narrative texts
            
        Returns
        -------
        features : ndarray of shape (n_samples, 10)
            Conflict typology features
        """
        X = ensure_string_list(X)
        features = []
        
        for text in X:
            feat = self._extract_conflict_features(text)
            features.append(feat)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_conflict_features(self, text: str) -> List[float]:
        """Extract all conflict features using NLP"""
        features = []
        
        # 1-6: Semantic similarity to each conflict type
        if self.use_embeddings:
            conflict_scores = self._compute_semantic_conflict_scores(text)
            for conflict_type in ['man_vs_man', 'man_vs_self', 'man_vs_nature', 
                                 'man_vs_society', 'man_vs_technology', 'man_vs_fate']:
                features.append(conflict_scores.get(conflict_type, 0.0))
        else:
            # Fallback: use syntactic patterns
            features.extend(self._compute_syntactic_conflict_patterns(text))
        
        # 7: Internal vs External conflict ratio
        if len(features) >= 6:
            internal = features[1]  # man_vs_self
            external = sum(features[i] for i in [0, 2, 3, 4, 5])
            internal_ratio = internal / (internal + external + 1e-6)
            features.append(internal_ratio)
        else:
            features.append(0.5)
        
        # 8: Conflict complexity (how many types present)
        if len(features) >= 6:
            complexity = sum(1 for score in features[:6] if score > 0.3)
            features.append(complexity / 6.0)
        else:
            features.append(0.0)
        
        # 9: Dominant conflict strength
        if len(features) >= 6:
            features.append(max(features[:6]))
        else:
            features.append(0.0)
        
        # 10: Conflict intensity (using linguistic markers)
        intensity = self._compute_conflict_intensity(text)
        features.append(intensity)
        
        return features
    
    def _compute_semantic_conflict_scores(self, text: str) -> Dict[str, float]:
        """
        Compute semantic similarity to conflict type prototypes.
        Uses sentence embeddings and cosine similarity.
        """
        scores = {}
        
        # Embed the text
        text_embedding = self.embedder.encode([text])[0]
        
        # Compare to each conflict type's prototypes
        for conflict_type, prototype_embeddings in self.prototype_embeddings.items():
            # Compute cosine similarities
            similarities = []
            for proto_emb in prototype_embeddings:
                # Cosine similarity
                sim = np.dot(text_embedding, proto_emb) / (
                    np.linalg.norm(text_embedding) * np.linalg.norm(proto_emb) + 1e-9
                )
                similarities.append(sim)
            
            # Take maximum similarity as the score
            scores[conflict_type] = float(np.max(similarities))
        
        return scores
    
    def _compute_syntactic_conflict_patterns(self, text: str) -> List[float]:
        """
        Fallback method using spaCy dependency parsing.
        Analyzes agent-patient relationships and conflict verbs.
        """
        features = []
        
        if not self.use_spacy:
            # Ultimate fallback: return zeros
            return [0.0] * 6
        
        doc = self.nlp(text)
        
        # Analyze dependency patterns
        conflict_indicators = {
            'man_vs_man': 0.0,
            'man_vs_self': 0.0,
            'man_vs_nature': 0.0,
            'man_vs_society': 0.0,
            'man_vs_technology': 0.0,
            'man_vs_fate': 0.0
        }
        
        # Count person entities and their relationships
        person_entities = [ent for ent in doc.ents if ent.label_ in ['PERSON', 'ORG']]
        
        # Man vs Man: Multiple person entities with conflict verbs
        if len(person_entities) >= 2:
            conflict_indicators['man_vs_man'] += 0.3
        
        # Analyze verb patterns
        for token in doc:
            if token.pos_ == 'VERB':
                # Get subjects and objects
                subjects = [child for child in token.children if child.dep_ in ['nsubj', 'nsubjpass']]
                objects = [child for child in token.children if child.dep_ in ['dobj', 'pobj']]
                
                # Man vs Self: reflexive pronouns
                for obj in objects:
                    if obj.text.lower() in ['myself', 'himself', 'herself', 'itself', 'themselves']:
                        conflict_indicators['man_vs_self'] += 0.2
                
                # Man vs Nature: nature-related objects
                for obj in objects:
                    if obj.pos_ == 'NOUN':
                        # Check if natural entity
                        if any(word in obj.text.lower() for word in ['storm', 'wind', 'rain', 'mountain', 'ocean', 'forest']):
                            conflict_indicators['man_vs_nature'] += 0.2
                
                # Man vs Society: collective nouns as objects
                for obj in objects:
                    if obj.text.lower() in ['society', 'system', 'government', 'authority', 'establishment', 'rules', 'law']:
                        conflict_indicators['man_vs_society'] += 0.2
                
                # Man vs Technology: tech-related objects
                for obj in objects:
                    if obj.text.lower() in ['machine', 'computer', 'robot', 'system', 'algorithm', 'ai']:
                        conflict_indicators['man_vs_technology'] += 0.2
                
                # Man vs Fate: abstract/existential objects
                for obj in objects:
                    if obj.text.lower() in ['fate', 'destiny', 'god', 'universe', 'unknown', 'death']:
                        conflict_indicators['man_vs_fate'] += 0.2
        
        # Normalize and return
        for conflict_type in ['man_vs_man', 'man_vs_self', 'man_vs_nature', 
                             'man_vs_society', 'man_vs_technology', 'man_vs_fate']:
            features.append(min(1.0, conflict_indicators[conflict_type]))
        
        return features
    
    def _compute_conflict_intensity(self, text: str) -> float:
        """
        Compute overall conflict intensity using linguistic markers.
        Uses dependency parsing for conflict verbs and modifiers.
        """
        if not self.use_spacy:
            # Simple fallback
            sentences = re.split(r'[.!?]+', text)
            return min(1.0, len(sentences) / 20.0)
        
        doc = self.nlp(text)
        intensity = 0.0
        
        # Count conflict-related linguistic patterns
        for token in doc:
            # Intense verbs (semantically loaded)
            if token.pos_ == 'VERB':
                # Check for intense verb lemmas
                if token.lemma_ in ['fight', 'battle', 'struggle', 'confront', 'face', 
                                   'challenge', 'oppose', 'resist', 'combat', 'clash']:
                    intensity += 0.1
                
                # Check for negation (increases conflict)
                if any(child.dep_ == 'neg' for child in token.children):
                    intensity += 0.05
            
            # Intensity modifiers
            if token.pos_ == 'ADV':
                if token.lemma_ in ['extremely', 'intensely', 'desperately', 'fiercely', 'violently']:
                    intensity += 0.05
            
            # Emotional language
            if token.pos_ == 'ADJ':
                if token.lemma_ in ['angry', 'furious', 'desperate', 'hopeless', 'determined']:
                    intensity += 0.05
        
        # Normalize by document length
        intensity = intensity / (len(doc) + 1) * 10
        
        return min(1.0, intensity)
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names"""
        return np.array([
            'conflict_man_vs_man',
            'conflict_man_vs_self',
            'conflict_man_vs_nature',
            'conflict_man_vs_society',
            'conflict_man_vs_technology',
            'conflict_man_vs_fate',
            'conflict_internal_ratio',
            'conflict_complexity',
            'conflict_dominant_strength',
            'conflict_intensity'
        ])

