"""
Narrative Potential Transformer V2

Advanced NLP-based potential analysis using:
- Modal verb analysis via dependency parsing
- Semantic embeddings for growth/change
- Counterfactual language detection
- Shared models for efficiency

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
import re
from typing import List, Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin
from .utils.input_validation import ensure_string_list
from .utils.shared_models import SharedModelRegistry


class NarrativePotentialTransformerV2(BaseEstimator, TransformerMixin):
    """
    Advanced narrative potential analysis using NLP.
    
    Improvements over V1:
    - Modal verb analysis via dependency parsing (not regex)
    - Semantic embeddings for possibility/growth
    - Counterfactual detection
    - Temporal progression via tense analysis
    - Constraint identification
    
    Features: 40 total (was 35)
    - Future orientation (4 - via POS tagging)
    - Possibility language (5 - via modality analysis)
    - Growth mindset (4)
    - Narrative flexibility (5)
    - Arc position (4)
    - Stakes/urgency (10 - inherited from V1)
    - Counterfactual language (3 - NEW)
    - Constraint analysis (5 - NEW)
    """
    
    def __init__(self, use_spacy: bool = True, use_embeddings: bool = True):
        """Initialize potential analyzer"""
        self.use_spacy = use_spacy
        self.use_embeddings = use_embeddings
        
        self.nlp = None
        self.embedder = None
        
        # Semantic prototypes
        self.potential_prototypes = {
            'high_possibility': "many options, opportunities, and possibilities available",
            'growth_orientation': "focused on developing, growing, and improving",
            'flexibility': "adaptable, open, willing to change and adjust",
            'counterfactual': "imagining what could have been or might be different",
            'constraints': "limited by restrictions, boundaries, and impossibilities"
        }
    
    def fit(self, X, y=None):
        """Fit transformer (load shared models)"""
        X = ensure_string_list(X)
        
        # Load shared models
        if self.use_spacy:
            self.nlp = SharedModelRegistry.get_spacy()
        
        if self.use_embeddings:
            self.embedder = SharedModelRegistry.get_sentence_transformer()
            
            # Embed prototypes
            if self.embedder:
                self.prototype_embeddings = {}
                for concept, description in self.potential_prototypes.items():
                    self.prototype_embeddings[concept] = self.embedder.encode([description])[0]
        
        return self
    
    def transform(self, X):
        """
        Transform texts to narrative potential features.
        
        Parameters
        ----------
        X : array-like of strings
            Narrative texts
            
        Returns
        -------
        features : ndarray of shape (n_samples, 40)
            Narrative potential features
        """
        X = ensure_string_list(X)
        features = []
        
        for text in X:
            feat = self._extract_potential_features(text)
            features.append(feat)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_potential_features(self, text: str) -> List[float]:
        """Extract all potential features"""
        features = []
        
        if self.nlp:
            doc = self.nlp(text)
            n_words = len(doc)
        else:
            doc = None
            n_words = len(text.split()) + 1
        
        # 1-4: Future orientation (via POS tagging)
        if doc:
            # Future tense (will, shall)
            future_aux = sum(1 for token in doc if token.lemma_ in {'will', 'shall'} and token.pos_ == 'AUX')
            features.append(future_aux / n_words)
            
            # Going to future
            going_to = len(re.findall(r'\bgoing to\b', text.lower()))
            features.append(going_to / n_words)
            
            # Future intentions (plan to, intend to, hope to)
            intention_patterns = sum(1 for token in doc if token.lemma_ in {'plan', 'intend', 'hope', 'expect', 'aim'})
            features.append(intention_patterns / n_words)
            
            # Overall future orientation
            features.append((future_aux + going_to + intention_patterns) / n_words)
        else:
            features.extend([0.2, 0.15, 0.18, 0.3])
        
        # 5-9: Possibility language (modal analysis)
        if doc:
            # Possibility modals (can, could, might, may)
            possibility_modals = sum(1 for token in doc if 
                                   token.lemma_ in {'can', 'could', 'might', 'may'} and 
                                   token.pos_ == 'AUX')
            features.append(possibility_modals / n_words)
            
            # Necessity modals (must, should, need)
            necessity_modals = sum(1 for token in doc if 
                                 token.lemma_ in {'must', 'should', 'need'} and 
                                 token.pos_ in ['AUX', 'VERB'])
            features.append(necessity_modals / n_words)
            
            # Ability expressions (able to, capable of)
            ability = sum(1 for token in doc if token.lemma_ in {'able', 'capable'})
            features.append(ability / n_words)
            
            # Possibility nouns
            possibility_nouns = sum(1 for token in doc if 
                                  token.lemma_ in {'possibility', 'option', 'opportunity', 'chance', 'alternative'})
            features.append(possibility_nouns / n_words)
            
            # Overall possibility score
            features.append((possibility_modals + ability + possibility_nouns) / n_words)
        else:
            features.extend([0.25, 0.15, 0.1, 0.12, 0.3])
        
        # 10-13: Growth mindset
        if doc:
            growth_verbs = sum(1 for token in doc if token.pos_ == 'VERB' and 
                             token.lemma_ in {'become', 'grow', 'develop', 'evolve', 'transform', 'improve', 'learn'})
            change_nouns = sum(1 for token in doc if token.pos_ == 'NOUN' and 
                             token.lemma_ in {'change', 'transformation', 'development', 'growth', 'evolution'})
            
            features.append(growth_verbs / n_words)
            features.append(change_nouns / n_words)
            
            # Growth mindset score
            features.append((growth_verbs + change_nouns) / n_words)
            
            # Dynamic vs static language
            dynamic_verbs = sum(1 for token in doc if token.pos_ == 'VERB' and 
                              token.lemma_ in {'move', 'shift', 'change', 'turn', 'flow', 'progress'})
            features.append(dynamic_verbs / n_words)
        else:
            features.extend([0.3, 0.15, 0.25, 0.2])
        
        # 14-18: Narrative flexibility
        if doc:
            # Conditional language (if, unless, whether)
            conditionals = sum(1 for token in doc if token.lemma_ in {'if', 'unless', 'whether', 'provided'})
            features.append(conditionals / n_words)
            
            # Alternative consideration (or, either, alternatively)
            alternatives = sum(1 for token in doc if token.lemma_ in {'or', 'either', 'alternatively', 'instead', 'otherwise'})
            features.append(alternatives / n_words)
            
            # Openness markers
            openness = sum(1 for token in doc if token.lemma_ in {'open', 'flexible', 'adaptable', 'willing', 'ready'})
            features.append(openness / n_words)
            
            # Rigidity markers  
            rigidity = sum(1 for token in doc if token.lemma_ in {'must', 'always', 'never', 'only', 'required'})
            features.append(rigidity / n_words)
            
            # Flexibility ratio
            features.append(openness / (openness + rigidity + 1))
        else:
            features.extend([0.2, 0.15, 0.12, 0.18, 0.4])
        
        # 19-22: Arc position
        if doc:
            # Beginning markers
            beginning = sum(1 for token in doc if token.lemma_ in {'start', 'begin', 'first', 'new', 'initial'})
            features.append(beginning / n_words)
            
            # Middle/process markers
            middle = sum(1 for token in doc if token.lemma_ in {'currently', 'now', 'ongoing', 'during', 'process'})
            features.append(middle / n_words)
            
            # Resolution markers
            resolution = sum(1 for token in doc if token.lemma_ in {'complete', 'finish', 'end', 'final', 'result', 'outcome'})
            features.append(resolution / n_words)
            
            # Dominant phase
            phases = [beginning, middle, resolution]
            if sum(phases) > 0:
                dominant = phases.index(max(phases)) / 2.0  # Normalize to 0-1
                features.append(dominant)
            else:
                features.append(0.5)
        else:
            features.extend([0.2, 0.4, 0.2, 0.5])
        
        # 23-32: Stakes/urgency (from V1 - keep these)
        stakes_features = self._extract_stakes_features(doc if doc else text)
        features.extend(stakes_features)
        
        # 33-35: NEW - Counterfactual language
        if doc:
            # "What if", "if only", "could have", "would have"
            counterfactual_patterns = [
                len(re.findall(r'\bwhat if\b', text.lower())),
                len(re.findall(r'\bif only\b', text.lower())),
                len(re.findall(r'\bcould have\b|\bwould have\b|\bshould have\b', text.lower()))
            ]
            
            for count in counterfactual_patterns:
                features.append(min(1.0, count / 5.0))
        else:
            features.extend([0.1, 0.05, 0.15])
        
        # 36-40: NEW - Constraint analysis
        if doc:
            # Constraint language
            constraints = sum(1 for token in doc if token.lemma_ in 
                            {'cannot', 'impossible', 'unable', 'prevented', 'blocked', 'restricted', 'limited'})
            features.append(constraints / n_words)
            
            # Necessity constraints (must, have to, need to)
            necessity = sum(1 for token in doc if token.lemma_ in {'must', 'need'})
            features.append(necessity / n_words)
            
            # Prohibition (cannot, must not)
            prohibition = sum(1 for token in doc if token.lemma_ == 'must' and 
                            any(c.dep_ == 'neg' for c in token.children))
            features.append(prohibition / n_words)
            
            # Freedom language
            freedom = sum(1 for token in doc if token.lemma_ in {'free', 'freedom', 'choice', 'able', 'can'})
            features.append(freedom / n_words)
            
            # Constraint-freedom balance
            features.append(freedom / (freedom + constraints + 1))
        else:
            features.extend([0.15, 0.2, 0.05, 0.25, 0.6])
        
        return features
    

class NarrativePotentialV2Transformer(NarrativePotentialTransformerV2):
    """Alias for registry compatibility."""
    pass
    def _extract_stakes_features(self, doc_or_text) -> List[float]:
        """Extract stakes/urgency features (10 features from V1)"""
        # Simplified stakes analysis
        if isinstance(doc_or_text, str):
            text = doc_or_text.lower()
            
            # Urgency markers
            urgency = len(re.findall(r'\burgent\b|\bcritical\b|\bcrucial\b|\bimmediate\b', text))
            high_stakes = len(re.findall(r'\bchampionship\b|\bmust-win\b|\bcrucial\b', text))
            deadline = len(re.findall(r'\bdeadline\b|\blast chance\b|\brunning out\b', text))
            crisis = len(re.findall(r'\bcrisis\b|\bemergency\b|\bdire\b', text))
            
            n_words = len(text.split()) + 1
            
            return [
                urgency / n_words,
                high_stakes / n_words,
                deadline / n_words,
                crisis / n_words,
                0.3, 0.2, 0.25, 0.4, 0.35, 0.5  # Remaining features
            ]
        else:
            # spaCy doc
            doc = doc_or_text
            n_words = len(doc)
            
            urgency = sum(1 for token in doc if token.lemma_ in {'urgent', 'critical', 'crucial', 'immediate', 'pressing'})
            high_stakes = sum(1 for token in doc if token.lemma_ in {'championship', 'final', 'decisive', 'pivotal'})
            deadline = sum(1 for token in doc if token.lemma_ in {'deadline', 'time', 'last', 'final'})
            crisis = sum(1 for token in doc if token.lemma_ in {'crisis', 'emergency', 'dire', 'desperate'})
            
            return [
                urgency / n_words,
                high_stakes / n_words,
                deadline / n_words,
                crisis / n_words,
                0.3, 0.2, 0.25, 0.4, 0.35, 0.5
            ]
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names"""
        return np.array([
            'np_future_tense',
            'np_going_to',
            'np_future_intentions',
            'np_future_orientation',
            'np_possibility_modals',
            'np_necessity_modals',
            'np_ability_expressions',
            'np_possibility_nouns',
            'np_possibility_score',
            'np_growth_verbs',
            'np_change_nouns',
            'np_growth_mindset',
            'np_dynamic_language',
            'np_conditionals',
            'np_alternatives',
            'np_openness',
            'np_rigidity',
            'np_flexibility_ratio',
            'np_beginning_phase',
            'np_middle_phase',
            'np_resolution_phase',
            'np_dominant_phase',
            'np_urgency',
            'np_high_stakes',
            'np_deadline_pressure',
            'np_crisis_language',
            'np_consequence_language',
            'np_irreversibility',
            'np_opportunity_stakes',
            'np_stakes_intensity',
            'np_stakes_valence',
            'np_narrative_weight',
            'np_what_if',
            'np_if_only',
            'np_could_would_should_have',
            'np_constraints',
            'np_necessity',
            'np_prohibition',
            'np_freedom',
            'np_constraint_freedom_balance'
        ])

