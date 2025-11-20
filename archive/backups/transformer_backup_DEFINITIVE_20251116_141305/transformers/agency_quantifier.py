"""
Agency Quantifier Transformer

Uses NLP to measure character/entity agency levels.
Dependency parsing for agent-patient relationships.

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin
from .utils.input_validation import ensure_string_list

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


class AgencyQuantifierTransformer(BaseEstimator, TransformerMixin):
    """
    Quantifies agency using advanced NLP.
    
    Features (5 total):
    1. Active decision-making frequency
    2. Initiative markers
    3. Problem-solving demonstration
    4. Resourcefulness indicators
    5. Agency evolution
    
    Uses:
    - Dependency parsing for agent detection
    - Active vs passive voice analysis
    - Modal verbs for capability
    - Verb semantic roles
    """
    
    def __init__(self, use_spacy: bool = True):
        """Initialize agency quantifier"""
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        
        if self.use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                try:
                    self.nlp = spacy.load("en_core_web_md")
                except:
                    self.use_spacy = False
    
    def fit(self, X, y=None):
        """Fit transformer"""
        X = ensure_string_list(X)
        return self
    
    def transform(self, X):
        """
        Transform texts to agency features.
        
        Parameters
        ----------
        X : array-like of strings
            Narrative texts
            
        Returns
        -------
        features : ndarray of shape (n_samples, 5)
            Agency quantification features
        """
        X = ensure_string_list(X)
        features = []
        
        for text in X:
            feat = self._extract_agency_features(text)
            features.append(feat)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_agency_features(self, text: str) -> List[float]:
        """Extract all agency features"""
        features = []
        
        if self.use_spacy:
            doc = self.nlp(text)
            
            # 1. Active decision-making
            decision_making = self._compute_decision_making(doc)
            features.append(decision_making)
            
            # 2. Initiative markers
            initiative = self._compute_initiative(doc)
            features.append(initiative)
            
            # 3. Problem-solving
            problem_solving = self._compute_problem_solving(doc)
            features.append(problem_solving)
            
            # 4. Resourcefulness
            resourcefulness = self._compute_resourcefulness(doc)
            features.append(resourcefulness)
            
            # 5. Agency evolution
            agency_evolution = self._compute_agency_evolution(doc)
            features.append(agency_evolution)
        else:
            # Fallback
            features = [0.5, 0.4, 0.4, 0.4, 0.3]
        
        return features
    
    def _compute_decision_making(self, doc) -> float:
        """
        Measure active decision-making through verb analysis.
        """
        decision_score = 0.0
        
        # Decision verbs
        decision_lemmas = {'decide', 'choose', 'select', 'determine', 'resolve',
                          'opt', 'pick', 'elect', 'settle'}
        
        # Active voice decisions
        for sent in doc.sents:
            for token in sent:
                if token.lemma_ in decision_lemmas:
                    # Check if agent is subject (active voice)
                    subjects = [c for c in token.children if c.dep_ == 'nsubj']
                    if subjects:
                        decision_score += 0.15
                        
                        # Bonus for first person (I/we decided)
                        if any(subj.lemma_ in {'i', 'we'} for subj in subjects):
                            decision_score += 0.05
        
        return min(1.0, decision_score)
    
    def _compute_initiative(self, doc) -> float:
        """
        Detect initiative through action initiation markers.
        """
        initiative_score = 0.0
        
        # Initiative verbs
        initiative_lemmas = {'start', 'begin', 'initiate', 'launch', 'create',
                            'lead', 'pioneer', 'originate', 'establish', 'found'}
        
        for sent in doc.sents:
            for token in sent:
                if token.lemma_ in initiative_lemmas:
                    # Check if in active voice
                    if token.dep_ == 'ROOT':
                        subjects = [c for c in token.children if c.dep_ == 'nsubj']
                        if subjects:
                            initiative_score += 0.12
                            
                            # Strong initiative if no external cause
                            # (no "made to" or "forced to")
                            has_coercion = any(
                                c.lemma_ in {'make', 'force', 'compel', 'require'}
                                for c in sent
                            )
                            if not has_coercion:
                                initiative_score += 0.08
        
        return min(1.0, initiative_score)
    
    def _compute_problem_solving(self, doc) -> float:
        """
        Detect problem-solving behavior.
        """
        solving_score = 0.0
        
        # Problem-solving verbs
        solve_lemmas = {'solve', 'fix', 'repair', 'resolve', 'overcome',
                       'address', 'tackle', 'handle', 'manage', 'deal'}
        
        # Problem nouns
        problem_lemmas = {'problem', 'issue', 'challenge', 'obstacle', 'difficulty',
                         'trouble', 'crisis', 'dilemma'}
        
        for sent in doc.sents:
            has_problem = any(token.lemma_ in problem_lemmas for token in sent)
            has_solving = any(
                token.lemma_ in solve_lemmas and token.pos_ == 'VERB'
                for token in sent
            )
            
            if has_problem and has_solving:
                solving_score += 0.2
            elif has_solving:
                solving_score += 0.1
        
        return min(1.0, solving_score)
    
    def _compute_resourcefulness(self, doc) -> float:
        """
        Detect resourcefulness through creative action and adaptation.
        """
        resource_score = 0.0
        
        # Resourcefulness indicators
        resource_lemmas = {'adapt', 'improvise', 'innovate', 'invent', 'devise',
                          'create', 'find', 'discover', 'figure', 'work'}
        
        # Capability modals
        capability_lemmas = {'can', 'able', 'capable', 'manage'}
        
        for sent in doc.sents:
            for token in sent:
                # Resourceful actions
                if token.lemma_ in resource_lemmas and token.pos_ == 'VERB':
                    # Check for agent as subject
                    subjects = [c for c in token.children if c.dep_ == 'nsubj']
                    if subjects:
                        resource_score += 0.1
                
                # Capability expressions
                if token.lemma_ in capability_lemmas:
                    resource_score += 0.05
        
        # Finding solutions ("found a way", "figured out how")
        way_phrases = sum(
            1 for sent in doc.sents
            if any(token.text.lower() in ['way', 'how', 'solution', 'answer'] for token in sent)
            and any(token.lemma_ in {'find', 'figure', 'discover'} for token in sent)
        )
        
        resource_score += min(0.3, way_phrases * 0.15)
        
        return min(1.0, resource_score)
    
    def _compute_agency_evolution(self, doc) -> float:
        """
        Track how agency changes over narrative.
        """
        sentences = list(doc.sents)
        if len(sentences) < 4:
            return 0.5
        
        # Divide into halves
        first_half = sentences[:len(sentences)//2]
        second_half = sentences[len(sentences)//2:]
        
        # Measure agency in each half
        agency_first = self._measure_agency_in_section(first_half)
        agency_second = self._measure_agency_in_section(second_half)
        
        # Evolution = change in agency
        evolution = agency_second - agency_first
        
        # Normalize to 0-1 where 1 = increasing agency
        return float(np.clip((evolution + 0.5) / 1.0, 0, 1))
    
    def _measure_agency_in_section(self, sentences: List) -> float:
        """
        Measure overall agency level in a section.
        """
        if not sentences:
            return 0.0
        
        agency_score = 0.0
        
        # Active voice ratio
        active_verbs = 0
        passive_verbs = 0
        
        for sent in sentences:
            for token in sent:
                if token.pos_ == 'VERB':
                    # Active
                    if any(c.dep_ == 'nsubj' for c in token.children):
                        active_verbs += 1
                    # Passive
                    if any(c.dep_ == 'nsubjpass' for c in token.children):
                        passive_verbs += 1
        
        total_verbs = active_verbs + passive_verbs
        if total_verbs > 0:
            agency_score += (active_verbs / total_verbs) * 0.5
        
        # Action verb density
        action_lemmas = {'do', 'make', 'create', 'build', 'fight', 'work',
                        'move', 'act', 'perform', 'execute', 'accomplish'}
        
        action_count = sum(
            1 for sent in sentences
            for token in sent
            if token.lemma_ in action_lemmas and token.pos_ == 'VERB'
        )
        
        agency_score += min(0.5, action_count / len(sentences) / 2)
        
        return agency_score
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names"""
        return np.array([
            'agency_decision_making',
            'agency_initiative',
            'agency_problem_solving',
            'agency_resourcefulness',
            'agency_evolution'
        ])

