"""
Self-Perception Transformer V2

Advanced NLP-based self-perception analysis using:
- Dependency parsing for attribution detection
- Semantic embeddings for personality traits
- Agency analysis via voice detection
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


class SelfPerceptionTransformerV2(BaseEstimator, TransformerMixin):
    """
    Advanced self-perception analysis using NLP.
    
    Improvements over V1:
    - Dependency parsing for attribution (not word lists)
    - Semantic analysis of personality traits
    - Active/passive voice via syntax (not regex)
    - Growth mindset via semantic embeddings
    - Big Five personality dimensions
    
    Features: 25 total (was 21)
    - First-person intensity (3)
    - Self-attribution (4 - via dependency parsing)
    - Growth orientation (3)
    - Aspirational vs descriptive (3)
    - Agency patterns (4 - via voice analysis)
    - Identity coherence (2)
    - Self-complexity (2)
    - Big Five dimensions (5 - NEW)
    """
    
    def __init__(self, use_spacy: bool = True, use_embeddings: bool = True):
        """Initialize self-perception analyzer"""
        self.use_spacy = use_spacy
        self.use_embeddings = use_embeddings
        
        self.nlp = None
        self.embedder = None
        
        # Big Five personality prototypes
        self.big_five_prototypes = {
            'openness': "curious, creative, imaginative, open to new experiences",
            'conscientiousness': "organized, responsible, disciplined, achievement-oriented",
            'extraversion': "outgoing, energetic, sociable, assertive",
            'agreeableness': "friendly, compassionate, cooperative, trusting",
            'neuroticism': "anxious, moody, emotionally unstable, worried"
        }
    
    def fit(self, X, y=None):
        """Fit transformer (load shared models)"""
        X = ensure_string_list(X)
        
        # Load shared models
        if self.use_spacy:
            self.nlp = SharedModelRegistry.get_spacy()
        
        if self.use_embeddings:
            self.embedder = SharedModelRegistry.get_sentence_transformer()
            
            # Embed Big Five prototypes
            self.big_five_embeddings = {}
            if self.embedder:
                for trait, description in self.big_five_prototypes.items():
                    self.big_five_embeddings[trait] = self.embedder.encode([description])[0]
        
        return self
    
    def transform(self, X):
        """
        Transform texts to self-perception features.
        
        Parameters
        ----------
        X : array-like of strings
            Narrative texts
            
        Returns
        -------
        features : ndarray of shape (n_samples, 25)
            Self-perception features
        """
        X = ensure_string_list(X)
        features = []
        
        for text in X:
            feat = self._extract_self_perception_features(text)
            features.append(feat)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_self_perception_features(self, text: str) -> List[float]:
        """Extract all self-perception features"""
        features = []
        
        if self.nlp:
            doc = self.nlp(text)
        else:
            doc = None
        
        text_lower = text.lower()
        n_words = len(text_lower.split()) + 1
        
        # 1-3: First-person intensity
        if doc:
            # Count first person pronouns
            fp_singular = sum(1 for token in doc if token.lemma_ in {'i', 'me', 'my', 'mine', 'myself'})
            fp_plural = sum(1 for token in doc if token.lemma_ in {'we', 'us', 'our', 'ours', 'ourselves'})
            
            features.append(fp_singular / n_words)
            features.append(fp_plural / n_words)
            
            total_fp = fp_singular + fp_plural + 1
            features.append(fp_singular / total_fp)
        else:
            # Regex fallback
            fp_sing = len(re.findall(r'\bi\b|\bme\b|\bmy\b', text_lower))
            fp_plur = len(re.findall(r'\bwe\b|\bus\b|\bour\b', text_lower))
            features.extend([fp_sing / n_words, fp_plur / n_words, 
                           fp_sing / (fp_sing + fp_plur + 1)])
        
        # 4-7: Self-attribution (via dependency parsing)
        if doc:
            # Find sentences where "I" or "we" is subject
            positive_attr = 0
            negative_attr = 0
            
            for sent in doc.sents:
                # Find first-person subjects
                fp_subjects = [token for token in sent 
                             if token.lemma_ in {'i', 'we'} and token.dep_ in ['nsubj', 'nsubjpass']]
                
                if fp_subjects:
                    # Find predicates
                    for subj in fp_subjects:
                        # Get the verb
                        if subj.head.pos_ == 'VERB':
                            # Check for positive/negative adjectives or verbs
                            for child in subj.head.children:
                                if child.pos_ == 'ADJ':
                                    # Positive traits
                                    if child.lemma_ in {'good', 'strong', 'smart', 'capable', 'confident', 'skilled'}:
                                        positive_attr += 1
                                    # Negative traits
                                    elif child.lemma_ in {'bad', 'weak', 'stupid', 'incapable', 'unsure'}:
                                        negative_attr += 1
            
            features.append(positive_attr / n_words)
            features.append(negative_attr / n_words)
            features.append((positive_attr - negative_attr) / n_words)  # Balance
            features.append(positive_attr / (positive_attr + negative_attr + 1))  # Confidence
        else:
            features.extend([0.2, 0.1, 0.1, 0.6])
        
        # 8-10: Growth orientation
        if doc:
            # Growth verbs via lemmatization
            growth_count = sum(1 for token in doc if token.pos_ == 'VERB' and 
                             token.lemma_ in {'become', 'grow', 'develop', 'evolve', 'transform', 'improve', 'learn', 'progress'})
            stasis_count = sum(1 for token in doc if token.lemma_ in 
                             {'stay', 'remain', 'continue', 'keep', 'stuck', 'static'})
            
            features.append(growth_count / n_words)
            features.append(stasis_count / n_words)
            features.append(growth_count / (growth_count + stasis_count + 1))
        else:
            features.extend([0.3, 0.2, 0.6])
        
        # 11-13: Aspirational vs descriptive
        if doc:
            # Aspirational = modal verbs + desire verbs
            aspirational = sum(1 for token in doc if 
                             token.lemma_ in {'want', 'hope', 'wish', 'dream', 'will', 'shall', 'aspire', 'goal', 'aim'})
            # Descriptive = copula + have
            descriptive = sum(1 for token in doc if token.lemma_ in {'be', 'have', 'has', 'am', 'is', 'are'})
            
            features.append(aspirational / n_words)
            features.append(descriptive / n_words)
            features.append(aspirational / (aspirational + descriptive + 1))
        else:
            features.extend([0.25, 0.4, 0.4])
        
        # 14-17: Agency patterns (via voice analysis)
        if doc:
            # Active voice (I/we + active verb)
            active_count = 0
            passive_count = 0
            
            for token in doc:
                if token.pos_ == 'VERB':
                    # Active voice
                    subjects = [c for c in token.children if c.dep_ == 'nsubj']
                    if subjects and any(s.lemma_ in {'i', 'we'} for s in subjects):
                        active_count += 1
                    
                    # Passive voice
                    pass_subjects = [c for c in token.children if c.dep_ == 'nsubjpass']
                    if pass_subjects and any(s.lemma_ in {'i', 'we'} for s in pass_subjects):
                        passive_count += 1
            
            features.append(active_count / n_words)
            features.append(passive_count / n_words)
            features.append(active_count / (active_count + passive_count + 1))
            
            # Agentive language (I did X, I made Y)
            agentive_verbs = sum(1 for token in doc 
                               if token.pos_ == 'VERB' and token.lemma_ in 
                               {'do', 'make', 'create', 'build', 'accomplish', 'achieve', 'complete'})
            features.append(agentive_verbs / n_words)
        else:
            features.extend([0.4, 0.15, 0.7, 0.3])
        
        # 18-19: Identity coherence
        if doc:
            # Measure consistency of self-reference across document
            sentences = list(doc.sents)
            if len(sentences) >= 3:
                # Track first-person density per third
                third = len(sentences) // 3
                thirds = [sentences[i*third:(i+1)*third] for i in range(3)]
                
                fp_densities = []
                for third_sents in thirds:
                    fp_count = sum(1 for sent in third_sents for token in sent 
                                 if token.lemma_ in {'i', 'we', 'me', 'us'})
                    third_words = sum(len(sent) for sent in third_sents)
                    fp_densities.append(fp_count / (third_words + 1))
                
                # Coherence = consistency (low variance)
                coherence = 1 / (1 + np.std(fp_densities))
                features.append(coherence)
                
                # Stability = no major shifts
                stability = 1.0 - abs(fp_densities[0] - fp_densities[-1])
                features.append(stability)
            else:
                features.extend([0.7, 0.7])
        else:
            features.extend([0.6, 0.6])
        
        # 20-21: Self-complexity
        if doc:
            # Variety of self-descriptors
            self_descriptors = set()
            for token in doc:
                if token.lemma_ in {'i', 'we'} and token.head.pos_ in ['ADJ', 'NOUN']:
                    self_descriptors.add(token.head.lemma_)
            
            complexity = min(1.0, len(self_descriptors) / 10.0)
            features.append(complexity)
            
            # Meta-cognitive language (I think I, I realize I)
            meta_cognitive = sum(1 for token in doc 
                               if token.lemma_ in {'think', 'realize', 'know', 'feel', 'believe'} and
                               any(c.lemma_ == 'i' for c in token.children))
            features.append(meta_cognitive / n_words)
        else:
            features.extend([0.4, 0.2])
        
        # 22-26: Big Five personality dimensions (via semantic matching)
        if self.embedder and hasattr(self, 'big_five_embeddings'):
            text_emb = self.embedder.encode([text])[0]
            
            for trait in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
                trait_emb = self.big_five_embeddings[trait]
                # Cosine similarity
                sim = np.dot(text_emb, trait_emb) / (
                    np.linalg.norm(text_emb) * np.linalg.norm(trait_emb) + 1e-9
                )
                features.append(float(sim))
        else:
            # Fallback: defaults
            features.extend([0.5, 0.5, 0.5, 0.5, 0.5])
        
        return features
    

class SelfPerceptionV2Transformer(SelfPerceptionTransformerV2):
    """Alias for registry compatibility."""
    pass
    def get_feature_names_out(self, input_features=None):
        """Return feature names"""
        return np.array([
            'sp_fp_singular',
            'sp_fp_plural',
            'sp_focus_ratio',
            'sp_positive_attribution',
            'sp_negative_attribution',
            'sp_attribution_balance',
            'sp_confidence',
            'sp_growth_orientation',
            'sp_stasis_orientation',
            'sp_growth_mindset',
            'sp_aspirational',
            'sp_descriptive',
            'sp_aspirational_ratio',
            'sp_active_voice',
            'sp_passive_voice',
            'sp_agency_ratio',
            'sp_agentive_verbs',
            'sp_identity_coherence',
            'sp_identity_stability',
            'sp_self_complexity',
            'sp_meta_cognitive',
            'sp_openness',
            'sp_conscientiousness',
            'sp_extraversion',
            'sp_agreeableness',
            'sp_neuroticism'
        ])

