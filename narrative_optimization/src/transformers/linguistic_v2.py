"""
Linguistic Patterns Transformer V2

Advanced NLP-based linguistic analysis using:
- Full POS tagging and dependency parsing
- Syntactic complexity via parse trees
- Discourse markers and connectives
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


class LinguisticPatternsTransformerV2(BaseEstimator, TransformerMixin):
    """
    Advanced linguistic analysis using full NLP pipeline.
    
    Improvements over V1:
    - Full POS tagging (not regex patterns)
    - Dependency parsing for voice (not pattern matching)
    - Syntactic complexity (parse tree depth)
    - Discourse connectives analysis
    - Credibility markers via syntax
    
    Features: 45 total (was 36)
    - Narrative voice (5 - via POS)
    - Temporal orientation (5 - via tense tags)
    - Agency patterns (4 - via dependencies)
    - Emotional trajectory (3)
    - Linguistic complexity (8 - via parse trees)
    - Evolution tracking (9 - trajectory analysis)
    - Credibility/authority (10 - via syntax)
    - Discourse structure (5 - NEW)
    """
    
    def __init__(self, use_spacy: bool = True):
        """Initialize linguistic analyzer"""
        self.use_spacy = use_spacy
        self.nlp = None
    
    def fit(self, X, y=None):
        """Fit transformer (load shared models)"""
        X = ensure_string_list(X)
        
        # Load shared spaCy model
        if self.use_spacy:
            self.nlp = SharedModelRegistry.get_spacy()
        
        return self
    
    def transform(self, X):
        """
        Transform texts to linguistic features.
        
        Parameters
        ----------
        X : array-like of strings
            Narrative texts
            
        Returns
        -------
        features : ndarray of shape (n_samples, 45)
            Linguistic features
        """
        X = ensure_string_list(X)
        features = []
        
        for text in X:
            feat = self._extract_linguistic_features(text)
            features.append(feat)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_linguistic_features(self, text: str) -> List[float]:
        """Extract all linguistic features"""
        features = []
        
        if self.nlp:
            doc = self.nlp(text)
            n_words = len(doc)
        else:
            doc = None
            n_words = len(text.split()) + 1
        
        # 1-5: Narrative voice (via POS tagging)
        if doc:
            # First person pronouns
            first_person = sum(1 for token in doc if token.lemma_ in 
                             {'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves'})
            features.append(first_person / n_words)
            
            # Second person
            second_person = sum(1 for token in doc if token.lemma_ in {'you', 'your', 'yours', 'yourself', 'yourselves'})
            features.append(second_person / n_words)
            
            # Third person
            third_person = sum(1 for token in doc if token.lemma_ in 
                             {'he', 'she', 'they', 'him', 'her', 'them', 'his', 'her', 'their'})
            features.append(third_person / n_words)
            
            # Voice consistency (entropy)
            total_person = first_person + second_person + third_person + 1
            person_dist = np.array([first_person, second_person, third_person]) + 0.1
            person_dist = person_dist / person_dist.sum()
            voice_entropy = -np.sum(person_dist * np.log(person_dist + 1e-10))
            features.append(voice_entropy / np.log(3))
            
            # Narrative immediacy (present tense + second person = direct address)
            present_verbs = sum(1 for token in doc if token.pos_ == 'VERB' and token.tag_ in ['VB', 'VBP', 'VBZ'])
            immediacy = (present_verbs / n_words + second_person / n_words) / 2
            features.append(immediacy)
        else:
            features.extend([0.3, 0.1, 0.4, 0.6, 0.3])
        
        # 6-10: Temporal orientation (via tense tags)
        if doc:
            # Past tense (VBD, VBN)
            past_verbs = sum(1 for token in doc if token.pos_ == 'VERB' and token.tag_ in ['VBD', 'VBN'])
            features.append(past_verbs / n_words)
            
            # Present tense (VB, VBP, VBZ, VBG)
            present_verbs = sum(1 for token in doc if token.pos_ == 'VERB' and token.tag_ in ['VB', 'VBP', 'VBZ', 'VBG'])
            features.append(present_verbs / n_words)
            
            # Future (will, shall + base form)
            future_verbs = sum(1 for token in doc if token.lemma_ in {'will', 'shall'})
            features.append(future_verbs / n_words)
            
            # Temporal balance (entropy)
            total_tense = past_verbs + present_verbs + future_verbs + 1
            tense_dist = np.array([past_verbs, present_verbs, future_verbs]) + 0.1
            tense_dist = tense_dist / tense_dist.sum()
            temporal_entropy = -np.sum(tense_dist * np.log(tense_dist + 1e-10))
            features.append(temporal_entropy / np.log(3))
            
            # Progressive aspect (ongoing action)
            progressive = sum(1 for token in doc if token.tag_ == 'VBG')
            features.append(progressive / n_words)
        else:
            features.extend([0.35, 0.4, 0.15, 0.7, 0.2])
        
        # 11-14: Agency patterns (via dependency parsing)
        if doc:
            # Active voice (nsubj before verb)
            active_count = 0
            passive_count = 0
            
            for token in doc:
                if token.pos_ == 'VERB' and token.dep_ == 'ROOT':
                    # Check for nsubj (active) vs nsubjpass (passive)
                    has_active_subj = any(c.dep_ == 'nsubj' for c in token.children)
                    has_passive_subj = any(c.dep_ == 'nsubjpass' for c in token.children)
                    
                    if has_active_subj:
                        active_count += 1
                    if has_passive_subj:
                        passive_count += 1
            
            features.append(active_count / n_words)
            features.append(passive_count / n_words)
            features.append(active_count / (active_count + passive_count + 1))
            
            # Agentless passives (by-phrase absent)
            agentless_passive = sum(1 for token in doc 
                                  if any(c.dep_ == 'nsubjpass' for c in token.children) and
                                  not any(c.dep_ == 'agent' for c in token.children))
            features.append(agentless_passive / n_words)
        else:
            features.extend([0.5, 0.2, 0.7, 0.1])
        
        # 15-17: Emotional trajectory (via sentiment)
        if doc:
            # Positive adjectives
            pos_adj = sum(1 for token in doc if token.pos_ == 'ADJ' and 
                        token.lemma_ in {'good', 'great', 'excellent', 'wonderful', 'happy', 'positive'})
            features.append(pos_adj / n_words)
            
            # Negative adjectives
            neg_adj = sum(1 for token in doc if token.pos_ == 'ADJ' and 
                        token.lemma_ in {'bad', 'terrible', 'awful', 'horrible', 'sad', 'negative'})
            features.append(neg_adj / n_words)
            
            # Sentiment balance
            features.append((pos_adj - neg_adj) / n_words)
        else:
            features.extend([0.2, 0.15, 0.05])
        
        # 18-25: Linguistic complexity (via parse trees)
        if doc:
            # Parse tree depth (average and max)
            depths = []
            for sent in doc.sents:
                for token in sent:
                    # Count ancestors to root
                    depth = 0
                    current = token
                    while current.head != current:
                        depth += 1
                        current = current.head
                        if depth > 20:  # Prevent infinite loops
                            break
                    depths.append(depth)
            
            if depths:
                features.append(np.mean(depths) / 10.0)  # Normalize
                features.append(max(depths) / 15.0)
            else:
                features.extend([0.5, 0.5])
            
            # Subordinate clauses
            subordinates = sum(1 for token in doc if token.dep_ in ['ccomp', 'xcomp', 'advcl', 'acl'])
            features.append(subordinates / n_words)
            
            # Relative clauses
            relatives = sum(1 for token in doc if token.dep_ in ['relcl'])
            features.append(relatives / n_words)
            
            # Coordination (and, or, but)
            coordination = sum(1 for token in doc if token.pos_ == 'CCONJ')
            features.append(coordination / n_words)
            
            # Prepositions (complexity indicator)
            prepositions = sum(1 for token in doc if token.pos_ == 'ADP')
            features.append(prepositions / n_words)
            
            # Determiners
            determiners = sum(1 for token in doc if token.pos_ == 'DET')
            features.append(determiners / n_words)
            
            # Overall syntactic complexity
            complexity = (subordinates + relatives + coordination) / n_words
            features.append(complexity)
        else:
            features.extend([0.5, 0.6, 0.3, 0.2, 0.4, 0.35, 0.3, 0.4])
        
        # 26-34: Evolution tracking (trajectory across document)
        if doc:
            evolution_features = self._extract_evolution_features(doc)
            features.extend(evolution_features)
        else:
            features.extend([0.0] * 9)
        
        # 35-44: Credibility/authority markers (via syntax)
        if doc:
            credibility_features = self._extract_credibility_features(doc)
            features.extend(credibility_features)
        else:
            features.extend([0.3] * 10)
        
        # 45: NEW - Discourse structure (coherence)
        if doc:
            # Discourse connectives
            connectives = sum(1 for token in doc if token.lemma_ in 
                            {'however', 'therefore', 'thus', 'moreover', 'furthermore', 
                             'nevertheless', 'consequently', 'meanwhile', 'additionally'})
            features.append(connectives / n_words)
        else:
            features.append(0.2)
        
        return features
    

class LinguisticPatternsV2Transformer(LinguisticPatternsTransformerV2):
    """Alias for registry compatibility."""
    pass
    def _extract_evolution_features(self, doc) -> List[float]:
        """Track linguistic feature evolution across document"""
        sentences = list(doc.sents)
        
        if len(sentences) < 3:
            return [0.0] * 9
        
        # Divide into thirds
        third = len(sentences) // 3
        thirds = [sentences[i*third:(i+1)*third] for i in range(3)]
        
        # Track first-person usage
        fp_trajectory = []
        for third_sents in thirds:
            fp = sum(1 for sent in third_sents for token in sent 
                    if token.lemma_ in {'i', 'we', 'me', 'us'})
            third_words = sum(len(sent) for sent in third_sents)
            fp_trajectory.append(fp / (third_words + 1))
        
        # Trend
        fp_trend = fp_trajectory[-1] - fp_trajectory[0]
        features = [fp_trend]
        
        # Track complexity
        complexity_trajectory = []
        for third_sents in thirds:
            subordinates = sum(1 for sent in third_sents for token in sent 
                             if token.dep_ in ['ccomp', 'xcomp', 'advcl'])
            third_words = sum(len(sent) for sent in third_sents)
            complexity_trajectory.append(subordinates / (third_words + 1))
        
        complexity_trend = complexity_trajectory[-1] - complexity_trajectory[0]
        features.append(complexity_trend)
        
        # Track sentiment
        sentiment_trajectory = []
        for third_sents in thirds:
            pos = sum(1 for sent in third_sents for token in sent 
                     if token.pos_ == 'ADJ' and token.lemma_ in {'good', 'great', 'wonderful'})
            neg = sum(1 for sent in third_sents for token in sent 
                     if token.pos_ == 'ADJ' and token.lemma_ in {'bad', 'terrible', 'awful'})
            third_words = sum(len(sent) for sent in third_sents)
            sentiment_trajectory.append((pos - neg) / (third_words + 1))
        
        sentiment_trend = sentiment_trajectory[-1] - sentiment_trajectory[0]
        features.append(sentiment_trend)
        
        # Variability measures
        features.append(np.std(fp_trajectory))
        features.append(np.std(complexity_trajectory))
        features.append(np.std(sentiment_trajectory))
        
        # Consistency (inverse variability)
        features.append(1 / (1 + np.std(fp_trajectory)))
        features.append(1 / (1 + np.std(complexity_trajectory)))
        features.append(1 / (1 + np.std(sentiment_trajectory)))
        
        return features
    
    def _extract_credibility_features(self, doc) -> List[float]:
        """Extract credibility/authority markers"""
        n_words = len(doc)
        features = []
        
        # 1. Expert terminology
        expert_terms = sum(1 for token in doc if token.lemma_ in 
                         {'research', 'study', 'analysis', 'data', 'evidence', 'findings'})
        features.append(expert_terms / n_words)
        
        # 2. Hedging (epistemic uncertainty)
        hedging = sum(1 for token in doc if token.lemma_ in 
                    {'maybe', 'possibly', 'perhaps', 'probably', 'likely', 'might', 'could', 'may'})
        features.append(hedging / n_words)
        
        # 3. Certainty markers
        certainty = sum(1 for token in doc if token.lemma_ in 
                      {'certainly', 'definitely', 'absolutely', 'clearly', 'obviously', 'undoubtedly'})
        features.append(certainty / n_words)
        
        # 4. Hedging-certainty balance
        total_epistemic = hedging + certainty
        features.append(certainty / total_epistemic if total_epistemic > 0 else 0.5)
        
        # 5. Citation patterns ("according to", "research shows")
        citations = len(re.findall(r'according to|research shows|studies indicate', text.lower()))
        features.append(citations / n_words)
        
        # 6. Credentials (Dr., Prof., expert)
        credentials = sum(1 for token in doc if token.text in {'Dr.', 'Prof.'} or 
                        token.lemma_ in {'expert', 'specialist', 'researcher', 'professor'})
        features.append(credentials / n_words)
        
        # 7. Technical jargon (long words)
        long_words = sum(1 for token in doc if len(token.text) > 12)
        features.append(long_words / n_words)
        
        # 8. Precision language (numbers)
        numbers = sum(1 for token in doc if token.pos_ == 'NUM' or token.like_num)
        features.append(numbers / n_words)
        
        # 9. Evidence markers
        evidence = sum(1 for token in doc if token.lemma_ in 
                     {'prove', 'demonstrate', 'show', 'indicate', 'reveal', 'confirm', 'validate'})
        features.append(evidence / n_words)
        
        # 10. Overall credibility score
        credibility = (expert_terms + certainty + citations + credentials + evidence) / (n_words * 5)
        features.append(credibility)
        
        return features
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names"""
        return np.array([
            # Voice
            'ling_first_person',
            'ling_second_person',
            'ling_third_person',
            'ling_voice_entropy',
            'ling_narrative_immediacy',
            
            # Temporal
            'ling_past_tense',
            'ling_present_tense',
            'ling_future_tense',
            'ling_temporal_balance',
            'ling_progressive_aspect',
            
            # Agency
            'ling_active_voice',
            'ling_passive_voice',
            'ling_agency_ratio',
            'ling_agentless_passive',
            
            # Emotion
            'ling_positive_adjectives',
            'ling_negative_adjectives',
            'ling_sentiment_balance',
            
            # Complexity
            'ling_parse_depth_mean',
            'ling_parse_depth_max',
            'ling_subordinate_clauses',
            'ling_relative_clauses',
            'ling_coordination',
            'ling_prepositions',
            'ling_determiners',
            'ling_syntactic_complexity',
            
            # Evolution
            'ling_fp_trend',
            'ling_complexity_trend',
            'ling_sentiment_trend',
            'ling_fp_variability',
            'ling_complexity_variability',
            'ling_sentiment_variability',
            'ling_fp_consistency',
            'ling_complexity_consistency',
            'ling_sentiment_consistency',
            
            # Credibility
            'ling_expert_terminology',
            'ling_hedging',
            'ling_certainty',
            'ling_epistemic_balance',
            'ling_citations',
            'ling_credentials',
            'ling_technical_jargon',
            'ling_precision_numbers',
            'ling_evidence_markers',
            'ling_overall_credibility',
            
            # Discourse
            'ling_discourse_connectives'
        ])

