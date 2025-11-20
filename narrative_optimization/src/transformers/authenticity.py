"""
Authenticity/Truth Transformer

Extracts specificity, consistency, authenticity markers, and truth signals.
Critical for high-stakes domains where deception/exaggeration matters.

Core insight: Authentic narratives have verifiable details, consistent voice,
and appropriate confidence calibration.
"""

import numpy as np
import re
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Dict, Any


class AuthenticityTransformer(BaseEstimator, TransformerMixin):
    """
    Extract authenticity and truth signals from narrative text.
    
    Captures:
    1. Specificity vs. Vagueness - concrete details, quantification
    2. Consistency - contradictions, alignment, tone
    3. Authenticity Markers - unique voice, imperfection, anecdotes
    4. Truth Signals - verifiability, evidence, appropriate confidence
    
    ~30 features total
    """
    
    def __init__(self):
        """Initialize authenticity markers and patterns"""
        
        # Specificity markers
        self.concrete_detail_markers = [
            r'\b\d+\b',  # Numbers
            r'\$\d+',  # Money
            r'\d+%',  # Percentages
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',  # Dates
            r'\b\d{4}\b',  # Years
            r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',  # Days
        ]
        
        # Vagueness/hedging markers
        self.hedging_words = [
            'maybe', 'perhaps', 'possibly', 'probably', 'might', 'could', 'would',
            'sort of', 'kind of', 'somewhat', 'relatively', 'fairly', 'rather',
            'generally', 'typically', 'usually', 'often', 'sometimes', 'occasionally'
        ]
        
        # Qualifiers and uncertainty
        self.uncertainty_markers = [
            'unclear', 'uncertain', 'unsure', 'unknown', 'unclear', 'ambiguous',
            'vague', 'roughly', 'approximately', 'about', 'around', 'nearly',
            'almost', 'practically'
        ]
        
        # Abstract/generic claims
        self.abstract_words = [
            'thing', 'stuff', 'something', 'anything', 'everything', 'nothing',
            'someone', 'anyone', 'everyone', 'nobody', 'somewhere', 'anywhere',
            'everywhere', 'nowhere', 'good', 'bad', 'nice', 'great', 'amazing'
        ]
        
        # Verifiable claim markers
        self.evidence_markers = [
            'study', 'research', 'data', 'evidence', 'proof', 'statistic',
            'according to', 'source', 'reference', 'citation', 'documented',
            'measured', 'tested', 'verified', 'confirmed', 'validated'
        ]
        
        # Clichés and templates (inauthenticity)
        self.cliches = [
            'at the end of the day', 'think outside the box', 'game changer',
            'paradigm shift', 'synergy', 'leverage', 'bandwidth', 'circle back',
            'take it to the next level', 'low-hanging fruit', 'move the needle',
            'best of breed', 'bleeding edge', 'core competency', 'deep dive',
            'drinking the kool-aid', 'giving 110%', 'hit the ground running',
            'it is what it is', 'touch base', 'win-win', 'value-add'
        ]
        
        # Overpromising patterns
        self.overpromising_markers = [
            'guarantee', 'guaranteed', 'never', 'always', 'absolutely', 'definitely',
            'certainly', 'surely', 'obviously', 'clearly', 'undoubtedly',
            'revolutionary', 'unprecedented', 'game-changing', 'transformative',
            'best ever', 'perfect', 'flawless', 'ultimate', 'impossible to',
            'nothing like', 'first of its kind', 'one of a kind'
        ]
        
        # Authenticity markers (vulnerability, imperfection)
        self.authenticity_markers = [
            'honestly', 'truth', 'admit', 'confess', 'acknowledge', 'realize',
            'learned', 'mistake', 'wrong', 'failed', 'struggled', 'difficult',
            'challenge', 'flaw', 'limitation', 'weakness', 'imperfect'
        ]
        
        # First person (personal anecdote)
        self.first_person = ['i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours']
        
        # Contradiction indicators
        self.contradiction_words = [
            'but', 'however', 'although', 'though', 'yet', 'nevertheless',
            'nonetheless', 'despite', 'in spite of', 'on the other hand',
            'conversely', 'alternatively', 'whereas'
        ]
        
    def fit(self, X, y=None):
        """Fit transformer (no-op, lexicon-based)"""
        return self
    
    def transform(self, X):
        """
        Transform texts into authenticity features.
        
        Parameters
        ----------
        X : array-like of strings
            Narrative texts
            
        Returns
        -------
        features : ndarray of shape (n_samples, 30)
            Authenticity features
        """
        features = []
        
        for text in X:
            text_lower = text.lower()
            words = re.findall(r'\b\w+\b', text_lower)
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            feat_dict = {}
            
            # === 1. SPECIFICITY VS. VAGUENESS (8 features) ===
            
            # Concrete details (numbers, dates, specific names)
            concrete_count = sum(len(re.findall(pattern, text_lower)) for pattern in self.concrete_detail_markers)
            feat_dict['concrete_detail_density'] = concrete_count / (len(words) + 1)
            
            # Abstract/generic words
            abstract_count = sum(1 for word in words if word in self.abstract_words)
            feat_dict['abstract_claim_density'] = abstract_count / (len(words) + 1)
            
            # Specificity ratio
            feat_dict['specificity_ratio'] = concrete_count / (abstract_count + 1)
            
            # Quantification usage (numbers and metrics)
            number_count = len(re.findall(r'\b\d+\b', text))
            feat_dict['quantification_usage'] = number_count / (len(sentences) + 1)
            
            # Hedging language
            hedging_count = sum(1 for word in words if word in self.hedging_words)
            feat_dict['hedging_density'] = hedging_count / (len(words) + 1)
            
            # Uncertainty markers
            uncertainty_count = sum(1 for word in words if word in self.uncertainty_markers)
            feat_dict['uncertainty_level'] = uncertainty_count / (len(words) + 1)
            
            # Precision (specific vs. vague numbers)
            precise_numbers = len(re.findall(r'\b\d{2,}\b', text))  # Multi-digit numbers are more precise
            round_numbers = len(re.findall(r'\b\d*00+\b', text))  # Round numbers might be estimates
            feat_dict['number_precision'] = precise_numbers / (round_numbers + 1)
            
            # Adjective density (excessive adjectives = lack of substance)
            adjective_pattern = r'\b(very|extremely|incredibly|amazingly|absolutely|totally|completely|utterly)\s+\w+'
            adjective_count = len(re.findall(adjective_pattern, text_lower))
            feat_dict['excessive_adjectives'] = adjective_count / (len(sentences) + 1)
            
            # === 2. CONSISTENCY (7 features) ===
            
            # Contradiction density (but, however, although)
            contradiction_count = sum(1 for word in words if word in self.contradiction_words)
            feat_dict['contradiction_markers'] = contradiction_count / (len(sentences) + 1)
            
            # Tone consistency (variation in sentence sentiment - use simple proxy)
            sentence_lengths = [len(s.split()) for s in sentences]
            if sentence_lengths:
                feat_dict['structural_consistency'] = 1.0 - (np.std(sentence_lengths) / (np.mean(sentence_lengths) + 1))
            else:
                feat_dict['structural_consistency'] = 1.0
            
            # Temporal consistency (past/present/future mix)
            past_tense = len(re.findall(r'\b\w+ed\b', text_lower))
            present_tense = len(re.findall(r'\b(am|is|are|being)\b', text_lower))
            future_tense = len(re.findall(r'\b(will|shall|going to)\b', text_lower))
            
            total_tense = past_tense + present_tense + future_tense + 1
            tense_entropy = -sum((count/total_tense) * np.log2(count/total_tense + 0.001) 
                               for count in [past_tense, present_tense, future_tense] if count > 0)
            feat_dict['temporal_consistency'] = 1.0 - (tense_entropy / 2.0)  # Normalize
            
            # Voice consistency (first person vs. third person)
            first_person_count = sum(1 for word in words if word in self.first_person)
            third_person_count = sum(1 for word in words if word in ['he', 'she', 'they', 'it'])
            
            if first_person_count + third_person_count > 0:
                voice_dominance = max(first_person_count, third_person_count) / (first_person_count + third_person_count)
                feat_dict['voice_consistency'] = voice_dominance
            else:
                feat_dict['voice_consistency'] = 1.0
            
            # Claim-evidence alignment (claims followed by support)
            claim_words = ['believe', 'think', 'claim', 'assert', 'argue', 'maintain']
            claim_count = sum(1 for word in words if word in claim_words)
            evidence_count = sum(1 for word in words if word in self.evidence_markers)
            
            feat_dict['claim_evidence_ratio'] = evidence_count / (claim_count + 1)
            
            # Repetition (excessive repetition = padding/lack of substance)
            word_counts = Counter(words)
            common_words = [w for w, c in word_counts.items() if c > 2 and len(w) > 3]
            feat_dict['excessive_repetition'] = len(common_words) / (len(set(words)) + 1)
            
            # Logical flow (sentence connectors)
            connectors = ['therefore', 'thus', 'hence', 'consequently', 'because', 'since', 
                         'as a result', 'for this reason', 'accordingly']
            connector_count = sum(1 for word in words if word in connectors)
            feat_dict['logical_flow'] = connector_count / (len(sentences) + 1)
            
            # === 3. AUTHENTICITY MARKERS (8 features) ===
            
            # Unique voice (vocabulary diversity)
            vocab_size = len(set(words))
            feat_dict['vocabulary_diversity'] = vocab_size / (len(words) + 1)
            
            # Imperfection acknowledgment
            authenticity_count = sum(1 for word in words if word in self.authenticity_markers)
            feat_dict['imperfection_acknowledgment'] = authenticity_count / (len(words) + 1)
            
            # Personal anecdotes (first person + narrative)
            first_person_density = first_person_count / (len(words) + 1)
            feat_dict['personal_anecdote_density'] = first_person_density
            
            # Cliché density (inverse = authenticity)
            cliche_count = sum(1 for cliche in self.cliches if cliche in text_lower)
            feat_dict['cliche_density'] = cliche_count / (len(sentences) + 1)
            feat_dict['originality_score'] = 1.0 - min(1.0, cliche_count / 3)
            
            # Specific storytelling (narrative elements)
            narrative_markers = ['when', 'then', 'after', 'before', 'during', 'while', 'until']
            narrative_count = sum(1 for word in words if word in narrative_markers)
            feat_dict['narrative_specificity'] = narrative_count / (len(sentences) + 1)
            
            # Conversational tone (questions, direct address)
            question_count = text.count('?')
            feat_dict['conversational_tone'] = question_count / (len(sentences) + 1)
            
            # Length variance (uniform length = templated)
            if len(sentence_lengths) > 1:
                feat_dict['length_variance'] = np.std(sentence_lengths) / (np.mean(sentence_lengths) + 1)
            else:
                feat_dict['length_variance'] = 0.0
            
            # === 4. TRUTH SIGNALS (7 features) ===
            
            # Verifiable claims
            verifiable_count = evidence_count  # Reuse from earlier
            feat_dict['verifiable_claim_density'] = verifiable_count / (len(sentences) + 1)
            
            # Evidence citations
            citation_pattern = r'\([12]\d{3}\)|et al\.|according to'
            citation_count = len(re.findall(citation_pattern, text))
            feat_dict['citation_density'] = citation_count / (len(sentences) + 1)
            
            # Appropriate qualification
            # High certainty claims should have evidence; uncertain claims should hedge
            overpromising_count = sum(1 for word in words if word in self.overpromising_markers)
            feat_dict['overpromising_density'] = overpromising_count / (len(words) + 1)
            
            # Confidence calibration (certainty matched to evidence)
            certainty_words = ['definitely', 'certainly', 'absolutely', 'clearly', 'obviously']
            certainty_count = sum(1 for word in words if word in certainty_words)
            
            # Well-calibrated: high certainty = high evidence, OR low certainty = low evidence
            if certainty_count > hedging_count:
                # High certainty - should have evidence
                feat_dict['confidence_calibration'] = evidence_count / (certainty_count + 1)
            else:
                # Appropriate uncertainty
                feat_dict['confidence_calibration'] = 1.0
            
            # Methodological rigor (for scientific/technical claims)
            rigor_markers = ['method', 'approach', 'process', 'procedure', 'analysis', 
                            'measurement', 'experiment', 'test', 'validation']
            rigor_count = sum(1 for word in words if word in rigor_markers)
            feat_dict['methodological_rigor'] = rigor_count / (len(sentences) + 1)
            
            # Falsifiability (testable claims)
            testable_markers = ['predict', 'expect', 'should', 'would', 'if.*then', 'result in']
            testable_count = sum(1 for marker in testable_markers if marker in text_lower)
            feat_dict['falsifiability'] = testable_count / (len(sentences) + 1)
            
            # Nuance (acknowledging complexity)
            nuance_markers = ['complex', 'nuanced', 'depends', 'varies', 'context', 
                            'sometimes', 'may', 'might', 'could']
            nuance_count = sum(1 for word in words if word in nuance_markers)
            feat_dict['nuance_acknowledgment'] = nuance_count / (len(words) + 1)
            
            # Convert to feature vector
            features.append(list(feat_dict.values()))
        
        return np.array(features)
    
    def get_feature_names_out(self, input_features=None):
        """Get feature names for output"""
        feature_names = [
            # Specificity (8)
            'concrete_detail_density',
            'abstract_claim_density',
            'specificity_ratio',
            'quantification_usage',
            'hedging_density',
            'uncertainty_level',
            'number_precision',
            'excessive_adjectives',
            
            # Consistency (7)
            'contradiction_markers',
            'structural_consistency',
            'temporal_consistency',
            'voice_consistency',
            'claim_evidence_ratio',
            'excessive_repetition',
            'logical_flow',
            
            # Authenticity markers (8)
            'vocabulary_diversity',
            'imperfection_acknowledgment',
            'personal_anecdote_density',
            'cliche_density',
            'originality_score',
            'narrative_specificity',
            'conversational_tone',
            'length_variance',
            
            # Truth signals (7)
            'verifiable_claim_density',
            'citation_density',
            'overpromising_density',
            'confidence_calibration',
            'methodological_rigor',
            'falsifiability',
            'nuance_acknowledgment'
        ]
        
        return np.array([f'authenticity_{name}' for name in feature_names])

