"""
Expertise/Authority Transformer

Extracts domain knowledge depth, credibility signals, epistemic stance, and authority markers.
Unlocks academic and professional domains (grants, papers, expert profiles).

Core insight: Expertise signals through technical depth, appropriate confidence, and credibility markers.
"""

import numpy as np
import re
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from .utils.input_validation import ensure_string_list


class ExpertiseAuthorityTransformer(BaseEstimator, TransformerMixin):
    """
    Extract expertise and authority features from narrative text.
    
    Captures:
    1. Domain Knowledge Depth - terminology, jargon, sophistication
    2. Credibility Signals - citations, methods, qualifications
    3. Epistemic Stance - certainty calibration, humility, nuance
    4. Authority Markers - credentials, affiliations, endorsements
    
    ~32 features total
    """
    
    def __init__(self):
        """Initialize expertise markers"""
        
        # Technical/academic terminology markers
        self.technical_markers = [
            'methodology', 'framework', 'paradigm', 'hypothesis', 'theory',
            'analysis', 'synthesis', 'empirical', 'systematic', 'comprehensive',
            'furthermore', 'moreover', 'therefore', 'thus', 'hence', 'consequently'
        ]
        
        # Field-specific jargon indicators (meta-markers)
        self.jargon_indicators = [
            r'\b\w+ology\b',  # -ology words
            r'\b\w+tion\b',   # -tion words
            r'\b\w+ism\b',    # -ism words
            r'\b\w+ness\b',   # -ness words
            r'\b[A-Z]{2,}\b'  # Acronyms
        ]
        
        # Citation and evidence markers
        self.citation_markers = [
            'study', 'research', 'paper', 'article', 'journal', 'publication',
            'et al', 'according to', 'found that', 'showed that', 'demonstrated',
            'evidence', 'data', 'findings', 'results', 'analysis', 'meta-analysis'
        ]
        
        # Method rigor indicators
        self.rigor_markers = [
            'method', 'methodology', 'approach', 'procedure', 'protocol',
            'design', 'experimental', 'control', 'sample', 'measurement',
            'statistical', 'significant', 'p-value', 'correlation', 'regression',
            'validated', 'reliable', 'replicable', 'systematic', 'rigorous'
        ]
        
        # Qualification statements
        self.qualification_markers = [
            'phd', 'doctor', 'professor', 'researcher', 'scientist', 'expert',
            'specialist', 'authority', 'pioneer', 'leader', 'founder', 'director',
            'certified', 'licensed', 'accredited', 'fellow', 'member'
        ]
        
        # Track record references
        self.track_record_markers = [
            'published', 'authored', 'developed', 'created', 'invented', 'discovered',
            'led', 'directed', 'managed', 'pioneered', 'established', 'founded',
            'years of experience', 'decades', 'extensive experience'
        ]
        
        # Epistemic humility
        self.humility_markers = [
            'may', 'might', 'could', 'possibly', 'potentially', 'suggest',
            'indicate', 'appear', 'seem', 'likely', 'preliminary', 'tentative',
            'requires further', 'more research', 'limitation', 'caveat'
        ]
        
        # Overconfidence markers
        self.overconfidence_markers = [
            'definitely', 'certainly', 'absolutely', 'undoubtedly', 'clearly',
            'obviously', 'proved', 'proven', 'fact', 'truth', 'guarantee'
        ]
        
        # Nuance markers
        self.nuance_markers = [
            'complex', 'nuanced', 'depends', 'context', 'varies', 'multifaceted',
            'however', 'although', 'while', 'whereas', 'on the other hand',
            'both', 'neither', 'spectrum', 'continuum', 'trade-off'
        ]
        
        # Institutional affiliation markers
        self.institution_markers = [
            'university', 'college', 'institute', 'laboratory', 'center',
            'department', 'school', 'academy', 'society', 'association',
            'foundation', 'organization', 'consortium'
        ]
        
        # Expert endorsement markers
        self.endorsement_markers = [
            'recognized', 'awarded', 'honored', 'distinguished', 'renowned',
            'acclaimed', 'peer-reviewed', 'cited', 'referenced', 'featured'
        ]
        
    def fit(self, X, y=None):
        """Fit transformer"""
        return self
    
    def transform(self, X):
        """Transform texts into expertise features"""
        features = []
        
        # Ensure X is list of strings
        X = ensure_string_list(X)
        
        for text in X:
            # Ensure text is string
            text = str(text) if not isinstance(text, str) else text
            text_lower = text.lower()
            words = re.findall(r'\b\w+\b', text_lower)
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            feat_dict = {}
            
            # === 1. DOMAIN KNOWLEDGE DEPTH (10 features) ===
            
            # Technical terminology density
            technical_count = sum(1 for w in words if w in self.technical_markers)
            feat_dict['technical_terminology'] = technical_count / (len(words) + 1)
            
            # Jargon/specialized language (using patterns)
            jargon_count = sum(len(re.findall(pattern, text)) for pattern in self.jargon_indicators)
            feat_dict['jargon_density'] = jargon_count / (len(words) + 1)
            
            # Acronym usage
            acronym_count = len(re.findall(r'\b[A-Z]{2,}\b', text))
            feat_dict['acronym_usage'] = acronym_count / (len(sentences) + 1)
            
            # Long words (sophistication proxy)
            long_words = [w for w in words if len(w) > 8]
            feat_dict['word_sophistication'] = len(long_words) / (len(words) + 1)
            
            # Concept density (nouns ending in -tion, -ment, -ness)
            concept_patterns = [r'\b\w+tion\b', r'\b\w+ment\b', r'\b\w+ness\b']
            concept_count = sum(len(re.findall(p, text_lower)) for p in concept_patterns)
            feat_dict['concept_density'] = concept_count / (len(words) + 1)
            
            # Vocabulary sophistication (unique words ratio)
            vocab_diversity = len(set(words)) / (len(words) + 1)
            feat_dict['vocabulary_sophistication'] = vocab_diversity
            
            # Latin/Greek roots (common in academic writing)
            latin_greek = ['per', 'ante', 'post', 'meta', 'proto', 'pseudo', 'quasi']
            latin_count = sum(1 for w in words if any(w.startswith(prefix) for prefix in latin_greek))
            feat_dict['classical_roots'] = latin_count / (len(words) + 1)
            
            # Passive voice (common in academic writing)
            passive_markers = ['was', 'were', 'been', 'being']
            passive_count = sum(1 for w in words if w in passive_markers)
            feat_dict['passive_voice_density'] = passive_count / (len(words) + 1)
            
            # Subordinate clauses (complex sentence structure)
            subordinate_markers = ['that', 'which', 'who', 'whom', 'where', 'when', 'while']
            subordinate_count = sum(1 for w in words if w in subordinate_markers)
            feat_dict['sentence_complexity'] = subordinate_count / (len(sentences) + 1)
            
            # Field-specific insider language (measured by low-frequency words)
            word_counts = Counter(words)
            rare_words = [w for w, c in word_counts.items() if c == 1 and len(w) > 5]
            feat_dict['insider_language'] = len(rare_words) / (len(words) + 1)
            
            # === 2. CREDIBILITY SIGNALS (8 features) ===
            
            # Citation patterns
            citation_count = sum(1 for w in words if w in self.citation_markers)
            feat_dict['citation_density'] = citation_count / (len(sentences) + 1)
            
            # Explicit citations (years in parentheses, et al)
            explicit_citations = len(re.findall(r'\([12]\d{3}\)|et al\.', text))
            feat_dict['explicit_citations'] = explicit_citations / (len(sentences) + 1)
            
            # Method rigor
            rigor_count = sum(1 for w in words if w in self.rigor_markers)
            feat_dict['methodological_rigor'] = rigor_count / (len(words) + 1)
            
            # Quantitative evidence (statistics, numbers)
            number_count = len(re.findall(r'\b\d+\.?\d*\b', text))
            feat_dict['quantitative_evidence'] = number_count / (len(sentences) + 1)
            
            # Qualification statements
            qual_count = sum(1 for w in words if w in self.qualification_markers)
            feat_dict['qualification_signals'] = qual_count / (len(words) + 1)
            
            # Track record references
            track_record_count = sum(1 for w in words if w in self.track_record_markers)
            feat_dict['track_record_mentions'] = track_record_count / (len(words) + 1)
            
            # Third-party validation
            validation_words = ['validated', 'verified', 'confirmed', 'endorsed', 'approved']
            validation_count = sum(1 for w in words if w in validation_words)
            feat_dict['external_validation'] = validation_count / (len(words) + 1)
            
            # Peer review mentions
            peer_review_markers = ['peer-reviewed', 'peer reviewed', 'refereed', 'reviewed by']
            peer_review = sum(1 for marker in peer_review_markers if marker in text_lower)
            feat_dict['peer_review_signals'] = float(peer_review > 0)
            
            # === 3. EPISTEMIC STANCE (8 features) ===
            
            # Epistemic humility
            humility_count = sum(1 for w in words if w in self.humility_markers)
            feat_dict['epistemic_humility'] = humility_count / (len(words) + 1)
            
            # Overconfidence
            overconfident_count = sum(1 for w in words if w in self.overconfidence_markers)
            feat_dict['overconfidence'] = overconfident_count / (len(words) + 1)
            
            # Confidence calibration (humility vs overconfidence balance)
            feat_dict['confidence_calibration'] = humility_count / (overconfident_count + 1)
            
            # Nuance and caveats
            nuance_count = sum(1 for w in words if w in self.nuance_markers)
            feat_dict['nuance_acknowledgment'] = nuance_count / (len(words) + 1)
            
            # Limitation acknowledgment
            limitation_words = ['limitation', 'limit', 'constraint', 'caveat', 'however', 'but']
            limitation_count = sum(1 for w in words if w in limitation_words)
            feat_dict['limitation_acknowledgment'] = limitation_count / (len(sentences) + 1)
            
            # Certainty gradient (measured by modal verb usage)
            certainty_levels = {
                'must': 1.0, 'will': 0.9, 'should': 0.7, 'can': 0.5,
                'may': 0.3, 'might': 0.2, 'could': 0.2
            }
            modal_scores = [certainty_levels.get(w, 0) for w in words]
            feat_dict['average_certainty'] = np.mean(modal_scores) if modal_scores else 0.5
            
            # Conditional reasoning (if-then statements)
            conditional_count = len(re.findall(r'\bif\b.*\bthen\b', text_lower))
            feat_dict['conditional_reasoning'] = conditional_count / (len(sentences) + 1)
            
            # Falsifiability (testable claims)
            testable_markers = ['test', 'predict', 'expect', 'hypothesis', 'falsifiable']
            testable_count = sum(1 for w in words if w in testable_markers)
            feat_dict['falsifiability'] = testable_count / (len(sentences) + 1)
            
            # === 4. AUTHORITY MARKERS (6 features) ===
            
            # Institutional affiliations
            institution_count = sum(1 for w in words if w in self.institution_markers)
            feat_dict['institutional_affiliation'] = institution_count / (len(words) + 1)
            
            # Credential signals
            credential_patterns = [r'\bphd\b', r'\bmd\b', r'\bdr\.', r'\bprof\.', r'\bdr\b']
            credential_count = sum(len(re.findall(p, text_lower)) for p in credential_patterns)
            feat_dict['credential_signals'] = credential_count / (len(sentences) + 1)
            
            # Publication history
            publication_words = ['published', 'authored', 'co-authored', 'journal', 'book']
            publication_count = sum(1 for w in words if w in publication_words)
            feat_dict['publication_history'] = publication_count / (len(words) + 1)
            
            # Expert endorsements
            endorsement_count = sum(1 for w in words if w in self.endorsement_markers)
            feat_dict['expert_endorsement'] = endorsement_count / (len(words) + 1)
            
            # Awards and honors
            award_words = ['award', 'prize', 'honor', 'grant', 'fellowship', 'medal']
            award_count = sum(1 for w in words if w in award_words)
            feat_dict['awards_honors'] = award_count / (len(words) + 1)
            
            # Years of experience
            experience_patterns = [r'\b\d+\+?\s*years?', r'decades? of', r'extensive experience']
            experience_count = sum(len(re.findall(p, text_lower)) for p in experience_patterns)
            feat_dict['experience_signals'] = experience_count / (len(sentences) + 1)
            
            features.append(list(feat_dict.values()))
        
        return np.array(features)
    
    def get_feature_names_out(self, input_features=None):
        """Get feature names"""
        names = [
            # Domain knowledge (10)
            'technical_terminology', 'jargon_density', 'acronym_usage',
            'word_sophistication', 'concept_density', 'vocabulary_sophistication',
            'classical_roots', 'passive_voice_density', 'sentence_complexity', 'insider_language',
            
            # Credibility signals (8)
            'citation_density', 'explicit_citations', 'methodological_rigor',
            'quantitative_evidence', 'qualification_signals', 'track_record_mentions',
            'external_validation', 'peer_review_signals',
            
            # Epistemic stance (8)
            'epistemic_humility', 'overconfidence', 'confidence_calibration',
            'nuance_acknowledgment', 'limitation_acknowledgment', 'average_certainty',
            'conditional_reasoning', 'falsifiability',
            
            # Authority markers (6)
            'institutional_affiliation', 'credential_signals', 'publication_history',
            'expert_endorsement', 'awards_honors', 'experience_signals'
        ]
        
        return np.array([f'expertise_authority_{n}' for n in names])

