"""
Judicial Rhetoric Transformer

Analyzes judicial writing style and rhetorical patterns:
- Formality level and register
- Voice (active vs passive, first-person plural)
- Certainty vs hedging
- Legal terminology density
- Sentence structure complexity
- Opinion coherence and flow

Features: 40

Author: Narrative Optimization Framework
Date: November 17, 2025
"""

import numpy as np
import re
from typing import List
from sklearn.base import BaseEstimator, TransformerMixin
from ..utils.input_validation import ensure_string_list


class JudicialRhetoricTransformer(BaseEstimator, TransformerMixin):
    """
    Extracts judicial rhetoric and writing style features.
    
    Judicial opinions have distinct rhetorical patterns:
    - High formality ("We hold" not "I think")
    - Passive voice for objectivity
    - Legal terminology (jurisdiction, remand, etc.)
    - Measured certainty (not overconfident)
    - Structured presentation
    
    Tests: Does better judicial writing → more influence?
    
    Features: 40
    """
    
    def __init__(self):
        # Formality markers
        self.formal_markers = [
            r'\bpetitioner\b', r'\brespondent\b', r'\bappellant\b', r'\bappellee\b',
            r'\bherein\b', r'\baforesaid\b', r'\bthereof\b', r'\bwherein\b'
        ]
        
        # Informal markers (should be rare)
        self.informal_markers = [
            r'\bkinda\b', r'\bsorta\b', r'\bstuff\b', r'\bthings\b', r'\bguys\b'
        ]
        
        # Legal terminology
        self.legal_terms = [
            r'\bjurisdiction\b', r'\b remand\b', r'\baffirm\b', r'\breverse\b',
            r'\bvacate\b', r'\bsubject matter\b', r'\bstanding\b', r'\bjusticiab',
            r'\bmootness\b', r'\bripeness\b', r'\bdicta\b', r'\bholding\b'
        ]
        
        # Certainty markers
        self.strong_certainty = [
            r'\bclearly\b', r'\bobviously\b', r'\bundoubtedly\b', r'\binescapably\b',
            r'\bplainly\b', r'\bunmistakab', r'\bincontrovertib'
        ]
        
        self.weak_certainty = [
            r'\bmay\b', r'\bmight\b', r'\bcould\b', r'\bperhaps\b',
            r'\bpossibly\b', r'\bseems?\b', r'\bappears?\b'
        ]
    
    def fit(self, X, y=None):
        """Fit transformer."""
        return self
    
    def transform(self, X):
        """
        Transform legal texts to judicial rhetoric features.
        
        Parameters
        ----------
        X : list of str
            Legal opinion texts
        
        Returns
        -------
        features : ndarray, shape (n_samples, 40)
        """
        X = ensure_string_list(X)
        features = []
        
        for text in X:
            feat = self._extract_features(text)
            features.append(feat)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_features(self, text: str) -> List[float]:
        """Extract judicial rhetoric features."""
        if not text or len(text) < 100:
            return [0.0] * 40
        
        text_lower = text.lower()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        n_sentences = len(sentences)
        words = text.split()
        n_words = len(words)
        
        features = []
        
        # 1-5: Formality level
        formal_count = self._count_patterns(text_lower, self.formal_markers)
        features.append(formal_count / (n_sentences + 1))
        
        informal_count = self._count_patterns(text_lower, self.informal_markers)
        features.append(informal_count / (n_sentences + 1))
        
        # Formality score (high = formal)
        if formal_count + informal_count > 0:
            formality = formal_count / (formal_count + informal_count + 1)
        else:
            formality = 0.8  # Default high for judicial writing
        features.append(formality)
        
        # Legal jargon density
        legal_term_count = self._count_patterns(text_lower, self.legal_terms)
        features.append(legal_term_count / (n_sentences + 1))
        features.append(legal_term_count / (n_words + 1))
        
        # 6-10: Voice patterns
        # First-person plural (judicial "we")
        we_pattern = r'\b[Ww]e\s+(hold|find|conclude|affirm|reverse|grant|deny)'
        we_count = len(re.findall(we_pattern, text))
        features.append(we_count / (n_sentences + 1))
        
        # First-person singular (rare in opinions, used in dissents)
        i_pattern = r'\b[Ii]\s+(believe|think|would|disagree|dissent)'
        i_count = len(re.findall(i_pattern, text))
        features.append(i_count / (n_sentences + 1))
        
        # Passive voice (objectivity marker)
        passive_markers = [r'was\s+\w+ed\b', r'were\s+\w+ed\b', r'is\s+\w+ed\b', r'are\s+\w+ed\b']
        passive_count = sum(len(re.findall(m, text_lower)) for m in passive_markers)
        features.append(passive_count / (n_sentences + 1))
        
        # Active vs passive ratio
        # Rough heuristic: passive > active in judicial writing
        if passive_count + we_count > 0:
            active_ratio = we_count / (passive_count + we_count)
        else:
            active_ratio = 0.3  # Default low (more passive)
        features.append(active_ratio)
        
        # Impersonal constructions ("It is", "There is")
        impersonal_pattern = r'\b(It is|There is|There are)\b'
        impersonal_count = len(re.findall(impersonal_pattern, text))
        features.append(impersonal_count / (n_sentences + 1))
        
        # 11-15: Certainty vs hedging
        strong_cert_count = self._count_patterns(text_lower, self.strong_certainty)
        features.append(strong_cert_count / (n_sentences + 1))
        
        weak_cert_count = self._count_patterns(text_lower, self.weak_certainty)
        features.append(weak_cert_count / (n_sentences + 1))
        
        # Certainty ratio (strong vs weak)
        if strong_cert_count + weak_cert_count > 0:
            certainty_ratio = strong_cert_count / (strong_cert_count + weak_cert_count)
        else:
            certainty_ratio = 0.6  # Default moderate-high
        features.append(certainty_ratio)
        
        # Qualification density (all hedging)
        qualifiers = [r'\bsome\b', r'\bcertain\b', r'\boften\b', r'\bsometimes\b', r'\btypically\b']
        qualifier_count = sum(self._count_pattern(text_lower, q) for q in qualifiers)
        features.append(qualifier_count / (n_words + 1))
        
        # Absolute language (never, always, all)
        absolute_pattern = r'\b(never|always|all|none|every|no one)\b'
        absolute_count = self._count_pattern(text_lower, absolute_pattern)
        features.append(absolute_count / (n_words + 1))
        
        # 16-20: Sentence structure
        # Average sentence length
        avg_sent_length = n_words / (n_sentences + 1)
        features.append(min(1.0, avg_sent_length / 35))  # Judicial avg ~25-35 words
        
        # Sentence length variance (some short, some long)
        sent_lengths = [len(s.split()) for s in sentences]
        sent_length_std = np.std(sent_lengths) if sent_lengths else 0
        features.append(min(1.0, sent_length_std / 20))
        
        # Short declarative sentences (emphasis)
        short_sentences = sum(1 for s in sentences if len(s.split()) < 15)
        features.append(short_sentences / (n_sentences + 1))
        
        # Complex sentences (multiple clauses)
        complex_markers = [',', ';', '—', 'which', 'that', 'who']
        complex_count = sum(1 for s in sentences 
                          if sum(s.count(m) for m in complex_markers) > 3)
        features.append(complex_count / (n_sentences + 1))
        
        # Semicolon use (sophisticated punctuation)
        semicolon_count = text.count(';')
        features.append(semicolon_count / (n_sentences + 1))
        
        # 21-25: Rhetorical devices
        # Rhetorical questions (rare but powerful)
        rhetorical_q = text.count('?')
        features.append(rhetorical_q / (n_sentences + 1))
        
        # Parallelism (not only...but also)
        parallel_pattern = r'\bnot only.*but also\b'
        parallel_count = len(re.findall(parallel_pattern, text_lower))
        features.append(parallel_count / (n_sentences + 1))
        
        # Triadic structure (X, Y, and Z)
        triad_pattern = r'\w+,\s+\w+,\s+and\s+\w+'
        triad_count = len(re.findall(triad_pattern, text_lower))
        features.append(triad_count / (n_sentences + 1))
        
        # Contrasts (not X but Y)
        contrast_pattern = r'\bnot\s+\w+\s+but\s+\w+'
        contrast_count = len(re.findall(contrast_pattern, text_lower))
        features.append(contrast_count / (n_sentences + 1))
        
        # Emphasis through repetition
        # Count repeated words (excluding common words)
        word_counts = {}
        for word in words:
            if len(word) > 4 and word.lower() not in {'that', 'this', 'which', 'their', 'there', 'these', 'those'}:
                word_lower = word.lower()
                word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
        
        repeated_words = sum(1 for count in word_counts.values() if count > 2)
        features.append(repeated_words / (n_sentences + 1))
        
        # 26-30: Opinion structure
        # Section markers (Part I, Part II, etc.)
        section_pattern = r'\b(Part|Section|I{1,3}|IV|V)\b'
        section_count = len(re.findall(section_pattern, text))
        features.append(section_count / (n_sentences + 1))
        
        # Numbered subsections
        numbered_pattern = r'\b\d+\.\s+[A-Z]'
        numbered_count = len(re.findall(numbered_pattern, text))
        features.append(numbered_count / (n_sentences + 1))
        
        # Topic sentences (First, Second, Finally)
        topic_pattern = r'\b(First|Second|Third|Fourth|Finally)\b'
        topic_count = len(re.findall(topic_pattern, text))
        features.append(topic_count / (n_sentences + 1))
        
        # Transitional phrases
        transition_pattern = r'\b(Moreover|Furthermore|However|Nevertheless|Accordingly|Thus)\b'
        transition_count = len(re.findall(transition_pattern, text))
        features.append(transition_count / (n_sentences + 1))
        
        # Concluding language
        conclusion_pattern = r'\b(In conclusion|To sum|In summary|Therefore|For these reasons)\b'
        conclusion_count = len(re.findall(conclusion_pattern, text, re.IGNORECASE))
        features.append(conclusion_count / (n_sentences + 1))
        
        # 31-35: Readability proxies
        # Latinate vs Anglo-Saxon (formality)
        latinate_markers = [r'tion\b', r'sion\b', r'ment\b', r'ence\b', r'ance\b']
        latinate_count = sum(self._count_pattern(text_lower, m) for m in latinate_markers)
        features.append(latinate_count / (n_words + 1))
        
        # Nominalizations (verbosity indicator)
        nominalization_pattern = r'\b\w+(tion|sion|ment|ance|ence)\b'
        nominalization_count = len(re.findall(nominalization_pattern, text_lower))
        features.append(nominalization_count / (n_words + 1))
        
        # Average word length (complexity)
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        features.append(min(1.0, avg_word_length / 8))
        
        # Long words (>10 chars, legal terminology)
        long_words = sum(1 for w in words if len(w) > 10)
        features.append(long_words / (n_words + 1))
        
        # Lexical diversity (vocabulary richness)
        unique_words = len(set(w.lower() for w in words))
        lexical_diversity = unique_words / (n_words + 1)
        features.append(lexical_diversity)
        
        # 36-40: Opinion coherence
        # Demonstrative references (this, that, these, those)
        demonstrative_pattern = r'\b(this|that|these|those)\s+\w+'
        demonstrative_count = len(re.findall(demonstrative_pattern, text_lower))
        features.append(demonstrative_count / (n_sentences + 1))
        
        # Backward references (former, latter, above, below)
        reference_pattern = r'\b(former|latter|above|below|supra|infra)\b'
        reference_count = self._count_pattern(text_lower, reference_pattern)
        features.append(reference_count / (n_sentences + 1))
        
        # Logical flow markers (therefore, thus, hence)
        flow_pattern = r'\b(therefore|thus|hence|accordingly|consequently)\b'
        flow_count = self._count_pattern(text_lower, flow_pattern)
        features.append(flow_count / (n_sentences + 1))
        
        # Consistency in terminology (reuse of key terms)
        # Measure by looking at top words repeated
        if len(word_counts) > 0:
            max_repetition = max(word_counts.values())
            features.append(min(1.0, max_repetition / 20))
        else:
            features.append(0.0)
        
        # Paragraph structure proxy (line breaks)
        # For this, we'd need actual formatting, so use sentence clustering heuristic
        # Assume paragraphs ~5-7 sentences
        paragraph_estimate = n_sentences / 6
        features.append(min(1.0, paragraph_estimate / 30))
        
        return features
    
    def _count_pattern(self, text: str, pattern: str) -> int:
        """Count regex pattern."""
        try:
            return len(re.findall(pattern, text, re.IGNORECASE))
        except:
            return 0
    
    def _count_patterns(self, text: str, patterns: List[str]) -> int:
        """Count multiple patterns."""
        return sum(self._count_pattern(text, p) for p in patterns)
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names."""
        names = [
            # Formality (1-5)
            'formal_language_density',
            'informal_language_density',
            'formality_score',
            'legal_terminology_density_sent',
            'legal_terminology_density_word',
            
            # Voice (6-10)
            'judicial_we_density',
            'first_person_singular',
            'passive_voice_density',
            'active_passive_ratio',
            'impersonal_construction_density',
            
            # Certainty (11-15)
            'strong_certainty_density',
            'weak_certainty_density',
            'certainty_ratio',
            'qualifier_density',
            'absolute_language_density',
            
            # Sentence structure (16-20)
            'avg_sentence_length_normalized',
            'sentence_length_variance',
            'short_declarative_ratio',
            'complex_sentence_ratio',
            'semicolon_sophistication',
            
            # Rhetorical devices (21-25)
            'rhetorical_question_density',
            'parallelism_density',
            'triadic_structure_density',
            'contrast_structure_density',
            'emphasis_through_repetition',
            
            # Opinion structure (26-30)
            'section_marker_density',
            'numbered_subsection_density',
            'topic_sentence_markers',
            'transitional_phrase_density',
            'concluding_language_density',
            
            # Readability (31-35)
            'latinate_vocabulary',
            'nominalization_density',
            'avg_word_length_normalized',
            'long_word_density',
            'lexical_diversity',
            
            # Coherence (36-40)
            'demonstrative_reference_density',
            'backward_reference_density',
            'logical_flow_marker_density',
            'terminological_consistency',
            'paragraph_structure_estimate'
        ]
        
        return np.array(names[:40])


if __name__ == '__main__':
    test_text = """
    We hold that the petitioner has established standing to bring this claim.
    The Constitution clearly protects this fundamental right. As we established
    in our prior cases, the test is well-settled. Therefore, we affirm the
    judgment of the Court of Appeals. Justice Smith dissents.
    """
    
    transformer = JudicialRhetoricTransformer()
    features = transformer.transform([test_text])
    feature_names = transformer.get_feature_names_out()
    
    print("\nJudicial Rhetoric Features:")
    print("="*80)
    for name, value in zip(feature_names, features[0]):
        if value > 0.01:
            print(f"{name:40s}: {value:.4f}")
    print("="*80)

