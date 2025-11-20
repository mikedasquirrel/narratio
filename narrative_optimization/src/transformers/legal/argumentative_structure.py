"""
Argumentative Structure Transformer

Analyzes legal argumentative structure:
- Claim-evidence-warrant patterns
- Logical connectives and reasoning chains
- Counterargument acknowledgment and rebuttal
- Syllogistic structure
- Burden of proof allocation

Features: 60

Author: Narrative Optimization Framework
Date: November 17, 2025
"""

import numpy as np
import re
from typing import List
from sklearn.base import BaseEstimator, TransformerMixin
from ..utils.input_validation import ensure_string_list


class ArgumentativeStructureTransformer(BaseEstimator, TransformerMixin):
    """
    Extracts argumentative structure features from legal text.
    
    Key components:
    - Claims (assertions being made)
    - Evidence (support for claims)
    - Warrants (logical connections)
    - Rebuttals (counterargument handling)
    - Logical structure (syllogisms, if-then, causation)
    
    Legal arguments follow Toulmin model:
    Claim → Evidence → Warrant → Backing → Rebuttal → Qualifier
    
    Features: 60
    """
    
    def __init__(self):
        # Claim indicators
        self.claim_markers = [
            r'\b(we hold|we find|we conclude|the court holds|it is clear|therefore)',
            r'\b(must|shall|requires|mandates|dictates)',
            r'\b(the law requires|the constitution|the statute)'
        ]
        
        # Evidence indicators
        self.evidence_markers = [
            r'\b(the evidence|the record|the facts|testimony|exhibits?)',
            r'\b(shows|demonstrates|proves|establishes|indicates)',
            r'\b(according to|based on|in light of)'
        ]
        
        # Warrant indicators (logical connections)
        self.warrant_markers = [
            r'\b(because|since|given that|in that|for the reason)',
            r'\b(it follows|thus|hence|accordingly|consequently)',
            r'\b(this means|this requires|this demonstrates)'
        ]
        
        # Rebuttal indicators
        self.rebuttal_markers = [
            r'\b(however|nevertheless|despite|notwithstanding|although)',
            r'\b(the dissent|opposing counsel|petitioner argues)',
            r'\b(we reject|we disagree|this argument fails|unconvincing)'
        ]
        
        # Logical connectives
        self.logical_connectives = {
            'causal': [r'\b(because|since|as|due to|caused by)'],
            'conditional': [r'\b(if|when|unless|provided|assuming)'],
            'adversative': [r'\b(but|however|yet|nevertheless|although)'],
            'additive': [r'\b(and|moreover|furthermore|additionally|also)'],
            'sequential': [r'\b(first|second|then|next|finally)']
        }
    
    def fit(self, X, y=None):
        """Fit transformer (no-op for this transformer)."""
        return self
    
    def transform(self, X):
        """
        Transform legal texts to argumentative structure features.
        
        Parameters
        ----------
        X : list of str
            Legal opinion texts
        
        Returns
        -------
        features : ndarray, shape (n_samples, 60)
        """
        X = ensure_string_list(X)
        features = []
        
        for text in X:
            feat = self._extract_features(text)
            features.append(feat)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_features(self, text: str) -> List[float]:
        """Extract all argumentative structure features."""
        if not text or len(text) < 100:
            return [0.0] * 60
        
        text_lower = text.lower()
        words = text.split()
        n_words = len(words)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        n_sentences = len(sentences)
        
        features = []
        
        # 1-5: Claim features
        claim_count = self._count_patterns(text_lower, self.claim_markers)
        features.append(claim_count / (n_sentences + 1))  # Claims per sentence
        features.append(claim_count / 10)  # Absolute claim count (normalized)
        
        # Claim strength indicators
        features.append(self._count_pattern(text_lower, r'\bwe hold\b') / (n_sentences + 1))
        features.append(self._count_pattern(text_lower, r'\bwe conclude\b') / (n_sentences + 1))
        features.append(self._count_pattern(text_lower, r'\bit is clear\b') / (n_sentences + 1))
        
        # 6-10: Evidence features
        evidence_count = self._count_patterns(text_lower, self.evidence_markers)
        features.append(evidence_count / (n_sentences + 1))
        features.append(evidence_count / 10)
        
        # Evidence types
        features.append(self._count_pattern(text_lower, r'\bthe record\b') / (n_sentences + 1))
        features.append(self._count_pattern(text_lower, r'\bthe evidence\b') / (n_sentences + 1))
        features.append(self._count_pattern(text_lower, r'\btestimony\b') / (n_sentences + 1))
        
        # 11-15: Warrant features (logical connections)
        warrant_count = self._count_patterns(text_lower, self.warrant_markers)
        features.append(warrant_count / (n_sentences + 1))
        features.append(warrant_count / 10)
        
        # Logical reasoning strength
        features.append(self._count_pattern(text_lower, r'\bbecause\b') / (n_words + 1))
        features.append(self._count_pattern(text_lower, r'\btherefore\b') / (n_words + 1))
        features.append(self._count_pattern(text_lower, r'\bthus\b') / (n_words + 1))
        
        # 16-20: Rebuttal features
        rebuttal_count = self._count_patterns(text_lower, self.rebuttal_markers)
        features.append(rebuttal_count / (n_sentences + 1))
        features.append(rebuttal_count / 10)
        
        # Counterargument handling
        features.append(self._count_pattern(text_lower, r'\bwe reject\b') / (n_sentences + 1))
        features.append(self._count_pattern(text_lower, r'\bthe dissent\b') / (n_sentences + 1))
        features.append(self._count_pattern(text_lower, r'\balthough\b') / (n_words + 1))
        
        # 21-30: Logical connective distribution
        for conn_type, patterns in self.logical_connectives.items():
            count = self._count_patterns(text_lower, patterns)
            features.append(count / (n_sentences + 1))
            features.append(count / (n_words + 1))
        
        # 31-35: Syllogistic structure
        # If X, then Y patterns
        if_then_count = len(re.findall(r'\bif\b.*\bthen\b', text_lower))
        features.append(if_then_count / (n_sentences + 1))
        
        # All X are Y patterns
        all_are_count = len(re.findall(r'\ball\b.*\bare\b', text_lower))
        features.append(all_are_count / (n_sentences + 1))
        
        # Must/shall (normative force)
        features.append(self._count_pattern(text_lower, r'\bmust\b') / (n_words + 1))
        features.append(self._count_pattern(text_lower, r'\bshall\b') / (n_words + 1))
        features.append(self._count_pattern(text_lower, r'\brequires\b') / (n_words + 1))
        
        # 36-40: Argument complexity
        # Average sentence length (complex arguments)
        avg_sent_length = n_words / (n_sentences + 1)
        features.append(min(1.0, avg_sent_length / 30))
        
        # Subordinate clauses (complexity)
        subordinate_markers = [r'\bwhich\b', r'\bthat\b', r'\bwho\b', r'\bwhom\b']
        subordinate_count = sum(self._count_pattern(text_lower, m) for m in subordinate_markers)
        features.append(subordinate_count / (n_sentences + 1))
        
        # Parenthetical references (citations in argument)
        parenthetical_count = len(re.findall(r'\([^)]{10,}\)', text))
        features.append(parenthetical_count / (n_sentences + 1))
        
        # Qualifying language (hedges)
        hedge_words = [r'\bmay\b', r'\bcould\b', r'\bpossibly\b', r'\bperhaps\b', r'\bseems\b']
        hedge_count = sum(self._count_pattern(text_lower, h) for h in hedge_words)
        features.append(hedge_count / (n_words + 1))
        
        # Certainty language (strong claims)
        certainty_words = [r'\bclearly\b', r'\bobviously\b', r'\bundoubtedly\b', r'\bcertainly\b']
        certainty_count = sum(self._count_pattern(text_lower, c) for c in certainty_words)
        features.append(certainty_count / (n_words + 1))
        
        # 41-45: Precedent integration in argument
        # Citation patterns (case law references)
        v_pattern = r'\b[A-Z][a-z]+\s+v\.?\s+[A-Z][a-z]+'  # Case names
        case_citations = len(re.findall(v_pattern, text))
        features.append(case_citations / (n_sentences + 1))
        
        # Year citations (temporal references)
        year_pattern = r'\b(19|20)\d{2}\b'
        year_citations = len(re.findall(year_pattern, text))
        features.append(year_citations / (n_sentences + 1))
        
        # "In X," citation opening
        in_citation_count = len(re.findall(r'\bIn\s+[A-Z][a-z]+', text))
        features.append(in_citation_count / (n_sentences + 1))
        
        # See/See also (citation signals)
        see_count = self._count_pattern(text_lower, r'\bsee\b')
        features.append(see_count / (n_sentences + 1))
        
        # Cf. (comparison citations)
        cf_count = self._count_pattern(text, r'\bCf\.')
        features.append(cf_count / (n_sentences + 1))
        
        # 46-50: Rebuttal sophistication
        # Distinguishing precedent
        distinguish_count = self._count_pattern(text_lower, r'\bdistinguish')
        features.append(distinguish_count / (n_sentences + 1))
        
        # Conceding points
        concede_count = self._count_pattern(text_lower, r'\b(concede|granted|admittedly|to be sure)')
        features.append(concede_count / (n_sentences + 1))
        
        # Weighing competing interests
        balance_count = self._count_pattern(text_lower, r'\b(balance|weigh|competing|interests)')
        features.append(balance_count / (n_sentences + 1))
        
        # Policy arguments
        policy_count = self._count_pattern(text_lower, r'\b(policy|public interest|social|practical)')
        features.append(policy_count / (n_sentences + 1))
        
        # Slippery slope warnings
        slippery_count = self._count_pattern(text_lower, r'\b(would lead to|slippery slope|open the door|flood)')
        features.append(slippery_count / (n_sentences + 1))
        
        # 51-60: Overall argument quality indicators
        # Question-answer structure (dialectical)
        question_count = text.count('?')
        features.append(question_count / (n_sentences + 1))
        
        # Enumeration (structured argument)
        enum_count = len(re.findall(r'\b(first|second|third|fourth|finally)\b', text_lower))
        features.append(enum_count / (n_sentences + 1))
        
        # Legal terminology density
        legal_terms = [
            r'\bjurisdiction\b', r'\bconstitution', r'\bstatute', r'\bprecedent',
            r'\bholding\b', r'\bdicta\b', r'\bremand', r'\baffirm', r'\breverse'
        ]
        legal_term_count = sum(self._count_pattern(text_lower, term) for term in legal_terms)
        features.append(legal_term_count / (n_words + 1))
        
        # Ratio of claims to evidence (assertion vs support)
        claim_evidence_ratio = claim_count / (evidence_count + 1)
        features.append(min(2.0, claim_evidence_ratio) / 2.0)  # Normalized
        
        # Ratio of warrants to claims (reasoning density)
        warrant_claim_ratio = warrant_count / (claim_count + 1)
        features.append(min(2.0, warrant_claim_ratio) / 2.0)
        
        # Rebuttal comprehensiveness (rebuttals per claim)
        rebuttal_claim_ratio = rebuttal_count / (claim_count + 1)
        features.append(min(2.0, rebuttal_claim_ratio) / 2.0)
        
        # Argument density (all argument markers per sentence)
        total_arg_markers = claim_count + evidence_count + warrant_count + rebuttal_count
        features.append(total_arg_markers / (n_sentences + 1))
        
        # Hedge-certainty balance
        hedge_certainty_ratio = hedge_count / (certainty_count + 1)
        features.append(min(3.0, hedge_certainty_ratio) / 3.0)
        
        # Logical connective diversity (use all 5 types?)
        connective_types_used = sum(1 for conn_type, patterns in self.logical_connectives.items() 
                                    if self._count_patterns(text_lower, patterns) > 0)
        features.append(connective_types_used / 5)
        
        # Citation density (precedent integration)
        citation_density = case_citations / (n_sentences + 1)
        features.append(min(1.0, citation_density))
        
        return features
    
    def _count_pattern(self, text: str, pattern: str) -> int:
        """Count occurrences of regex pattern."""
        try:
            return len(re.findall(pattern, text, re.IGNORECASE))
        except:
            return 0
    
    def _count_patterns(self, text: str, patterns: List[str]) -> int:
        """Count occurrences of multiple patterns."""
        total = 0
        for pattern in patterns:
            total += self._count_pattern(text, pattern)
        return total
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names."""
        names = [
            # Claims (1-5)
            'claims_per_sentence',
            'claim_count_normalized',
            'we_hold_density',
            'we_conclude_density',
            'it_is_clear_density',
            
            # Evidence (6-10)
            'evidence_per_sentence',
            'evidence_count_normalized',
            'record_references',
            'evidence_references',
            'testimony_references',
            
            # Warrants (11-15)
            'warrants_per_sentence',
            'warrant_count_normalized',
            'because_density',
            'therefore_density',
            'thus_density',
            
            # Rebuttals (16-20)
            'rebuttals_per_sentence',
            'rebuttal_count_normalized',
            'we_reject_density',
            'dissent_references',
            'although_density',
            
            # Logical connectives (21-30)
            'causal_connectives_sent',
            'causal_connectives_word',
            'conditional_connectives_sent',
            'conditional_connectives_word',
            'adversative_connectives_sent',
            'adversative_connectives_word',
            'additive_connectives_sent',
            'additive_connectives_word',
            'sequential_connectives_sent',
            'sequential_connectives_word',
            
            # Syllogistic (31-35)
            'if_then_patterns',
            'all_are_patterns',
            'must_density',
            'shall_density',
            'requires_density',
            
            # Complexity (36-40)
            'avg_sentence_complexity',
            'subordinate_clause_density',
            'parenthetical_density',
            'hedge_word_density',
            'certainty_word_density',
            
            # Precedent integration (41-45)
            'case_citation_density',
            'year_citation_density',
            'in_citation_density',
            'see_signal_density',
            'cf_signal_density',
            
            # Rebuttal sophistication (46-50)
            'distinguish_precedent',
            'concession_patterns',
            'balance_competing_interests',
            'policy_arguments',
            'slippery_slope_warnings',
            
            # Overall quality (51-60)
            'question_density',
            'enumeration_density',
            'legal_terminology_density',
            'claim_evidence_ratio',
            'warrant_claim_ratio',
            'rebuttal_claim_ratio',
            'argument_density',
            'hedge_certainty_balance',
            'logical_diversity',
            'citation_integration'
        ]
        
        return np.array(names[:60])


if __name__ == '__main__':
    # Test transformer
    test_text = """
    We hold that the Constitution protects a woman's right to choose. 
    The evidence shows this is a fundamental liberty interest. Because 
    this right is fundamental, the State must show a compelling interest. 
    Although the dissent argues otherwise, we reject this view as 
    inconsistent with precedent. In Roe v. Wade (1973), the Court 
    established this framework. Therefore, we affirm the lower court's 
    decision.
    """
    
    transformer = ArgumentativeStructureTransformer()
    features = transformer.transform([test_text])
    feature_names = transformer.get_feature_names_out()
    
    print("\nArgumentative Structure Features:")
    print("="*80)
    for name, value in zip(feature_names, features[0]):
        if value > 0:
            print(f"{name:40s}: {value:.4f}")
    print("="*80)
    print(f"Total features: {len(features[0])}")

