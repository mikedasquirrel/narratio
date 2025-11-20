"""
Precedential Narrative Transformer

Analyzes how legal narratives invoke precedent and authority:
- Citation patterns and density
- Precedent framing (following vs distinguishing)
- Authority invocation (prior cases, constitutional framers, legal scholars)
- Historical narrative construction
- Stare decisis strength

Features: 45

Author: Narrative Optimization Framework
Date: November 17, 2025
"""

import numpy as np
import re
from typing import List, Set
from sklearn.base import BaseEstimator, TransformerMixin
from ..utils.input_validation import ensure_string_list


class PrecedentialNarrativeTransformer(BaseEstimator, TransformerMixin):
    """
    Extracts precedential narrative features from legal text.
    
    Key components:
    - Citation density and patterns
    - Precedent framing (following, distinguishing, overruling)
    - Authority types (Supreme Court, framers, scholars)
    - Historical narrative depth
    - Stare decisis invocation
    
    Legal reasoning relies heavily on precedent - this measures HOW
    precedent is narratively constructed and deployed.
    
    Features: 45
    """
    
    def __init__(self):
        # Precedent framing types
        self.following_markers = [
            r'\bfollows? from\b', r'\bconsistent with\b', r'\bin line with\b',
            r'\bas established in\b', r'\bas held in\b', r'\bapplying\b'
        ]
        
        self.distinguishing_markers = [
            r'\bdistinguish', r'\bunlike\b', r'\bdifferent from\b',
            r'\bdoes not apply\b', r'\binapplicable\b', r'\bnarrow'
        ]
        
        self.overruling_markers = [
            r'\boverrule', r'\boverturned?\b', r'\babrogated?\b',
            r'\bno longer good law\b', r'\breconsider\b'
        ]
        
        # Authority types
        self.authority_markers = {
            'supreme_court': [r'\bthis [Cc]ourt\b', r'\bwe have held\b', r'\bour precedents?\b'],
            'constitution': [r'\b[Cc]onstitution', r'\b[Ff]ramers?\b', r'\b[Ff]ounding\b', r'\bAmendment\b'],
            'scholars': [r'\bscholars?\b', r'\bprofessors?\b', r'\bcommentators?\b', r'\bacademic'],
            'historical': [r'\bhistory\b', r'\bhistorical', r'\btradition', r'\boriginally\b']
        }
        
        # Famous landmark cases (strong precedent)
        self.landmark_cases = [
            'marbury', 'brown', 'miranda', 'roe', 'gideon', 'mapp',
            'griswold', 'loving', 'korematsu', 'plessy', 'lochner'
        ]
    
    def fit(self, X, y=None):
        """Fit transformer."""
        return self
    
    def transform(self, X):
        """
        Transform legal texts to precedential narrative features.
        
        Parameters
        ----------
        X : list of str
            Legal opinion texts
        
        Returns
        -------
        features : ndarray, shape (n_samples, 45)
        """
        X = ensure_string_list(X)
        features = []
        
        for text in X:
            feat = self._extract_features(text)
            features.append(feat)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_features(self, text: str) -> List[float]:
        """Extract precedential narrative features."""
        if not text or len(text) < 100:
            return [0.0] * 45
        
        text_lower = text.lower()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        n_sentences = len(sentences)
        n_words = len(text.split())
        
        features = []
        
        # 1-5: Citation density and patterns
        # Case citations (Party v. Party pattern)
        case_pattern = r'\b[A-Z][a-z]+\s+v\.?\s+[A-Z][a-z]+'
        case_citations = len(re.findall(case_pattern, text))
        features.append(case_citations / (n_sentences + 1))
        features.append(min(1.0, case_citations / 50))  # Absolute (normalized)
        
        # Full citations with year and reporter
        full_citation_pattern = r'\d+\s+U\.S\.\s+\d+'
        full_citations = len(re.findall(full_citation_pattern, text))
        features.append(full_citations / (n_sentences + 1))
        
        # String citations (Id., Ibid., supra)
        string_citation_count = (
            text.count('Id.') + 
            text.count('Ibid.') + 
            self._count_pattern(text_lower, r'\bsupra\b')
        )
        features.append(string_citation_count / (n_sentences + 1))
        
        # Citation signals (See, See also, Cf., But see)
        signal_pattern = r'\b(See|Cf\.|But see|See also|Compare)'
        signal_count = len(re.findall(signal_pattern, text))
        features.append(signal_count / (n_sentences + 1))
        
        # 6-10: Precedent framing
        following_count = self._count_patterns(text_lower, self.following_markers)
        features.append(following_count / (n_sentences + 1))
        features.append(following_count / 10)
        
        distinguishing_count = self._count_patterns(text_lower, self.distinguishing_markers)
        features.append(distinguishing_count / (n_sentences + 1))
        
        overruling_count = self._count_patterns(text_lower, self.overruling_markers)
        features.append(overruling_count / (n_sentences + 1))
        
        # Following vs distinguishing ratio (precedent stance)
        if distinguishing_count + following_count > 0:
            follow_ratio = following_count / (following_count + distinguishing_count)
        else:
            follow_ratio = 0.5
        features.append(follow_ratio)
        
        # 11-20: Authority invocation by type
        for auth_type, patterns in self.authority_markers.items():
            count = self._count_patterns(text_lower, patterns)
            features.append(count / (n_sentences + 1))
            features.append(count / 20)  # Normalized absolute
        
        # 21-25: Historical narrative depth
        # Temporal references (years mentioned)
        year_pattern = r'\b(17|18|19|20)\d{2}\b'
        year_mentions = len(re.findall(year_pattern, text))
        features.append(year_mentions / (n_sentences + 1))
        
        # Century/era references
        era_pattern = r'\b(century|era|period|decades?|founding|original)'
        era_mentions = self._count_pattern(text_lower, era_pattern)
        features.append(era_mentions / (n_sentences + 1))
        
        # Historical figures
        founder_pattern = r'\b(Madison|Hamilton|Jefferson|Washington|Franklin|Adams)'
        founder_mentions = len(re.findall(founder_pattern, text))
        features.append(founder_mentions / (n_sentences + 1))
        
        # Temporal progression (early → later)
        temporal_progression = len(re.findall(r'\b(originally|initially|later|subsequently|ultimately)', text_lower))
        features.append(temporal_progression / (n_sentences + 1))
        
        # Historical narrative arc (beginning/middle/end markers)
        narrative_arc = len(re.findall(r'\b(began|developed|evolved|culminated|resulted)', text_lower))
        features.append(narrative_arc / (n_sentences + 1))
        
        # 26-30: Stare decisis strength
        # Explicit stare decisis references
        stare_decisis_count = self._count_pattern(text_lower, r'\bstare decisis\b')
        features.append(stare_decisis_count / (n_sentences + 1))
        
        # Precedent stability language
        stability_markers = [r'\bsettled\b', r'\blong-standing\b', r'\bwell-established\b', r'\bfirmly rooted\b']
        stability_count = sum(self._count_pattern(text_lower, m) for m in stability_markers)
        features.append(stability_count / (n_sentences + 1))
        
        # Reliance language
        reliance_pattern = r'\b(relied?|reliance|depend|expectations?)\b'
        reliance_count = self._count_pattern(text_lower, reliance_pattern)
        features.append(reliance_count / (n_sentences + 1))
        
        # Predictability/consistency language
        consistency_pattern = r'\b(consistent|predictable|uniformity|certainty)\b'
        consistency_count = self._count_pattern(text_lower, consistency_pattern)
        features.append(consistency_count / (n_sentences + 1))
        
        # Workability language (practical precedent)
        workability_pattern = r'\b(workable|practical|administrable|functional)\b'
        workability_count = self._count_pattern(text_lower, workability_pattern)
        features.append(workability_count / (n_sentences + 1))
        
        # 31-35: Landmark case references
        landmark_mentions = sum(1 for landmark in self.landmark_cases if landmark in text_lower)
        features.append(landmark_mentions / 10)  # Normalized (max ~10 landmarks)
        
        # Landmark case density
        features.append(landmark_mentions / (n_sentences + 1))
        
        # Citation age diversity (old + new cases)
        years_in_text = set(re.findall(year_pattern, text))
        if years_in_text:
            year_values = [int(y) for y in years_in_text if 1700 < int(y) < 2100]
            if year_values:
                year_range = max(year_values) - min(year_values)
                features.append(min(1.0, year_range / 200))  # 200 year range
                
                # Citation recency (how many recent?)
                if max(year_values) > 1900:
                    recent_citations = sum(1 for y in year_values if y > max(year_values) - 30)
                    features.append(recent_citations / len(year_values))
                else:
                    features.append(0.5)
            else:
                features.extend([0.0, 0.0])
        else:
            features.extend([0.0, 0.0])
        
        # Balance of old vs new precedent
        if case_citations > 0:
            old_precedent = self._count_pattern(text, r'(17|18)\d{2}')
            modern_precedent = self._count_pattern(text, r'(19[5-9]|20)\d{2}')
            if old_precedent + modern_precedent > 0:
                balance = modern_precedent / (old_precedent + modern_precedent)
            else:
                balance = 0.5
            features.append(balance)
        else:
            features.append(0.5)
        
        # 36-40: Precedent treatment sophistication
        # Analogical reasoning (like/similar/analogous)
        analogy_pattern = r'\b(like|similar|analogous|comparable|parallel)\b'
        analogy_count = self._count_pattern(text_lower, analogy_pattern)
        features.append(analogy_count / (n_sentences + 1))
        
        # Synthesis of precedents (multiple cases unified)
        synthesis_pattern = r'\b(taken together|synthesizing|harmonizing|reconciling)\b'
        synthesis_count = self._count_pattern(text_lower, synthesis_pattern)
        features.append(synthesis_count / (n_sentences + 1))
        
        # Precedent line construction (chain of cases)
        line_pattern = r'\b(line of cases|series of|string of|consistent line)\b'
        line_count = self._count_pattern(text_lower, line_pattern)
        features.append(line_count / (n_sentences + 1))
        
        # Evolution narrative (how law developed)
        evolution_pattern = r'\b(evolved|developed|progression|trajectory|arc)\b'
        evolution_count = self._count_pattern(text_lower, evolution_pattern)
        features.append(evolution_count / (n_sentences + 1))
        
        # Break from precedent (doctrinal shift)
        break_pattern = r'\b(depart|departure|break|shift|new direction)\b'
        break_count = self._count_pattern(text_lower, break_pattern)
        features.append(break_count / (n_sentences + 1))
        
        # 41-45: Authority weighting
        # Supreme Court self-reference strength
        self_ref_pattern = r'\b(this [Cc]ourt|we have|our [Cc]ases|our precedents?)\b'
        self_ref_count = len(re.findall(self_ref_pattern, text))
        features.append(self_ref_count / (n_sentences + 1))
        
        # Constitutional grounding
        const_pattern = r'\b[Cc]onstitution'
        const_count = len(re.findall(const_pattern, text))
        features.append(const_count / (n_sentences + 1))
        
        # Framers' intent invocation
        intent_pattern = r'\b(framers?.*intent|original (understanding|meaning)|ratifiers?)\b'
        intent_count = len(re.findall(intent_pattern, text_lower))
        features.append(intent_count / (n_sentences + 1))
        
        # Ratio of authority types (constitution vs precedent)
        if const_count + case_citations > 0:
            const_ratio = const_count / (const_count + case_citations)
        else:
            const_ratio = 0.5
        features.append(const_ratio)
        
        # Authority density (total authority invocations)
        total_authority = (
            self_ref_count + const_count + 
            case_citations + founder_mentions + 
            stability_count
        )
        features.append(total_authority / (n_sentences + 1))
        
        return features
    
    def _count_pattern(self, text: str, pattern: str) -> int:
        """Count regex pattern occurrences."""
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
            # Citations (1-5)
            'case_citation_density',
            'case_citation_count',
            'full_citation_density',
            'string_citation_density',
            'citation_signal_density',
            
            # Precedent framing (6-10)
            'following_precedent_density',
            'following_count',
            'distinguishing_precedent_density',
            'overruling_precedent_density',
            'follow_distinguish_ratio',
            
            # Authority types (11-20)
            'supreme_court_authority_density',
            'supreme_court_authority_count',
            'constitutional_authority_density',
            'constitutional_authority_count',
            'scholarly_authority_density',
            'scholarly_authority_count',
            'historical_authority_density',
            'historical_authority_count',
            
            # Historical narrative (21-25)
            'temporal_reference_density',
            'era_reference_density',
            'founder_reference_density',
            'temporal_progression_markers',
            'historical_arc_markers',
            
            # Stare decisis (26-30)
            'stare_decisis_explicit',
            'stability_language',
            'reliance_language',
            'consistency_language',
            'workability_language',
            
            # Landmark cases (31-35)
            'landmark_case_count',
            'landmark_case_density',
            'citation_temporal_range',
            'citation_recency',
            'old_modern_balance',
            
            # Sophistication (36-40)
            'analogical_reasoning',
            'precedent_synthesis',
            'precedent_line_construction',
            'evolution_narrative',
            'break_from_precedent',
            
            # Authority weighting (41-45)
            'self_reference_strength',
            'constitutional_grounding',
            'framers_intent_invocation',
            'constitution_precedent_ratio',
            'total_authority_density'
        ]
        
        return np.array(names[:45])


if __name__ == '__main__':
    # Test
    test_text = """
    This Court has long held that the Constitution protects fundamental rights.
    In Brown v. Board of Education (1954), we established that separate is 
    inherently unequal. Following that landmark precedent, our cases have 
    consistently applied strict scrutiny. The framers intended the Fourteenth 
    Amendment to protect individual liberty. As we held in Loving v. Virginia (1967),
    these principles apply with equal force here. Therefore, stare decisis compels
    us to follow our well-established precedents.
    """
    
    transformer = PrecedentialNarrativeTransformer()
    features = transformer.transform([test_text])
    feature_names = transformer.get_feature_names_out()
    
    print("\nPrecedential Narrative Features:")
    print("="*80)
    for name, value in zip(feature_names, features[0]):
        if value > 0.01:
            print(f"{name:40s}: {value:.4f}")
    print("="*80)

