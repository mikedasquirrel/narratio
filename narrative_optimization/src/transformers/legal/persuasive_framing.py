"""
Persuasive Framing Transformer

Analyzes persuasive and rhetorical framing in legal narratives:
- Emotional vs logical appeals
- Rights-based framing
- Harm narratives and severity
- Public policy arguments
- Slippery slope and parade of horribles
- Moral framing

Features: 50

Author: Narrative Optimization Framework  
Date: November 17, 2025
"""

import numpy as np
import re
from typing import List
from sklearn.base import BaseEstimator, TransformerMixin
from ..utils.input_validation import ensure_string_list


class PersuasiveFramingTransformer(BaseEstimator, TransformerMixin):
    """
    Extracts persuasive framing features from legal text.
    
    Tests HOW legal arguments frame issues:
    - Emotional (pathos) vs logical (logos) vs credibility (ethos)
    - Rights framing vs duties framing
    - Individual liberty vs collective good
    - Present harm vs future consequences
    
    This is where narrative framing potentially overrides evidence.
    
    Features: 50
    """
    
    def __init__(self):
        # Emotional appeal markers
        self.emotional_markers = {
            'fear': [r'\bdanger', r'\brisk', r'\bthreat', r'\bharm', r'\binjury'],
            'hope': [r'\bhope', r'\bpromise', r'\bopportunity', r'\bfuture', r'\bprogress'],
            'fairness': [r'\bfair', r'\bjust', r'\bequit', r'\bright', r'\bwrong'],
            'dignity': [r'\bdignity', r'\bhuman', r'\brespect', r'\bperson', r'\bindividual'],
            'outrage': [r'\babsurd', r'\bunthinkable', r'\bunacceptable', r'\bshocking']
        }
        
        # Logical appeal markers  
        self.logical_markers = [
            r'\blogical', r'\brational', r'\breasonable', r'\banalysis',
            r'\bsyllogism', r'\bdeduction', r'\binference'
        ]
        
        # Rights framing
        self.rights_frames = {
            'liberty': [r'\bliberty\b', r'\bfreedom\b', r'\bautonomy\b', r'\bchoice\b'],
            'equality': [r'\bequal', r'\bdiscriminat', r'\bfairness\b'],
            'privacy': [r'\bprivacy\b', r'\bprivate\b', r'\bpersonal\b', r'\bintimate\b'],
            'property': [r'\bproperty\b', r'\bown', r'\bpossession\b'],
            'speech': [r'\bspeech\b', r'\bexpression\b', r'\bfirst amendment\b']
        }
        
        # Harm narratives
        self.harm_markers = {
            'immediate': [r'\bimmediate', r'\bpresent\b', r'\bcurrent', r'\bnow\b'],
            'severe': [r'\bsevere', r'\bgrave\b', r'\bserious', r'\bextreme'],
            'irreparable': [r'\birrepar', r'\bpermanent', r'\birrevers', r'\blasting'],
            'widespread': [r'\bwidespread', r'\bmassive', r'\bextensive', r'\bsweeping']
        }
    
    def fit(self, X, y=None):
        """Fit transformer."""
        return self
    
    def transform(self, X):
        """
        Transform legal texts to persuasive framing features.
        
        Parameters
        ----------
        X : list of str
            Legal texts
        
        Returns
        -------
        features : ndarray, shape (n_samples, 50)
        """
        X = ensure_string_list(X)
        features = []
        
        for text in X:
            feat = self._extract_features(text)
            features.append(feat)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_features(self, text: str) -> List[float]:
        """Extract persuasive framing features."""
        if not text or len(text) < 100:
            return [0.0] * 50
        
        text_lower = text.lower()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        n_sentences = len(sentences)
        n_words = len(text.split())
        
        features = []
        
        # 1-10: Emotional appeal distribution
        total_emotional = 0
        for emotion, patterns in self.emotional_markers.items():
            count = self._count_patterns(text_lower, patterns)
            features.append(count / (n_sentences + 1))
            total_emotional += count
        
        # Total emotional density
        features.append(total_emotional / (n_sentences + 1))
        features.append(total_emotional / (n_words + 1))
        
        # Emotional diversity (use multiple emotion types?)
        emotion_types_used = sum(1 for emotion, patterns in self.emotional_markers.items() 
                                if self._count_patterns(text_lower, patterns) > 0)
        features.append(emotion_types_used / len(self.emotional_markers))
        
        # 11-13: Logical appeal
        logical_count = self._count_patterns(text_lower, self.logical_markers)
        features.append(logical_count / (n_sentences + 1))
        
        # Emotional vs logical ratio (pathos vs logos)
        if total_emotional + logical_count > 0:
            emotional_ratio = total_emotional / (total_emotional + logical_count)
        else:
            emotional_ratio = 0.5
        features.append(emotional_ratio)
        
        # Balanced appeal (uses both)
        balanced_score = 1.0 - abs(emotional_ratio - 0.5) * 2  # Max at 0.5 balance
        features.append(balanced_score)
        
        # 14-23: Rights framing by type
        total_rights = 0
        for right_type, patterns in self.rights_frames.items():
            count = self._count_patterns(text_lower, patterns)
            features.append(count / (n_sentences + 1))
            total_rights += count
        
        # Total rights framing density
        features.append(total_rights / (n_sentences + 1))
        
        # Rights diversity
        rights_types_used = sum(1 for right_type, patterns in self.rights_frames.items()
                               if self._count_patterns(text_lower, patterns) > 0)
        features.append(rights_types_used / len(self.rights_frames))
        
        # 24-31: Harm narrative structure
        for harm_type, patterns in self.harm_markers.items():
            count = self._count_patterns(text_lower, patterns)
            features.append(count / (n_sentences + 1))
        
        # Total harm narrative
        total_harm = sum(self._count_patterns(text_lower, patterns) 
                        for patterns in self.harm_markers.values())
        features.append(total_harm / (n_sentences + 1))
        
        # Harm specificity (immediate + severe)
        immediate = self._count_patterns(text_lower, self.harm_markers['immediate'])
        severe = self._count_patterns(text_lower, self.harm_markers['severe'])
        harm_specificity = (immediate + severe) / (total_harm + 1)
        features.append(harm_specificity)
        
        # Harm scope (individual vs widespread)
        individual_harm = self._count_pattern(text_lower, r'\b(individual|person|plaintiff)')
        widespread_harm = self._count_patterns(text_lower, self.harm_markers['widespread'])
        if individual_harm + widespread_harm > 0:
            scope_ratio = widespread_harm / (individual_harm + widespread_harm)
        else:
            scope_ratio = 0.5
        features.append(scope_ratio)
        
        # 32-36: Public policy framing
        policy_pattern = r'\b(policy|public interest|societal|social impact|consequences?)\b'
        policy_count = self._count_pattern(text_lower, policy_pattern)
        features.append(policy_count / (n_sentences + 1))
        
        # Practical effects
        practical_pattern = r'\b(practical|real-world|actual|in practice|functionally)\b'
        practical_count = self._count_pattern(text_lower, practical_pattern)
        features.append(practical_count / (n_sentences + 1))
        
        # Economic arguments
        economic_pattern = r'\b(economic|financial|cost|burden|efficiency|market)\b'
        economic_count = self._count_pattern(text_lower, economic_pattern)
        features.append(economic_count / (n_sentences + 1))
        
        # Social arguments
        social_pattern = r'\b(social|cultural|community|society|traditions?)\b'
        social_count = self._count_pattern(text_lower, social_pattern)
        features.append(social_count / (n_sentences + 1))
        
        # Democratic/governance arguments
        democracy_pattern = r'\b(democratic|democracy|voter|election|representative|government)\b'
        democracy_count = self._count_pattern(text_lower, democracy_pattern)
        features.append(democracy_count / (n_sentences + 1))
        
        # 37-41: Slippery slope arguments
        slippery_slope_pattern = r'\b(slippery slope|open.*door|floodgates?|would lead to|next step)\b'
        slippery_count = self._count_pattern(text_lower, slippery_slope_pattern)
        features.append(slippery_count / (n_sentences + 1))
        
        # Parade of horribles
        horribles_pattern = r'\b(parade of horribles|absurd|unthinkable|imagine if|what if)\b'
        horribles_count = self._count_pattern(text_lower, horribles_pattern)
        features.append(horribles_count / (n_sentences + 1))
        
        # Hypothetical consequences
        hypothetical_pattern = r'\b(would|could|might|may).*\b(result|lead|cause|create)\b'
        hypothetical_count = len(re.findall(hypothetical_pattern, text_lower))
        features.append(hypothetical_count / (n_sentences + 1))
        
        # Limiting principle challenges
        limiting_pattern = r'\b(limiting principle|where.*stop|line.*drawn|boundary)\b'
        limiting_count = self._count_pattern(text_lower, limiting_pattern)
        features.append(limiting_count / (n_sentences + 1))
        
        # Textual interpretation
        textual_pattern = r'\b(plain (meaning|text)|textu|ordinary meaning|dictionary)\b'
        textual_count = self._count_pattern(text_lower, textual_pattern)
        features.append(textual_count / (n_sentences + 1))
        
        # 42-46: Moral framing
        moral_pattern = r'\b(moral|immoral|ethical|right|wrong|justice)\b'
        moral_count = self._count_pattern(text_lower, moral_pattern)
        features.append(moral_count / (n_sentences + 1))
        
        # Values language
        values_pattern = r'\b(values?|principles?|ideals?|beliefs?|sacred)\b'
        values_count = self._count_pattern(text_lower, values_pattern)
        features.append(values_count / (n_sentences + 1))
        
        # Tradition/history framing
        tradition_pattern = r'\b(tradition|historical|longstanding|time-honored|always)\b'
        tradition_count = self._count_pattern(text_lower, tradition_pattern)
        features.append(tradition_count / (n_sentences + 1))
        
        # Progress/change framing
        progress_pattern = r'\b(progress|evolving|modern|contemporary|change|adapt)\b'
        progress_count = self._count_pattern(text_lower, progress_pattern)
        features.append(progress_count / (n_sentences + 1))
        
        # Tradition vs progress ratio
        if tradition_count + progress_count > 0:
            tradition_ratio = tradition_count / (tradition_count + progress_count)
        else:
            tradition_ratio = 0.5
        features.append(tradition_ratio)
        
        # 47-50: Overall framing strategy
        # Aggregate framing intensity
        total_framing = (
            total_emotional + logical_count + total_rights + 
            total_harm + policy_count + moral_count
        )
        features.append(total_framing / (n_sentences + 1))
        
        # Framing diversity (uses multiple frame types)
        frame_types = [
            total_emotional > 0,
            logical_count > 0,
            total_rights > 0,
            total_harm > 0,
            policy_count > 0,
            moral_count > 0
        ]
        features.append(sum(frame_types) / len(frame_types))
        
        # Narrative dominance (framing vs pure legal analysis)
        legal_technical = self._count_pattern(text_lower, r'\b(statute|regulation|provision|section|subsection)')
        if total_framing + legal_technical > 0:
            narrative_dominance = total_framing / (total_framing + legal_technical)
        else:
            narrative_dominance = 0.5
        features.append(narrative_dominance)
        
        # Persuasion intensity (strong persuasive language)
        persuasion_markers = [r'\bclearly\b', r'\bundoubtedly\b', r'\binescapably\b', r'\bcompels?\b']
        persuasion_count = sum(self._count_pattern(text_lower, m) for m in persuasion_markers)
        features.append(persuasion_count / (n_sentences + 1))
        
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
            # Emotional appeals (1-10)
            'fear_appeal_density',
            'hope_appeal_density',
            'fairness_appeal_density',
            'dignity_appeal_density',
            'outrage_appeal_density',
            'total_emotional_density_sent',
            'total_emotional_density_word',
            'emotional_diversity',
            
            # Logical appeals (11-13)
            'logical_appeal_density',
            'emotional_logical_ratio',
            'balanced_appeal_score',
            
            # Rights framing (14-23)
            'liberty_framing',
            'equality_framing',
            'privacy_framing',
            'property_framing',
            'speech_framing',
            'total_rights_framing',
            'rights_diversity',
            
            # Harm narrative (24-31)
            'immediate_harm',
            'severe_harm',
            'irreparable_harm',
            'widespread_harm',
            'total_harm_narrative',
            'harm_specificity',
            'individual_vs_widespread',
            
            # Policy framing (32-36)
            'public_policy_density',
            'practical_effects_density',
            'economic_arguments',
            'social_arguments',
            'democratic_governance',
            
            # Slippery slope (37-41)
            'slippery_slope_warnings',
            'parade_of_horribles',
            'hypothetical_consequences',
            'limiting_principle_challenges',
            'textual_interpretation',
            
            # Moral framing (42-46)
            'moral_framing',
            'values_language',
            'tradition_framing',
            'progress_framing',
            'tradition_progress_ratio',
            
            # Overall strategy (47-50)
            'total_framing_intensity',
            'framing_diversity',
            'narrative_vs_technical',
            'persuasion_intensity'
        ]
        
        return np.array(names[:50])


if __name__ == '__main__':
    test_text = """
    The fundamental liberty of individuals to make personal decisions
    about their own bodies is a right deeply rooted in our tradition.
    The State's restriction would lead to severe and irreparable harm.
    If we allow this, it would open the floodgates to government intrusion
    into the most intimate aspects of personal life. The parade of horribles
    is clear: unlimited state power over individual choice. This is not merely
    a policy question—it is a matter of basic human dignity and constitutional
    protection. The practical effects would be devastating for millions.
    """
    
    transformer = PersuasiveFramingTransformer()
    features = transformer.transform([test_text])
    feature_names = transformer.get_feature_names_out()
    
    print("\nPersuasive Framing Features:")
    print("="*80)
    for name, value in zip(feature_names, features[0]):
        if value > 0.01:
            print(f"{name:40s}: {value:.4f}")
    print("="*80)

