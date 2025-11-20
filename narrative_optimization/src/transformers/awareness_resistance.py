"""
Awareness Resistance Transformer (θ Measurement)

Measures awareness resistance (θ) - free will resistance to nominative gravity.
θ is central to three-force model: Д = ة - θ - λ

θ represents conscious awareness and resistance to narrative pull.
Higher θ means more awareness and deliberate resistance.

Author: Narrative Integration System
Date: November 2025
"""

import re
import numpy as np
from typing import List, Dict, Optional
from collections import Counter

from .base import NarrativeTransformer


class AwarenessResistanceTransformer(NarrativeTransformer):
    """
    Extracts features measuring awareness resistance (θ).
    
    Theory: Д = ة - θ - λ (regular) or Д = ة + θ - λ (prestige)
    - θ = awareness × [field_studies + obviousness] × social_cost
    - At instance level: Extract awareness markers from text
    
    The awareness resistance determines how much conscious resistance
    exists to nominative gravity. High awareness can suppress (regular)
    or amplify (prestige) narrative effects.
    
    Features Extracted (15 total):
    1. Meta-awareness language density ("I know this is...", "Aware that...")
    2. Self-awareness indicators ("I realize", "I understand", "I recognize")
    3. Skepticism markers ("However", "But", "Despite", "Although")
    4. Questioning language ("Why", "How", "What if")
    5. Doubt indicators ("Uncertain", "Unsure", "Maybe")
    6. Critical thinking indicators ("Analyze", "Evaluate", "Consider")
    7. Evidence-seeking language ("Research", "Study", "Investigate")
    8. Reasoning markers ("Because", "Therefore", "Thus")
    9. Nominative determinism mentions (explicit awareness)
    10. Bias/stereotype awareness (meta-cognitive awareness)
    11. Educational indicators (academic language density)
    12. Citation/reference markers (scholarly awareness)
    13. Sophistication markers (complex reasoning patterns)
    14. Counter-narrative language (resistance to expected patterns)
    15. Awareness resistance score θ ∈ [0, 1]
    
    Parameters
    ----------
    detect_field_awareness : bool, default=True
        Whether to detect field-specific awareness (nominative determinism, etc.)
    
    Examples
    --------
    >>> transformer = AwarenessResistanceTransformer()
    >>> features = transformer.fit_transform(narratives)
    >>> 
    >>> # Check awareness resistance
    >>> theta_values = features[:, 14]  # Column 15 is θ
    >>> print(f"Average θ: {theta_values.mean():.2f}")
    >>> 
    >>> # High θ (~0.8) = highly aware population (psychology, academics)
    >>> # Low θ (~0.2) = low awareness (general public)
    """
    
    def __init__(self, detect_field_awareness: bool = True):
        super().__init__(
            narrative_id="awareness_resistance",
            description="Measures θ (awareness resistance) for three-force model Д = ة - θ - λ"
        )
        
        self.detect_field_awareness = detect_field_awareness
        
        # Meta-awareness patterns (general + domain-specific)
        self.meta_awareness_patterns = [
            # General awareness
            r'\bi know\b', r'\bi realize\b', r'\bi understand\b',
            r'\baware that\b', r'\baware of\b', r'\bconscious\b',
            r'\brecognize\b', r'\bperceive\b', r'\bappreciate\b',
            r'\bconsciousness\b', r'\bawareness\b', r'\bknowing\b',
            # Golf/Tennis mental game
            r'\bmental game\b', r'\bbetween the ears\b', r'\ball mental\b',
            r'\bpressure situation\b', r'\bclutch performance\b', r'\bchoking\b',
            r'\bconfidence\b', r'\bmindset\b', r'\bpsychological edge\b',
            r'\bmental toughness\b', r'\bmental strength\b', r'\bhead game\b',
            # Sports awareness
            r'\bbulletin board material\b', r'\boverlooked\b', r'\bunderestimated\b',
            r'\bnarrative\b', r'\bstoryline\b', r'\bscript\b',
            r'\brespect\b', r'\bdisrespect\b', r'\bdoubt\b',
            r'\bunderdog\b', r'\boverrated\b', r'\bunderrated\b'
        ]
        
        # Self-awareness indicators
        self.self_awareness_patterns = [
            r'\bi realize\b', r'\bi understand\b', r'\bi recognize\b',
            r'\bi see\b', r'\bi notice\b', r'\bi observe\b',
            r'\bmy awareness\b', r'\bmy understanding\b', r'\bmy perception\b',
            r'\bself-aware\b', r'\bself-conscious\b', r'\bintrospective\b'
        ]
        
        # Skepticism markers
        self.skepticism_patterns = [
            r'\bhowever\b', r'\bbut\b', r'\bdespite\b', r'\balthough\b',
            r'\byet\b', r'\bnevertheless\b', r'\bnonetheless\b',
            r'\bcontrary\b', r'\bwhereas\b', r'\bwhile\b',
            r'\bdespite the fact\b', r'\bin spite of\b', r'\bnotwithstanding\b'
        ]
        
        # Questioning language
        self.questioning_patterns = [
            r'\bwhy\b', r'\bhow\b', r'\bwhat if\b', r'\bwhat about\b',
            r'\bwhy not\b', r'\bhow come\b', r'\bwhat\'s the\b',
            r'\bis it\b', r'\bare we\b', r'\bdo we\b', r'\bcan we\b',
            r'\bshould we\b', r'\bmust we\b', r'\bquestion\b', r'\bwonder\b'
        ]
        
        # Doubt indicators
        self.doubt_patterns = [
            r'\buncertain\b', r'\bunsure\b', r'\bmaybe\b', r'\bperhaps\b',
            r'\bpossibly\b', r'\bpotentially\b', r'\bmight\b', r'\bcould\b',
            r'\bdoubt\b', r'\bskeptical\b', r'\bsuspicious\b', r'\bwary\b',
            r'\bhesitant\b', r'\bunclear\b', r'\bambiguous\b', r'\bvague\b'
        ]
        
        # Critical thinking indicators
        self.critical_thinking_patterns = [
            r'\banalyze\b', r'\bevaluate\b', r'\bconsider\b', r'\bexamine\b',
            r'\bassess\b', r'\bscrutinize\b', r'\breview\b', r'\bstudy\b',
            r'\binvestigate\b', r'\bexplore\b', r'\bquestion\b', r'\bchallenge\b',
            r'\bcritique\b', r'\bappraise\b', r'\bjudge\b', r'\bweigh\b'
        ]
        
        # Evidence-seeking language
        self.evidence_patterns = [
            r'\bresearch\b', r'\bstudy\b', r'\binvestigate\b', r'\bexamine\b',
            r'\bdata\b', r'\bevidence\b', r'\bproof\b', r'\bfacts\b',
            r'\bfindings\b', r'\bresults\b', r'\banalysis\b', r'\bstatistics\b',
            r'\bsurvey\b', r'\bexperiment\b', r'\btest\b', r'\bverify\b'
        ]
        
        # Reasoning markers
        self.reasoning_patterns = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bconsequently\b', r'\bas a result\b', r'\bdue to\b',
            r'\bowing to\b', r'\bfor this reason\b', r'\bso\b',
            r'\baccordingly\b', r'\bit follows\b', r'\bimplies\b', r'\bsuggests\b'
        ]
        
        # Field-specific awareness (nominative determinism, bias, stereotypes)
        self.field_awareness_patterns = [
            # Psychology/Academia
            r'\bnominative determinism\b', r'\bname effect\b', r'\bname bias\b',
            r'\bstereotype\b', r'\bbias\b', r'\bprejudice\b', r'\bdiscrimination\b',
            r'\bunconscious bias\b', r'\bimplicit bias\b', r'\bsocial construct\b',
            r'\bcultural conditioning\b', r'\bsocialization\b', r'\bconditioning\b',
            r'\bconfirmation bias\b', r'\bcognitive bias\b', r'\bimplicit association\b',
            r'\bstereotype threat\b', r'\bpriming effect\b', r'\bobserver effect\b',
            r'\bmeta-analysis\b', r'\bsystematic review\b', r'\bpeer-reviewed\b',
            r'\bempirical evidence\b', r'\bstatistical significance\b'
        ]
        
        # Educational indicators
        self.educational_patterns = [
            r'\bacademic\b', r'\bscholarly\b', r'\bresearch\b', r'\bstudy\b',
            r'\bthesis\b', r'\bdissertation\b', r'\bpublication\b', r'\bjournal\b',
            r'\bpeer-reviewed\b', r'\bempirical\b', r'\btheoretical\b',
            r'\bphilosophical\b', r'\bintellectual\b', r'\banalytical\b'
        ]
        
        # Citation/reference markers
        self.citation_patterns = [
            r'\bcited\b', r'\breference\b', r'\bsource\b', r'\bliterature\b',
            r'\baccording to\b', r'\bas stated\b', r'\bas noted\b',
            r'\bsee\b', r'\bcf\b', r'\bibid\b', r'\bet al\b',
            r'\bauthor\b', r'\bpublication\b', r'\bjournal\b'
        ]
        
        # Sophistication markers (complex reasoning)
        self.sophistication_patterns = [
            r'\bnevertheless\b', r'\bnonetheless\b', r'\bmoreover\b',
            r'\bfurthermore\b', r'\badditionally\b', r'\bconversely\b',
            r'\bparadoxically\b', r'\bironically\b', r'\bnotably\b',
            r'\bimportantly\b', r'\bsignificantly\b', r'\bnotably\b',
            r'\bcomplex\b', r'\bnuanced\b', r'\bsubtle\b', r'\bintricate\b'
        ]
        
        # Counter-narrative language (resistance to expected patterns)
        self.counter_narrative_patterns = [
            r'\bcontrary to\b', r'\bunexpectedly\b', r'\bsurprisingly\b',
            r'\bdefies\b', r'\bchallenges\b', r'\bcontradicts\b',
            r'\bgoes against\b', r'\bruns counter\b', r'\bbreaks with\b',
            r'\bunconventional\b', r'\batypical\b', r'\bnon-traditional\b'
        ]
        
        # Domain statistics
        self.domain_mean_theta_ = None
        self.domain_std_theta_ = None
    
    def _count_pattern_matches(self, text: str, patterns: List[str]) -> int:
        """Count matches for regex patterns."""
        text_lower = text.lower()
        count = 0
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            count += len(matches)
        return count
    
    def fit(self, X, y=None):
        """
        Fit transformer to data.
        
        Parameters
        ----------
        X : array-like of str
            Training narratives
        y : array-like, optional
            Target values (not used)
        
        Returns
        -------
        self : AwarenessResistanceTransformer
            Fitted transformer
        """
        # Convert to list if needed
        if isinstance(X, np.ndarray):
            X = X.tolist()
        
        # Extract features for all samples
        all_theta_values = []
        for text in X:
            if isinstance(text, str):
                features = self._extract_awareness_features(text)
                all_theta_values.append(features['awareness_resistance_theta'])
        
        # Compute domain statistics
        if all_theta_values:
            self.domain_mean_theta_ = np.mean(all_theta_values)
            self.domain_std_theta_ = np.std(all_theta_values)
            self.metadata['domain_mean_theta'] = float(self.domain_mean_theta_)
            self.metadata['domain_std_theta'] = float(self.domain_std_theta_)
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Transform narratives to awareness resistance features.
        
        Parameters
        ----------
        X : array-like of str
            Narratives to transform
        
        Returns
        -------
        features : ndarray
            Array of shape (n_samples, 15) with awareness resistance features
        """
        self._validate_fitted()
        
        # Convert to list if needed
        if isinstance(X, np.ndarray):
            X = X.tolist()
        
        features_list = []
        for text in X:
            if isinstance(text, str):
                feat_dict = self._extract_awareness_features(text)
                # Extract features in consistent order
                features_list.append([
                    feat_dict['meta_awareness_density'],
                    feat_dict['self_awareness_density'],
                    feat_dict['skepticism_density'],
                    feat_dict['questioning_density'],
                    feat_dict['doubt_density'],
                    feat_dict['critical_thinking_density'],
                    feat_dict['evidence_seeking_density'],
                    feat_dict['reasoning_density'],
                    feat_dict['field_awareness_density'],
                    feat_dict['bias_awareness_density'],
                    feat_dict['educational_density'],
                    feat_dict['citation_density'],
                    feat_dict['sophistication_density'],
                    feat_dict['counter_narrative_density'],
                    feat_dict['awareness_resistance_theta']
                ])
            else:
                # Return zeros if not string
                features_list.append([0.0] * 15)
        
        return np.array(features_list)
    
    def _extract_awareness_features(self, text: str) -> Dict[str, float]:
        """
        Extract all awareness resistance features from text.
        
        Parameters
        ----------
        text : str
            Input narrative text
        
        Returns
        -------
        features : dict
            Dictionary of feature values
        """
        text_lower = text.lower()
        words = text_lower.split()
        n_words = len(words)
        
        # Normalize by word count (per 100 words)
        normalization = n_words / 100.0 if n_words > 0 else 1.0
        
        features = {}
        
        # 1. Meta-awareness language density
        meta_count = self._count_pattern_matches(text_lower, self.meta_awareness_patterns)
        features['meta_awareness_density'] = meta_count / normalization
        
        # 2. Self-awareness indicators
        self_aware_count = self._count_pattern_matches(text_lower, self.self_awareness_patterns)
        features['self_awareness_density'] = self_aware_count / normalization
        
        # 3. Skepticism markers
        skepticism_count = self._count_pattern_matches(text_lower, self.skepticism_patterns)
        features['skepticism_density'] = skepticism_count / normalization
        
        # 4. Questioning language
        questioning_count = self._count_pattern_matches(text_lower, self.questioning_patterns)
        features['questioning_density'] = questioning_count / normalization
        
        # 5. Doubt indicators
        doubt_count = self._count_pattern_matches(text_lower, self.doubt_patterns)
        features['doubt_density'] = doubt_count / normalization
        
        # 6. Critical thinking indicators
        critical_count = self._count_pattern_matches(text_lower, self.critical_thinking_patterns)
        features['critical_thinking_density'] = critical_count / normalization
        
        # 7. Evidence-seeking language
        evidence_count = self._count_pattern_matches(text_lower, self.evidence_patterns)
        features['evidence_seeking_density'] = evidence_count / normalization
        
        # 8. Reasoning markers
        reasoning_count = self._count_pattern_matches(text_lower, self.reasoning_patterns)
        features['reasoning_density'] = reasoning_count / normalization
        
        # 9. Field-specific awareness (nominative determinism, bias)
        if self.detect_field_awareness:
            field_aware_count = self._count_pattern_matches(text_lower, self.field_awareness_patterns)
            features['field_awareness_density'] = field_aware_count / normalization
        else:
            features['field_awareness_density'] = 0.0
        
        # 10. Bias/stereotype awareness (subset of field awareness)
        bias_patterns = [p for p in self.field_awareness_patterns if 'bias' in p or 'stereotype' in p]
        bias_count = self._count_pattern_matches(text_lower, bias_patterns)
        features['bias_awareness_density'] = bias_count / normalization
        
        # 11. Educational indicators
        educational_count = self._count_pattern_matches(text_lower, self.educational_patterns)
        features['educational_density'] = educational_count / normalization
        
        # 12. Citation/reference markers
        citation_count = self._count_pattern_matches(text_lower, self.citation_patterns)
        features['citation_density'] = citation_count / normalization
        
        # 13. Sophistication markers
        sophistication_count = self._count_pattern_matches(text_lower, self.sophistication_patterns)
        features['sophistication_density'] = sophistication_count / normalization
        
        # 14. Counter-narrative language
        counter_narrative_count = self._count_pattern_matches(text_lower, self.counter_narrative_patterns)
        features['counter_narrative_density'] = counter_narrative_count / normalization
        
        # 15. Compute θ score (0-1)
        # Weighted combination of awareness indicators
        theta_score = self._compute_theta_score(features)
        features['awareness_resistance_theta'] = theta_score
        
        return features
    
    def _compute_theta_score(self, features: Dict[str, float]) -> float:
        """
        Compute θ (awareness resistance) score from features.
        
        Formula: Weighted combination of awareness indicators
        Higher values = more awareness/resistance
        
        Parameters
        ----------
        features : dict
            Extracted feature values
        
        Returns
        -------
        theta : float
            Awareness resistance score [0, 1]
        """
        # Weights for different awareness components
        weights = {
            'meta_awareness_density': 0.15,
            'self_awareness_density': 0.10,
            'skepticism_density': 0.12,
            'questioning_density': 0.08,
            'doubt_density': 0.05,
            'critical_thinking_density': 0.15,
            'evidence_seeking_density': 0.10,
            'reasoning_density': 0.08,
            'field_awareness_density': 0.10,  # High weight for field-specific awareness
            'bias_awareness_density': 0.05,
            'educational_density': 0.05,
            'citation_density': 0.03,
            'sophistication_density': 0.02,
            'counter_narrative_density': 0.02
        }
        
        # Compute weighted sum
        weighted_sum = sum(
            features.get(key, 0.0) * weight
            for key, weight in weights.items()
        )
        
        # Normalize to [0, 1] using sigmoid-like function
        # Typical range: 0-5 per 100 words, normalize to [0, 1]
        theta = 1.0 / (1.0 + np.exp(-weighted_sum * 2.0))  # Sigmoid normalization
        
        # Clamp to [0, 1]
        theta = max(0.0, min(1.0, theta))
        
        return theta
    
    def _generate_interpretation(self) -> str:
        """Generate human-readable interpretation."""
        if self.domain_mean_theta_ is None:
            return "No data fitted yet."
        
        mean_theta = self.domain_mean_theta_
        
        if mean_theta < 0.3:
            level = "low"
            interpretation = "Low awareness resistance - population shows minimal conscious resistance to narrative effects."
        elif mean_theta < 0.5:
            level = "moderate"
            interpretation = "Moderate awareness resistance - some conscious awareness but limited resistance."
        elif mean_theta < 0.7:
            level = "high"
            interpretation = "High awareness resistance - significant conscious awareness and resistance to narrative pull."
        else:
            level = "very high"
            interpretation = "Very high awareness resistance - strong meta-awareness and deliberate resistance (e.g., psychology, academics)."
        
        return f"Domain shows {level} awareness resistance (θ={mean_theta:.3f}). {interpretation}"
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return [
            'meta_awareness_density',
            'self_awareness_density',
            'skepticism_density',
            'questioning_density',
            'doubt_density',
            'critical_thinking_density',
            'evidence_seeking_density',
            'reasoning_density',
            'field_awareness_density',
            'bias_awareness_density',
            'educational_density',
            'citation_density',
            'sophistication_density',
            'counter_narrative_density',
            'awareness_resistance_theta'
        ]

