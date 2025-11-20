"""
Anticipatory Commitment Transformer

Names aren't just descriptive (what you ARE) but promissory (what you WILL BE).
Captures temporal commitment, self-fulfilling prophecies, and trajectory-locking effects.

Research Foundation:
- Papal names → policy alignment (r ≈ 0.85-0.90) - name predicts future actions
- Pen names → genre alignment - horror writers choose harsh names BEFORE writing horror
- Revolutionary names → revolutionary action - "Lenin" (strong) vs birth name (weak)
- Stage names predict performance style
- Startup pivots harder with committed names

Core Insight:
Names create TEMPORAL COMMITMENTS that lock trajectories:
- Choosing "Bitcoin" (digital + coin) commits to specific narrative
- Cannot pivot to unrelated purpose without massive cost
- Name = promissory note about future
- Self-fulfilling: Name creates expectations → expectations shape behavior → behavior fulfills name

This explains synchronicity: Not mystical - ANTICIPATORY signaling + confirmation bias.
"""

from typing import List, Dict, Any
import numpy as np
import re
from collections import Counter
from .base import NarrativeTransformer
from .utils.input_validation import ensure_string_list, ensure_string


class AnticipatoryCommunicationTransformer(NarrativeTransformer):
    """
    Analyzes anticipatory signaling and temporal commitment in narratives.
    
    Tests hypothesis that names function as promissory signals about future
    trajectories, creating self-fulfilling prophecies and locking commitments.
    
    Features extracted (25):
    - Future-commitment language (will, promise, commit)
    - Trajectory-locking indicators (focused, dedicated, specialized)
    - Promissory strength (confident future statements)
    - Self-fulfilling markers (becoming, manifesting)
    - Identity-destiny alignment (who I am → what I'll do)
    - Temporal binding (long-term orientation)
    - Expectation-setting language
    - Pivot flexibility (can change course?)
    - Path dependency indicators
    - Inevitability framing
    
    Parameters
    ----------
    measure_synchronicity : bool
        Whether to compute synchronicity potential scores
    """
    
    def __init__(self, measure_synchronicity: bool = True):
        super().__init__(
            narrative_id="anticipatory_commitment",
            description="Anticipatory commitment: names as promissory signals about future trajectories"
        )
        
        self.measure_synchronicity = measure_synchronicity
        
        # Future commitment language (strong promises)
        self.commitment_language = [
            'commit', 'committed', 'promise', 'pledge', 'vow', 'swear', 'dedicate',
            'devoted', 'determine', 'resolve', 'intend', 'aim', 'destined', 'meant to'
        ]
        
        # Trajectory-locking language (specialization, focus)
        self.trajectory_lock = [
            'focused', 'specialized', 'dedicated', 'exclusive', 'solely', 'only',
            'purely', 'entirely', 'completely', 'always', 'forever', 'permanent',
            'locked in', 'committed', 'bound', 'tied to', 'defined by'
        ]
        
        # Promissory strength (will + certainty)
        self.promissory_markers = [
            'will', 'shall', 'going to', 'destined', 'meant to', 'designed to',
            'built to', 'intended to', 'guaranteed', 'ensure', 'deliver', 'promise'
        ]
        
        # Self-fulfilling language
        self.self_fulfilling = [
            'become', 'becoming', 'manifest', 'manifesting', 'fulfill', 'realize',
            'actualize', 'embody', 'transform into', 'evolve into', 'grow into',
            'live up to', 'make real', 'bring to life'
        ]
        
        # Identity-destiny coupling
        self.identity_destiny = [
            'born to', 'made for', 'destined', 'calling', 'purpose', 'meant to be',
            'natural', 'innate', 'inherent', 'essential', 'core', 'fundamental'
        ]
        
        # Long-term orientation
        self.long_term_markers = [
            'legacy', 'lasting', 'enduring', 'permanent', 'forever', 'eternal',
            'generations', 'centuries', 'decades', 'long-term', 'sustained', 'ongoing'
        ]
        
        # Expectation-setting
        self.expectation_language = [
            'expect', 'anticipate', 'foresee', 'predict', 'envision', 'imagine',
            'project', 'forecast', 'assume', 'presume', 'suppose'
        ]
        
        # Pivot flexibility (ability to change)
        self.flexibility_markers = [
            'adapt', 'adjust', 'flexible', 'versatile', 'diverse', 'range',
            'variety', 'multiple', 'alternative', 'pivot', 'shift', 'evolve'
        ]
        
        # Path dependency
        self.path_dependence = [
            'history', 'roots', 'origins', 'began', 'started', 'initially',
            'originally', 'foundation', 'established', 'tradition', 'heritage'
        ]
        
        # Inevitability framing
        self.inevitability_markers = [
            'inevitable', 'certain', 'sure', 'definitely', 'undoubtedly', 'clearly',
            'obviously', 'necessarily', 'must', 'bound to', 'destined', 'fate'
        ]
        
    def fit(self, X, y=None):
        """
        Learn anticipatory patterns from corpus.
        
        Parameters
        ----------
        X : list of str
            Text documents
        y : ignored
        
        Returns
        -------
        self
        """
        # Ensure X is list of strings
        X = ensure_string_list(X)
        
        # Corpus statistics
        commitment_distribution = []
        promissory_distribution = []
        
        for text in X:
            text_lower = text.lower()
            
            commitment_count = sum(1 for m in self.commitment_language if m in text_lower)
            promissory_count = sum(1 for m in self.promissory_markers if m in text_lower)
            
            commitment_distribution.append(commitment_count)
            promissory_distribution.append(promissory_count)
        
        # Metadata
        self.metadata['avg_commitment'] = np.mean(commitment_distribution) if commitment_distribution else 0
        self.metadata['avg_promissory'] = np.mean(promissory_distribution) if promissory_distribution else 0
        self.metadata['high_commitment_rate'] = sum(1 for c in commitment_distribution if c > 2) / max(1, len(commitment_distribution))
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Transform documents to anticipatory commitment features.
        
        Parameters
        ----------
        X : list of str
            Documents to transform
        
        Returns
        -------
        features : array, shape (n_samples, 25)
            Anticipatory commitment feature matrix
        """
        self._validate_fitted()
        
        # Ensure X is list of strings
        X = ensure_string_list(X)
        
        
        features_list = []
        
        for text in X:
            doc_features = self._extract_anticipatory_features(text)
            features_list.append(doc_features)
        
        return np.array(features_list)
    
    def _extract_anticipatory_features(self, text: str) -> np.ndarray:
        """Extract all 25 anticipatory commitment features."""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        word_count = max(1, len(words))
        
        features = []
        
        # === TEMPORAL COMMITMENT (7 features) ===
        
        # 1. Commitment language density
        commitment_count = sum(1 for m in self.commitment_language if m in text_lower)
        features.append(commitment_count / word_count * 100)
        
        # 2. Trajectory-locking language
        lock_count = sum(1 for m in self.trajectory_lock if m in text_lower)
        features.append(lock_count / word_count * 100)
        
        # 3. Promissory strength (will + guarantees)
        promissory_count = sum(1 for m in self.promissory_markers if m in text_lower)
        features.append(promissory_count / word_count * 100)
        
        # 4. Long-term orientation
        long_term_count = sum(1 for m in self.long_term_markers if m in text_lower)
        features.append(long_term_count / word_count * 100)
        
        # 5. Overall commitment intensity
        total_commitment = commitment_count + lock_count + promissory_count + long_term_count
        features.append(total_commitment / word_count * 100)
        
        # 6. Commitment-flexibility balance
        flexibility_count = sum(1 for m in self.flexibility_markers if m in text_lower)
        total_cf = total_commitment + flexibility_count
        commitment_ratio = total_commitment / total_cf if total_cf > 0 else 0.5
        features.append(commitment_ratio)  # 0 = flexible, 1 = committed
        
        # 7. Temporal binding strength (how locked to future path)
        # High commitment + low flexibility = strong binding
        binding_strength = features[4] * (1.0 - flexibility_count / word_count * 100)
        features.append(binding_strength)
        
        # === PROMISSORY SIGNALING (6 features) ===
        
        # 8. Identity-destiny coupling
        id_destiny_count = sum(1 for m in self.identity_destiny if m in text_lower)
        features.append(id_destiny_count / word_count * 100)
        
        # 9. Self-fulfilling language
        self_fulfill_count = sum(1 for m in self.self_fulfilling if m in text_lower)
        features.append(self_fulfill_count / word_count * 100)
        
        # 10. Expectation-setting density
        expectation_count = sum(1 for m in self.expectation_language if m in text_lower)
        features.append(expectation_count / word_count * 100)
        
        # 11. Inevitability framing
        inevitability_count = sum(1 for m in self.inevitability_markers if m in text_lower)
        features.append(inevitability_count / word_count * 100)
        
        # 12. Promissory credibility (promises + evidence language)
        evidence_words = ['proven', 'demonstrated', 'shown', 'track record', 'history', 'evidence']
        evidence_count = sum(1 for m in evidence_words if m in text_lower)
        credibility = evidence_count / max(1, promissory_count + 1)
        features.append(credibility)
        
        # 13. Promise-delivery gap (high promises, low evidence = gap)
        gap = promissory_count / max(1, evidence_count + 1)
        features.append(min(10.0, gap))
        
        # === SELF-FULFILLING DYNAMICS (6 features) ===
        
        # 14. Confirmation bias potential
        # High expectations + inevitability = confirmation bias setup
        confirmation_potential = features[10] * features[11]
        features.append(confirmation_potential)
        
        # 15. Behavioral lock-in (identity → action coupling)
        # "I am X" + "X does Y" = behavioral lock-in
        identity_mentions = text_lower.count('i am') + text_lower.count("i'm") + text_lower.count('we are')
        action_verbs = len(re.findall(r'\b(do|act|perform|execute|deliver|achieve|accomplish)\b', text_lower))
        lock_in = identity_mentions * action_verbs / max(1, word_count)
        features.append(lock_in)
        
        # 16. Self-concept reinforcement
        # Becoming language = active self-shaping
        reinforcement = features[9] * features[8]  # self-fulfilling × identity-destiny
        features.append(reinforcement)
        
        # 17. Prophecy strength (prediction + certainty)
        # Strong prophecy = confident future statement
        prophecy = features[10] * features[11]  # expectation × inevitability
        features.append(prophecy)
        
        # 18. Reality construction (manifesting + actualization language)
        construction_words = ['create', 'build', 'make', 'construct', 'shape', 'form', 'craft']
        construction_count = sum(1 for m in construction_words if m in text_lower)
        reality_construction = construction_count * features[9]  # construction × self-fulfilling
        features.append(reality_construction)
        
        # 19. Synchronicity potential (name-outcome alignment likelihood)
        # High if: strong identity-destiny + commitment + self-fulfilling
        if self.measure_synchronicity:
            synchronicity = (features[8] + features[0] + features[9]) / 3.0
            features.append(synchronicity)
        else:
            features.append(0.0)
        
        # === PATH DEPENDENCY (6 features) ===
        
        # 20. Path dependence strength
        path_dep_count = sum(1 for m in self.path_dependence if m in text_lower)
        features.append(path_dep_count / word_count * 100)
        
        # 21. Historical constraint (past locks future)
        # High path dependence + low flexibility = constrained
        historical_constraint = features[19] * (1.0 - commitment_ratio)
        features.append(historical_constraint)
        
        # 22. Pivot difficulty (how hard to change trajectory)
        # High commitment + high path dependence = difficult pivot
        pivot_difficulty = features[0] * features[19]
        features.append(pivot_difficulty)
        
        # 23. Sunk cost trap potential
        # High commitment + high evidence of past = sunk cost psychology
        sunk_cost_trap = features[0] * features[19] / 100
        features.append(sunk_cost_trap)
        
        # 24. Escape velocity required (energy to break trajectory)
        # Composite of all locking factors
        escape_velocity = (
            features[6] +   # temporal binding
            features[21] +  # historical constraint
            features[17]    # prophecy strength
        ) / 3.0
        features.append(escape_velocity)
        
        # 25. Overall anticipatory lock-in
        # How much does this narrative lock future trajectory?
        lock_in_score = (
            features[0] +   # commitment
            features[6] +   # binding
            features[8] +   # identity-destiny
            features[11] +  # inevitability
            features[18]    # synchronicity potential
        ) / 5.0
        features.append(lock_in_score)
        
        return np.array(features)
    
    def get_feature_names(self) -> List[str]:
        """Return names of all 25 features."""
        return [
            # Temporal commitment (7)
            'commitment_language', 'trajectory_locking', 'promissory_strength',
            'long_term_orientation', 'commitment_intensity', 'commitment_flexibility_ratio',
            'temporal_binding_strength',
            
            # Promissory signaling (6)
            'identity_destiny_coupling', 'self_fulfilling_language', 'expectation_setting',
            'inevitability_framing', 'promissory_credibility', 'promise_delivery_gap',
            
            # Self-fulfilling dynamics (6)
            'confirmation_bias_potential', 'behavioral_lock_in', 'self_concept_reinforcement',
            'prophecy_strength', 'reality_construction', 'synchronicity_potential',
            
            # Path dependency (6)
            'path_dependence', 'historical_constraint', 'pivot_difficulty',
            'sunk_cost_trap', 'escape_velocity_required', 'overall_anticipatory_lock_in'
        ]
    
    def interpret_features(self, features: np.ndarray) -> Dict[str, Any]:
        """Interpret anticipatory commitment features in plain English."""
        interpretation = {
            'summary': self._generate_summary(features),
            'features': {},
            'insights': []
        }
        
        # Commitment intensity
        commitment = features[4]
        if commitment > 5.0:
            interpretation['insights'].append("STRONG temporal commitment - trajectory highly locked")
        elif commitment < 1.0:
            interpretation['insights'].append("Weak commitment - high flexibility, easy to pivot")
        
        # Synchronicity potential
        if self.measure_synchronicity:
            sync_potential = features[18]
            if sync_potential > 3.0:
                interpretation['insights'].append(f"HIGH synchronicity potential ({sync_potential:.1f}) - name likely to predict outcomes")
        
        # Lock-in
        lock_in = features[24]
        if lock_in > 4.0:
            interpretation['insights'].append("Strong anticipatory lock-in - self-fulfilling prophecy likely")
        elif lock_in < 1.0:
            interpretation['insights'].append("Low lock-in - outcome not predetermined by narrative")
        
        # Pivot difficulty
        pivot_diff = features[21]
        if pivot_diff > 5.0:
            interpretation['insights'].append("Very difficult to pivot - strong path dependency")
        
        return interpretation
    
    def _generate_summary(self, features: np.ndarray) -> str:
        """Generate plain English summary."""
        commitment = features[4]
        lock_in = features[24]
        sync_potential = features[18]
        pivot_diff = features[21]
        
        if lock_in > 4.0 and sync_potential > 3.0:
            return f"Strong anticipatory lock-in ({lock_in:.1f}): High commitment, self-fulfilling potential. Name predicts trajectory (synchronicity: {sync_potential:.1f})."
        elif commitment < 2.0 and pivot_diff < 2.0:
            return f"Flexible trajectory: Low commitment ({commitment:.1f}), easy to pivot. Outcome not predetermined."
        elif path_dependency := features[19] > 3.0:
            return f"Path-dependent: Historical constraints strong, future locked by past."
        else:
            return f"Moderate lock-in: Commitment {commitment:.1f}, lock-in {lock_in:.1f}, some trajectory constraint."

