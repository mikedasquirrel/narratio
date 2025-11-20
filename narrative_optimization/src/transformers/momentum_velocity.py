"""
Momentum & Velocity Transformer

Analyzes narrative momentum, change velocity, and trajectory.
"""

from typing import List
import numpy as np

from .base import NarrativeTransformer
from .utils.input_validation import ensure_string_list
from .utils.shared_models import SharedModelRegistry


class MomentumVelocityTransformer(NarrativeTransformer):
    """
    Extract narrative momentum and velocity features.

    Features:
        1. Narrative momentum
        2. Velocity of change
        3. Acceleration indicators
        4. Trajectory predictors
        5. Momentum sustainability
    """

    FEATURE_NAMES = [
        'momentum_narrative_momentum',
        'momentum_velocity_of_change',
        'momentum_acceleration',
        'momentum_trajectory',
        'momentum_sustainability',
    ]

    def __init__(self, use_spacy: bool = True):
        super().__init__(
            narrative_id='momentum_velocity',
            description='Quantifies narrative momentum, velocity, and trajectory.'
        )
        self.request_spacy = use_spacy
        self.nlp = None

    def fit(self, X, y=None):
        ensure_string_list(X)
        if self.request_spacy:
            self.nlp = SharedModelRegistry.get_spacy()
        self.metadata['feature_names'] = self.FEATURE_NAMES
        self.is_fitted_ = True
        return self

    def transform(self, X):
        self._validate_fitted()
        texts = ensure_string_list(X)
        outputs = np.zeros((len(texts), len(self.FEATURE_NAMES)), dtype=np.float32)
        for i, text in enumerate(texts):
            outputs[i] = self._extract_features(text)
        return outputs

    def _extract_features(self, text: str) -> np.ndarray:
        if not self.nlp:
            return np.array([0.5, 0.4, 0.3, 0.5, 0.4], dtype=np.float32)
        doc = self.nlp(text)
        sentences = list(doc.sents)
        return np.array(
            [
                self._compute_momentum(sentences),
                self._compute_velocity(sentences),
                self._compute_acceleration(sentences),
                self._compute_trajectory(sentences),
                self._compute_sustainability(sentences),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _compute_momentum(sentences: List) -> float:
        if not sentences:
            return 0.0
        action_verbs = sum(
            1
            for sent in sentences
            for token in sent
            if token.pos_ == 'VERB'
            and token.dep_ in {'ROOT', 'conj'}
            and token.lemma_ not in {'be', 'have', 'do'}
        )
        progressive = sum(1 for sent in sentences for token in sent if token.tag_ == 'VBG')
        tempo_markers = {'next', 'then', 'after', 'following', 'soon', 'later', 'eventually'}
        tempo = sum(1 for sent in sentences for tok in sent if tok.lemma_ in tempo_markers)
        score = (
            min(0.4, action_verbs / (len(sentences) + 1) / 3)
            + min(0.3, tempo / (len(sentences) + 1) * 3)
            + min(0.3, progressive / (len(sentences) + 1) * 2)
        )
        return float(min(1.0, score))

    @staticmethod
    def _compute_velocity(sentences: List) -> float:
        if len(sentences) < 4:
            return 0.0
        quarter = max(1, len(sentences) // 4)
        change_terms = {'change', 'transform', 'shift', 'evolve', 'progress', 'advance'}
        change_counts = []
        for i in range(4):
            segment = sentences[i * quarter:(i + 1) * quarter]
            if not segment:
                change_counts.append(0)
                continue
            count = sum(1 for sent in segment for token in sent if token.lemma_ in change_terms)
            change_counts.append(count / len(segment))
        velocity = np.mean(change_counts)
        return float(min(1.0, velocity * 5))

    @staticmethod
    def _compute_acceleration(sentences: List) -> float:
        if len(sentences) < 4:
            return 0.0
        quarter = max(1, len(sentences) // 4)
        first = sentences[:quarter]
        last = sentences[-quarter:]
        first_density = (
            sum(1 for sent in first for token in sent if token.pos_ == 'VERB' and token.dep_ != 'aux') / (len(first) + 1)
        )
        last_density = (
            sum(1 for sent in last for token in sent if token.pos_ == 'VERB' and token.dep_ != 'aux') / (len(last) + 1)
        )
        if first_density == 0:
            return float(min(1.0, last_density))
        acceleration = (last_density - first_density) / (first_density + 0.1)
        return float(np.clip(acceleration, 0, 1))

    @staticmethod
    def _compute_trajectory(sentences: List) -> float:
        if len(sentences) < 3:
            return 0.5
        third = max(1, len(sentences) // 3)
        positive = {'good', 'better', 'improve', 'win', 'succeed', 'achieve'}
        negative = {'bad', 'worse', 'decline', 'lose', 'fail', 'struggle'}
        sentiments = []
        for i in range(3):
            segment = sentences[i * third:(i + 1) * third]
            pos = sum(1 for sent in segment for token in sent if token.lemma_ in positive)
            neg = sum(1 for sent in segment for token in sent if token.lemma_ in negative)
            sentiments.append(pos - neg)
        if len(sentiments) >= 2:
            trend = sentiments[-1] - sentiments[0]
            return float(np.clip((trend + 3) / 6, 0, 1))
        return 0.5

    @staticmethod
    def _compute_sustainability(sentences: List) -> float:
        if len(sentences) < 5:
            return 0.5
        segments = 5
        segment_size = max(1, len(sentences) // segments)
        densities = []
        for i in range(segments):
            segment = sentences[i * segment_size:(i + 1) * segment_size]
            if not segment:
                continue
            actions = sum(1 for sent in segment for token in sent if token.pos_ == 'VERB' and token.dep_ == 'ROOT')
            densities.append(actions / len(segment))
        if not densities:
            return 0.5
        variance = np.var(densities)
        mean = np.mean(densities)
        if mean <= 0:
            return 0.5
        cv = (variance ** 0.5) / mean
        return float(1 / (1 + cv))

    def get_feature_names(self) -> List[str]:
        return self.FEATURE_NAMES

    def _generate_interpretation(self) -> str:
        return (
            "Captures narrative motion by analyzing action density, change markers, "
            "sentiment trajectory, and the sustainability of momentum."
        )

