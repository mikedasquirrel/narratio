"""
Matchup Advantage Transformer

Analyzes competitive narratives to detect style contrasts, scheme advantages,
and counter-strategy language. Designed for sports domains but can be applied
to any competitive context.
"""

from typing import List, Dict
import numpy as np

from .base import NarrativeTransformer
from .utils.input_validation import ensure_string_list
from .utils.shared_models import SharedModelRegistry

try:
    from sentence_transformers import SentenceTransformer  # noqa: F401
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class MatchupAdvantageTransformer(NarrativeTransformer):
    """
    Extracts matchup advantage features from competitive narratives.

    Features:
        1. Style matchup contrast
        2. Strength vs weakness analysis
        3. Historical matchup references
        4. Scheme/strategy advantage
        5. Pace/tempo compatibility
        6. Strategic contrast markers
        7. Counter-style effectiveness
        8. Overall favorability index
    """

    FEATURE_NAMES = [
        'matchup_style_contrast',
        'matchup_strength_weakness',
        'matchup_historical_pattern',
        'matchup_scheme_advantage',
        'matchup_tempo_compatibility',
        'matchup_strategic_contrast',
        'matchup_counter_effectiveness',
        'matchup_favorability_index'
    ]

    def __init__(self, use_spacy: bool = True, use_embeddings: bool = True):
        super().__init__(
            narrative_id='matchup_advantage',
            description='Analyzes matchup language for style, tempo, and strategy.'
        )
        self.request_spacy = use_spacy
        self.request_embeddings = use_embeddings and SENTENCE_TRANSFORMERS_AVAILABLE
        self.nlp = None
        self.embedder = None
        self.prototype_embeddings: Dict[str, np.ndarray] = {}
        self.style_prototypes = {
            'aggressive_vs_defensive': [
                "attacking style versus defensive approach",
                "offensive mindset against conservative strategy",
                "aggressive play confronting cautious tactics",
            ],
            'fast_vs_slow': [
                "fast-paced game versus slow methodical play",
                "quick tempo against deliberate pace",
                "rapid style facing patient approach",
            ],
            'experience_vs_youth': [
                "veteran experience against youthful energy",
                "seasoned professional facing rising talent",
                "established presence versus emerging force",
            ],
            'power_vs_finesse': [
                "physical power versus technical skill",
                "raw strength against refined technique",
                "brute force facing tactical precision",
            ],
        }

    def fit(self, X, y=None):
        ensure_string_list(X)
        if self.request_spacy:
            self.nlp = SharedModelRegistry.get_spacy()
        if self.request_embeddings:
            self.embedder = SharedModelRegistry.get_sentence_transformer()
        if self.embedder:
            for matchup, examples in self.style_prototypes.items():
                self.prototype_embeddings[matchup] = self.embedder.encode(examples)
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
        if self.nlp:
            doc = self.nlp(text)
        else:
            return np.full(len(self.FEATURE_NAMES), 0.0, dtype=np.float32)

        style_score = self._compute_style_matchup(text, doc)
        strength_weakness = self._detect_strength_weakness_matchup(doc)
        historical_match = self._compute_historical_pattern(doc)
        scheme_advantage = self._detect_scheme_advantage(doc)
        tempo_compatibility = self._compute_tempo_compatibility(doc)
        strategic_contrast = self._compute_strategic_contrast(doc)
        counter_effectiveness = self._compute_counter_effectiveness(doc)
        favorability = np.mean(
            [
                style_score,
                strength_weakness,
                scheme_advantage,
                tempo_compatibility,
                counter_effectiveness,
            ]
        )

        return np.array(
            [
                style_score,
                strength_weakness,
                historical_match,
                scheme_advantage,
                tempo_compatibility,
                strategic_contrast,
                counter_effectiveness,
                favorability,
            ],
            dtype=np.float32,
        )

    def _compute_style_matchup(self, text: str, doc) -> float:
        score = 0.0
        if self.embedder and self.prototype_embeddings:
            text_emb = self.embedder.encode([text])[0]
            max_sim = 0.0
            for proto_embeddings in self.prototype_embeddings.values():
                sims = proto_embeddings @ text_emb / (
                    np.linalg.norm(proto_embeddings, axis=1) * np.linalg.norm(text_emb) + 1e-9
                )
                max_sim = max(max_sim, sims.max())
            score = float(max_sim)

        contrast_markers = {'versus', 'against', 'while', 'whereas', 'however'}
        for sent in doc.sents:
            if any(token.lemma_ in contrast_markers for token in sent):
                score += 0.1
        return float(min(1.0, score))

    @staticmethod
    def _detect_strength_weakness_matchup(doc) -> float:
        strength = {'strength', 'strong', 'advantage', 'superior', 'dominant'}
        weakness = {'weakness', 'weak', 'vulnerable', 'struggle', 'inferior'}
        strength_count = sum(1 for token in doc if token.lemma_ in strength)
        weakness_count = sum(1 for token in doc if token.lemma_ in weakness)
        if strength_count and weakness_count:
            total = strength_count + weakness_count
            balance = 1.0 - abs(strength_count - weakness_count) / total
            return float(balance)
        return 0.0

    @staticmethod
    def _compute_historical_pattern(doc) -> float:
        historical = {'history', 'previous', 'past', 'traditionally', 'record', 'last'}
        count = sum(1 for token in doc if token.lemma_ in historical)
        sentences = max(len(list(doc.sents)), 1)
        return float(min(1.0, (count / sentences) * 0.5))

    @staticmethod
    def _detect_scheme_advantage(doc) -> float:
        tactical = {'strategy', 'tactic', 'scheme', 'gameplan', 'plan', 'system'}
        advantage = {'advantage', 'favor', 'benefit', 'suit', 'help'}
        score = 0.0
        for sent in doc.sents:
            if any(token.lemma_ in tactical for token in sent) and any(
                token.lemma_ in advantage for token in sent
            ):
                score += 0.2
        return float(min(1.0, score))

    @staticmethod
    def _compute_tempo_compatibility(doc) -> float:
        tempo = {'pace', 'tempo', 'speed', 'fast', 'slow', 'quick', 'methodical'}
        count = sum(1 for token in doc if token.lemma_ in tempo)
        sentences = max(len(list(doc.sents)), 1)
        return float(min(1.0, (count / sentences) * 0.5))

    @staticmethod
    def _compute_strategic_contrast(doc) -> float:
        contrast_score = 0.0
        for token in doc:
            if token.tag_ in {'JJR', 'RBR'} or token.lemma_ == 'than':
                contrast_score += 0.1
        return float(min(1.0, contrast_score))

    @staticmethod
    def _compute_counter_effectiveness(doc) -> float:
        counter = {'counter', 'neutralize', 'negate', 'nullify', 'adjust', 'respond'}
        score = 0.0
        for token in doc:
            if token.lemma_ in counter:
                score += 0.15
                if any(child.lemma_ in {'can', 'able', 'capable', 'effective'} for child in token.children):
                    score += 0.1
        return float(min(1.0, score))

    def get_feature_names(self) -> List[str]:
        return self.FEATURE_NAMES

    def _generate_interpretation(self) -> str:
        return (
            "Measures competitive narrative signals such as style contrasts, "
            "strength-vs-weakness framing, and counter-strategy language to "
            "estimate matchup favorability."
        )

