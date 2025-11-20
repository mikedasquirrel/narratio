"""
Reputation & Prestige Transformer

Universal reputation/status analysis for prestige-driven domains.
"""

from typing import List, Dict, Union
import numpy as np

from .base import NarrativeTransformer
from .utils.shared_models import SharedModelRegistry
from .utils.input_validation import ensure_string_list


class ReputationPrestigeTransformer(NarrativeTransformer):
    """Extract reputation and prestige features from narratives or dict inputs."""

    FEATURE_NAMES = [
        'reputation_score',
        'reputation_prestige_level',
        'reputation_legacy',
        'reputation_award_history',
        'reputation_peer_recognition',
        'reputation_public_recognition',
        'reputation_media_coverage',
        'reputation_influencer_status',
        'reputation_authority_markers',
        'reputation_endorsement_quality',
        'reputation_scandal_score',
        'reputation_trajectory',
    ]

    def __init__(self, use_spacy: bool = True, use_embeddings: bool = True):
        super().__init__(
            narrative_id='reputation_prestige',
            description='Analyzes prestige, awards, and recognition signals.'
        )
        self.request_spacy = use_spacy
        self.request_embeddings = use_embeddings
        self.nlp = None
        self.embedder = None
        self.prototype_embeddings: Dict[str, np.ndarray] = {}
        self.prestige_prototypes = {
            'elite': "elite status with highest recognition and prestige",
            'legendary': "legendary reputation with historic significance",
            'authority': "authoritative position with expert recognition",
            'influencer': "influential reach with broad impact",
            'controversial': "controversial reputation with mixed public perception",
        }

    def fit(self, X, y=None):
        ensure_string_list(X)
        if self.request_spacy:
            self.nlp = SharedModelRegistry.get_spacy()
        if self.request_embeddings:
            self.embedder = SharedModelRegistry.get_sentence_transformer()
        if self.embedder:
            for concept, description in self.prestige_prototypes.items():
                self.prototype_embeddings[concept] = self.embedder.encode([description])[0]
        self.metadata['feature_names'] = self.FEATURE_NAMES
        self.is_fitted_ = True
        return self

    def transform(self, X):
        self._validate_fitted()
        features = []
        for item in X:
            features.append(self._extract_features(item))
        return np.array(features, dtype=np.float32)

    def _extract_features(self, item: Union[str, Dict]) -> List[float]:
        if isinstance(item, str):
            text = item
            metrics = {}
        elif isinstance(item, dict):
            text = str(item.get('text', item.get('narrative', '')))
            metrics = item
        else:
            text = str(item)
            metrics = {}

        doc = self.nlp(text[:5000]) if self.nlp and text else None

        return [
            self._compute_reputation_score(text, doc, metrics),
            self._compute_prestige_level(text, doc, metrics),
            self._compute_legacy_indicators(doc),
            self._compute_award_history(text, doc, metrics),
            self._compute_peer_recognition(text, doc, metrics),
            self._compute_public_recognition(text, doc, metrics),
            self._compute_media_coverage(text, doc, metrics),
            self._compute_influencer_status(text, doc, metrics),
            self._compute_authority_markers(doc),
            self._compute_endorsement_quality(doc),
            self._compute_scandal_score(doc),
            self._compute_reputation_trajectory(text, doc, metrics),
        ]

    @staticmethod
    def _compute_reputation_score(text: str, doc, metrics: Dict) -> float:
        if 'reputation_score' in metrics:
            return float(metrics['reputation_score'])
        keywords = {'famous', 'renowned', 'acclaimed', 'celebrated', 'respected', 'esteemed'}
        if doc:
            count = sum(1 for token in doc if token.lemma_ in keywords)
            return float(min(1.0, count / (len(doc) + 1) * 10))
        return 0.3 if text else 0.0

    def _compute_prestige_level(self, text: str, doc, metrics: Dict) -> float:
        if 'prestige_level' in metrics:
            return float(metrics['prestige_level'])
        score = 0.0
        if self.embedder and self.prototype_embeddings and text:
            text_emb = self.embedder.encode([text[:1000]])[0]
            elite_emb = self.prototype_embeddings.get('elite')
            if elite_emb is not None:
                sim = np.dot(text_emb, elite_emb) / (
                    np.linalg.norm(text_emb) * np.linalg.norm(elite_emb) + 1e-9
                )
                score += float(max(0.0, sim))
        prestige_terms = {'elite', 'prestigious', 'exclusive', 'distinguished'}
        if doc:
            count = sum(1 for token in doc if token.lemma_ in prestige_terms)
            score += min(0.3, count / (len(doc) + 1) * 10)
        return float(min(1.0, score))

    @staticmethod
    def _compute_legacy_indicators(doc) -> float:
        if not doc:
            return 0.3
        legacy_terms = {'legacy', 'legend', 'historic', 'iconic', 'immortal'}
        count = sum(1 for token in doc if token.lemma_ in legacy_terms)
        return float(min(1.0, count / (len(doc) + 1) * 10))

    @staticmethod
    def _compute_award_history(text: str, doc, metrics: Dict) -> float:
        if 'award_count' in metrics:
            return float(min(1.0, metrics['award_count'] / 10))
        if doc:
            award_terms = {'award', 'prize', 'honor', 'winner', 'champion', 'medal'}
            count = sum(1 for token in doc if token.lemma_ in award_terms)
            return float(min(1.0, count / (len(doc) + 1) * 10))
        return 0.3 if text else 0.0

    @staticmethod
    def _compute_peer_recognition(text: str, doc, metrics: Dict) -> float:
        if 'peer_recognition' in metrics:
            return float(metrics['peer_recognition'])
        peer_terms = {'peer', 'expert', 'professional', 'colleague', 'admired'}
        if doc:
            count = sum(1 for token in doc if token.lemma_ in peer_terms)
            return float(min(1.0, count / (len(doc) + 1) * 10))
        return 0.4 if text else 0.0

    @staticmethod
    def _compute_public_recognition(text: str, doc, metrics: Dict) -> float:
        if 'followers' in metrics:
            return float(min(1.0, metrics['followers'] / 1_000_000))
        public_terms = {'popular', 'celebrity', 'star', 'household'}
        if doc:
            count = sum(1 for token in doc if token.lemma_ in public_terms)
            return float(min(1.0, count / (len(doc) + 1) * 10))
        return 0.3 if text else 0.0

    @staticmethod
    def _compute_media_coverage(text: str, doc, metrics: Dict) -> float:
        if 'media_mentions' in metrics:
            return float(min(1.0, metrics['media_mentions'] / 1000))
        media_terms = {'media', 'press', 'headline', 'coverage', 'news'}
        if doc:
            count = sum(1 for token in doc if token.lemma_ in media_terms)
            return float(min(1.0, count / (len(doc) + 1) * 10))
        return 0.3 if text else 0.0

    @staticmethod
    def _compute_influencer_status(text: str, doc, metrics: Dict) -> float:
        if 'influence_score' in metrics:
            return float(metrics['influence_score'])
        influence_terms = {'influence', 'impact', 'shape', 'change', 'inspire'}
        if doc:
            count = sum(1 for token in doc if token.lemma_ in influence_terms)
            return float(min(1.0, count / (len(doc) + 1) * 10))
        return 0.4 if text else 0.0

    @staticmethod
    def _compute_authority_markers(doc) -> float:
        if not doc:
            return 0.3
        titles = {'Dr.', 'Prof.', 'Sir', 'Dame'}
        authority_terms = {'expert', 'authority', 'specialist', 'master'}
        title_count = sum(1 for token in doc if token.text in titles)
        authority_count = sum(1 for token in doc if token.lemma_ in authority_terms)
        return float(min(1.0, (title_count + authority_count) / (len(doc) + 1) * 10))

    @staticmethod
    def _compute_endorsement_quality(doc) -> float:
        if not doc:
            return 0.3
        endorsement_terms = {'endorse', 'support', 'back', 'sponsor', 'recommend'}
        count = sum(1 for token in doc if token.lemma_ in endorsement_terms)
        return float(min(1.0, count / (len(doc) + 1) * 10))

    @staticmethod
    def _compute_scandal_score(doc) -> float:
        if not doc:
            return 0.0
        scandal_terms = {'scandal', 'controversy', 'accused', 'allegation', 'criticized'}
        count = sum(1 for token in doc if token.lemma_ in scandal_terms)
        return float(min(1.0, count / (len(doc) + 1) * 10))

    @staticmethod
    def _compute_reputation_trajectory(text: str, doc, metrics: Dict) -> float:
        if 'reputation_trend' in metrics:
            return float(metrics['reputation_trend'])
        if not doc:
            return 0.5
        rising = {'rising', 'growing', 'ascending', 'climbing'}
        falling = {'falling', 'declining', 'fading', 'waning'}
        pos = sum(1 for token in doc if token.lemma_ in rising)
        neg = sum(1 for token in doc if token.lemma_ in falling)
        total = pos + neg
        if total > 0:
            return float(pos / total)
        return 0.5

    def get_feature_names(self) -> List[str]:
        return self.FEATURE_NAMES

    def _generate_interpretation(self) -> str:
        return (
            "Measures reputation, prestige, and recognition cues by analyzing "
            "awards, media coverage, authority markers, and reputation trajectory."
        )

