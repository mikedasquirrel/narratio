"""
Universal Themes Transformer

Uses semantic embeddings and NLP to detect universal narrative themes.
No hardcoded words - all semantic similarity based.

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from .utils.input_validation import ensure_string_list

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


class UniversalThemesTransformer(BaseEstimator, TransformerMixin):
    """
    Detects 20 universal narrative themes using semantic embeddings.
    
    Uses sentence transformers to compute semantic similarity between
    narrative text and theme prototypes. No hardcoded word lists.
    
    Themes:
    1. Good vs Evil
    2. Love vs Hate
    3. Freedom vs Oppression
    4. Individual vs Society
    5. Coming of Age
    6. Redemption
    7. Sacrifice
    8. Identity/Self-Discovery
    9. Power/Corruption
    10. Survival
    11. Truth vs Deception
    12. Justice vs Revenge
    13. Loss of Innocence
    14. Hope vs Despair
    15. Legacy
    16. Transformation
    17. Betrayal
    18. Loyalty
    19. Ambition
    20. Mortality
    
    Total: 20 features
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize theme detector.
        
        Parameters
        ----------
        model_name : str
            Sentence transformer model to use
        """
        self.model_name = model_name
        self.embedder = None
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedder = SentenceTransformer(model_name)
            except:
                pass
        
        # Theme prototypes - multiple semantic descriptions per theme
        self.theme_prototypes = {
            'good_vs_evil': [
                "a battle between good and evil forces",
                "moral righteousness confronting corruption",
                "heroes fighting villains for what is right",
                "light versus darkness in moral terms",
                "ethical choices between right and wrong"
            ],
            'love_vs_hate': [
                "love confronting hatred and prejudice",
                "affection overcoming animosity",
                "compassion versus cruelty",
                "romantic love challenging enmity",
                "emotional connection defying hostility"
            ],
            'freedom_vs_oppression': [
                "fighting for liberty against tyranny",
                "escaping bondage and captivity",
                "resisting authoritarian control",
                "pursuing independence from domination",
                "liberation from systemic oppression"
            ],
            'individual_vs_society': [
                "person challenging social conformity",
                "individual defying collective expectations",
                "standing alone against crowd pressure",
                "personal identity versus social norms",
                "nonconformity challenging establishment"
            ],
            'coming_of_age': [
                "youth maturing into adulthood",
                "adolescent discovering adult realities",
                "growing up and losing childhood",
                "transition from innocence to experience",
                "young person learning life lessons"
            ],
            'redemption': [
                "seeking forgiveness for past wrongs",
                "atoning for previous mistakes",
                "transformation from wrongdoing to righteousness",
                "earning second chance through change",
                "moral recovery and renewal"
            ],
            'sacrifice': [
                "giving up something valuable for greater good",
                "personal loss for benefit of others",
                "selfless act requiring great cost",
                "surrendering desires for higher purpose",
                "paying price for meaningful cause"
            ],
            'identity_self_discovery': [
                "searching for true self and purpose",
                "discovering personal identity and meaning",
                "journey to understand who one really is",
                "uncovering authentic self beneath surface",
                "quest for self-knowledge and understanding"
            ],
            'power_corruption': [
                "authority leading to moral decay",
                "influence corrupting ethical principles",
                "absolute power destroying integrity",
                "ambition compromising values",
                "leadership breeding corruption"
            ],
            'survival': [
                "fighting to stay alive against odds",
                "enduring extreme circumstances to live",
                "persevering through life-threatening danger",
                "struggling to continue existing",
                "overcoming threats to survival"
            ],
            'truth_vs_deception': [
                "honesty confronting lies and manipulation",
                "reality versus illusion and falsehood",
                "uncovering hidden truths behind deception",
                "authenticity challenging pretense",
                "revelation of truth against concealment"
            ],
            'justice_vs_revenge': [
                "righteous punishment versus personal vengeance",
                "legal fairness confronting retribution",
                "measured justice versus violent payback",
                "moral accountability versus vindictive anger",
                "fair consequences versus vengeful retaliation"
            ],
            'loss_of_innocence': [
                "naive worldview shattered by harsh reality",
                "childhood purity destroyed by experience",
                "optimistic beliefs crushed by truth",
                "innocent perspective corrupted by knowledge",
                "pure ideals tarnished by world"
            ],
            'hope_vs_despair': [
                "optimism fighting against hopelessness",
                "faith persisting despite bleakness",
                "maintaining belief against overwhelming odds",
                "light of possibility in darkness",
                "resilient optimism confronting nihilism"
            ],
            'legacy': [
                "impact left on world after death",
                "memory and influence passing to future",
                "lasting contribution beyond lifetime",
                "inheritance of values to next generation",
                "enduring mark on history"
            ],
            'transformation': [
                "fundamental change in nature or character",
                "complete metamorphosis of being",
                "radical shift in identity or understanding",
                "evolution into different state",
                "profound personal revolution"
            ],
            'betrayal': [
                "trust violated by close ally",
                "loyalty broken by confidante",
                "faith destroyed by trusted person",
                "allegiance abandoned for advantage",
                "confidence shattered by deception"
            ],
            'loyalty': [
                "unwavering commitment despite adversity",
                "faithful devotion through hardship",
                "steadfast allegiance to person or cause",
                "dedication that endures challenges",
                "reliable support through difficulties"
            ],
            'ambition': [
                "driving desire to achieve greatness",
                "relentless pursuit of success and status",
                "hunger for accomplishment and recognition",
                "determination to reach highest goals",
                "aspiration propelling action forward"
            ],
            'mortality': [
                "confronting inevitability of death",
                "grappling with life's finite nature",
                "facing one's own impermanence",
                "awareness of existence ending",
                "acceptance of human fragility"
            ]
        }
        
        # Embed all prototypes
        self.prototype_embeddings = {}
        if self.embedder is not None:
            for theme, descriptions in self.theme_prototypes.items():
                self.prototype_embeddings[theme] = self.embedder.encode(descriptions)
    
    def fit(self, X, y=None):
        """Fit transformer (preprocessing only)"""
        X = ensure_string_list(X)
        return self
    
    def transform(self, X):
        """
        Transform texts to universal theme features.
        
        Parameters
        ----------
        X : array-like of strings
            Narrative texts
            
        Returns
        -------
        features : ndarray of shape (n_samples, 20)
            Theme detection features
        """
        X = ensure_string_list(X)
        features = []
        
        for text in X:
            feat = self._extract_theme_features(text)
            features.append(feat)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_theme_features(self, text: str) -> List[float]:
        """Extract theme similarity scores using embeddings"""
        features = []
        
        if self.embedder is None or not self.prototype_embeddings:
            # Fallback: return zeros
            return [0.0] * 20
        
        # Embed the text
        text_embedding = self.embedder.encode([text])[0]
        
        # Compute similarity to each theme's prototypes
        theme_order = [
            'good_vs_evil', 'love_vs_hate', 'freedom_vs_oppression',
            'individual_vs_society', 'coming_of_age', 'redemption',
            'sacrifice', 'identity_self_discovery', 'power_corruption',
            'survival', 'truth_vs_deception', 'justice_vs_revenge',
            'loss_of_innocence', 'hope_vs_despair', 'legacy',
            'transformation', 'betrayal', 'loyalty', 'ambition', 'mortality'
        ]
        
        for theme in theme_order:
            prototype_embeddings = self.prototype_embeddings[theme]
            
            # Compute cosine similarities
            similarities = []
            for proto_emb in prototype_embeddings:
                # Cosine similarity
                sim = np.dot(text_embedding, proto_emb) / (
                    np.linalg.norm(text_embedding) * np.linalg.norm(proto_emb) + 1e-9
                )
                similarities.append(sim)
            
            # Take maximum similarity (strongest match)
            theme_score = float(np.max(similarities))
            features.append(theme_score)
        
        return features
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names"""
        return np.array([
            'theme_good_vs_evil',
            'theme_love_vs_hate',
            'theme_freedom_vs_oppression',
            'theme_individual_vs_society',
            'theme_coming_of_age',
            'theme_redemption',
            'theme_sacrifice',
            'theme_identity_self_discovery',
            'theme_power_corruption',
            'theme_survival',
            'theme_truth_vs_deception',
            'theme_justice_vs_revenge',
            'theme_loss_of_innocence',
            'theme_hope_vs_despair',
            'theme_legacy',
            'theme_transformation',
            'theme_betrayal',
            'theme_loyalty',
            'theme_ambition',
            'theme_mortality'
        ])

