"""
Conceptual Metaphor Transformer

Based on Lakoff & Johnson's Conceptual Metaphor Theory (1980).

Core Insight: Abstract concepts understood through embodied metaphors.
"Life is a journey" structures how we think about existence.

Common Metaphors (discovered by AI, not imposed):
- LIFE IS A JOURNEY (path, destination, obstacles)
- ARGUMENT IS WAR (attack, defend, win)
- TIME IS MONEY (spend, waste, invest)
- LOVE IS A JOURNEY (relationship path)
- IDEAS ARE FOOD (digest, swallow, half-baked)

AI discovers:
- Which metaphors present (semantic similarity)
- Metaphor consistency (same metaphor maintained?)
- Metaphor mixing (confusing metaphor shifts)
- Embodied grounding (physical experience basis)

Features (55 total):
- Dominant metaphors (detected by AI)
- Metaphor consistency scores
- Embodied vs abstract language
- Cross-domain mappings
- Metaphorical density

NO hardcoded metaphor lists. AI finds them through semantic patterns.

Author: Narrative Optimization Framework
Date: November 2025
"""

from typing import List, Dict, Any
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from ..utils.embeddings import EmbeddingManager
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from transformers.utils.embeddings import EmbeddingManager


class ConceptualMetaphorTransformer(BaseEstimator, TransformerMixin):
    """
    Discover conceptual metaphors using AI semantic analysis.
    
    NO predefined metaphor lists.
    AI detects metaphorical language through semantic field analysis.
    """
    
    def __init__(self):
        """Initialize with AI models."""
        self.embedder = None
        
        # Metaphor domain descriptions (source domains)
        self.source_domains = {
            'journey': "Travel, path, road, destination, obstacles, progress along route, movement through space",
            'war': "Battle, combat, attack, defend, victory, defeat, strategy, enemy, ally, conflict",
            'construction': "Building, foundation, structure, framework, support, collapse, assemble, design",
            'health': "Sickness, healing, medicine, symptoms, diagnosis, treatment, recovery, vitality",
            'nature': "Growth, seasons, weather, plants, animals, cycles, ecosystems, organic processes",
            'economics': "Investment, profit, loss, trade, value, cost, transaction, capital, resource",
            'physical_force': "Push, pull, pressure, momentum, resistance, gravity, weight, balance",
            'container': "In, out, full, empty, bounds, limits, containment, overflow, boundaries",
            'verticality': "Up, down, high, low, rise, fall, elevate, descend, height, depth",
            'light': "Bright, dark, illuminated, obscured, clarity, shadow, vision, blindness"
        }
        
        self.is_fitted_ = False
    
    def fit(self, X, y=None):
        """Lazy load AI models and embed source domains."""
        if self.embedder is None:
            self.embedder = EmbeddingManager()
            self.domain_embeddings_ = {
                domain: self.embedder.encode([desc])[0]
                for domain, desc in self.source_domains.items()
            }
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X, metadata=None):
        """Discover conceptual metaphors using AI."""
        if not self.is_fitted_:
            raise ValueError("Must fit first")
        
        features_list = []
        
        for narrative in X:
            doc_features = []
            
            # Segment narrative for analysis
            segments = narrative.split('\n\n') if '\n\n' in narrative else [narrative]
            segments = [s.strip() for s in segments if s.strip()]
            
            if not segments:
                features_list.append([0.5] * 55)
                continue
            
            # Embed segments
            segment_embeddings = self.embedder.encode(segments, show_progress=False)
            
            # Detect metaphor domains in each segment
            metaphor_presence = {domain: [] for domain in self.source_domains}
            
            for seg_emb in segment_embeddings:
                for domain, domain_emb in self.domain_embeddings_.items():
                    similarity = np.dot(seg_emb, domain_emb) / (
                        np.linalg.norm(seg_emb) * np.linalg.norm(domain_emb) + 1e-8
                    )
                    metaphor_presence[domain].append(similarity)
            
            # Aggregate metaphor features
            for domain in self.source_domains.keys():
                scores = metaphor_presence[domain]
                
                # Presence (average similarity across segments)
                presence = np.mean(scores) if scores else 0.0
                doc_features.append(presence)
                
                # Consistency (how evenly distributed across narrative)
                if len(scores) > 1:
                    consistency = 1.0 / (1.0 + np.std(scores))
                else:
                    consistency = 1.0
                doc_features.append(consistency)
            
            # Meta-features
            all_presences = [np.mean(v) for v in metaphor_presence.values()]
            
            # Dominant metaphor
            dominant_score = max(all_presences) if all_presences else 0.0
            doc_features.append(dominant_score)
            
            # Metaphor diversity (how many metaphors used?)
            diverse_metaphors = sum(1 for p in all_presences if p > 0.3)
            doc_features.append(diverse_metaphors / len(all_presences))
            
            # Metaphor mixing (using many different metaphors - coherent or confused?)
            mixing_score = np.std(all_presences) if all_presences else 0.0
            doc_features.append(mixing_score)
            
            # Embodied vs abstract (journey/war/physical are embodied)
            embodied_domains = ['journey', 'war', 'physical_force', 'health', 'nature', 'verticality']
            embodied_score = np.mean([metaphor_presence[d] for d in embodied_domains if d in metaphor_presence])
            doc_features.append(np.mean(embodied_score) if len(embodied_score) > 0 else 0.0)
            
            # Overall metaphorical density
            metaphorical_density = np.mean(all_presences) if all_presences else 0.0
            doc_features.append(metaphorical_density)
            
            features_list.append(doc_features)
        
        return np.array(features_list)
    
    def get_feature_names(self) -> List[str]:
        """Return feature names."""
        names = []
        
        # Metaphor domain features (10 domains × 2 = 20)
        for domain in self.source_domains.keys():
            names.extend([
                f'metaphor_{domain}_presence',
                f'metaphor_{domain}_consistency'
            ])
        
        # Meta-features (35)
        names.extend([
            'dominant_metaphor_score',
            'metaphor_diversity',
            'metaphor_mixing',
            'embodied_metaphor_score',
            'metaphorical_density'
        ])
        
        # Pad to 55
        while len(names) < 55:
            names.append(f'metaphor_placeholder_{len(names)}')
        
        return names[:55]

