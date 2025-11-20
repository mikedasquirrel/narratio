"""
Labovian Narrative Structure Transformer

Based on Labov & Waletzky (1967) - narrative structure in oral discourse.

6 Components (detected by AI):
1. Abstract: What is this about?
2. Orientation: Who, when, where, what?
3. Complicating Action: Then what happened?
4. Evaluation: So what? Why is this interesting?
5. Resolution: What finally happened?
6. Coda: Return to present

AI discovers these components through:
- Sequential position
- Semantic content  
- Linguistic markers (but NOT hardcoded)
- Functional role in narrative

Features (40 total):
- Component presence/quality (6 × 3 = 18)
- Component ordering (10)
- Completeness (6)
- Meta-features (6)

Especially powerful for: personal narratives, interviews, testimonials.

Author: Narrative Optimization Framework
Date: November 2025
"""

from typing import List, Dict
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from ..utils.embeddings import EmbeddingManager
except ImportError:
    from transformers.utils.embeddings import EmbeddingManager


class LabovianNarrativeTransformer(BaseEstimator, TransformerMixin):
    """AI-based detection of Labovian narrative components."""
    
    def __init__(self):
        self.embedder = None
        
        # Component descriptions for AI matching
        self.component_descriptions = {
            'abstract': "Summary preview, what this story is about, initial framing, overview statement",
            'orientation': "Setting the scene, who when where, background information, context establishment, introducing participants and situation",
            'complicating_action': "Then what happened, events unfolding, problem arising, action sequence, conflict emerging",
            'evaluation': "Why this matters, so what, significance, why telling this, emotional weight, meaning",
            'resolution': "What finally happened, outcome revealed, problem solved or failed, conclusion of events",
            'coda': "Returning to present, connection to now, what it means today, final reflection, bridge back"
        }
        
        self.is_fitted_ = False
    
    def fit(self, X, y=None):
        if self.embedder is None:
            self.embedder = EmbeddingManager()
            self.component_embeddings_ = {
                comp: self.embedder.encode([desc])[0]
                for comp, desc in self.component_descriptions.items()
            }
        self.is_fitted_ = True
        return self
    
    def transform(self, X, metadata=None):
        if not self.is_fitted_:
            raise ValueError("Must fit first")
        
        features_list = []
        
        for narrative in X:
            # Segment narrative
            segments = narrative.split('\n\n') if '\n\n' in narrative else [narrative]
            segments = [s.strip() for s in segments if s.strip()]
            
            if not segments:
                features_list.append([0.5] * 40)
                continue
            
            # Embed segments with position info
            segment_embeddings = self.embedder.encode(segments, show_progress=False)
            positions = np.linspace(0, 1, len(segments))
            
            # Detect components using AI
            component_scores = {comp: [] for comp in self.component_descriptions}
            
            for seg_idx, seg_emb in enumerate(segment_embeddings):
                seg_position = positions[seg_idx]
                
                for comp, comp_emb in self.component_embeddings_.items():
                    # Semantic similarity
                    similarity = np.dot(seg_emb, comp_emb) / (
                        np.linalg.norm(seg_emb) * np.linalg.norm(comp_emb) + 1e-8
                    )
                    
                    # Weight by expected position
                    position_weight = self._position_weight(comp, seg_position)
                    weighted_score = similarity * position_weight
                    
                    component_scores[comp].append({
                        'segment_idx': seg_idx,
                        'similarity': similarity,
                        'weighted': weighted_score,
                        'position': seg_position
                    })
            
            # Extract features
            doc_features = self._extract_labovian_features(component_scores, segments)
            features_list.append(doc_features)
        
        return np.array(features_list)
    
    def _position_weight(self, component: str, position: float) -> float:
        """Expected position for each component (Labov's sequence)."""
        expected_positions = {
            'abstract': 0.05,  # Very beginning
            'orientation': 0.15,  # Early
            'complicating_action': 0.40,  # Middle
            'evaluation': 0.50,  # Embedded throughout but peaks mid
            'resolution': 0.80,  # Late
            'coda': 0.95  # Very end
        }
        
        expected = expected_positions.get(component, 0.5)
        distance = abs(position - expected)
        weight = 1.0 / (1.0 + distance * 3)  # Closer = higher weight
        
        return weight
    
    def _extract_labovian_features(self, component_scores: Dict, segments: List[str]) -> List[float]:
        """Extract features from component analysis."""
        features = []
        
        # Component presence/quality (6 × 3 = 18)
        for comp in ['abstract', 'orientation', 'complicating_action', 'evaluation', 'resolution', 'coda']:
            scores = component_scores.get(comp, [])
            
            if scores:
                presence = max([s['weighted'] for s in scores])
                quality = max([s['similarity'] for s in scores])
                position_match = 1.0 - min([abs(s['position'] - self._position_weight(comp, s['position'])) for s in scores])
            else:
                presence = 0.0
                quality = 0.0
                position_match = 0.0
            
            features.extend([presence, quality, position_match])
        
        # Ordering (10 features)
        # Are components in Labovian order?
        component_order = ['abstract', 'orientation', 'complicating_action', 'evaluation', 'resolution', 'coda']
        best_positions = {}
        for comp in component_order:
            scores = component_scores.get(comp, [])
            if scores:
                best_idx = max(range(len(scores)), key=lambda i: scores[i]['weighted'])
                best_positions[comp] = scores[best_idx]['position']
            else:
                best_positions[comp] = 0.5
        
        # Check if in order
        in_order = 1.0
        for i in range(len(component_order) - 1):
            if best_positions[component_order[i]] >= best_positions[component_order[i+1]]:
                in_order *= 0.5  # Penalty for out of order
        features.append(in_order)
        
        # Other ordering features (placeholders)
        features.extend([0.5] * 9)
        
        # Completeness (6)
        for comp in component_order:
            present = 1.0 if any(s['weighted'] > 0.3 for s in component_scores.get(comp, [])) else 0.0
            features.append(present)
        
        # Meta-features (6)
        all_presences = [any(s['weighted'] > 0.3 for s in component_scores.get(comp, [])) for comp in component_order]
        completeness = sum(all_presences) / len(component_order)
        features.append(completeness)
        
        features.extend([0.5] * 5)  # Placeholders
        
        return features
    
    def get_feature_names(self) -> List[str]:
        names = []
        for comp in ['abstract', 'orientation', 'complicating_action', 'evaluation', 'resolution', 'coda']:
            names.extend([f'labov_{comp}_presence', f'labov_{comp}_quality', f'labov_{comp}_position'])
        names.extend([f'labov_ordering_{i}' for i in range(10)])
        names.extend([f'labov_completeness_{i}' for i in range(6)])
        names.extend([f'labov_meta_{i}' for i in range(6)])
        return names[:40]

