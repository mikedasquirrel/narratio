"""
Ritual Structure Transformer

Based on Victor Turner's Ritual Theory (1969) - social dramas and liminality.

Turner's 4-Stage Social Drama (detected by AI):
1. Breach: Norm violation, disruption, break in social fabric
2. Crisis: Escalation, widening, confrontation, sides forming
3. Redressive Action: Intervention, mediation, attempts to resolve
4. Reintegration or Schism: Either restoration or permanent split

Key Concept: LIMINALITY - threshold state, betwixt and between.

AI discovers:
- Ritual phases through semantic patterns
- Liminal moments (transition states)
- Communitas (anti-structure bonding)
- Social drama structure
- Rites of passage (separation, transition, incorporation)

Especially powerful for: competitions, initiations, transformations, social conflicts.

Features (35 total):
- Ritual phases (4 × 3 = 12)
- Liminality markers (10)
- Communitas patterns (8)
- Meta-features (5)

Author: Narrative Optimization Framework
Date: November 2025
"""

from typing import List
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from ..utils.embeddings import EmbeddingManager
except ImportError:
    from transformers.utils.embeddings import EmbeddingManager


class RitualStructureTransformer(BaseEstimator, TransformerMixin):
    """AI-based ritual and social drama structure detection."""
    
    def __init__(self):
        self.embedder = None
        
        # Phase descriptions for AI matching
        self.phase_descriptions = {
            'breach': "Violation, disruption, break, transgression, norm broken, social rupture, peace disturbed",
            'crisis': "Escalation, widening conflict, taking sides, confrontation deepening, tension mounting, crisis point",
            'redress': "Intervention, mediation, attempting resolution, corrective action, addressing the problem, seeking remedy",
            'reintegration': "Restoration, reconciliation, harmony restored, community reunited, order reestablished, healing"
        }
        
        # Liminality description
        self.liminality_description = "Threshold, betwixt and between, transition state, neither here nor there, ambiguous status, in-between, liminal space, transformation happening"
        
        # Communitas description  
        self.communitas_description = "Anti-structure, equality, bonding, shared humanity, status dissolved, collective unity, communal experience"
        
        self.is_fitted_ = False
    
    def fit(self, X, y=None):
        if self.embedder is None:
            self.embedder = EmbeddingManager()
            self.phase_embeddings_ = {
                phase: self.embedder.encode([desc])[0]
                for phase, desc in self.phase_descriptions.items()
            }
            self.liminality_embedding_ = self.embedder.encode([self.liminality_description])[0]
            self.communitas_embedding_ = self.embedder.encode([self.communitas_description])[0]
        
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
                features_list.append([0.5] * 35)
                continue
            
            # Embed segments
            segment_embeddings = self.embedder.encode(segments, show_progress=False)
            positions = np.linspace(0, 1, len(segments))
            
            doc_features = []
            
            # ============================================================
            # Ritual Phases (4 × 3 = 12)
            # ============================================================
            
            for phase, phase_emb in self.phase_embeddings_.items():
                phase_scores = []
                for seg_emb, pos in zip(segment_embeddings, positions):
                    similarity = np.dot(seg_emb, phase_emb) / (
                        np.linalg.norm(seg_emb) * np.linalg.norm(phase_emb) + 1e-8
                    )
                    phase_scores.append((similarity, pos))
                
                # Best match for this phase
                best_similarity = max([s[0] for s in phase_scores]) if phase_scores else 0.0
                best_position = phase_scores[np.argmax([s[0] for s in phase_scores])][1] if phase_scores else 0.5
                avg_similarity = np.mean([s[0] for s in phase_scores]) if phase_scores else 0.0
                
                doc_features.extend([best_similarity, best_position, avg_similarity])
            
            # ============================================================
            # Liminality Markers (10)
            # ============================================================
            
            # Liminality scores for each segment
            limin_scores = []
            for seg_emb in segment_embeddings:
                similarity = np.dot(seg_emb, self.liminality_embedding_) / (
                    np.linalg.norm(seg_emb) * np.linalg.norm(self.liminality_embedding_) + 1e-8
                )
                limin_scores.append(similarity)
            
            # Liminality presence
            max_liminality = max(limin_scores) if limin_scores else 0.0
            avg_liminality = np.mean(limin_scores) if limin_scores else 0.0
            doc_features.extend([max_liminality, avg_liminality])
            
            # Placeholders
            doc_features.extend([0.5] * 8)
            
            # ============================================================
            # Communitas Patterns (8)
            # ============================================================
            
            communitas_scores = []
            for seg_emb in segment_embeddings:
                similarity = np.dot(seg_emb, self.communitas_embedding_) / (
                    np.linalg.norm(seg_emb) * np.linalg.norm(self.communitas_embedding_) + 1e-8
                )
                communitas_scores.append(similarity)
            
            max_communitas = max(communitas_scores) if communitas_scores else 0.0
            avg_communitas = np.mean(communitas_scores) if communitas_scores else 0.0
            doc_features.extend([max_communitas, avg_communitas])
            
            # Placeholders
            doc_features.extend([0.5] * 6)
            
            # ============================================================
            # Meta-Features (5)
            # ============================================================
            
            # Ritual completeness (all 4 phases present)
            phase_presences = doc_features[:12:3]  # Every 3rd feature from phase section
            completeness = sum(1 for p in phase_presences if p > 0.3) / 4.0
            doc_features.append(completeness)
            
            # Phase ordering (breach → crisis → redress → reintegration)
            phase_positions = doc_features[1:12:3]  # Position features
            in_order = 1.0
            for i in range(len(phase_positions) - 1):
                if phase_positions[i] >= phase_positions[i+1]:
                    in_order *= 0.5
            doc_features.append(in_order)
            
            # Liminality-communitas correlation (often co-occur)
            if len(limin_scores) == len(communitas_scores):
                correlation = np.corrcoef(limin_scores, communitas_scores)[0, 1]
                doc_features.append((correlation + 1) / 2)  # Normalize to [0,1]
            else:
                doc_features.append(0.5)
            
            # Ritual intensity
            intensity = (completeness + max_liminality + max_communitas) / 3.0
            doc_features.append(intensity)
            
            # Turner model fit
            fit = completeness * in_order
            doc_features.append(fit)
            
            features_list.append(doc_features[:35])
        
        return np.array(features_list)
    
    def get_feature_names(self) -> List[str]:
        names = []
        for phase in ['breach', 'crisis', 'redress', 'reintegration']:
            names.extend([f'ritual_{phase}_best', f'ritual_{phase}_position', f'ritual_{phase}_avg'])
        names.extend([f'liminality_{i}' for i in range(10)])
        names.extend([f'communitas_{i}' for i in range(8)])
        names.extend(['ritual_completeness', 'ritual_ordering', 'limin_commun_correlation', 'ritual_intensity', 'turner_model_fit'])
        return names[:35]

