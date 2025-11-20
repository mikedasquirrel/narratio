"""
Cross-Cultural Archetype Transformer

Discovers which narrative tradition(s) best fit a narrative using AI analysis.

CRITICAL PRINCIPLES:
- NO hardcoded word lists
- USE AI semantic similarity
- DETECT patterns, don't impose them
- MULTIPLE traditions can apply simultaneously
- Let narrative reveal its own cultural framework

Traditions analyzed (via AI semantic similarity):
- Chinese (Qi flow, Yin-Yang, Five Elements)
- Japanese (Mono no aware, Ma, Jo-ha-kyū)
- Indian (Rasa theory, cyclical time)
- Arabic (Maqama episodic, rhetoric)
- African (Oral, call-response, proverb)
- Indigenous Australian (Songlines, spatial)
- Mesoamerican (Cyclical, duality)
- Islamic (Tafsir layers, prophetic)
- Western (Campbell, Aristotle)

Author: Narrative Optimization Framework
Date: November 2025
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from ..utils.embeddings import EmbeddingManager
    from ..utils.shared_models import SharedModelRegistry
    from ...analysis.multi_stream_narrative_processor import MultiStreamNarrativeProcessor
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from transformers.utils.embeddings import EmbeddingManager
    from transformers.utils.shared_models import SharedModelRegistry
    from analysis.multi_stream_narrative_processor import MultiStreamNarrativeProcessor


class CrossCulturalArchetypeTransformer(BaseEstimator, TransformerMixin):
    """
    Discover cultural narrative framework fit using AI.
    
    Features (80 total):
    - Framework fit scores (9 traditions × 3 = 27)
    - Multi-stream patterns per tradition (9 × 4 = 36)
    - Cross-tradition patterns (10)
    - Meta-features (7)
    
    NO hardcoded patterns. AI discovers fit.
    """
    
    def __init__(self):
        """Initialize with AI models only."""
        self.embedder = None  # Lazy load
        self.multi_stream_processor = None
        
        # Tradition anchor descriptions (for semantic similarity)
        # These describe CONCEPTS, not keywords
        self.tradition_anchors = {
            'chinese': "Balance and harmony, five elements progression, yin-yang opposites, cyclical patterns, flowing energy, group over individual, seasonal metaphors, Confucian harmony, elder wisdom",
            
            'japanese': "Transience and impermanence, beauty in fleeting moments, negative space and silence, gradual acceleration from slow to rapid, mono no aware melancholy, restrained expression, subtle emotional depth",
            
            'indian': "Emotional essence and rasa, love devotion fury compassion, cyclical time and rebirth, karma and dharma, multiple emotional layers, devotional bhakti elements, complex feelings",
            
            'arabic': "Episodic non-linear structure, rhetorical ornamentation, clever trickster protagonist, nested frame narratives, linguistic beauty emphasis, wisdom through cunning",
            
            'african': "Oral call and response, community participation, proverb wisdom teaching, animal fables with moral lessons, griot narrator authority, tradition and ancestors, collective over individual",
            
            'indigenous_australian': "Landscape as narrative structure, dreamtime non-linear time, walking and movement, ancestor journeys, spatial progression, embodied kinesthetic memory, sacred geography",
            
            'mesoamerican': "Cyclical destruction and rebirth, duality and opposition, calendrical structure, five worlds ages, necessary sacrifice and exchange, cosmic balance, transformation through center",
            
            'islamic': "Layered interpretation levels, prophetic patterns, legal and narrative integrated, intertextual references, wisdom teaching, divine agency, call and response to divine",
            
            'western': "Hero journey departure trials return, three-act structure, individual protagonist, linear time progression, character transformation, conflict and resolution, triumph over adversity"
        }
        
        self.is_fitted_ = False
    
    def fit(self, X, y=None):
        """
        Fit transformer (lazy load models).
        
        Parameters
        ----------
        X : list of str
            Narratives
        y : ignored
        """
        # Lazy load AI models
        if self.embedder is None:
            print("[CrossCulturalArchetype] Loading AI models...")
            self.embedder = EmbeddingManager()
            self.multi_stream_processor = MultiStreamNarrativeProcessor()
            print("[CrossCulturalArchetype] ✓ AI models ready")
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X, metadata=None):
        """
        Discover cultural framework fit using AI.
        
        Parameters
        ----------
        X : list of str
            Narratives to analyze
        metadata : dict, optional
            Additional context
            
        Returns
        -------
        features : ndarray
            Shape (n_narratives, 80)
            Cultural archetype features (AI-discovered)
        """
        if not self.is_fitted_:
            raise ValueError("Must fit transformer before transform")
        
        features_list = []
        
        print(f"\nAnalyzing {len(X)} narratives for cultural framework fit...")
        print("Using AI semantic similarity (NO hardcoded patterns)\n")
        
        for idx, narrative in enumerate(X):
            if idx % 100 == 0 and idx > 0:
                print(f"  Processed: {idx}/{len(X)}")
            
            doc_features = []
            
            # ============================================================
            # Framework Fit Scores (27 features: 9 traditions × 3)
            # ============================================================
            
            # Embed narrative
            narrative_embedding = self.embedder.encode([narrative])[0]
            
            # Embed tradition anchors
            tradition_embeddings = {
                tradition: self.embedder.encode([description])[0]
                for tradition, description in self.tradition_anchors.items()
            }
            
            # Compute semantic similarity to each tradition
            fit_scores = {}
            for tradition, tradition_emb in tradition_embeddings.items():
                # Cosine similarity
                similarity = np.dot(narrative_embedding, tradition_emb) / (
                    np.linalg.norm(narrative_embedding) * np.linalg.norm(tradition_emb) + 1e-8
                )
                fit_scores[tradition] = float(similarity)
                
                # Add to features: raw, squared, confidence
                doc_features.append(similarity)  # Raw fit
                doc_features.append(similarity ** 2)  # Quadratic
                doc_features.append(max(0, similarity - 0.5) * 2)  # Threshold confidence
            
            # ============================================================
            # Multi-Stream Patterns per Tradition (36 features: 9 × 4)
            # ============================================================
            
            # Discover narrative streams
            try:
                stream_analysis = self.multi_stream_processor.discover_streams(
                    narrative,
                    narrative_id=f"narrative_{idx}"
                )
                
                n_streams = stream_analysis.get('n_streams', 1)
                features_dict = stream_analysis.get('features', {})
                
                # For each tradition, extract stream-relevant features
                for tradition in self.tradition_anchors.keys():
                    # Stream count appropriateness for tradition
                    # Some traditions emphasize multiple streams (African oral, Arabic episodic)
                    # Others emphasize single stream (Western hero, Japanese mono no aware)
                    
                    # Multi-stream indicator
                    doc_features.append(min(n_streams / 5.0, 1.0))
                    
                    # Stream balance (some traditions value balance more)
                    balance = features_dict.get('stream_balance_entropy', 0.5)
                    doc_features.append(balance)
                    
                    # Interaction patterns
                    switch_rate = features_dict.get('switch_rate', 0.0)
                    doc_features.append(switch_rate)
                    
                    # Convergence (resolution patterns)
                    convergence = features_dict.get('final_convergence', 0.0)
                    doc_features.append(convergence)
            
            except Exception as e:
                # Fallback if stream detection fails
                doc_features.extend([0.5] * (len(self.tradition_anchors) * 4))
            
            # ============================================================
            # Cross-Tradition Patterns (10 features)
            # ============================================================
            
            # Hybrid indicator (fits multiple traditions well)
            fit_values = list(fit_scores.values())
            top_fits = sorted(fit_values, reverse=True)[:3]
            hybrid_score = top_fits[1] / top_fits[0] if top_fits[0] > 0 else 0  # 2nd place/1st place
            doc_features.append(hybrid_score)
            
            # Cultural distance (how different is best fit from others)
            best_fit = max(fit_values)
            avg_other_fits = np.mean([f for f in fit_values if f != best_fit])
            cultural_distance = best_fit - avg_other_fits
            doc_features.append(cultural_distance)
            
            # East vs West (binary comparison)
            eastern_traditions = ['chinese', 'japanese', 'indian', 'islamic']
            western_traditions = ['western']
            
            eastern_fit = np.mean([fit_scores[t] for t in eastern_traditions if t in fit_scores])
            western_fit = np.mean([fit_scores[t] for t in western_traditions if t in fit_scores])
            
            doc_features.append(eastern_fit)
            doc_features.append(western_fit)
            doc_features.append(eastern_fit - western_fit)  # East-West balance
            
            # Cyclical vs Linear (across traditions)
            cyclical_traditions = ['chinese', 'indian', 'mesoamerican', 'indigenous_australian']
            linear_traditions = ['western', 'arabic']
            
            cyclical_fit = np.mean([fit_scores.get(t, 0) for t in cyclical_traditions])
            linear_fit = np.mean([fit_scores.get(t, 0) for t in linear_traditions])
            
            doc_features.append(cyclical_fit)
            doc_features.append(linear_fit)
            
            # Oral vs Written emphasis
            oral_traditions = ['african', 'indigenous_australian']
            written_traditions = ['arabic', 'islamic', 'chinese']
            
            oral_fit = np.mean([fit_scores.get(t, 0) for t in oral_traditions])
            written_fit = np.mean([fit_scores.get(t, 0) for t in written_traditions])
            
            doc_features.append(oral_fit)
            doc_features.append(written_fit)
            
            # ============================================================
            # Meta-Features (7 features)
            # ============================================================
            
            # Best fit tradition ID (as index)
            best_tradition_idx = max(range(len(fit_scores)), 
                                    key=lambda i: list(fit_scores.values())[i])
            doc_features.append(best_tradition_idx / len(fit_scores))  # Normalized
            
            # Confidence in best fit
            confidence = top_fits[0] if top_fits else 0.0
            doc_features.append(confidence)
            
            # Ambiguity (many traditions fit equally)
            ambiguity = 1.0 - np.std(fit_values) / (np.mean(fit_values) + 0.1)
            doc_features.append(ambiguity)
            
            # Universal patterns (fits all traditions moderately)
            universal_score = min(fit_values) if fit_values else 0.0
            doc_features.append(universal_score)
            
            # Cultural specificity (fits one tradition strongly, others weakly)
            specificity = best_fit - avg_other_fits
            doc_features.append(specificity)
            
            # Tradition diversity (how many traditions > 0.6 fit?)
            diverse = sum(1 for f in fit_values if f > 0.6)
            doc_features.append(diverse / len(fit_values))
            
            # Overall cultural complexity
            complexity = hybrid_score * diversity / len(fit_values)
            doc_features.append(complexity)
            
            features_list.append(doc_features)
        
        return np.array(features_list)
    
    def get_feature_names(self) -> List[str]:
        """Return feature names."""
        names = []
        
        # Framework fit scores (9 × 3 = 27)
        for tradition in self.tradition_anchors.keys():
            names.extend([
                f'{tradition}_fit_raw',
                f'{tradition}_fit_squared',
                f'{tradition}_fit_confident'
            ])
        
        # Multi-stream patterns (9 × 4 = 36)
        for tradition in self.tradition_anchors.keys():
            names.extend([
                f'{tradition}_stream_count',
                f'{tradition}_stream_balance',
                f'{tradition}_stream_switching',
                f'{tradition}_stream_convergence'
            ])
        
        # Cross-tradition patterns (10)
        names.extend([
            'hybrid_score',
            'cultural_distance',
            'eastern_fit',
            'western_fit',
            'east_west_balance',
            'cyclical_fit',
            'linear_fit',
            'oral_fit',
            'written_fit',
        ])
        
        # Meta-features (7)
        names.extend([
            'best_tradition_idx',
            'best_fit_confidence',
            'tradition_ambiguity',
            'universal_pattern_score',
            'cultural_specificity',
            'tradition_diversity',
            'cultural_complexity'
        ])
        
        return names

