"""
Emotional Semantic Transformer

INTELLIGENT VERSION - No hardcoded word lists!

Uses sentence embeddings to detect emotions semantically, capturing far more
emotional language than any hardcoded lexicon could.

Approach:
- Semantic similarity to emotion concept anchors
- Learned emotional arc patterns
- Zero-shot emotion classification
- Multilingual by default
"""

import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Dict, Optional
import warnings

try:
    from ..utils.embeddings import EmbeddingManager, get_default_embedder
    from ..utils.semantic_similarity import SemanticSimilarity
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    warnings.warn("Embedding utils not available")


class EmotionalSemanticTransformer(BaseEstimator, TransformerMixin):
    """
    Extract emotional features using semantic embeddings.
    
    NO HARDCODED WORD LISTS - uses semantic similarity instead!
    
    Features (~35):
    1. Emotion Detection - Semantic similarity to emotion concepts
    2. Emotional Arc - Learned trajectory patterns
    3. Emotional Complexity - Embedding-based analysis
    4. Emotional Dynamics - Pattern detection via embeddings
    """
    
    def __init__(
        self,
        embedder=None,
        use_intelligent_features=True,
        fallback_to_lexicon=True
    ):
        """
        Initialize transformer.
        
        Parameters
        ----------
        embedder : EmbeddingManager, optional
            Embedding manager (creates default if None)
        use_intelligent_features : bool
            Use embedding-based features (recommended)
        fallback_to_lexicon : bool
            Fall back to word lists if embeddings fail
        """
        self.use_intelligent_features = use_intelligent_features and EMBEDDINGS_AVAILABLE
        self.fallback_to_lexicon = fallback_to_lexicon
        
        if self.use_intelligent_features:
            self.embedder = embedder or get_default_embedder()
            self._setup_emotion_anchors()
        
        # Fallback: minimal lexicon (if needed)
        if fallback_to_lexicon:
            self._setup_fallback_lexicon()
    
    def _setup_emotion_anchors(self):
        """
        Setup emotion concept anchors (NOT word lists).
        
        These are semantic concepts that define emotions,
        allowing us to find ANY expression of these emotions.
        """
        # Define emotions as semantic concepts
        self.emotion_concepts = {
            'joy': 'happiness, joy, delight, pleasure, contentment, satisfaction, bliss',
            'sadness': 'sadness, grief, sorrow, melancholy, despair, heartbreak, misery',
            'anger': 'anger, rage, fury, frustration, hostility, indignation, wrath',
            'fear': 'fear, anxiety, worry, terror, dread, apprehension, panic',
            'trust': 'trust, confidence, faith, security, reliability, assurance',
            'disgust': 'disgust, revulsion, repulsion, aversion, loathing',
            'surprise': 'surprise, astonishment, shock, amazement, wonder',
            'anticipation': 'hope, expectation, anticipation, eagerness, excitement, optimism'
        }
        
        # Arc anchors
        self.arc_concepts = {
            'hope': 'hope, optimism, positive outlook, bright future',
            'despair': 'despair, hopelessness, defeat, devastation',
            'struggle': 'struggle, difficulty, challenge, hardship',
            'triumph': 'triumph, victory, success, achievement, breakthrough',
            'peace': 'peace, calm, resolution, serenity, contentment'
        }
        
        # Precompute anchor embeddings
        self.emotion_anchors = {
            emotion: self.embedder.get_anchor_embedding(concept)
            for emotion, concept in self.emotion_concepts.items()
        }
        
        self.arc_anchors = {
            concept: self.embedder.get_anchor_embedding(desc)
            for concept, desc in self.arc_concepts.items()
        }
    
    def _setup_fallback_lexicon(self):
        """Minimal lexicon for fallback"""
        self.emotion_words = {
            'positive': ['happy', 'joy', 'love', 'good', 'great', 'wonderful'],
            'negative': ['sad', 'angry', 'fear', 'bad', 'terrible', 'awful']
        }
    
    def fit(self, X, y=None):
        """Fit transformer"""
        return self
    
    def transform(self, X):
        """
        Transform texts into emotional features.
        
        Parameters
        ----------
        X : array-like of strings
            
        Returns
        -------
        features : ndarray of shape (n_samples, ~35)
        """
        if self.use_intelligent_features:
            return self._transform_intelligent(X)
        else:
            return self._transform_fallback(X)
    
    def _transform_intelligent(self, X):
        """Transform using embeddings (intelligent approach)"""
        features = []
        
        for text in X:
            # Split into sentences for arc analysis
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
            
            if len(sentences) == 0:
                # Empty text - return zeros
                features.append([0.0] * 35)
                continue
            
            feat_dict = {}
            
            # === 1. SEMANTIC EMOTION DETECTION (10 features) ===
            
            # Get text embedding
            text_embedding = self.embedder.encode([text])[0]
            
            # Compute similarity to each emotion anchor
            for emotion, anchor_emb in self.emotion_anchors.items():
                similarity = np.dot(text_embedding, anchor_emb) / (
                    np.linalg.norm(text_embedding) * np.linalg.norm(anchor_emb) + 1e-8
                )
                feat_dict[f'emotion_{emotion}_semantic'] = max(0, similarity)
            
            # Emotional diversity (how many emotions are present)
            emotion_scores = [feat_dict[f'emotion_{e}_semantic'] for e in self.emotion_anchors.keys()]
            active_emotions = sum(1 for score in emotion_scores if score > 0.3)
            feat_dict['emotional_diversity_semantic'] = active_emotions / len(self.emotion_anchors)
            
            # Emotional valence (positive vs negative)
            positive = ['joy', 'trust', 'anticipation']
            negative = ['sadness', 'anger', 'fear', 'disgust']
            
            pos_score = np.mean([feat_dict[f'emotion_{e}_semantic'] for e in positive])
            neg_score = np.mean([feat_dict[f'emotion_{e}_semantic'] for e in negative])
            
            feat_dict['emotional_valence_semantic'] = pos_score - neg_score
            
            # === 2. SEMANTIC ARC STRUCTURE (10 features) ===
            
            # Get sentence embeddings
            sent_embeddings = self.embedder.encode(sentences) if len(sentences) > 0 else np.zeros((0, self.embedder.embedding_dim))
            
            if len(sent_embeddings) > 0:
                # Compute arc from hope → despair → triumph
                arc_hope = SemanticSimilarity.detect_semantic_arc(
                    sent_embeddings,
                    self.arc_anchors['hope'],
                    self.arc_anchors['despair']
                )
                feat_dict['arc_hope_to_despair'] = arc_hope['trajectory']
                
                arc_struggle = SemanticSimilarity.detect_semantic_arc(
                    sent_embeddings,
                    self.arc_anchors['struggle'],
                    self.arc_anchors['triumph']
                )
                feat_dict['arc_struggle_to_triumph'] = arc_struggle['trajectory']
                
                # Emotional momentum (increasing or decreasing intensity)
                emotion_intensities = []
                for sent_emb in sent_embeddings:
                    # Average similarity to all emotion anchors
                    intensities = [
                        np.dot(sent_emb, anchor) / (np.linalg.norm(sent_emb) * np.linalg.norm(anchor) + 1e-8)
                        for anchor in self.emotion_anchors.values()
                    ]
                    emotion_intensities.append(max(intensities))
                
                feat_dict['emotional_intensity_mean'] = np.mean(emotion_intensities)
                feat_dict['emotional_intensity_peak'] = np.max(emotion_intensities)
                feat_dict['emotional_intensity_range'] = np.max(emotion_intensities) - np.min(emotion_intensities)
                
                # Trajectory (rising, falling, u-shaped)
                if len(emotion_intensities) >= 3:
                    third = len(emotion_intensities) // 3
                    early = np.mean(emotion_intensities[:third])
                    middle = np.mean(emotion_intensities[third:2*third])
                    late = np.mean(emotion_intensities[2*third:])
                    
                    feat_dict['trajectory_rising'] = float(late > early)
                    feat_dict['trajectory_u_shaped'] = float((early + late) / 2 > middle)
                    feat_dict['trajectory_magnitude'] = late - early
                else:
                    feat_dict['trajectory_rising'] = 0.5
                    feat_dict['trajectory_u_shaped'] = 0.0
                    feat_dict['trajectory_magnitude'] = 0.0
                
                # Volatility (rapid emotional shifts)
                if len(emotion_intensities) > 1:
                    changes = [abs(emotion_intensities[i] - emotion_intensities[i-1]) 
                              for i in range(1, len(emotion_intensities))]
                    feat_dict['emotional_volatility_semantic'] = np.mean(changes)
                else:
                    feat_dict['emotional_volatility_semantic'] = 0.0
            else:
                # No sentences - zeros
                feat_dict['arc_hope_to_despair'] = 0.0
                feat_dict['arc_struggle_to_triumph'] = 0.0
                feat_dict['emotional_intensity_mean'] = 0.0
                feat_dict['emotional_intensity_peak'] = 0.0
                feat_dict['emotional_intensity_range'] = 0.0
                feat_dict['trajectory_rising'] = 0.5
                feat_dict['trajectory_u_shaped'] = 0.0
                feat_dict['trajectory_magnitude'] = 0.0
                feat_dict['emotional_volatility_semantic'] = 0.0
            
            # === 3. SEMANTIC COMPLEXITY (8 features) ===
            
            # Mixed emotions (high similarity to multiple contradictory emotions)
            joy_score = feat_dict['emotion_joy_semantic']
            sadness_score = feat_dict['emotion_sadness_semantic']
            feat_dict['bittersweet_semantic'] = min(joy_score, sadness_score)
            
            # Emotional ambiguity (similar to multiple emotions)
            emotion_score_array = np.array(emotion_scores)
            feat_dict['emotional_ambiguity'] = np.std(emotion_score_array)
            
            # Emotional specificity (one dominant vs. diffuse)
            if emotion_score_array.sum() > 0:
                feat_dict['emotional_specificity_semantic'] = emotion_score_array.max() / (emotion_score_array.sum() + 1e-8)
            else:
                feat_dict['emotional_specificity_semantic'] = 0.0
            
            # Vulnerability (semantic similarity to vulnerability concept)
            vulnerability_anchor = self.embedder.get_anchor_embedding(
                "vulnerability, weakness, insecurity, exposure, fragility"
            )
            vulnerability_score = np.dot(text_embedding, vulnerability_anchor) / (
                np.linalg.norm(text_embedding) * np.linalg.norm(vulnerability_anchor) + 1e-8
            )
            feat_dict['vulnerability_semantic'] = max(0, vulnerability_score)
            
            # Catharsis (semantic similarity to breakthrough/resolution)
            catharsis_anchor = self.embedder.get_anchor_embedding(
                "breakthrough, realization, catharsis, release, resolution, understanding"
            )
            catharsis_score = np.dot(text_embedding, catharsis_anchor) / (
                np.linalg.norm(text_embedding) * np.linalg.norm(catharsis_anchor) + 1e-8
            )
            feat_dict['catharsis_semantic'] = max(0, catharsis_score)
            
            # Empathy (similarity to shared human experience)
            empathy_anchor = self.embedder.get_anchor_embedding(
                "shared experience, universal, we all feel, human connection, relate to"
            )
            empathy_score = np.dot(text_embedding, empathy_anchor) / (
                np.linalg.norm(text_embedding) * np.linalg.norm(empathy_anchor) + 1e-8
            )
            feat_dict['empathy_triggers_semantic'] = max(0, empathy_score)
            
            # Authenticity of emotion (vs. performed/fake)
            authentic_emotion_anchor = self.embedder.get_anchor_embedding(
                "genuine emotion, authentic feeling, real experience, vulnerable, honest"
            )
            authentic_score = np.dot(text_embedding, authentic_emotion_anchor) / (
                np.linalg.norm(text_embedding) * np.linalg.norm(authentic_emotion_anchor) + 1e-8
            )
            feat_dict['emotional_authenticity_semantic'] = max(0, authentic_score)
            
            # Emotional maturity
            mature_emotion_anchor = self.embedder.get_anchor_embedding(
                "nuanced feeling, complex emotion, emotional intelligence, self-awareness"
            )
            mature_score = np.dot(text_embedding, mature_emotion_anchor) / (
                np.linalg.norm(text_embedding) * np.linalg.norm(mature_emotion_anchor) + 1e-8
            )
            feat_dict['emotional_maturity_semantic'] = max(0, mature_score)
            
            # === 4. DYNAMIC FEATURES (7 features) ===
            
            # Emotional density (overall emotional content)
            all_emotion_scores = [feat_dict[f'emotion_{e}_semantic'] for e in self.emotion_anchors.keys()]
            feat_dict['emotional_density_semantic'] = np.mean(all_emotion_scores)
            
            # Emotional contrast (semantic distance between strongest emotions)
            if len(all_emotion_scores) > 1:
                feat_dict['emotional_contrast_semantic'] = max(all_emotion_scores) - min(all_emotion_scores)
            else:
                feat_dict['emotional_contrast_semantic'] = 0.0
            
            # Emotional coherence (consistency of emotional message)
            if len(sent_embeddings) > 1:
                # Compute pairwise similarities between sentences
                sent_sims = []
                for i in range(len(sent_embeddings) - 1):
                    sim = np.dot(sent_embeddings[i], sent_embeddings[i+1]) / (
                        np.linalg.norm(sent_embeddings[i]) * np.linalg.norm(sent_embeddings[i+1]) + 1e-8
                    )
                    sent_sims.append(sim)
                feat_dict['emotional_coherence_semantic'] = np.mean(sent_sims)
            else:
                feat_dict['emotional_coherence_semantic'] = 1.0
            
            # Emotional resolution (ending valence)
            if len(sent_embeddings) > 0:
                final_emb = sent_embeddings[-1]
                final_pos_sim = np.mean([
                    np.dot(final_emb, self.emotion_anchors[e]) / (
                        np.linalg.norm(final_emb) * np.linalg.norm(self.emotion_anchors[e]) + 1e-8
                    )
                    for e in ['joy', 'trust', 'anticipation']
                ])
                feat_dict['emotional_resolution_positive_semantic'] = max(0, final_pos_sim)
            else:
                feat_dict['emotional_resolution_positive_semantic'] = 0.0
            
            # Emotional momentum (building intensity)
            if len(emotion_intensities) > 1:
                momentum = emotion_intensities[-1] - emotion_intensities[0]
                feat_dict['emotional_momentum_semantic'] = momentum
            else:
                feat_dict['emotional_momentum_semantic'] = 0.0
            
            # Emotional saturation (too much emotion = performative)
            feat_dict['emotional_saturation_semantic'] = float(feat_dict['emotional_density_semantic'] > 0.5)
            
            # Emotional nuance (multiple emotions present)
            feat_dict['emotional_nuance_semantic'] = len([s for s in all_emotion_scores if s > 0.2]) / len(all_emotion_scores)
            
            # Convert to feature vector
            features.append(list(feat_dict.values()))
        
        return np.array(features)
    
    def _transform_fallback(self, X):
        """Fallback to simple lexicon-based (minimal)"""
        features = []
        
        for text in X:
            text_lower = text.lower()
            words = re.findall(r'\b\w+\b', text_lower)
            
            feat_dict = {}
            
            # Simple positive/negative
            pos_count = sum(1 for w in words if w in self.emotion_words['positive'])
            neg_count = sum(1 for w in words if w in self.emotion_words['negative'])
            
            total = pos_count + neg_count + 1
            
            feat_dict['positive_density'] = pos_count / len(words)
            feat_dict['negative_density'] = neg_count / len(words)
            feat_dict['valence'] = (pos_count - neg_count) / total
            
            # Pad to match intelligent feature count
            while len(feat_dict) < 35:
                feat_dict[f'fallback_{len(feat_dict)}'] = 0.0
            
            features.append(list(feat_dict.values()))
        
        return np.array(features)
    
    def get_feature_names_out(self, input_features=None):
        """Get feature names"""
        names = [
            # Emotion detection (8)
            'emotion_joy_semantic', 'emotion_sadness_semantic', 'emotion_anger_semantic',
            'emotion_fear_semantic', 'emotion_trust_semantic', 'emotion_disgust_semantic',
            'emotion_surprise_semantic', 'emotion_anticipation_semantic',
            
            # Diversity and valence (2)
            'emotional_diversity_semantic', 'emotional_valence_semantic',
            
            # Arc structure (9)
            'arc_hope_to_despair', 'arc_struggle_to_triumph',
            'emotional_intensity_mean', 'emotional_intensity_peak', 'emotional_intensity_range',
            'trajectory_rising', 'trajectory_u_shaped', 'trajectory_magnitude',
            'emotional_volatility_semantic',
            
            # Complexity (8)
            'bittersweet_semantic', 'emotional_ambiguity', 'emotional_specificity_semantic',
            'vulnerability_semantic', 'catharsis_semantic', 'empathy_triggers_semantic',
            'emotional_authenticity_semantic', 'emotional_maturity_semantic',
            
            # Dynamics (7)
            'emotional_density_semantic', 'emotional_contrast_semantic',
            'emotional_coherence_semantic', 'emotional_resolution_positive_semantic',
            'emotional_momentum_semantic', 'emotional_saturation_semantic',
            'emotional_nuance_semantic'
        ]
        
        return np.array([f'emotional_semantic_{n}' for n in names])

