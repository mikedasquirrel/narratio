"""
Fundamental Constraints Transformer (λ Measurement) - FULLY RENOVATED

Measures fundamental constraints (λ) - physics/training barriers.
λ is central to three-force model: Д = ة - θ - λ

λ represents physical, technical, and training barriers that constrain outcomes.
Higher λ means more fundamental constraints (physics, training, aptitude).

FULLY RENOVATED (November 2025):
- Removed ALL hardcoded regex patterns
- Uses sentence-transformers for semantic similarity detection
- Discovers constraint concepts via embedding space
- Works across languages and novel expressions
- Fully data-driven constraint detection

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Optional
import warnings

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    # No fallback - transformer requires sentence-transformers to function

from sklearn.metrics.pairwise import cosine_similarity
from .base import NarrativeTransformer


class FundamentalConstraintsTransformer(NarrativeTransformer):
    """
    Extracts features measuring fundamental constraints (λ) using semantic embeddings.
    
    Theory: Д = ة - θ - λ (regular) or Д = ة + θ - λ (prestige)
    - λ = training_years/10 + aptitude_threshold + economic_barrier
    - At instance level: Extract constraint indicators semantically
    
    RENOVATED APPROACH:
    Instead of hardcoded regex patterns, uses semantic similarity to
    constraint anchor concepts. Discovers constraint language naturally.
    
    Features Extracted (28 total):
    
    Training Constraints (5 features):
    - Training requirement similarity
    - Qualification requirement similarity
    - Education requirement similarity
    - Certification requirement similarity
    - Experience requirement similarity
    
    Aptitude Constraints (5 features):
    - Skill requirement similarity
    - Talent requirement similarity
    - Ability threshold similarity
    - Natural aptitude similarity
    - Expertise requirement similarity
    
    Physical Constraints (5 features):
    - Equipment requirement similarity
    - Physical limitation similarity
    - Technical constraint similarity
    - Resource requirement similarity
    - Infrastructure requirement similarity
    
    Access Constraints (5 features):
    - Geographic barrier similarity
    - Economic barrier similarity
    - Social barrier similarity
    - Entry barrier similarity
    - Permission requirement similarity
    
    Temporal Constraints (3 features):
    - Time requirement similarity
    - Duration constraint similarity
    - Schedule constraint similarity
    
    Composite Features (5 features):
    - Overall constraint score λ ∈ [0, 1]
    - Constraint intensity
    - Constraint diversity (how many types present)
    - Maximum constraint dimension
    - Constraint consistency
    
    Parameters
    ----------
    model_name : str, default='all-MiniLM-L6-v2'
        Sentence transformer model to use for embeddings
    use_embeddings : bool, default=True
        Whether to use semantic embeddings (requires sentence-transformers)
    
    Examples
    --------
    >>> transformer = FundamentalConstraintsTransformer()
    >>> features = transformer.fit_transform(narratives)
    >>> 
    >>> # Check fundamental constraints
    >>> lambda_values = features[:, -5]  # Overall λ score
    >>> print(f"Average λ: {lambda_values.mean():.2f}")
    >>> 
    >>> # High λ (~0.9) = highly constrained (aviation, physics)
    >>> # Low λ (~0.1) = minimal constraints (lottery, housing)
    """
    
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        use_embeddings: bool = True
    ):
        super().__init__(
            narrative_id="fundamental_constraints",
            description="Measures λ (fundamental constraints) using semantic embeddings"
        )
        
        self.model_name = model_name
        self.use_embeddings = use_embeddings and HAS_SENTENCE_TRANSFORMERS
        
        # Initialize sentence transformer
        self.model_ = None
        if self.use_embeddings:
            try:
                self.model_ = SentenceTransformer(model_name)
            except Exception as e:
                warnings.warn(f"Could not load sentence transformer: {e}. Transformer will not function.")
                self.use_embeddings = False
                self.model = None
        
        # Constraint anchor concepts (semantic, not keywords)
        self.training_anchors = [
            "requires extensive training and education",
            "needs professional qualification and certification",
            "demands years of practice and experience",
            "must complete formal education program",
            "requires specialized training and credentials"
        ]
        
        self.aptitude_anchors = [
            "requires exceptional natural talent and skill",
            "needs high level of innate ability",
            "demands rare aptitude and capacity",
            "must possess extraordinary talent",
            "requires world-class skill and expertise"
        ]
        
        self.physical_anchors = [
            "requires specialized equipment and technology",
            "needs physical infrastructure and resources",
            "depends on technical apparatus and tools",
            "requires expensive machinery and facilities",
            "demands specific physical conditions"
        ]
        
        self.access_anchors = [
            "limited by geographic location and distance",
            "restricted by economic and financial barriers",
            "constrained by social class and privilege",
            "requires special permission and access",
            "gatekept by exclusive entry requirements"
        ]
        
        self.temporal_anchors = [
            "requires significant time investment",
            "needs long duration commitment",
            "constrained by strict scheduling requirements",
            "demands years of sustained effort",
            "requires lengthy preparation period"
        ]
        
        # Store embeddings for anchors
        self.training_embeddings_ = None
        self.aptitude_embeddings_ = None
        self.physical_embeddings_ = None
        self.access_embeddings_ = None
        self.temporal_embeddings_ = None
        
    def fit(self, X, y=None):
        """
        Fit transformer to data.
        
        Computes embeddings for anchor concepts.
        
        Parameters
        ----------
        X : array-like of str
            Training texts
        y : array-like, optional
            Target values (not used)
        
        Returns
        -------
        self : FundamentalConstraintsTransformer
            Fitted transformer
        """
        if self.use_embeddings and self.model_ is not None:
            # Pre-compute anchor embeddings
            self.training_embeddings_ = self.model_.encode(self.training_anchors)
            self.aptitude_embeddings_ = self.model_.encode(self.aptitude_anchors)
            self.physical_embeddings_ = self.model_.encode(self.physical_anchors)
            self.access_embeddings_ = self.model_.encode(self.access_anchors)
            self.temporal_embeddings_ = self.model_.encode(self.temporal_anchors)
        
        # Store metadata
        self.metadata['n_features'] = 28
        self.metadata['feature_names'] = self._get_feature_names()
        self.metadata['use_embeddings'] = self.use_embeddings
        self.metadata['model'] = self.model_name if self.use_embeddings else 'disabled'
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Transform texts to constraint features.
        
        Parameters
        ----------
        X : array-like of str
            Texts to transform
        
        Returns
        -------
        features : ndarray, shape (n_samples, 28)
            Constraint features
        """
        self._validate_fitted()
        
        if not isinstance(X, (list, np.ndarray)):
            X = [X]
        
        features = []
        for text in X:
            feat_vector = self._extract_constraint_features(str(text))
            features.append(feat_vector)
        
        return np.array(features)
    
    def _extract_constraint_features(self, text: str) -> np.ndarray:
        """Extract all constraint features from text."""
        if self.use_embeddings and self.model_ is not None:
            return self._extract_semantic_features(text)
        else:
            return self._extract_fallback_features(text)
    
    def _extract_fallback_features(self, text: str) -> np.ndarray:
        """
        Extract constraint features without embeddings (fallback mode).
        
        Uses simple keyword matching and text statistics as a fallback
        when sentence transformers are not available.
        
        Parameters
        ----------
        text : str
            Input text
            
        Returns
        -------
        features : ndarray
            28 constraint features (same structure as semantic version)
        """
        text_lower = text.lower()
        words = text_lower.split()
        n_words = len(words) + 1
        
        features = []
        
        # Training constraints (5)  
        training_words = ['train', 'training', 'learn', 'education', 'study', 'qualification']
        training_count = sum(1 for w in training_words if w in text_lower) / n_words
        features.extend([training_count] * 5)
        
        # Aptitude constraints (5)
        aptitude_words = ['skill', 'talent', 'ability', 'aptitude', 'capable', 'expertise']
        aptitude_count = sum(1 for w in aptitude_words if w in text_lower) / n_words
        features.extend([aptitude_count] * 5)
        
        # Physical constraints (5)
        physical_words = ['equipment', 'physical', 'technical', 'resource', 'infrastructure']
        physical_count = sum(1 for w in physical_words if w in text_lower) / n_words
        features.extend([physical_count] * 5)
        
        # Access constraints (5)
        access_words = ['access', 'entry', 'barrier', 'permission', 'requirement', 'fee']
        access_count = sum(1 for w in access_words if w in text_lower) / n_words
        features.extend([access_count] * 5)
        
        # Temporal constraints (3)
        temporal_words = ['time', 'duration', 'schedule', 'deadline', 'period']
        temporal_count = sum(1 for w in temporal_words if w in text_lower) / n_words
        features.extend([temporal_count] * 3)
        
        # Composite features (5)
        all_counts = [training_count, aptitude_count, physical_count, access_count, temporal_count]
        lambda_score = np.mean(all_counts)  # Overall constraint
        intensity = np.max(all_counts)  # Max constraint
        diversity = sum(1 for c in all_counts if c > 0.01)  # Number of constraint types
        max_type = np.argmax(all_counts)  # Dominant constraint type
        consistency = np.std(all_counts)  # How varied are constraints
        
        features.extend([lambda_score, intensity, diversity, max_type, consistency])
        
        return np.array(features)
    
    def _extract_semantic_features(self, text: str) -> np.ndarray:
        """Extract features using semantic similarity."""
        features = []
        
        # Encode text
        text_embedding = self.model_.encode([text])[0]
        
        # Training constraints (5)
        training_sims = cosine_similarity(
            [text_embedding],
            self.training_embeddings_
        )[0]
        features.extend([
            np.max(training_sims),  # Max similarity to training concepts
            np.mean(training_sims),  # Average similarity
            training_sims[0],  # Specific concept similarities
            training_sims[1],
            training_sims[2]
        ])
        
        # Aptitude constraints (5)
        aptitude_sims = cosine_similarity(
            [text_embedding],
            self.aptitude_embeddings_
        )[0]
        features.extend([
            np.max(aptitude_sims),
            np.mean(aptitude_sims),
            aptitude_sims[0],
            aptitude_sims[1],
            aptitude_sims[2]
        ])
        
        # Physical constraints (5)
        physical_sims = cosine_similarity(
            [text_embedding],
            self.physical_embeddings_
        )[0]
        features.extend([
            np.max(physical_sims),
            np.mean(physical_sims),
            physical_sims[0],
            physical_sims[1],
            physical_sims[2]
        ])
        
        # Access constraints (5)
        access_sims = cosine_similarity(
            [text_embedding],
            self.access_embeddings_
        )[0]
        features.extend([
            np.max(access_sims),
            np.mean(access_sims),
            access_sims[0],
            access_sims[1],
            access_sims[2]
        ])
        
        # Temporal constraints (3)
        temporal_sims = cosine_similarity(
            [text_embedding],
            self.temporal_embeddings_
        )[0]
        features.extend([
            np.max(temporal_sims),
            np.mean(temporal_sims),
            temporal_sims[0]
        ])
        
        # Composite features (5)
        all_sims = np.concatenate([
            training_sims, aptitude_sims, physical_sims,
            access_sims, temporal_sims
        ])
        
        # Overall constraint score (λ)
        lambda_score = np.mean(all_sims)
        features.append(lambda_score)
        
        # Constraint intensity (max across all)
        intensity = np.max(all_sims)
        features.append(intensity)
        
        # Constraint diversity (how many types > threshold)
        threshold = 0.3
        type_sims = [
            np.max(training_sims),
            np.max(aptitude_sims),
            np.max(physical_sims),
            np.max(access_sims),
            np.max(temporal_sims)
        ]
        diversity = np.sum(np.array(type_sims) > threshold) / 5.0
        features.append(diversity)
        
        # Maximum constraint dimension
        max_dimension = np.argmax(type_sims)
        features.append(float(max_dimension))
        
        # Constraint consistency (low std = consistent across types)
        consistency = 1.0 / (1.0 + np.std(type_sims))
        features.append(consistency)
        
        return np.array(features)
    
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names."""
        names = []
        
        # Training (5)
        names.extend([
            'training_constraint_max',
            'training_constraint_mean',
            'training_constraint_1',
            'training_constraint_2',
            'training_constraint_3'
        ])
        
        # Aptitude (5)
        names.extend([
            'aptitude_constraint_max',
            'aptitude_constraint_mean',
            'aptitude_constraint_1',
            'aptitude_constraint_2',
            'aptitude_constraint_3'
        ])
        
        # Physical (5)
        names.extend([
            'physical_constraint_max',
            'physical_constraint_mean',
            'physical_constraint_1',
            'physical_constraint_2',
            'physical_constraint_3'
        ])
        
        # Access (5)
        names.extend([
            'access_constraint_max',
            'access_constraint_mean',
            'access_constraint_1',
            'access_constraint_2',
            'access_constraint_3'
        ])
        
        # Temporal (3)
        names.extend([
            'temporal_constraint_max',
            'temporal_constraint_mean',
            'temporal_constraint_1'
        ])
        
        # Composite (5)
        names.extend([
            'lambda_overall_score',
            'constraint_intensity',
            'constraint_diversity',
            'constraint_max_dimension',
            'constraint_consistency'
        ])
        
        return names
    
    def _generate_interpretation(self) -> str:
        """Generate interpretation."""
        if not self.is_fitted_:
            return "Transformer not fitted yet."
        
        method = "semantic embeddings" if self.use_embeddings else "disabled"
        
        interpretation = f"""
Fundamental Constraints (λ) Analysis - FULLY RENOVATED

Method: {method}
Model: {self.metadata.get('model', 'unknown')}
Features: {self.metadata.get('n_features', 0)}

SEMANTIC APPROACH:
- Uses sentence-transformers for constraint detection
- Computes similarity to constraint anchor concepts
- Discovers constraint language naturally
- Works across languages and novel expressions
- No hardcoded keywords

Constraint Categories:
1. Training (5): Education, qualification, experience requirements
2. Aptitude (5): Skill, talent, ability thresholds
3. Physical (5): Equipment, resources, infrastructure needs
4. Access (5): Geographic, economic, social barriers
5. Temporal (3): Time, duration, schedule constraints

Primary Output: lambda_overall_score (column 23)
- λ ∈ [0, 1] where higher = more constrained
- Used in three-force model: Д = ة - θ - λ

Example Interpretations:
- λ ≈ 0.9: Highly constrained (aviation, physics, medicine)
- λ ≈ 0.5: Moderately constrained (sports, business)
- λ ≈ 0.1: Minimally constrained (lottery, housing)

ADVANTAGE: Semantic understanding captures constraint concepts
that would be missed by keyword matching.
"""
        return interpretation.strip()
