"""
Genome (ж) Structure - Complete DNA of Story Instances

The genome is the complete feature vector containing ALL aspects of a story:
1. Nominative elements (names, proper nouns, labels)
2. Archetypal elements (distance from domain Ξ)
3. Historial elements (historical narrative weight, position in narrative space)
4. Uniquity (gravitational pull toward rare/elusive narratives)

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity


class GenomeStructure:
    """
    Defines the complete structure of ж (genome) for a story instance.
    
    ж = [nominative | archetypal | historial | uniquity]
    
    Components:
    -----------
    Nominative (N features):
        - Proper nouns, names, labels
        - Explicit categorical features
        - Surface-level identifiers
    
    Archetypal (A features):
        - Distance from domain Ξ
        - Archetype pattern matches
        - Sub-archetype scores
        - Contextual boosts
    
    Historial (H features):
        - Position in historical narrative space
        - Similarity to past narratives
        - Temporal momentum/decay
        - Narrative lineage (what came before)
        - Historical gravity (pull from precedent)
    
    Uniquity (U features):
        - Rarity score (how unique is this pattern?)
        - Elusive narrative pull (desire for unseen patterns)
        - Novelty gradient (departure from common)
        - **CONSTANT**: Universal pull toward uniqueness
    
    Total genome: ж ∈ R^(N+A+H+U)
    """
    
    def __init__(self, n_nominative: int, n_archetypal: int, n_historial: int = 10, n_uniquity: int = 5):
        self.n_nominative = n_nominative
        self.n_archetypal = n_archetypal
        self.n_historial = n_historial
        self.n_uniquity = n_uniquity
        
        self.total_features = n_nominative + n_archetypal + n_historial + n_uniquity
        
        # Index ranges for each component
        self.nominative_range = (0, n_nominative)
        self.archetypal_range = (n_nominative, n_nominative + n_archetypal)
        self.historial_range = (n_nominative + n_archetypal, 
                                n_nominative + n_archetypal + n_historial)
        self.uniquity_range = (n_nominative + n_archetypal + n_historial, self.total_features)
    
    def get_nominative(self, genome: np.ndarray) -> np.ndarray:
        """Extract nominative component from genome."""
        return genome[self.nominative_range[0]:self.nominative_range[1]]
    
    def get_archetypal(self, genome: np.ndarray) -> np.ndarray:
        """Extract archetypal component from genome."""
        return genome[self.archetypal_range[0]:self.archetypal_range[1]]
    
    def get_historial(self, genome: np.ndarray) -> np.ndarray:
        """Extract historial component from genome."""
        return genome[self.historial_range[0]:self.historial_range[1]]
    
    def get_uniquity(self, genome: np.ndarray) -> np.ndarray:
        """Extract uniquity component from genome."""
        return genome[self.uniquity_range[0]:self.uniquity_range[1]]
    
    def assemble_genome(
        self,
        nominative: np.ndarray,
        archetypal: np.ndarray,
        historial: np.ndarray,
        uniquity: np.ndarray
    ) -> np.ndarray:
        """Assemble complete genome from components."""
        return np.concatenate([nominative, archetypal, historial, uniquity])


class HistorialCalculator:
    """
    Calculates historial features - historical narrative weight and positioning.
    
    Historial features capture:
    - Where does this narrative sit in the historical space of all narratives?
    - How similar is it to past successful/failed narratives?
    - What is its temporal momentum?
    - What historical gravity does it experience?
    
    Examples
    --------
    >>> calculator = HistorialCalculator()
    >>> calculator.fit(historical_narratives, historical_outcomes)
    >>> historial_features = calculator.transform(new_narrative)
    """
    
    def __init__(self):
        self.historical_narratives = None
        self.historical_outcomes = None
        self.historical_embeddings = None
        self.temporal_weights = None
        
    def fit(self, texts: List[str], outcomes: np.ndarray, timestamps: Optional[np.ndarray] = None):
        """
        Learn historical narrative space.
        
        Parameters
        ----------
        texts : list of str
            Historical narratives
        outcomes : ndarray
            Historical outcomes
        timestamps : ndarray, optional
            Timestamps (for temporal weighting)
        """
        # Handle empty input
        if len(texts) == 0:
            raise ValueError("Cannot fit on empty texts")
        
        # Filter empty texts
        valid_indices = [i for i, t in enumerate(texts) if t and len(str(t).strip()) > 0]
        if len(valid_indices) == 0:
            raise ValueError("No valid (non-empty) texts found")
        
        texts_filtered = [texts[i] for i in valid_indices]
        outcomes_filtered = outcomes[valid_indices]
        
        if timestamps is not None:
            timestamps_filtered = timestamps[valid_indices]
        else:
            timestamps_filtered = None
        
        self.historical_narratives = texts_filtered
        self.historical_outcomes = outcomes_filtered
        
        # Handle single sample
        if len(texts_filtered) == 1:
            # Single sample - create minimal embedding
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=10)
            # Need at least 2 samples for TF-IDF, so pad
            texts_padded = texts_filtered + [texts_filtered[0]]
            outcomes_padded = np.concatenate([outcomes_filtered, outcomes_filtered])
            self.historical_embeddings = vectorizer.fit_transform(texts_padded).toarray()[:1]
            self.temporal_weights = np.array([1.0])
            return self
        
        # Create embeddings (simplified - use TF-IDF as proxy)
        from sklearn.feature_extraction.text import TfidfVectorizer
        try:
            vectorizer = TfidfVectorizer(max_features=100, min_df=1)
            self.historical_embeddings = vectorizer.fit_transform(texts_filtered).toarray()
        except Exception as e:
            # Fallback to simpler vectorizer
            vectorizer = TfidfVectorizer(max_features=50, min_df=1, token_pattern=r'\b\w+\b')
            self.historical_embeddings = vectorizer.fit_transform(texts_filtered).toarray()
        
        # Temporal weights (more recent = higher weight)
        if timestamps_filtered is not None:
            # Use configurable decay function
            from .temporal_decay import TemporalDecay, DecayType
            self.temporal_weights = TemporalDecay.exponential_decay(
                timestamps_filtered,
                half_life=None,  # Auto-calculate
                max_time=None    # Auto-calculate
            )
        else:
            self.temporal_weights = np.ones(len(texts_filtered))
        
        return self
    
    def transform(self, text: str) -> np.ndarray:
        """
        Calculate historial features for a new narrative.
        
        Parameters
        ----------
        text : str
            New narrative
        
        Returns
        -------
        ndarray
            Historial features (10 total):
            1. distance_to_historical_winners
            2. distance_to_historical_losers
            3. similarity_to_recent_narratives
            4. historical_precedent_strength
            5. narrative_lineage_score
            6. temporal_momentum
            7. historical_gravity_pull
            8. pattern_recency
            9. historical_novelty
            10. precedent_outcome_correlation
        """
        if self.historical_embeddings is None or len(self.historical_embeddings) == 0:
            # Return zeros if not fitted
            return np.zeros(10)
        
        # Vectorize new text using same vectorizer logic
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Recreate vectorizer with same parameters
        try:
            vectorizer = TfidfVectorizer(max_features=100, min_df=1)
            all_texts = self.historical_narratives + [text]
            all_embeddings = vectorizer.fit_transform(all_texts).toarray()
            new_embedding = all_embeddings[-1]
            hist_embeddings = all_embeddings[:-1]
        except Exception:
            # Fallback: use pre-computed embeddings and approximate new embedding
            vectorizer = TfidfVectorizer(max_features=50, min_df=1, token_pattern=r'\b\w+\b')
            all_texts = self.historical_narratives + [text]
            all_embeddings = vectorizer.fit_transform(all_texts).toarray()
            new_embedding = all_embeddings[-1]
            hist_embeddings = all_embeddings[:-1]
            
            # Align dimensions if needed
            if hist_embeddings.shape[1] != self.historical_embeddings.shape[1]:
                # Use historical embeddings if dimensions match
                if len(self.historical_narratives) == hist_embeddings.shape[0]:
                    hist_embeddings = self.historical_embeddings
                else:
                    # Pad or truncate
                    min_dim = min(hist_embeddings.shape[1], self.historical_embeddings.shape[1])
                    hist_embeddings = hist_embeddings[:, :min_dim]
                    new_embedding = new_embedding[:min_dim]
        
        features = np.zeros(10)
        
        # 1. Distance to historical winners (lower = follows winning pattern)
        winner_mask = self.historical_outcomes > np.median(self.historical_outcomes)
        if winner_mask.sum() > 0:
            winner_embeddings = hist_embeddings[winner_mask]
            distances = np.linalg.norm(winner_embeddings - new_embedding, axis=1)
            features[0] = np.mean(distances)
        
        # 2. Distance to historical losers (higher = avoids losing pattern)
        loser_mask = ~winner_mask
        if loser_mask.sum() > 0:
            loser_embeddings = hist_embeddings[loser_mask]
            distances = np.linalg.norm(loser_embeddings - new_embedding, axis=1)
            features[1] = np.mean(distances)
        
        # 3. Similarity to recent narratives (temporal weighting)
        similarities = cosine_similarity([new_embedding], hist_embeddings)[0]
        weighted_similarity = np.average(similarities, weights=self.temporal_weights)
        features[2] = weighted_similarity
        
        # 4. Historical precedent strength (how much history supports this?)
        precedent_strength = np.sum(similarities > 0.5) / len(similarities)
        features[3] = precedent_strength
        
        # 5. Narrative lineage score (connection to successful lineages)
        if winner_mask.sum() > 0:
            winner_similarities = similarities[winner_mask]
            features[4] = np.max(winner_similarities) if len(winner_similarities) > 0 else 0.0
        
        # 6. Temporal momentum (similarity to recent winners)
        recent_mask = self.temporal_weights > 0.7
        if recent_mask.sum() > 0 and winner_mask.sum() > 0:
            recent_winners = recent_mask & winner_mask
            if recent_winners.sum() > 0:
                features[5] = np.mean(similarities[recent_winners])
        
        # 7. Historical gravity pull (weighted by outcomes and similarity)
        gravity = np.sum(similarities * self.historical_outcomes * self.temporal_weights)
        features[6] = gravity / (np.sum(self.temporal_weights) + 1e-8)
        
        # 8. Pattern recency (how recently was this pattern seen?)
        if similarities.max() > 0.5:
            most_similar_idx = similarities.argmax()
            features[7] = self.temporal_weights[most_similar_idx]
        
        # 9. Historical novelty (inverse of similarity to all history)
        features[8] = 1.0 - np.mean(similarities)
        
        # 10. Precedent outcome correlation (do similar past narratives predict success?)
        if len(similarities) > 0:
            # Correlate similarity with outcomes
            corr = np.corrcoef(similarities, self.historical_outcomes)[0, 1]
            features[9] = corr if not np.isnan(corr) else 0.0
        
        return features


class UniquityCalculator:
    """
    Calculates uniquity - gravitational pull toward rare/elusive narratives.
    
    Key insight: RARE narratives have HIGH gravitational pull.
    - Common narratives are overdone, low pull
    - Elusive narratives are desired, high pull
    - Novelty creates value
    
    Uniquity is partially CONSTANT - there's a universal pull toward freshness.
    
    Features:
    ---------
    1. Rarity score (how rare is this pattern?)
    2. Elusive narrative pull (desire for unseen patterns)
    3. Novelty gradient (departure from common)
    4. Pattern saturation penalty (is this overdone?)
    5. Uniquity constant (universal pull toward uniqueness ≈ 0.3)
    
    Examples
    --------
    >>> calculator = UniquityCalculator()
    >>> calculator.fit(all_past_narratives)
    >>> uniquity = calculator.transform(new_narrative)
    >>> print(f"Rarity score: {uniquity[0]:.3f}")
    """
    
    UNIQUITY_CONSTANT = 0.3  # Universal pull toward uniqueness
    
    def __init__(self):
        self.pattern_frequencies = None
        self.common_patterns = None
        self.rare_patterns = None
        
    def fit(self, texts: List[str]):
        """
        Learn pattern frequencies from historical narratives.
        
        Parameters
        ----------
        texts : list of str
            Historical narratives
        """
        # Handle empty input
        if len(texts) == 0:
            # Initialize with empty patterns
            self.pattern_frequencies = Counter()
            self.common_patterns = set()
            self.rare_patterns = set()
            return self
        
        # Filter empty texts
        valid_texts = [t for t in texts if t and len(str(t).strip()) > 0]
        
        if len(valid_texts) == 0:
            self.pattern_frequencies = Counter()
            self.common_patterns = set()
            self.rare_patterns = set()
            return self
        
        # Extract n-grams as patterns
        all_patterns = []
        for text in valid_texts:
            try:
                words = str(text).lower().split()
                if len(words) < 2:
                    continue
                # Bigrams and trigrams
                bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
                trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)] if len(words) >= 3 else []
                all_patterns.extend(bigrams + trigrams)
            except Exception:
                # Skip problematic texts
                continue
        
        if len(all_patterns) == 0:
            self.pattern_frequencies = Counter()
            self.common_patterns = set()
            self.rare_patterns = set()
            return self
        
        # Count frequencies
        self.pattern_frequencies = Counter(all_patterns)
        total_patterns = sum(self.pattern_frequencies.values())
        
        if total_patterns == 0:
            self.common_patterns = set()
            self.rare_patterns = set()
            return self
        
        # Normalize to probabilities
        for pattern in self.pattern_frequencies:
            self.pattern_frequencies[pattern] /= total_patterns
        
        # Identify common (top 20%) and rare (bottom 20%) patterns
        sorted_patterns = sorted(self.pattern_frequencies.items(), key=lambda x: x[1], reverse=True)
        n_common = max(1, int(len(sorted_patterns) * 0.2))
        n_rare = max(1, int(len(sorted_patterns) * 0.2))
        
        self.common_patterns = set([p for p, _ in sorted_patterns[:n_common]])
        self.rare_patterns = set([p for p, _ in sorted_patterns[-n_rare:]])
        
        return self
    
    def transform(self, text: str) -> np.ndarray:
        """
        Calculate uniquity features for a narrative.
        
        Enhanced with:
        - Better rarity calculation
        - Elusive narrative detection (patterns that SHOULD exist but don't)
        - Temporal uniquity (how rare was this AT THE TIME?)
        
        Parameters
        ----------
        text : str
            Narrative to analyze
        
        Returns
        -------
        ndarray
            Uniquity features (5 total):
            1. rarity_score: How rare are the patterns?
            2. elusive_narrative_pull: Inverse frequency (desire for rare)
            3. novelty_gradient: Unseen patterns (first of kind)
            4. saturation_penalty: Overdone patterns (lower = more saturated)
            5. uniquity_constant: Universal pull toward uniqueness (~0.3)
        """
        features = np.zeros(5)
        
        if not text or len(str(text).strip()) == 0:
            features[4] = self.UNIQUITY_CONSTANT
            return features
        
        words = str(text).lower().split()
        if len(words) < 2:
            features[4] = self.UNIQUITY_CONSTANT
            return features
        
        # Extract patterns from this text
        bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)] if len(words) >= 3 else []
        text_patterns = bigrams + trigrams
        
        if len(text_patterns) == 0:
            features[4] = self.UNIQUITY_CONSTANT
            return features
        
        # 1. Rarity score (how many rare patterns?)
        rare_count = sum(1 for p in text_patterns if p in self.rare_patterns)
        features[0] = rare_count / len(text_patterns) if text_patterns else 0.0
        
        # 2. Elusive narrative pull (inverse frequency score)
        # Enhanced: Weight by how rare (very rare = very high pull)
        pattern_rarities = []
        for pattern in text_patterns:
            if pattern in self.pattern_frequencies:
                # Rarity = 1 - frequency (rare patterns have high score)
                # Square it to emphasize very rare patterns
                rarity = 1.0 - self.pattern_frequencies[pattern]
                pattern_rarities.append(rarity ** 2)  # Square for emphasis
            else:
                # Unseen pattern = maximum rarity
                pattern_rarities.append(1.0)
        
        features[1] = np.mean(pattern_rarities) if pattern_rarities else 0.5
        
        # 3. Novelty gradient (presence of unseen patterns)
        # Enhanced: "First of its kind" detection
        unseen_count = sum(1 for p in text_patterns if p not in self.pattern_frequencies)
        features[2] = unseen_count / len(text_patterns) if text_patterns else 0.0
        
        # 4. Pattern saturation penalty (how many overdone patterns?)
        # Enhanced: Penalty increases with saturation
        common_count = sum(1 for p in text_patterns if p in self.common_patterns)
        saturation = common_count / len(text_patterns) if text_patterns else 0.0
        # Square saturation for stronger penalty
        features[3] = 1.0 - (saturation ** 2)  # Penalty (lower = more saturated)
        
        # 5. Uniquity constant (universal pull toward uniqueness)
        # This is the CONSTANT gravitational pull toward rare/novel narratives
        features[4] = self.UNIQUITY_CONSTANT
        
        return features


class CompleteGenomeExtractor:
    """
    Extracts complete genome (ж) including all four components:
    nominative, archetypal, historial, and uniquity.
    
    This is the COMPLETE DNA of a story instance.
    
    ENHANCED (Nov 2025): Now extracts π_effective and complexity_factors
    for each instance.
    
    Examples
    --------
    >>> extractor = CompleteGenomeExtractor(
    ...     nominative_transformer=nominative_transformer,
    ...     archetypal_transformer=archetypal_transformer
    ... )
    >>> extractor.fit(historical_texts, historical_outcomes)
    >>> genome = extractor.transform(new_text)
    >>> print(f"Genome shape: {genome.shape}")
    >>> print(f"Nominative: {genome[:10]}")
    >>> print(f"Archetypal: {genome[10:20]}")
    >>> print(f"Historial: {genome[20:30]}")
    >>> print(f"Uniquity: {genome[30:]}")
    """
    
    def __init__(
        self,
        nominative_transformer,
        archetypal_transformer,
        historial_calculator: Optional[HistorialCalculator] = None,
        uniquity_calculator: Optional[UniquityCalculator] = None,
        domain_config = None,
        complexity_scorer = None
    ):
        self.nominative_transformer = nominative_transformer
        self.archetypal_transformer = archetypal_transformer
        self.historial_calculator = historial_calculator or HistorialCalculator()
        self.uniquity_calculator = uniquity_calculator or UniquityCalculator()
        self.domain_config = domain_config
        self.complexity_scorer = complexity_scorer
        
        self.genome_structure = None
        self.pi_base = None
        
        # Get domain π if config available
        if self.domain_config:
            self.pi_base = self.domain_config.get_pi()
        
    def fit(self, texts: List[str], outcomes: np.ndarray, timestamps: Optional[np.ndarray] = None):
        """
        Fit all genome extractors.
        
        Parameters
        ----------
        texts : list of str
            Historical narratives
        outcomes : ndarray
            Historical outcomes
        timestamps : ndarray, optional
            Timestamps for temporal weighting
        """
        # Fit nominative
        print(f"    [1/4] Fitting nominative transformer on {len(texts)} samples...", end=" ", flush=True)
        self.nominative_transformer.fit(texts, outcomes)
        print("✓")
        
        # Fit archetypal
        print(f"    [2/4] Fitting archetypal transformer (learning domain Ξ)...", end=" ", flush=True)
        self.archetypal_transformer.fit(texts, outcomes)
        print("✓")
        
        # Fit historial
        print(f"    [3/4] Fitting historial calculator (narrative lineage)...", end=" ", flush=True)
        self.historial_calculator.fit(texts, outcomes, timestamps)
        print("✓")
        
        # Fit uniquity
        print(f"    [4/4] Fitting uniquity calculator (rarity patterns)...", end=" ", flush=True)
        self.uniquity_calculator.fit(texts)
        print("✓")
        
        # Determine genome structure
        sample_nom = self.nominative_transformer.transform([texts[0]])
        sample_arch = self.archetypal_transformer.transform([texts[0]])
        
        self.genome_structure = GenomeStructure(
            n_nominative=sample_nom.shape[1],
            n_archetypal=sample_arch.shape[1],
            n_historial=10,
            n_uniquity=5
        )
        
        return self
    
    def transform(self, texts: List[str], return_metadata: bool = False):
        """
        Extract complete genomes for texts.
        
        ENHANCED: Now also computes π_effective and complexity_factors.
        
        Parameters
        ----------
        texts : list of str
            Texts to extract genomes from
        return_metadata : bool
            If True, returns (genomes, metadata) where metadata includes
            π_effective and complexity_factors for each instance
        
        Returns
        -------
        genomes : ndarray
            Complete genomes, shape (n_texts, total_genome_features)
        metadata : dict (if return_metadata=True)
            {
                'pi_effective': ndarray,
                'pi_base': float,
                'complexity_factors': list of dict
            }
        """
        n = len(texts)
        
        # Extract nominative
        print(f"    [1/5] Extracting nominative features for {n} samples...", end=" ", flush=True)
        nominative = self.nominative_transformer.transform(texts)
        print("✓")
        
        # Extract archetypal
        print(f"    [2/5] Extracting archetypal features (distance from Ξ)...", end=" ", flush=True)
        archetypal = self.archetypal_transformer.transform(texts)
        print("✓")
        
        # Extract historial and uniquity for each text
        print(f"    [3/5] Computing historial features (narrative history)...", end=" ", flush=True)
        historial_list = []
        uniquity_list = []
        
        # Show progress for large datasets
        show_progress = n > 100
        progress_interval = max(1, n // 10)
        
        for i, text in enumerate(texts):
            if show_progress and i > 0 and i % progress_interval == 0:
                print(f"{i}/{n}", end="...", flush=True)
            
            historial = self.historial_calculator.transform(text)
            uniquity = self.uniquity_calculator.transform(text)
            historial_list.append(historial)
            uniquity_list.append(uniquity)
        
        historial = np.array(historial_list)
        uniquity = np.array(uniquity_list)
        print("✓")
        
        # Compute complexity and π_effective if scorer available
        print(f"    [4/5] Computing complexity and π_effective...", end=" ", flush=True)
        complexity_values = []
        pi_effective_values = []
        complexity_factors_list = []
        
        if self.complexity_scorer and self.domain_config:
            # Import StoryInstance temporarily for complexity calculation
            from pathlib import Path
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from core.story_instance import StoryInstance
            
            for i, text in enumerate(texts):
                # Create temporary instance for complexity scoring
                temp_instance = StoryInstance(
                    instance_id=f"temp_{i}",
                    domain=self.domain_config.domain_name if hasattr(self.domain_config, 'domain_name') else "unknown",
                    narrative_text=text
                )
                
                # Calculate complexity
                complexity = self.complexity_scorer.calculate_complexity(temp_instance, text)
                complexity_values.append(complexity)
                complexity_factors_list.append(temp_instance.complexity_factors)
                
                # Calculate π_effective
                pi_eff = self.domain_config.calculate_effective_pi(complexity)
                pi_effective_values.append(pi_eff)
        else:
            # No complexity scorer - use base π for all
            for _ in texts:
                complexity_values.append(0.5)  # Default medium complexity
                pi_effective_values.append(self.pi_base if self.pi_base else 0.5)
                complexity_factors_list.append({})
        
        print("✓")
        
        # Assemble complete genomes
        print(f"    [5/5] Assembling complete genomes...", end=" ", flush=True)
        genomes = np.hstack([nominative, archetypal, historial, uniquity])
        print("✓")
        
        if return_metadata:
            metadata = {
                'pi_effective': np.array(pi_effective_values),
                'pi_base': self.pi_base if self.pi_base else 0.5,
                'complexity_factors': complexity_factors_list,
                'complexity_values': np.array(complexity_values)
            }
            return genomes, metadata
        else:
            return genomes
    
    def get_feature_names(self) -> Dict[str, List[str]]:
        """Get feature names for all genome components."""
        # Handle transformers that may not have get_feature_names method
        try:
            nom_features = self.nominative_transformer.get_feature_names()
        except AttributeError:
            # Fallback: use generic names
            nom_features = [f'nominative_{i}' for i in range(self.genome_structure.n_nominative)]
        
        return {
            'nominative': nom_features,
            'archetypal': self.archetypal_transformer.get_feature_names(),
            'historial': [
                'distance_to_historical_winners',
                'distance_to_historical_losers',
                'similarity_to_recent',
                'precedent_strength',
                'narrative_lineage',
                'temporal_momentum',
                'historical_gravity',
                'pattern_recency',
                'historical_novelty',
                'precedent_outcome_correlation'
            ],
            'uniquity': [
                'rarity_score',
                'elusive_narrative_pull',
                'novelty_gradient',
                'saturation_penalty',
                'uniquity_constant'
            ]
        }

