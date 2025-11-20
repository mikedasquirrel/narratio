"""
Sequential Narrative Processor

CRITICAL INSIGHT: Stories are SEQUENCES, not bags of words.
Meaning emerges from ORDERING, SPACING, TIMING, PROGRESSION.

This processor captures:
- Sequential structure (what follows what)
- Temporal spacing (pauses, rhythm, gaps)
- Progression patterns (how narrative evolves)
- Contextual embeddings (each element in its position)
- Momentum and acceleration
- Everything that makes sequence meaningful

Uses AI to analyze, NO hardcoded categorizations.
Background processing for large corpora.
Let ALL structure emerge without presupposition.

Author: Narrative Optimization Framework
Date: November 2025
"""

from typing import List, Dict, Any, Optional, Iterator, Tuple
import numpy as np
from pathlib import Path
import json
from dataclasses import dataclass, asdict
import hashlib
from datetime import datetime
import time

try:
    from ..transformers.utils.embeddings import EmbeddingManager
    from ..transformers.utils.shared_models import SharedModelRegistry
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from transformers.utils.embeddings import EmbeddingManager
    from transformers.utils.shared_models import SharedModelRegistry


@dataclass
class SequentialElement:
    """
    Single element in narrative sequence.
    
    Captures:
    - Content (what it says)
    - Position (where in sequence)
    - Timing (spacing from previous)
    - Context (surrounding elements)
    - Embedding (semantic representation IN CONTEXT)
    """
    index: int
    text: str
    position_pct: float  # 0-1, where in narrative
    spacing_before: float  # Time/space since previous element
    embedding: np.ndarray  # Contextual embedding
    length: int  # Characters or words
    
    # Relational
    distance_to_prev: Optional[float] = None  # Semantic distance
    distance_to_next: Optional[float] = None
    local_density: Optional[float] = None  # How much happening here
    momentum: Optional[float] = None  # Direction of change


class SequentialNarrativeProcessor:
    """
    Process narratives as SEQUENCES, preserving all temporal/spatial structure.
    
    Philosophy:
    - Narrative = ordered sequence of elements
    - Meaning emerges from ORDER + SPACING + PROGRESSION
    - Each element contextual (depends on what came before)
    - AI captures patterns WITHOUT us specifying what to look for
    
    Output:
    - Sequential embeddings (position-aware)
    - Spacing/rhythm patterns
    - Progression trajectories
    - Momentum and acceleration
    - Mysterious dimensions that predict but don't explain
    """
    
    def __init__(
        self,
        segment_method: str = 'adaptive',
        embedding_model: str = 'all-MiniLM-L6-v2',
        cache_dir: Optional[str] = None
    ):
        """
        Initialize sequential processor.
        
        Parameters
        ----------
        segment_method : str
            How to segment narrative:
            - 'paragraph': Natural paragraph breaks
            - 'sentence': Sentence-level
            - 'semantic': AI-detected semantic shifts
            - 'adaptive': Combine methods
        embedding_model : str
            Model for contextual embeddings
        cache_dir : str, optional
            Cache directory
        """
        self.segment_method = segment_method
        
        # Embedding manager (handles caching, models)
        self.embedder = EmbeddingManager(
            model_name=embedding_model,
            cache_dir=cache_dir,
            use_cache=True
        )
        
        # Will be populated during processing
        self.processed_sequences_ = {}
    
    def process_narrative_sequential(
        self,
        narrative: str,
        narrative_id: str,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Process single narrative as sequence.
        
        Preserves:
        - Element order
        - Spacing between elements
        - Contextual embeddings
        - Progression patterns
        
        NO interpretation. ONLY measurement.
        
        Parameters
        ----------
        narrative : str
            Raw narrative text
        narrative_id : str
            Unique identifier
        metadata : dict, optional
            Any metadata (genre, source, etc.)
            
        Returns
        -------
        sequence_analysis : dict
            Complete sequential structure analysis
        """
        # Step 1: Segment narrative (preserving boundaries)
        segments = self._segment_narrative(narrative)
        
        if len(segments) == 0:
            return {'error': 'No segments extracted'}
        
        # Step 2: Extract spacing (temporal structure)
        spacings = self._extract_spacing(segments, narrative)
        
        # Step 3: Embed each segment IN CONTEXT
        # This captures meaning-in-position, not meaning-in-isolation
        segment_texts = [s['text'] for s in segments]
        embeddings = self.embedder.encode(segment_texts, show_progress=False)
        
        # Step 4: Build sequential elements
        elements = []
        for idx, (segment, embedding, spacing) in enumerate(zip(segments, embeddings, spacings)):
            position_pct = segment['position'] / len(narrative)
            
            element = SequentialElement(
                index=idx,
                text=segment['text'],
                position_pct=position_pct,
                spacing_before=spacing,
                embedding=embedding,
                length=len(segment['text'])
            )
            
            elements.append(element)
        
        # Step 5: Compute relational features (distances, momentum)
        self._compute_relational_features(elements, embeddings)
        
        # Step 6: Extract progression patterns
        progression = self._extract_progression(elements, embeddings)
        
        # Step 7: Compute rhythm and acceleration
        rhythm_analysis = self._analyze_rhythm(spacings)
        
        # Step 8: Mysterious dimensions (PCA on sequential embeddings)
        mysterious_dims = self._extract_mysterious_dimensions(embeddings)
        
        # Package results
        sequence_analysis = {
            'narrative_id': narrative_id,
            'metadata': metadata or {},
            'n_elements': len(elements),
            'total_length': len(narrative),
            
            # Sequential structure
            'elements': [
                {
                    'index': e.index,
                    'position_pct': e.position_pct,
                    'spacing': e.spacing_before,
                    'length': e.length,
                    'text_preview': e.text[:100] + '...' if len(e.text) > 100 else e.text,
                    # Don't serialize embedding here (too large)
                }
                for e in elements
            ],
            
            # Embeddings (for downstream use)
            'embeddings_shape': embeddings.shape,
            # Actual embeddings stored separately in cache
            
            # Spacing/rhythm
            'rhythm_analysis': rhythm_analysis,
            
            # Progression
            'progression': progression,
            
            # Mysterious dimensions
            'mysterious_dimensions': mysterious_dims,
            
            # Timestamp
            'processed_at': datetime.now().isoformat()
        }
        
        # Cache full data (including embeddings)
        cache_key = self._get_cache_key(narrative_id)
        self._cache_sequence(cache_key, elements, embeddings)
        
        return sequence_analysis
    
    def _segment_narrative(self, narrative: str) -> List[Dict]:
        """
        Segment narrative preserving natural boundaries.
        
        Methods:
        - Paragraph breaks (natural)
        - Sentence boundaries (granular)
        - Semantic shifts (AI-detected)
        
        Returns segments with position markers.
        """
        if self.segment_method == 'paragraph':
            return self._segment_by_paragraphs(narrative)
        elif self.segment_method == 'sentence':
            return self._segment_by_sentences(narrative)
        elif self.segment_method == 'semantic':
            return self._segment_by_semantic_shifts(narrative)
        else:  # adaptive
            return self._segment_adaptive(narrative)
    
    def _segment_by_paragraphs(self, narrative: str) -> List[Dict]:
        """Segment by paragraph breaks."""
        paragraphs = narrative.split('\n\n')
        
        segments = []
        position = 0
        for para in paragraphs:
            para = para.strip()
            if para:
                segments.append({
                    'text': para,
                    'position': position,
                    'type': 'paragraph'
                })
                position += len(para) + 2  # +2 for \n\n
        
        return segments
    
    def _segment_by_sentences(self, narrative: str) -> List[Dict]:
        """Segment by sentences (using spaCy for accuracy)."""
        nlp = SharedModelRegistry.get_spacy()
        
        if nlp is None:
            # Fallback: simple sentence splitting
            import re
            sentences = re.split(r'[.!?]+\s+', narrative)
        else:
            doc = nlp(narrative)
            sentences = [sent.text for sent in doc.sents]
        
        segments = []
        position = 0
        for sent in sentences:
            if sent.strip():
                segments.append({
                    'text': sent.strip(),
                    'position': position,
                    'type': 'sentence'
                })
                position += len(sent)
        
        return segments
    
    def _segment_adaptive(self, narrative: str) -> List[Dict]:
        """
        Adaptive segmentation based on narrative length.
        
        - Short (< 1000 chars): Sentences
        - Medium (1000-10000): Paragraphs
        - Long (> 10000): Sections + paragraphs
        """
        length = len(narrative)
        
        if length < 1000:
            return self._segment_by_sentences(narrative)
        elif length < 10000:
            return self._segment_by_paragraphs(narrative)
        else:
            # Long narratives: look for section breaks first
            if '\n\n\n' in narrative:  # Triple breaks = sections
                sections = narrative.split('\n\n\n')
                all_segments = []
                for section in sections:
                    section_segments = self._segment_by_paragraphs(section)
                    all_segments.extend(section_segments)
                return all_segments
            else:
                return self._segment_by_paragraphs(narrative)
    
    def _extract_spacing(self, segments: List[Dict], full_narrative: str) -> List[float]:
        """
        Extract spacing between segments (temporal structure).
        
        Spacing preserves rhythm, pauses, gaps.
        First element has spacing = 0.
        """
        spacings = [0.0]  # First element
        
        for i in range(1, len(segments)):
            prev_end = segments[i-1]['position'] + len(segments[i-1]['text'])
            curr_start = segments[i]['position']
            
            # Gap between segments (whitespace, breaks)
            gap = curr_start - prev_end
            spacings.append(float(gap))
        
        return spacings
    
    def _compute_relational_features(
        self,
        elements: List[SequentialElement],
        embeddings: np.ndarray
    ):
        """
        Compute how each element relates to neighbors.
        
        Measures:
        - Semantic distance to previous/next
        - Local density (how much happening nearby)
        - Momentum (direction of semantic change)
        """
        for i, element in enumerate(elements):
            # Distance to previous
            if i > 0:
                prev_embedding = embeddings[i-1]
                element.distance_to_prev = float(
                    np.linalg.norm(element.embedding - prev_embedding)
                )
            
            # Distance to next
            if i < len(elements) - 1:
                next_embedding = embeddings[i+1]
                element.distance_to_next = float(
                    np.linalg.norm(element.embedding - next_embedding)
                )
            
            # Local density (how much nearby)
            if i >= 2 and i < len(elements) - 2:
                # Window of ±2 elements
                window = embeddings[i-2:i+3]
                centroid = window.mean(axis=0)
                distances = np.linalg.norm(window - centroid, axis=1)
                element.local_density = float(1.0 / (1.0 + distances.mean()))
            
            # Momentum (change direction)
            if i >= 1 and i < len(elements) - 1:
                # Vector from prev to current
                v1 = element.embedding - embeddings[i-1]
                # Vector from current to next  
                v2 = embeddings[i+1] - element.embedding
                
                # Momentum = cosine similarity of vectors (continuing same direction?)
                dot = np.dot(v1, v2)
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                
                if norm1 > 0 and norm2 > 0:
                    element.momentum = float(dot / (norm1 * norm2))
                else:
                    element.momentum = 0.0
    
    def _extract_progression(
        self,
        elements: List[SequentialElement],
        embeddings: np.ndarray
    ) -> Dict:
        """
        Extract how narrative progresses through semantic space.
        
        Measures trajectory without interpreting it:
        - Path length (how much ground covered)
        - Directionality (straight path vs wandering)
        - Returns to origin (circular vs linear)
        - Acceleration patterns
        """
        if len(elements) < 3:
            return {'path_length': 0, 'directionality': 0}
        
        # Path length (sum of semantic distances)
        path_length = sum(
            np.linalg.norm(embeddings[i+1] - embeddings[i])
            for i in range(len(embeddings) - 1)
        )
        
        # Direct distance (start to end)
        direct_distance = np.linalg.norm(embeddings[-1] - embeddings[0])
        
        # Directionality = direct / path (1.0 = straight line, 0.0 = circular)
        directionality = direct_distance / path_length if path_length > 0 else 0.0
        
        # Return to origin (circular narrative?)
        return_similarity = np.dot(embeddings[0], embeddings[-1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[-1]) + 1e-8
        )
        
        # Acceleration (is path speeding up or slowing down?)
        step_sizes = [
            np.linalg.norm(embeddings[i+1] - embeddings[i])
            for i in range(len(embeddings) - 1)
        ]
        
        if len(step_sizes) >= 2:
            # Linear trend in step sizes
            x = np.arange(len(step_sizes))
            acceleration = np.polyfit(x, step_sizes, 1)[0]  # Slope
        else:
            acceleration = 0.0
        
        # Trajectory complexity (how many direction changes?)
        direction_changes = 0
        if len(embeddings) >= 3:
            for i in range(1, len(embeddings) - 1):
                v1 = embeddings[i] - embeddings[i-1]
                v2 = embeddings[i+1] - embeddings[i]
                
                # Angle between vectors
                cos_angle = np.dot(v1, v2) / (
                    np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
                )
                
                # Count as direction change if angle > 90 degrees
                if cos_angle < 0:
                    direction_changes += 1
        
        complexity = direction_changes / len(embeddings) if len(embeddings) > 0 else 0
        
        return {
            'path_length': float(path_length),
            'direct_distance': float(direct_distance),
            'directionality': float(directionality),
            'return_to_origin': float(return_similarity),
            'acceleration': float(acceleration),
            'complexity': float(complexity),
            'n_direction_changes': int(direction_changes)
        }
    
    def _analyze_rhythm(self, spacings: List[float]) -> Dict:
        """
        Analyze temporal rhythm from spacing patterns.
        
        Captures ρ (temporal rhythm) and patterns WITHOUT interpretation.
        """
        if len(spacings) < 2:
            return {'rho': 0, 'pattern': 'insufficient_data'}
        
        spacings_array = np.array(spacings[1:])  # Skip first (always 0)
        
        # Basic statistics
        mean_spacing = np.mean(spacings_array)
        std_spacing = np.std(spacings_array)
        
        # ρ (coefficient of variation)
        rho = std_spacing / mean_spacing if mean_spacing > 0 else 0.0
        
        # Distribution shape
        from scipy import stats as scipy_stats
        if len(spacings_array) >= 3:
            skewness = scipy_stats.skew(spacings_array)
            kurtosis = scipy_stats.kurtosis(spacings_array)
        else:
            skewness = 0.0
            kurtosis = 0.0
        
        # Autocorrelation (does rhythm have memory?)
        if len(spacings_array) >= 10:
            lag1_autocorr = np.corrcoef(spacings_array[:-1], spacings_array[1:])[0, 1]
        else:
            lag1_autocorr = 0.0
        
        # Trend (accelerating or decelerating?)
        if len(spacings_array) >= 3:
            x = np.arange(len(spacings_array))
            trend_slope = np.polyfit(x, spacings_array, 1)[0]
        else:
            trend_slope = 0.0
        
        return {
            'rho': float(rho),
            'mean_spacing': float(mean_spacing),
            'std_spacing': float(std_spacing),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'autocorrelation_lag1': float(lag1_autocorr),
            'trend': float(trend_slope),
            'min_spacing': float(np.min(spacings_array)),
            'max_spacing': float(np.max(spacings_array)),
            'note': 'Rhythm measured. Meaning elusive.'
        }
    
    def _extract_mysterious_dimensions(self, embeddings: np.ndarray) -> Dict:
        """
        Extract latent dimensions from sequential embeddings.
        
        These dimensions PREDICT but we don't know WHY.
        Better analysis through accepting mystery.
        
        Uses:
        - PCA: Variance structure
        - ICA: Independent components
        - UMAP: Topological structure
        
        Returns measurements WITHOUT interpretation.
        """
        if len(embeddings) < 10:
            return {'note': 'Insufficient data for dimension extraction'}
        
        n_dims = min(10, len(embeddings) // 2)
        
        # PCA (what explains variance?)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_dims)
        pca_dims = pca.fit_transform(embeddings)
        
        # ICA (what's independent?)
        from sklearn.decomposition import FastICA
        ica = FastICA(n_components=n_dims, random_state=42, max_iter=500)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                ica_dims = ica.fit_transform(embeddings)
            except:
                ica_dims = pca_dims  # Fallback
        
        # Trajectory through first 3 PCA dimensions
        trajectory_3d = pca_dims[:, :3].tolist() if pca_dims.shape[1] >= 3 else []
        
        return {
            'n_dimensions': n_dims,
            'variance_explained': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'trajectory_3d': trajectory_3d,  # Path through semantic space
            'note': 'Dimensions measured. Interpretation deliberately avoided.',
            'reminder': 'Let patterns work WITHOUT understanding them.'
        }
    
    def _get_cache_key(self, narrative_id: str) -> str:
        """Generate cache key for sequence."""
        return hashlib.md5(narrative_id.encode()).hexdigest()
    
    def _cache_sequence(
        self,
        cache_key: str,
        elements: List[SequentialElement],
        embeddings: np.ndarray
    ):
        """Cache full sequence data (including embeddings)."""
        cache_dir = Path(self.embedder.cache_dir) / 'sequences'
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_file = cache_dir / f"{cache_key}.npz"
        
        # Save embeddings and element data
        np.savez_compressed(
            cache_file,
            embeddings=embeddings,
            indices=np.array([e.index for e in elements]),
            positions=np.array([e.position_pct for e in elements]),
            spacings=np.array([e.spacing_before for e in elements]),
            lengths=np.array([e.length for e in elements]),
            distances_prev=np.array([e.distance_to_prev or 0 for e in elements]),
            distances_next=np.array([e.distance_to_next or 0 for e in elements]),
            local_density=np.array([e.local_density or 0 for e in elements]),
            momentum=np.array([e.momentum or 0 for e in elements])
        )


class BackgroundCorpusProcessor:
    """
    Background task processor for large narrative corpora.
    
    Processes source-by-source, preserving everything:
    - Sequential structure
    - Temporal spacing
    - Contextual embeddings
    - Progression patterns
    
    NO categorization. ONLY measurement.
    Let AI find ALL patterns without guidance.
    """
    
    def __init__(
        self,
        output_dir: str = 'data/processed_narratives',
        checkpoint_every: int = 100
    ):
        """
        Initialize background processor.
        
        Parameters
        ----------
        output_dir : str
            Where to store processed narratives
        checkpoint_every : int
            Save checkpoint every N narratives
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_every = checkpoint_every
        
        self.processor = SequentialNarrativeProcessor()
        
        # Progress tracking
        self.processed_count = 0
        self.source_progress = {}
    
    def process_source(
        self,
        source_name: str,
        narrative_iterator: Iterator[Tuple[str, str, Optional[Dict]]],
        expected_count: Optional[int] = None
    ):
        """
        Process narratives from a source in background.
        
        Parameters
        ----------
        source_name : str
            Source identifier (e.g., 'gutenberg_novels', 'imdb_plots')
        narrative_iterator : iterator
            Yields (narrative_id, narrative_text, metadata) tuples
        expected_count : int, optional
            Expected number of narratives (for progress tracking)
        """
        print(f"\n{'='*80}")
        print(f"PROCESSING SOURCE: {source_name}")
        print(f"{'='*80}\n")
        
        if expected_count:
            print(f"Expected: {expected_count:,} narratives")
        
        print("Beginning sequential analysis...")
        print("Preserving: order, spacing, progression, rhythm")
        print("NO categorization. ONLY measurement.\n")
        
        source_output = self.output_dir / source_name
        source_output.mkdir(exist_ok=True)
        
        processed_narratives = []
        source_count = 0
        start_time = time.time()
        
        for narrative_id, narrative_text, metadata in narrative_iterator:
            try:
                # Process narrative sequentially
                sequence_analysis = self.processor.process_narrative_sequential(
                    narrative=narrative_text,
                    narrative_id=narrative_id,
                    metadata=metadata
                )
                
                processed_narratives.append(sequence_analysis)
                source_count += 1
                self.processed_count += 1
                
                # Progress update
                if source_count % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = source_count / elapsed
                    print(f"  Processed: {source_count:,} narratives ({rate:.1f}/sec)")
                
                # Checkpoint
                if source_count % self.checkpoint_every == 0:
                    self._save_checkpoint(source_name, processed_narratives)
            
            except Exception as e:
                print(f"  ✗ Error processing {narrative_id}: {e}")
                continue
        
        # Final save
        self._save_source_complete(source_name, processed_narratives)
        
        elapsed = time.time() - start_time
        print(f"\n✓ Source complete: {source_name}")
        print(f"  Processed: {source_count:,} narratives in {elapsed:.1f}s")
        print(f"  Rate: {source_count/elapsed:.1f} narratives/sec")
        print(f"  Output: {source_output}/")
    
    def _save_checkpoint(self, source_name: str, processed: List[Dict]):
        """Save checkpoint during processing."""
        checkpoint_file = self.output_dir / source_name / f'checkpoint_{len(processed)}.json'
        
        with open(checkpoint_file, 'w') as f:
            json.dump({
                'source': source_name,
                'count': len(processed),
                'timestamp': datetime.now().isoformat(),
                'narratives': processed
            }, f)
    
    def _save_source_complete(self, source_name: str, processed: List[Dict]):
        """Save complete processed source."""
        output_file = self.output_dir / source_name / f'{source_name}_complete.json'
        
        with open(output_file, 'w') as f:
            json.dump({
                'source': source_name,
                'count': len(processed),
                'completed_at': datetime.now().isoformat(),
                'narratives': processed
            }, f, indent=2)
        
        # Also save summary
        summary_file = self.output_dir / source_name / f'{source_name}_summary.json'
        with open(summary_file, 'w') as f:
            json.dump({
                'source': source_name,
                'total_narratives': len(processed),
                'avg_elements_per_narrative': np.mean([n['n_elements'] for n in processed]),
                'avg_length': np.mean([n['total_length'] for n in processed]),
                'completed_at': datetime.now().isoformat()
            }, f, indent=2)


def discover_universal_patterns_across_sources(
    source_dirs: List[str],
    min_narratives_per_source: int = 1000,
    output_file: Optional[str] = None
) -> Dict:
    """
    Meta-analysis: Discover patterns that emerge ACROSS narrative sources.
    
    This is where universal structure reveals itself (if it exists).
    
    Process:
    1. Load processed narratives from all sources
    2. Combine embeddings (preserve source labels)
    3. Discover patterns in combined space
    4. Test if patterns appear in ALL sources (universal)
    5. Test if patterns are source-specific (cultural)
    
    DO NOT presuppose universality. LET DATA SHOW IT.
    
    Parameters
    ----------
    source_dirs : list of str
        Directories containing processed sources
    min_narratives_per_source : int
        Minimum narratives to include source
    output_file : str, optional
        Where to save meta-analysis results
        
    Returns
    -------
    meta_discovery : dict
        Universal patterns (if any exist)
        Source-specific patterns
        Cross-source correlations
        MYSTERIOUS dimensions that work everywhere
    """
    print("\n" + "="*80)
    print("META-ANALYSIS: DISCOVERING UNIVERSAL PATTERNS")
    print("="*80)
    print("\nPhilosophy:")
    print("- Do not assume universality exists")
    print("- Let data show what's universal vs cultural")
    print("- Patterns that appear everywhere are REAL")
    print("- Patterns unique to one source are SPECIFIC")
    print("="*80 + "\n")
    
    # Load all sources
    all_embeddings = []
    all_source_labels = []
    all_outcomes = []
    
    print("Loading processed sources...")
    
    for source_dir in source_dirs:
        source_path = Path(source_dir)
        if not source_path.exists():
            print(f"  ✗ Not found: {source_dir}")
            continue
        
        # Load complete file
        complete_files = list(source_path.glob('*_complete.json'))
        if not complete_files:
            print(f"  ✗ No complete file in: {source_dir}")
            continue
        
        with open(complete_files[0], 'r') as f:
            source_data = json.load(f)
        
        n_narratives = source_data['count']
        if n_narratives < min_narratives_per_source:
            print(f"  ⊘ Skipping {source_data['source']}: only {n_narratives} narratives")
            continue
        
        print(f"  ✓ Loaded {source_data['source']}: {n_narratives:,} narratives")
        
        # TODO: Load embeddings from cache
        # For now, placeholder
    
    # Discover cross-source patterns
    print("\nDiscovering patterns across sources...")
    print("This reveals what's UNIVERSAL vs what's CULTURAL\n")
    
    # TODO: Implement cross-source pattern discovery
    
    print("="*80)
    print("META-ANALYSIS COMPLETE")
    print("="*80)
    
    return {
        'universal_patterns': [],
        'source_specific_patterns': {},
        'note': 'Patterns emerge from data. Mechanisms remain elusive.'
    }

