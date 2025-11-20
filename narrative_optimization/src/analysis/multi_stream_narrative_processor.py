"""
Multi-Stream Narrative Processor

CRITICAL REALIZATION: One "story" contains MANY simultaneous stories.

A novel contains:
- Main plot sequence
- Each character's arc (N different stories)
- Multiple subplot sequences
- Thematic progression (ideas evolving)
- Symbolic journey (metaphors developing)
- Relationship threads (each relationship is story)
- Background stories (world events, history)
- Temporal threads (past/present/future narratives)
- Tonal shifts (mood progressions)
- Structural meta-narrative (form itself tells story)

Each stream has its own:
- Sequence (different orderings)
- Spacing/rhythm
- Progression trajectory
- Beginning/middle/end
- Acceleration patterns

AI must detect and track ALL streams WITHOUT us specifying what they are.

Author: Narrative Optimization Framework
Date: November 2025
"""

from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
from pathlib import Path
import json
from dataclasses import dataclass, field
from collections import defaultdict
import warnings

try:
    from ..transformers.utils.embeddings import EmbeddingManager
    from ..transformers.utils.shared_models import SharedModelRegistry
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from transformers.utils.embeddings import EmbeddingManager
    from transformers.utils.shared_models import SharedModelRegistry


@dataclass
class NarrativeStream:
    """
    Single narrative thread within larger story.
    
    A stream is:
    - Coherent sequence (elements connected semantically)
    - Has its own rhythm and spacing
    - Has beginning, middle, end (not necessarily in that order)
    - Can weave in and out of main narrative
    - Has its own progression trajectory
    
    DO NOT name streams (character arc, plot, theme).
    Just detect: "Stream A", "Stream B", etc.
    Let AI discover what makes a coherent stream.
    """
    stream_id: int
    element_indices: List[int]  # Which elements belong to this stream
    embeddings: np.ndarray  # Embeddings of elements in this stream
    positions: List[float]  # Positions in overall narrative (0-1)
    
    # Stream characteristics (measured, not interpreted)
    coherence: float = 0.0  # How semantically connected
    continuity: float = 0.0  # How temporally continuous
    prominence: float = 0.0  # How much text/attention
    
    # Stream progression
    path_length: float = 0.0
    directionality: float = 0.0
    
    # Stream rhythm (separate from overall narrative rhythm)
    stream_rho: float = 0.0
    stream_spacing: List[float] = field(default_factory=list)


class MultiStreamNarrativeProcessor:
    """
    Detect and track MULTIPLE concurrent narrative streams.
    
    Process:
    1. Segment narrative into elements
    2. Embed all elements contextually
    3. Detect streams (clusters in semantic space that are also temporally related)
    4. Track each stream independently
    5. Analyze stream interactions (when streams converge/diverge)
    6. Extract multi-stream features
    
    NO presupposition about:
    - How many streams
    - What streams represent (plot? character? theme?)
    - Which is "main" vs "sub"
    
    Let AI discover natural stream structure.
    """
    
    def __init__(
        self,
        min_stream_length: int = 3,
        semantic_coherence_threshold: float = 0.6,
        embedding_model: str = 'all-MiniLM-L6-v2'
    ):
        """
        Initialize multi-stream processor.
        
        Parameters
        ----------
        min_stream_length : int
            Minimum elements to constitute stream
        semantic_coherence_threshold : float
            Minimum similarity to be part of stream
        embedding_model : str
            Model for embeddings
        """
        self.min_stream_length = min_stream_length
        self.coherence_threshold = semantic_coherence_threshold
        
        self.embedder = EmbeddingManager(
            model_name=embedding_model,
            use_cache=True
        )
    
    def discover_streams(
        self,
        narrative: str,
        narrative_id: str
    ) -> Dict[str, Any]:
        """
        Discover all narrative streams in text.
        
        Returns
        -------
        multi_stream_analysis : dict
            {
                'streams': List of discovered streams,
                'stream_count': How many concurrent stories,
                'interactions': When streams converge/diverge,
                'dominant_stream': Which stream is most prominent,
                'stream_balance': How balanced are streams,
                'weaving_pattern': How streams interleave
            }
        """
        print(f"\nDiscovering narrative streams in: {narrative_id}")
        print("  (Remember: one story contains many stories)")
        
        # Step 1: Segment narrative
        segments = self._segment_narrative(narrative)
        
        if len(segments) < self.min_stream_length * 2:
            return {'error': 'Narrative too short for multi-stream analysis'}
        
        print(f"  Segments: {len(segments)}")
        
        # Step 2: Embed all segments
        segment_texts = [s['text'] for s in segments]
        embeddings = self.embedder.encode(segment_texts, show_progress=False)
        
        print(f"  Embeddings: {embeddings.shape}")
        
        # Step 3: Detect streams (coherent + temporally related clusters)
        streams = self._detect_streams_unsupervised(
            segments,
            embeddings
        )
        
        print(f"  Streams discovered: {len(streams)}")
        
        # Step 4: Analyze each stream
        for stream in streams:
            self._analyze_stream(stream)
        
        # Step 5: Analyze stream interactions
        interactions = self._analyze_stream_interactions(streams, len(segments))
        
        # Step 6: Extract multi-stream features
        multi_stream_features = self._extract_multi_stream_features(
            streams,
            interactions,
            len(segments)
        )
        
        print(f"  ✓ Multi-stream analysis complete\n")
        
        result = {
            'narrative_id': narrative_id,
            'n_segments': len(segments),
            'n_streams': len(streams),
            'streams': [self._serialize_stream(s) for s in streams],
            'interactions': interactions,
            'features': multi_stream_features,
            'note': 'Streams detected by AI. Not labeled as plot/character/theme.',
            'reminder': 'Each stream is story. One narrative = many stories.'
        }
        
        return result
    
    def extract_stream_features_for_genome(
        self,
        narrative: str,
        narrative_id: str
    ) -> np.ndarray:
        """
        Extract stream features as feature vector for genome integration.
        
        Returns fixed-size feature vector (20 features) suitable for
        adding to StoryInstance genome as concurrent narrative component.
        
        Parameters
        ----------
        narrative : str
            Narrative text
        narrative_id : str
            Narrative identifier
        
        Returns
        -------
        ndarray
            Stream features (20 dimensions):
            [0]: stream_count (normalized)
            [1]: avg_stream_coherence
            [2]: avg_stream_continuity
            [3]: avg_stream_prominence
            [4]: dominant_stream_prominence
            [5]: stream_balance (entropy)
            [6]: interaction_density
            [7]: convergence_count (normalized)
            [8]: divergence_count (normalized)
            [9]: avg_path_length
            [10]: avg_directionality
            [11]: avg_stream_rho (rhythm)
            [12]: weaving_complexity
            [13]: max_stream_coherence
            [14]: min_stream_coherence
            [15]: stream_coherence_variance
            [16]: temporal_distribution_score
            [17]: semantic_distance_range
            [18]: multi_stream_quality_score
            [19]: narrative_richness (streams per segment)
        """
        result = self.discover_streams(narrative, narrative_id)
        
        if 'error' in result:
            # Return zeros if analysis failed
            return np.zeros(20)
        
        features = np.zeros(20)
        
        streams = result.get('streams', [])
        interactions = result.get('interactions', {})
        n_segments = result.get('n_segments', 1)
        
        if len(streams) == 0:
            return features
        
        # [0] Stream count (normalized by segments)
        features[0] = len(streams) / max(n_segments, 1)
        
        # [1-3] Average stream characteristics
        coherences = [s['coherence'] for s in streams if 'coherence' in s]
        continuities = [s['continuity'] for s in streams if 'continuity' in s]
        prominences = [s['prominence'] for s in streams if 'prominence' in s]
        
        features[1] = np.mean(coherences) if coherences else 0.0
        features[2] = np.mean(continuities) if continuities else 0.0
        features[3] = np.mean(prominences) if prominences else 0.0
        
        # [4] Dominant stream prominence
        features[4] = max(prominences) if prominences else 0.0
        
        # [5] Stream balance (entropy)
        if prominences:
            # Normalize prominences to probabilities
            prom_array = np.array(prominences)
            prom_norm = prom_array / (prom_array.sum() + 1e-8)
            # Calculate entropy
            entropy = -np.sum(prom_norm * np.log(prom_norm + 1e-8))
            # Normalize by max entropy
            max_entropy = np.log(len(prominences))
            features[5] = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # [6] Interaction density
        convergences = interactions.get('convergences', [])
        divergences = interactions.get('divergences', [])
        total_interactions = len(convergences) + len(divergences)
        features[6] = total_interactions / max(n_segments, 1)
        
        # [7-8] Convergence and divergence counts
        features[7] = len(convergences) / max(n_segments, 1)
        features[8] = len(divergences) / max(n_segments, 1)
        
        # [9-10] Path and directionality
        path_lengths = [s['path_length'] for s in streams if 'path_length' in s]
        directionalities = [s['directionality'] for s in streams if 'directionality' in s]
        
        features[9] = np.mean(path_lengths) if path_lengths else 0.0
        features[10] = np.mean(directionalities) if directionalities else 0.0
        
        # [11] Average stream rhythm
        rhos = [s['stream_rho'] for s in streams if 'stream_rho' in s]
        features[11] = np.mean(rhos) if rhos else 0.0
        
        # [12] Weaving complexity (how interwoven are streams)
        # Higher when streams frequently appear near each other
        if len(streams) > 1:
            positions_by_stream = []
            for s in streams:
                positions_by_stream.append(s.get('positions', []))
            
            # Calculate position overlap
            overlaps = 0
            for i in range(len(positions_by_stream)):
                for j in range(i+1, len(positions_by_stream)):
                    pos_i = set([int(p*100) for p in positions_by_stream[i]])
                    pos_j = set([int(p*100) for p in positions_by_stream[j]])
                    overlaps += len(pos_i & pos_j)
            
            features[12] = overlaps / (len(streams) * (len(streams)-1) / 2 + 1)
        
        # [13-15] Coherence statistics
        if coherences:
            features[13] = max(coherences)
            features[14] = min(coherences)
            features[15] = np.var(coherences)
        
        # [16] Temporal distribution (how evenly distributed)
        # Check if streams appear throughout narrative
        if len(streams) > 0:
            temporal_spans = []
            for s in streams:
                positions = s.get('positions', [])
                if len(positions) > 1:
                    span = max(positions) - min(positions)
                    temporal_spans.append(span)
            features[16] = np.mean(temporal_spans) if temporal_spans else 0.0
        
        # [17] Semantic distance range
        # How different are the streams from each other
        if len(coherences) > 1:
            features[17] = max(coherences) - min(coherences)
        
        # [18] Multi-stream quality score (aggregate)
        # Higher when streams are coherent, balanced, and well-distributed
        features[18] = (
            0.3 * features[1] +  # coherence
            0.2 * features[5] +  # balance
            0.2 * features[16] +  # distribution
            0.15 * features[12] +  # weaving
            0.15 * features[10]   # directionality
        )
        
        # [19] Narrative richness (streams per segment)
        features[19] = len(streams) / max(n_segments / 10, 1)  # Per 10 segments
        
        return features
    
    def _detect_streams_unsupervised(
        self,
        segments: List[Dict],
        embeddings: np.ndarray
    ) -> List[NarrativeStream]:
        """
        Detect narrative streams WITHOUT presupposing what they are.
        
        A stream is:
        - Semantically coherent (elements similar to each other)
        - Temporally related (appear throughout narrative, not just clustered)
        - Has progression (moves through semantic space)
        
        Method:
        1. Cluster by semantic similarity (coherence)
        2. Filter clusters that span narrative (not just local)
        3. Order elements by position (reconstruct stream sequence)
        4. Each valid cluster = discovered stream
        """
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics.pairwise import cosine_similarity
        
        n_segments = len(embeddings)
        
        # Similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Distance matrix (for clustering)
        distance_matrix = 1 - similarity_matrix
        
        # Hierarchical clustering (discovers natural groupings)
        # Use distance threshold so we don't need to specify n_clusters
        n_streams_estimated = max(5, n_segments // 20)  # Estimate
        
        clusterer = AgglomerativeClustering(
            n_clusters=n_streams_estimated,
            metric='precomputed',
            linkage='average'
        )
        
        stream_labels = clusterer.fit_predict(distance_matrix)
        
        # Build streams from clusters
        detected_streams = []
        
        for stream_id in set(stream_labels):
            # Elements in this stream
            stream_indices = [i for i, label in enumerate(stream_labels) if label == stream_id]
            
            # Filter: stream must span significant portion of narrative
            if len(stream_indices) < self.min_stream_length:
                continue  # Too short
            
            # Check temporal spread
            positions = [segments[i]['position'] for i in stream_indices]
            position_spread = (max(positions) - min(positions)) / max(positions)
            
            if position_spread < 0.3:  # Must span at least 30% of narrative
                continue  # Too localized (just a local topic, not a stream)
            
            # Valid stream
            stream_embeddings = embeddings[stream_indices]
            stream_positions = [segments[i]['position'] / segments[-1]['position'] 
                              for i in stream_indices]
            
            # Sort by position (reconstruct stream sequence)
            sorted_order = np.argsort(stream_positions)
            sorted_indices = [stream_indices[i] for i in sorted_order]
            sorted_embeddings = stream_embeddings[sorted_order]
            sorted_positions = [stream_positions[i] for i in sorted_order]
            
            # Compute coherence (average pairwise similarity)
            stream_similarity = cosine_similarity(stream_embeddings)
            coherence = (stream_similarity.sum() - len(stream_indices)) / (
                len(stream_indices) * (len(stream_indices) - 1) + 1e-8
            )
            
            # Compute continuity (temporal regularity)
            if len(sorted_positions) >= 2:
                position_gaps = np.diff(sorted_positions)
                continuity = 1.0 / (1.0 + np.std(position_gaps))
            else:
                continuity = 1.0
            
            # Prominence (how much text)
            prominence = len(stream_indices) / n_segments
            
            stream = NarrativeStream(
                stream_id=stream_id,
                element_indices=sorted_indices,
                embeddings=sorted_embeddings,
                positions=sorted_positions,
                coherence=float(coherence),
                continuity=float(continuity),
                prominence=float(prominence)
            )
            
            detected_streams.append(stream)
        
        # Sort streams by prominence (but DON'T call any "main")
        detected_streams.sort(key=lambda s: s.prominence, reverse=True)
        
        # Reassign IDs (stream_0 is most prominent, but not "main")
        for new_id, stream in enumerate(detected_streams):
            stream.stream_id = new_id
        
        return detected_streams
    
    def _analyze_stream(self, stream: NarrativeStream):
        """
        Analyze individual stream progression.
        
        Measures WITHOUT interpretation:
        - Path through semantic space
        - Direction (linear vs circular)
        - Rhythm of stream (separate from overall)
        """
        if len(stream.embeddings) < 2:
            return
        
        # Path length (semantic distance traveled)
        path_length = sum(
            np.linalg.norm(stream.embeddings[i+1] - stream.embeddings[i])
            for i in range(len(stream.embeddings) - 1)
        )
        stream.path_length = float(path_length)
        
        # Directionality (straight vs wandering)
        direct_distance = np.linalg.norm(
            stream.embeddings[-1] - stream.embeddings[0]
        )
        stream.directionality = float(
            direct_distance / path_length if path_length > 0 else 0.0
        )
        
        # Stream rhythm (spacing between appearances)
        if len(stream.positions) >= 2:
            stream.stream_spacing = list(np.diff(stream.positions))
            
            # ρ for this stream
            if len(stream.stream_spacing) >= 2:
                mean_spacing = np.mean(stream.stream_spacing)
                std_spacing = np.std(stream.stream_spacing)
                stream.stream_rho = float(
                    std_spacing / mean_spacing if mean_spacing > 0 else 0.0
                )
    
    def _analyze_stream_interactions(
        self,
        streams: List[NarrativeStream],
        n_total_segments: int
    ) -> Dict:
        """
        Analyze how streams interact with each other.
        
        Interactions:
        - Convergence: When streams become semantically similar
        - Divergence: When streams separate
        - Weaving: How streams alternate
        - Dominance shifts: Which stream prominent when
        - Resonance: Streams echoing each other
        
        DO NOT interpret as "plot and character converge at climax".
        JUST measure: "Stream 0 and Stream 1 converge at 75%".
        """
        if len(streams) < 2:
            return {'note': 'Single stream, no interactions'}
        
        interactions = []
        
        # Analyze each pair of streams
        for i in range(len(streams)):
            for j in range(i + 1, len(streams)):
                stream_a = streams[i]
                stream_b = streams[j]
                
                # Find temporal overlap points
                overlap_points = self._find_overlap_points(stream_a, stream_b, n_total_segments)
                
                # Measure convergence/divergence at overlap points
                for point in overlap_points:
                    # Get nearest elements from each stream to this position
                    a_idx = self._nearest_element(stream_a, point)
                    b_idx = self._nearest_element(stream_b, point)
                    
                    if a_idx is not None and b_idx is not None:
                        # Semantic distance between streams at this point
                        distance = np.linalg.norm(
                            stream_a.embeddings[a_idx] - stream_b.embeddings[b_idx]
                        )
                        
                        interactions.append({
                            'position': point,
                            'streams': [stream_a.stream_id, stream_b.stream_id],
                            'distance': float(distance),
                            'type': 'convergent' if distance < 0.5 else 'divergent'
                        })
        
        # Analyze weaving pattern (how streams alternate)
        weaving = self._analyze_weaving_pattern(streams, n_total_segments)
        
        # Analyze resonance (streams echoing each other)
        resonance = self._analyze_resonance(streams)
        
        return {
            'convergence_divergence_points': interactions,
            'weaving_pattern': weaving,
            'resonance': resonance,
            'note': 'Stream interactions measured. Meaning intentionally elusive.'
        }
    
    def _find_overlap_points(
        self,
        stream_a: NarrativeStream,
        stream_b: NarrativeStream,
        n_segments: int,
        n_points: int = 10
    ) -> List[float]:
        """Find positions where streams might interact."""
        # Sample positions across narrative
        return list(np.linspace(0, 1, n_points))
    
    def _nearest_element(
        self,
        stream: NarrativeStream,
        position: float
    ) -> Optional[int]:
        """Find stream element nearest to position."""
        if not stream.positions:
            return None
        
        distances = [abs(p - position) for p in stream.positions]
        nearest_idx = np.argmin(distances)
        
        return nearest_idx
    
    def _analyze_weaving_pattern(
        self,
        streams: List[NarrativeStream],
        n_segments: int
    ) -> Dict:
        """
        Analyze how streams weave through narrative.
        
        Pattern types (discovered, not presupposed):
        - Parallel: Streams progress simultaneously
        - Alternating: Streams take turns
        - Nested: One stream contains another
        - Braided: Streams intertwine regularly
        - Convergent: Streams start separate, merge
        - Divergent: Streams start together, separate
        """
        # Create timeline showing which stream at each position
        timeline = [None] * n_segments
        
        for stream in streams:
            for idx in stream.element_indices:
                if timeline[idx] is None:
                    timeline[idx] = [stream.stream_id]
                else:
                    timeline[idx].append(stream.stream_id)
        
        # Measure switching frequency
        switches = 0
        for i in range(1, len(timeline)):
            if timeline[i] != timeline[i-1]:
                switches += 1
        
        switch_rate = switches / len(timeline)
        
        # Measure parallel sections (multiple streams active)
        parallel_count = sum(1 for t in timeline if t and len(t) > 1)
        parallel_rate = parallel_count / len(timeline)
        
        # Measure stream dominance over time
        dominance_shifts = []
        current_dominant = None
        
        for pos_idx, active_streams in enumerate(timeline):
            if active_streams:
                # Most prominent stream at this position
                prominent = max(active_streams, key=lambda sid: streams[sid].prominence)
                
                if current_dominant != prominent:
                    dominance_shifts.append({
                        'position': pos_idx / len(timeline),
                        'from_stream': current_dominant,
                        'to_stream': prominent
                    })
                    current_dominant = prominent
        
        return {
            'switch_rate': float(switch_rate),
            'parallel_rate': float(parallel_rate),
            'dominance_shifts': dominance_shifts,
            'n_dominance_shifts': len(dominance_shifts),
            'note': 'Weaving pattern measured. DO NOT interpret as plot vs character.'
        }
    
    def _analyze_resonance(self, streams: List[NarrativeStream]) -> Dict:
        """
        Measure if streams echo/mirror each other.
        
        Resonance = streams having similar progression patterns even if
        semantically different content.
        
        Example (DO NOT hardcode this, let AI find it):
        - Main plot: tension rises, peaks, resolves
        - Character arc: confidence rises, peaks, resolves
        - Thematic idea: complexity rises, peaks, resolves
        
        Same SHAPE, different CONTENT.
        This is resonance. Measure it WITHOUT explaining it.
        """
        if len(streams) < 2:
            return {'resonance_score': 0.0}
        
        resonances = []
        
        for i in range(len(streams)):
            for j in range(i + 1, len(streams)):
                # Normalize both streams to comparable lengths
                stream_a_path = self._normalize_stream_trajectory(streams[i])
                stream_b_path = self._normalize_stream_trajectory(streams[j])
                
                if stream_a_path is None or stream_b_path is None:
                    continue
                
                # Compare progression shapes (correlation of trajectories)
                if len(stream_a_path) == len(stream_b_path):
                    resonance_score = np.corrcoef(stream_a_path, stream_b_path)[0, 1]
                    
                    resonances.append({
                        'streams': [streams[i].stream_id, streams[j].stream_id],
                        'resonance': float(resonance_score),
                        'interpretation': 'UNKNOWN - streams mirror each other for mysterious reasons'
                    })
        
        avg_resonance = np.mean([r['resonance'] for r in resonances]) if resonances else 0.0
        
        return {
            'stream_pairs': resonances,
            'avg_resonance': float(avg_resonance),
            'note': 'Streams echo each other. Why? Elusive.'
        }
    
    def _normalize_stream_trajectory(
        self,
        stream: NarrativeStream,
        n_points: int = 20
    ) -> Optional[np.ndarray]:
        """
        Extract stream's progression as normalized trajectory.
        
        Maps stream to fixed number of points for comparison.
        """
        if len(stream.embeddings) < 3:
            return None
        
        # Cumulative semantic distance (progression metric)
        distances = [0.0]
        for i in range(len(stream.embeddings) - 1):
            dist = np.linalg.norm(stream.embeddings[i+1] - stream.embeddings[i])
            distances.append(distances[-1] + dist)
        
        # Normalize to [0, 1]
        if distances[-1] > 0:
            distances = np.array(distances) / distances[-1]
        else:
            distances = np.array(distances)
        
        # Interpolate to fixed number of points
        interp_positions = np.linspace(0, 1, n_points)
        trajectory = np.interp(interp_positions, stream.positions, distances)
        
        return trajectory
    
    def _extract_multi_stream_features(
        self,
        streams: List[NarrativeStream],
        interactions: Dict,
        n_segments: int
    ) -> Dict:
        """
        Extract features from multi-stream structure.
        
        Features (NO interpretation):
        - Number of streams
        - Stream balance (dominance distribution)
        - Interaction frequency
        - Weaving complexity
        - Resonance levels
        - Stream rhythms
        - Convergence patterns
        
        These predict outcomes WITHOUT us knowing mechanism.
        """
        if not streams:
            return {}
        
        features = {
            # Basic structure
            'n_streams': len(streams),
            'n_streams_normalized': min(len(streams) / 10.0, 1.0),
            
            # Stream balance
            'stream_balance_entropy': self._compute_entropy([s.prominence for s in streams]),
            'dominant_stream_prominence': streams[0].prominence if streams else 0.0,
            
            # Stream characteristics (averaged)
            'avg_stream_coherence': float(np.mean([s.coherence for s in streams])),
            'avg_stream_continuity': float(np.mean([s.continuity for s in streams])),
            'avg_stream_path_length': float(np.mean([s.path_length for s in streams])),
            'avg_stream_directionality': float(np.mean([s.directionality for s in streams])),
            
            # Stream rhythm characteristics
            'avg_stream_rho': float(np.mean([s.stream_rho for s in streams if s.stream_rho > 0])),
            
            # Interactions
            'interaction_rate': float(len(interactions.get('convergence_divergence_points', [])) / n_segments),
            'switch_rate': float(interactions.get('weaving_pattern', {}).get('switch_rate', 0)),
            'parallel_rate': float(interactions.get('weaving_pattern', {}).get('parallel_rate', 0)),
            'avg_resonance': float(interactions.get('resonance', {}).get('avg_resonance', 0)),
            
            # Complexity
            'stream_complexity': float(len(streams) * interactions.get('weaving_pattern', {}).get('switch_rate', 0)),
            
            # Convergence (streams merging toward end)
            'final_convergence': self._measure_final_convergence(streams),
            
            'note': 'Multi-stream features. Work without interpretation.'
        }
        
        return features
    
    def _measure_final_convergence(self, streams: List[NarrativeStream]) -> float:
        """
        Measure if streams converge toward narrative end.
        
        Compare semantic distances between streams in:
        - First third vs last third
        
        Convergence = distances decrease over narrative.
        """
        if len(streams) < 2:
            return 0.0
        
        # Get stream embeddings in first third vs last third
        first_third_embeddings = []
        last_third_embeddings = []
        
        for stream in streams:
            # First third positions
            first_elements = [i for i, p in enumerate(stream.positions) if p < 0.33]
            if first_elements:
                first_third_embeddings.append(stream.embeddings[first_elements[-1]])
            
            # Last third positions
            last_elements = [i for i, p in enumerate(stream.positions) if p > 0.67]
            if last_elements:
                last_third_embeddings.append(stream.embeddings[last_elements[0]])
        
        if len(first_third_embeddings) < 2 or len(last_third_embeddings) < 2:
            return 0.0
        
        # Average pairwise distance in first vs last
        from sklearn.metrics.pairwise import euclidean_distances
        
        first_distances = euclidean_distances(first_third_embeddings)
        last_distances = euclidean_distances(last_third_embeddings)
        
        avg_first = first_distances[np.triu_indices_from(first_distances, k=1)].mean()
        avg_last = last_distances[np.triu_indices_from(last_distances, k=1)].mean()
        
        # Convergence = distances decreased
        convergence = (avg_first - avg_last) / avg_first if avg_first > 0 else 0.0
        
        return float(convergence)
    
    def _compute_entropy(self, values: List[float]) -> float:
        """Compute entropy (balance measure)."""
        if not values or sum(values) == 0:
            return 0.0
        
        probs = np.array(values) / sum(values)
        probs = probs[probs > 0]  # Remove zeros
        
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(len(probs))
        
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0
    
    def _segment_narrative(self, narrative: str) -> List[Dict]:
        """Segment narrative (paragraphs or sentences)."""
        # Use adaptive method
        paragraphs = narrative.split('\n\n')
        
        segments = []
        position = 0
        for para in paragraphs:
            para = para.strip()
            if para:
                segments.append({
                    'text': para,
                    'position': position
                })
                position += len(para) + 2
        
        return segments
    
    def _serialize_stream(self, stream: NarrativeStream) -> Dict:
        """Convert stream to serializable format."""
        return {
            'stream_id': stream.stream_id,
            'n_elements': len(stream.element_indices),
            'prominence': stream.prominence,
            'coherence': stream.coherence,
            'continuity': stream.continuity,
            'path_length': stream.path_length,
            'directionality': stream.directionality,
            'stream_rho': stream.stream_rho,
            'positions': stream.positions,
            'note': f'Stream {stream.stream_id}: Exists. Meaning unknown.'
        }


def process_narrative_with_all_streams(
    narrative: str,
    narrative_id: str,
    outcome: Optional[float] = None
) -> Dict:
    """
    Complete multi-stream analysis of single narrative.
    
    Use this for deep analysis of individual narratives.
    Discovers ALL concurrent stories within the story.
    
    Returns measurements WITHOUT interpretation.
    Let patterns remain mysterious.
    """
    processor = MultiStreamNarrativeProcessor()
    
    discovery = processor.discover_streams(narrative, narrative_id)
    
    if outcome is not None:
        discovery['outcome'] = outcome
        discovery['note_outcome'] = 'Outcome provided. Correlate streams with success separately.'
    
    return discovery


def batch_process_corpus_multi_stream(
    narratives: List[Tuple[str, str, Optional[float]]],
    output_dir: str = 'data/multi_stream_processed',
    checkpoint_every: int = 100
) -> str:
    """
    Process large corpus with multi-stream analysis.
    
    Parameters
    ----------
    narratives : list of (narrative_id, text, outcome) tuples
    output_dir : str
        Where to save results
    checkpoint_every : int
        Checkpoint frequency
        
    Returns
    -------
    output_path : str
        Where results are saved
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    processor = MultiStreamNarrativeProcessor()
    
    results = []
    
    print(f"\n{'='*80}")
    print(f"MULTI-STREAM BATCH PROCESSING")
    print(f"{'='*80}\n")
    print(f"Corpus size: {len(narratives):,} narratives")
    print(f"Each narrative analyzed for MULTIPLE concurrent stories")
    print(f"NO categorization. PURE discovery.\n")
    
    for idx, (narrative_id, text, outcome) in enumerate(narratives):
        if idx % 100 == 0 and idx > 0:
            print(f"  Processed: {idx:,}/{len(narratives):,}")
        
        try:
            discovery = processor.discover_streams(text, narrative_id)
            
            if outcome is not None:
                discovery['outcome'] = outcome
            
            results.append(discovery)
            
            # Checkpoint
            if idx > 0 and idx % checkpoint_every == 0:
                checkpoint_file = output_path / f'checkpoint_{idx}.json'
                with open(checkpoint_file, 'w') as f:
                    json.dump(results, f, indent=2)
        
        except Exception as e:
            print(f"  ✗ Error processing {narrative_id}: {e}")
            continue
    
    # Save complete results
    final_file = output_path / 'multi_stream_analysis_complete.json'
    with open(final_file, 'w') as f:
        json.dump({
            'n_narratives': len(results),
            'analyses': results,
            'summary': {
                'avg_streams_per_narrative': np.mean([r['n_streams'] for r in results]),
                'max_streams': max([r['n_streams'] for r in results]),
                'min_streams': min([r['n_streams'] for r in results])
            }
        }, f, indent=2)
    
    print(f"\n✓ Complete. Results: {final_file}")
    
    return str(final_file)

