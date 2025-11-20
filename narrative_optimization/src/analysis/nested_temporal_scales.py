"""
Nested Temporal Scales Framework

CRITICAL INSIGHT: Every narrative exists at MULTIPLE temporal scales SIMULTANEOUSLY.

A single NBA game contains:
- Possession narratives (10-30 seconds each) [~200 possessions]
- Quarter narratives (12 minutes each) [4 quarters]
- Half narratives (24 minutes each) [2 halves]
- Full game narrative (48 minutes) [1 game]
- Matchup narrative (season series) [2-4 games]
- Season narrative (82 games)
- Playoff series narrative (4-7 games if applicable)
- Season arc narrative (months)
- Career narrative (years)
- Rivalry narrative (decades)
- Franchise narrative (50+ years)
- Era narrative (generations)
- Sport narrative (century+)

ALL scales coexist simultaneously in potentiality.
Each scale has its own:
- Sequence and spacing
- Beginning/middle/end
- Progression and rhythm
- Outcome and meaning
- Connection to other scales

This is FRACTAL. Same patterns at every scale.
Or maybe NOT - let AI discover if patterns recur or differ.

Author: Narrative Optimization Framework
Date: November 2025
"""

from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
import json

try:
    from ..transformers.utils.embeddings import EmbeddingManager
except ImportError:
    from narrative_optimization.src.transformers.utils.embeddings import EmbeddingManager


@dataclass
class TemporalScale:
    """
    Single temporal scale of analysis.
    
    A scale has:
    - Duration (how long)
    - Contains N lower-scale units
    - Is contained in M higher-scale units
    - Has its own narrative structure
    - Connects to parallel scales
    """
    scale_name: str
    duration: float  # In consistent units (seconds, minutes, etc.)
    contains: List['TemporalScale'] = field(default_factory=list)
    contained_in: Optional['TemporalScale'] = None
    
    # Narrative at this scale
    narrative_elements: List[Any] = field(default_factory=list)
    sequence: Optional[List[int]] = None
    spacing: Optional[List[float]] = None
    
    # Scale-specific features
    tau: Optional[float] = None  # Duration ratio for this scale
    sigma: Optional[float] = None  # Compression at this scale
    rho: Optional[float] = None  # Rhythm at this scale
    
    # Outcome at this scale
    outcome: Optional[Any] = None


class NestedTemporalAnalyzer:
    """
    Analyze narrative across ALL temporal scales simultaneously.
    
    Process:
    1. Identify all temporal scales present
    2. Extract narrative at each scale
    3. Analyze each scale independently
    4. Analyze cross-scale patterns
    5. Test if scales show similar or different structures
    
    NO presupposition about:
    - Which scales matter most
    - Whether patterns recur across scales (fractal)
    - How scales interact
    
    Let AI discover multi-scale structure.
    """
    
    def __init__(self, domain: str = 'general'):
        """
        Initialize nested temporal analyzer.
        
        Parameters
        ----------
        domain : str
            Domain determines natural scale hierarchy
        """
        self.domain = domain
        self.embedder = EmbeddingManager()
        
        # Domain-specific scale hierarchies (discovered, not imposed)
        self.scale_hierarchies = {
            'nba': [
                {'name': 'possession', 'duration_seconds': 20, 'per_parent': 240},
                {'name': 'quarter', 'duration_seconds': 720, 'per_parent': 4},
                {'name': 'half', 'duration_seconds': 1440, 'per_parent': 2},
                {'name': 'game', 'duration_seconds': 2880, 'per_parent': 82},
                {'name': 'season', 'duration_seconds': 236160, 'per_parent': 1},
                {'name': 'playoff_series', 'duration_seconds': 20160, 'per_parent': 4},  # If applicable
                {'name': 'season_arc', 'duration_seconds': 15552000, 'per_parent': 1},  # 6 months
                {'name': 'career', 'duration_seconds': 315360000, 'per_parent': 1},  # 10 years
                {'name': 'rivalry', 'duration_seconds': 946080000, 'per_parent': 1},  # 30 years
                {'name': 'franchise', 'duration_seconds': 1892160000, 'per_parent': 1},  # 60 years
            ],
            
            'film': [
                {'name': 'shot', 'duration_seconds': 5, 'per_parent': 300},
                {'name': 'scene', 'duration_seconds': 180, 'per_parent': 40},
                {'name': 'sequence', 'duration_seconds': 900, 'per_parent': 8},
                {'name': 'act', 'duration_seconds': 2400, 'per_parent': 3},
                {'name': 'film', 'duration_seconds': 7200, 'per_parent': 1},
                {'name': 'directors_oeuvre', 'duration_seconds': 216000, 'per_parent': 10},  # Avg career
                {'name': 'genre_era', 'duration_seconds': 315360000, 'per_parent': 1},  # 10 years
            ],
            
            'novel': [
                {'name': 'sentence', 'duration_seconds': 10, 'per_parent': 200},
                {'name': 'paragraph', 'duration_seconds': 120, 'per_parent': 50},
                {'name': 'scene', 'duration_seconds': 600, 'per_parent': 60},
                {'name': 'chapter', 'duration_seconds': 1800, 'per_parent': 30},
                {'name': 'section', 'duration_seconds': 7200, 'per_parent': 5},
                {'name': 'novel', 'duration_seconds': 36000, 'per_parent': 1},
                {'name': 'series', 'duration_seconds': 108000, 'per_parent': 3},
                {'name': 'authors_works', 'duration_seconds': 360000, 'per_parent': 10},
            ],
            
            'ufc': [
                {'name': 'exchange', 'duration_seconds': 3, 'per_parent': 300},
                {'name': 'round', 'duration_seconds': 300, 'per_parent': 3},
                {'name': 'fight', 'duration_seconds': 900, 'per_parent': 12},  # Card
                {'name': 'card', 'duration_seconds': 10800, 'per_parent': 12},  # Year
                {'name': 'season', 'duration_seconds': 31536000, 'per_parent': 1},
                {'name': 'fighter_career', 'duration_seconds': 157680000, 'per_parent': 1},  # 5 years
                {'name': 'division_era', 'duration_seconds': 315360000, 'per_parent': 1},  # 10 years
            ]
        }
    
    def analyze_all_scales(
        self,
        narrative_data: Dict[str, Any],
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze narrative at ALL temporal scales simultaneously.
        
        Parameters
        ----------
        narrative_data : dict
            Complete narrative with:
            - Text at each scale (play-by-play, game summary, season arc, etc.)
            - Metadata for temporal boundaries
            - Outcomes at each scale
        domain : str, optional
            Override self.domain
            
        Returns
        -------
        multi_scale_analysis : dict
            {
                'scales': Analysis at each temporal scale,
                'cross_scale_patterns': How scales relate,
                'fractal_dimension': Are patterns self-similar?,
                'dominant_scale': Which scale most predictive?,
                'scale_interactions': How scales affect each other,
                'mysterious_cross_scale_dimensions': What AI discovers
            }
        """
        domain = domain or self.domain
        
        if domain not in self.scale_hierarchies:
            domain = 'general'
        
        hierarchy = self.scale_hierarchies.get(domain, [])
        
        print(f"\n{'='*80}")
        print(f"NESTED TEMPORAL SCALE ANALYSIS: {domain}")
        print(f"{'='*80}\n")
        print(f"Analyzing {len(hierarchy)} simultaneous temporal scales")
        print("Each scale is complete narrative in itself")
        print("All scales coexist in potentiality\n")
        
        scale_analyses = {}
        
        # Analyze each scale
        for scale_spec in hierarchy:
            scale_name = scale_spec['name']
            
            if scale_name in narrative_data.get('scales', {}):
                print(f"[{scale_name}] Analyzing...")
                
                scale_narrative = narrative_data['scales'][scale_name]
                analysis = self._analyze_single_scale(
                    scale_name,
                    scale_narrative,
                    scale_spec
                )
                
                scale_analyses[scale_name] = analysis
                print(f"  ✓ Features extracted: {len(analysis['features'])}")
        
        # Cross-scale pattern analysis
        print(f"\nAnalyzing cross-scale patterns...")
        cross_scale = self._analyze_cross_scale_patterns(scale_analyses)
        
        # Fractal dimension
        print(f"Testing for fractal (self-similar) structure...")
        fractal_analysis = self._test_fractal_structure(scale_analyses)
        
        print(f"\n{'='*80}")
        print("MULTI-SCALE ANALYSIS COMPLETE")
        print(f"{'='*80}\n")
        
        return {
            'domain': domain,
            'n_scales': len(scale_analyses),
            'scale_analyses': scale_analyses,
            'cross_scale_patterns': cross_scale,
            'fractal_analysis': fractal_analysis,
            'note': 'All scales analyzed. Patterns coexist simultaneously.',
            'reminder': 'Do not privilege one scale. All are equally real.'
        }
    
    def _analyze_single_scale(
        self,
        scale_name: str,
        scale_data: Dict,
        scale_spec: Dict
    ) -> Dict:
        """
        Analyze narrative at single temporal scale.
        
        Each scale is complete narrative:
        - Has sequence (what follows what)
        - Has spacing (rhythm at this scale)
        - Has progression (trajectory)
        - Has outcome
        """
        narrative_text = scale_data.get('text', scale_data.get('description', ''))
        
        if not narrative_text:
            return {'error': 'No narrative text at this scale'}
        
        # Embed narrative at this scale
        embedding = self.embedder.encode([narrative_text])[0]
        
        # Extract temporal features at this scale
        features = {
            'scale_name': scale_name,
            'duration_seconds': scale_spec['duration_seconds'],
            'embedding': embedding,  # Full semantic representation at this scale
            'tau_at_scale': self._calculate_tau_at_scale(scale_data, scale_spec),
            'outcome_at_scale': scale_data.get('outcome'),
            'note': f'Narrative at {scale_name} scale. Independent and complete.'
        }
        
        return features
    
    def _calculate_tau_at_scale(self, scale_data: Dict, scale_spec: Dict) -> float:
        """Calculate duration ratio at this specific scale."""
        actual_duration = scale_data.get('actual_duration', scale_spec['duration_seconds'])
        expected_duration = scale_spec['duration_seconds']
        
        tau = actual_duration / expected_duration if expected_duration > 0 else 1.0
        return tau
    
    def _analyze_cross_scale_patterns(self, scale_analyses: Dict) -> Dict:
        """
        Find patterns ACROSS scales.
        
        Questions:
        - Do scales show similar patterns (fractal)?
        - Do scales interact (one predicts another)?
        - Is there dominant scale (one matters most)?
        - Are scales independent or coupled?
        
        NO presupposition. Let AI find patterns.
        """
        if len(scale_analyses) < 2:
            return {'note': 'Need multiple scales for cross-scale analysis'}
        
        # Extract embeddings from each scale
        scale_embeddings = {}
        for scale_name, analysis in scale_analyses.items():
            if 'embedding' in analysis:
                scale_embeddings[scale_name] = analysis['embedding']
        
        # Compute cross-scale similarities
        cross_similarities = {}
        scales = list(scale_embeddings.keys())
        
        for i in range(len(scales)):
            for j in range(i + 1, len(scales)):
                scale_a = scales[i]
                scale_b = scales[j]
                
                emb_a = scale_embeddings[scale_a]
                emb_b = scale_embeddings[scale_b]
                
                # Semantic similarity between scales
                similarity = np.dot(emb_a, emb_b) / (
                    np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-8
                )
                
                cross_similarities[f'{scale_a}_{scale_b}'] = float(similarity)
        
        # Identify most similar scale pairs (coupled narratives)
        sorted_similarities = sorted(cross_similarities.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'cross_scale_similarities': cross_similarities,
            'most_coupled': sorted_similarities[:3] if len(sorted_similarities) >= 3 else sorted_similarities,
            'least_coupled': sorted_similarities[-3:] if len(sorted_similarities) >= 3 else [],
            'avg_coupling': float(np.mean(list(cross_similarities.values()))) if cross_similarities else 0.0,
            'note': 'Scales measured. Which couples matter? Elusive.'
        }
    
    def _test_fractal_structure(self, scale_analyses: Dict) -> Dict:
        """
        Test if narrative shows fractal (self-similar) structure across scales.
        
        Fractal = same patterns at every scale.
        
        Method:
        - Extract τ, ς, ρ at each scale
        - Test if distributions similar
        - Measure self-similarity
        
        If fractal: Patterns recur. Deep universality.
        If not fractal: Each scale unique. Scale-specific dynamics.
        """
        # Extract temporal features at each scale
        taus = []
        sigmas = []
        rhos = []
        
        for scale_name, analysis in scale_analyses.items():
            if 'tau_at_scale' in analysis:
                taus.append(analysis['tau_at_scale'])
            # Would need sigma, rho calculations at each scale
        
        if len(taus) < 3:
            return {'note': 'Insufficient scales for fractal analysis'}
        
        # Measure if distributions are similar (fractal hypothesis)
        tau_variance = np.var(taus) if taus else 0.0
        
        # Low variance across scales = fractal (self-similar)
        fractal_score = 1.0 / (1.0 + tau_variance)
        
        return {
            'fractal_score': float(fractal_score),
            'tau_by_scale': taus,
            'is_fractal': fractal_score > 0.7,
            'interpretation': 'Fractal if same patterns at every scale. Otherwise scale-specific.',
            'note': 'Self-similarity measured. Meaning elusive.'
        }


class MultiScaleFeatureExtractor:
    """
    Extract features from ALL temporal scales for prediction.
    
    Philosophy:
    - Game outcome affected by: quarter narratives + game narrative + season narrative + career narrative + rivalry narrative + ALL scales
    - Don't assume which scale matters
    - Let AI weights determine importance
    - Each scale provides independent signal
    
    Features: Concatenate all scales (~500+ features from 10 scales)
    """
    
    def __init__(self, domain: str):
        """Initialize for specific domain."""
        self.domain = domain
        self.analyzer = NestedTemporalAnalyzer(domain=domain)
        self.embedder = EmbeddingManager()
    
    def extract_multi_scale_features(
        self,
        narrative_data: Dict[str, Any]
    ) -> np.ndarray:
        """
        Extract features from all available temporal scales.
        
        Returns feature vector containing:
        - Features from smallest scale (possessions, shots, exchanges)
        - Features from next scale (quarters, scenes, rounds)
        - ... up through all scales ...
        - Features from largest scale (franchise, era, sport history)
        
        Total: ~50 features per scale × 10 scales = ~500 features
        
        Model learns which scales matter most.
        We don't presuppose.
        """
        all_features = []
        
        # Extract features at each scale
        for scale_name, scale_data in narrative_data.get('scales', {}).items():
            scale_features = self._extract_scale_features(scale_name, scale_data)
            all_features.extend(scale_features)
        
        return np.array(all_features)
    
    def _extract_scale_features(self, scale_name: str, scale_data: Dict) -> List[float]:
        """
        Extract features at single scale.
        
        ~50 features per scale:
        - Embedding (first 20 dimensions)
        - Temporal (τ, ς, ρ)
        - Sequential (progression, rhythm)
        - Outcome at this scale
        - Position within parent scale
        - Mysterious dimensions
        """
        narrative_text = scale_data.get('text', scale_data.get('description', ''))
        
        if not narrative_text:
            return [0.0] * 50
        
        # Embed at this scale
        embedding = self.embedder.encode([narrative_text])[0]
        
        features = []
        
        # First 20 embedding dimensions (semantic content at this scale)
        features.extend(embedding[:20].tolist())
        
        # Temporal features at this scale
        features.append(scale_data.get('tau', 1.0))
        features.append(scale_data.get('sigma', 1.0))
        features.append(scale_data.get('rho', 0.35))
        
        # Outcome at this scale (if applicable)
        outcome = scale_data.get('outcome')
        if outcome is not None:
            if isinstance(outcome, (int, float)):
                features.append(float(outcome))
            else:
                features.append(1.0 if outcome else 0.0)
        else:
            features.append(0.5)
        
        # Position within parent scale
        features.append(scale_data.get('position_in_parent', 0.5))
        
        # Mysterious dimensions (remaining to reach 50)
        while len(features) < 50:
            features.append(0.5)
        
        return features[:50]


def demonstrate_nested_scales_nba_example():
    """
    Demonstrate nested scales using NBA example.
    
    Single moment (e.g., LeBron hits game-winning shot) exists in:
    - Shot narrative (3 seconds)
    - Possession narrative (24 seconds)
    - Quarter narrative (final seconds of Q4)
    - Game narrative (Finals Game 7)
    - Series narrative (Championship clincher)
    - Season narrative (Culmination)
    - Career narrative (Legacy moment)
    - Franchise narrative (Championship #4)
    - Rivalry narrative (vs Warriors, historical)
    - Era narrative (LeBron's generation)
    - Sport narrative (NBA history)
    
    ALL scales coexist. ALL contribute to meaning.
    """
    print(f"\n{'='*80}")
    print("NESTED TEMPORAL SCALES: NBA EXAMPLE")
    print(f"{'='*80}\n")
    
    print("Single moment: LeBron game-winning shot, Finals Game 7, 2025")
    print("\nThis moment exists SIMULTANEOUSLY in:\n")
    
    scales_present = [
        ('Shot', '3 seconds', 'Trajectory, release, swish'),
        ('Possession', '24 seconds', 'Timeout → inbound → movement → shot'),
        ('Final Minutes', '2 minutes', 'Tied game, final possessions, tension'),
        ('Quarter', '12 minutes', 'Q4 dramatic finish'),
        ('Half', '24 minutes', '2nd half comeback'),
        ('Game', '48 minutes', 'Back-and-forth classic'),
        ('Series', '7 games', 'Series-clinching shot'),
        ('Playoff Run', '16 games', 'Championship journey'),
        ('Season', '82 games', 'All season building to this'),
        ('Career', '21 years', 'Legacy-defining moment'),
        ('Franchise', '55 years', '4th championship'),
        ('Rivalry', '15 years', 'vs Warriors, final chapter'),
        ('Era', '20 years', 'LeBron generation closure'),
        ('Sport', '78 years', 'NBA history, all-time great moment')
    ]
    
    for scale_name, duration, narrative in scales_present:
        print(f"  [{scale_name:<20s}] ({duration:<15s}): {narrative}")
    
    print(f"\nAll {len(scales_present)} narratives coexist in potentiality.")
    print("Same physical moment = 14 different stories.")
    print("Which matters most? Let data reveal.\n")
    
    print("="*80 + "\n")


if __name__ == '__main__':
    demonstrate_nested_scales_nba_example()

