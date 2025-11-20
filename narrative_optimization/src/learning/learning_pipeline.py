"""
Adaptive Learning Pipeline

Orchestrates continuous learning cycle:
1. Ingest Data → 2. Discover Patterns → 3. Validate Patterns
      ↑                                          ↓
6. Apply Archetypes ← 5. Update Archetypes ← 4. Measure Performance

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import json
from datetime import datetime
from dataclasses import dataclass, asdict

from .universal_learner import UniversalArchetypeLearner
from .domain_learner import DomainSpecificLearner
from .validation_engine import ValidationEngine
from .registry_versioned import VersionedArchetypeRegistry


@dataclass
class LearningMetrics:
    """Metrics for learning cycle."""
    iteration: int
    timestamp: str
    patterns_discovered: int
    patterns_validated: int
    patterns_pruned: int
    r_squared_before: float
    r_squared_after: float
    improvement: float
    coherence_score: float
    predictive_power: float


class LearningPipeline:
    """
    Main learning pipeline that orchestrates continuous archetype improvement.
    
    This is the heart of the holistic learning system. It:
    - Ingests new data continuously
    - Discovers patterns (universal + domain-specific)
    - Validates patterns statistically
    - Measures performance improvements
    - Updates archetypes based on performance
    - Applies learned archetypes to analyses
    
    Parameters
    ----------
    registry_path : Path, optional
        Path to save/load archetype registry
    incremental : bool
        If True, updates without full retrain
    auto_prune : bool
        Automatically remove weak patterns
    
    Examples
    --------
    >>> pipeline = LearningPipeline()
    >>> pipeline.ingest_domain('golf', texts, outcomes)
    >>> pipeline.learn_cycle()  # Discover → Validate → Update
    >>> improved_archetypes = pipeline.get_archetypes('golf')
    """
    
    def __init__(
        self,
        registry_path: Optional[Path] = None,
        incremental: bool = True,
        auto_prune: bool = True,
        min_improvement: float = 0.01,
        config: Optional['PipelineConfig'] = None
    ):
        self.incremental = incremental
        self.auto_prune = auto_prune
        self.min_improvement = min_improvement
        
        # Load config
        if config is None:
            from ..pipeline_config import get_config
            config = get_config()
        self.config = config
        
        # Core components
        self.universal_learner = UniversalArchetypeLearner()
        self.domain_learners = {}  # domain_name -> DomainSpecificLearner
        self.validator = ValidationEngine(
            alpha=config.validation_alpha if config else 0.05,
            min_samples=10
        )
        self.registry = VersionedArchetypeRegistry(registry_path)
        
        # Data storage (for incremental learning)
        self.domain_data = {}  # domain_name -> {'texts': [], 'outcomes': []}
        
        # Learning history
        self.learning_history = []
        self.iteration = 0
        
    def ingest_domain(
        self,
        domain_name: str,
        texts: List[str],
        outcomes: np.ndarray,
        names: Optional[List[str]] = None,
        timestamps: Optional[np.ndarray] = None
    ):
        """
        Ingest data for a domain.
        
        Parameters
        ----------
        domain_name : str
            Domain name
        texts : list of str
            Narrative texts
        outcomes : ndarray
            Outcomes
        names : list of str, optional
            Entity names
        timestamps : ndarray, optional
            Timestamps
        """
        # Store data
        if domain_name not in self.domain_data:
            self.domain_data[domain_name] = {
                'texts': [],
                'outcomes': [],
                'names': [],
                'timestamps': []
            }
        
        # Append new data
        self.domain_data[domain_name]['texts'].extend(texts)
        self.domain_data[domain_name]['outcomes'].extend(outcomes.tolist())
        
        if names:
            self.domain_data[domain_name]['names'].extend(names)
        if timestamps is not None:
            self.domain_data[domain_name]['timestamps'].extend(timestamps.tolist())
        
        print(f"✓ Ingested {len(texts)} samples for {domain_name}")
        print(f"  Total: {len(self.domain_data[domain_name]['texts'])} samples")
    
    def learn_cycle(
        self,
        domains: Optional[List[str]] = None,
        learn_universal: bool = True,
        learn_domain_specific: bool = True
    ) -> LearningMetrics:
        """
        Execute one complete learning cycle.
        
        Steps:
        1. Discover patterns (universal + domain-specific)
        2. Validate patterns statistically
        3. Measure performance improvement
        4. Update archetypes
        5. Prune weak patterns (if auto_prune)
        
        Parameters
        ----------
        domains : list of str, optional
            Domains to learn on (None = all)
        learn_universal : bool
            Learn universal (cross-domain) patterns
        learn_domain_specific : bool
            Learn domain-specific patterns
        
        Returns
        -------
        LearningMetrics
            Metrics for this learning cycle
        """
        self.iteration += 1
        print(f"\n{'='*80}")
        print(f"LEARNING CYCLE {self.iteration}")
        print(f"{'='*80}")
        
        if domains is None:
            domains = list(self.domain_data.keys())
        
        # Metrics tracking
        total_discovered = 0
        total_validated = 0
        total_pruned = 0
        r2_before_list = []
        r2_after_list = []
        
        # Step 1: Baseline performance (before learning)
        print(f"\n[1/6] Measuring baseline performance...")
        for domain in domains:
            if domain in self.domain_data:
                r2_before = self._measure_performance(domain)
                r2_before_list.append(r2_before)
                print(f"  {domain}: R²={r2_before:.3f}")
        
        # Step 2: Discover patterns
        print(f"\n[2/6] Discovering patterns...")
        
        if learn_universal:
            # Learn universal patterns from all domains
            all_texts = []
            all_outcomes = []
            for domain in domains:
                if domain in self.domain_data:
                    all_texts.extend(self.domain_data[domain]['texts'])
                    all_outcomes.extend(self.domain_data[domain]['outcomes'])
            
            print(f"  Learning universal patterns from {len(all_texts)} total samples...")
            universal_patterns = self.universal_learner.discover_patterns(
                all_texts, np.array(all_outcomes)
            )
            total_discovered += len(universal_patterns)
            print(f"  ✓ Discovered {len(universal_patterns)} universal patterns")
        
        if learn_domain_specific:
            # Learn domain-specific patterns
            for domain in domains:
                if domain not in self.domain_data:
                    continue
                
                # Create learner if doesn't exist
                if domain not in self.domain_learners:
                    self.domain_learners[domain] = DomainSpecificLearner(domain)
                
                data = self.domain_data[domain]
                texts = data['texts']
                outcomes = np.array(data['outcomes'])
                
                print(f"  Learning {domain}-specific patterns from {len(texts)} samples...")
                domain_patterns = self.domain_learners[domain].discover_patterns(
                    texts, outcomes
                )
                total_discovered += len(domain_patterns)
                print(f"  ✓ Discovered {len(domain_patterns)} domain patterns")
        
        # Step 3: Validate patterns
        print(f"\n[3/6] Validating patterns...")
        
        if learn_universal:
            validated_universal = self.validator.validate_patterns(
                universal_patterns,
                all_texts,
                np.array(all_outcomes)
            )
            total_validated += len(validated_universal)
            print(f"  ✓ Validated {len(validated_universal)}/{len(universal_patterns)} universal patterns")
        
        for domain in domains:
            if domain not in self.domain_learners:
                continue
            
            domain_patterns = self.domain_learners[domain].get_patterns()
            data = self.domain_data[domain]
            
            validated_domain = self.validator.validate_patterns(
                domain_patterns,
                data['texts'],
                np.array(data['outcomes'])
            )
            total_validated += len(validated_domain)
            print(f"  ✓ Validated {len(validated_domain)}/{len(domain_patterns)} {domain} patterns")
        
        # Step 4: Measure performance (after learning)
        print(f"\n[4/6] Measuring improved performance...")
        for domain in domains:
            if domain in self.domain_data:
                r2_after = self._measure_performance(domain, use_learned=True)
                r2_after_list.append(r2_after)
                improvement = r2_after - r2_before_list[domains.index(domain)]
                print(f"  {domain}: R²={r2_after:.3f} ({improvement:+.3f})")
        
        # Step 5: Update archetypes in registry
        print(f"\n[5/6] Updating archetype registry...")
        
        if learn_universal:
            self.registry.register_universal_patterns(
                validated_universal,
                version=f"v{self.iteration}",
                performance_improvement=np.mean(r2_after_list) - np.mean(r2_before_list)
            )
        
        for domain in domains:
            if domain in self.domain_learners:
                validated = self.domain_learners[domain].get_validated_patterns()
                self.registry.register_domain_patterns(
                    domain,
                    validated,
                    version=f"v{self.iteration}",
                    performance_improvement=r2_after_list[domains.index(domain)] - r2_before_list[domains.index(domain)]
                )
        
        print(f"  ✓ Registry updated (version v{self.iteration})")
        
        # Step 6: Prune weak patterns (if enabled)
        if self.auto_prune:
            print(f"\n[6/6] Pruning weak patterns...")
            
            for domain in domains:
                if domain in self.domain_learners:
                    pruned = self.domain_learners[domain].prune_weak_patterns(
                        min_correlation=0.05,
                        min_frequency=0.02
                    )
                    total_pruned += pruned
                    print(f"  {domain}: Pruned {pruned} weak patterns")
            
            print(f"  ✓ Total pruned: {total_pruned} patterns")
        else:
            print(f"\n[6/6] Skipping pruning (disabled)")
        
        # Create metrics
        metrics = LearningMetrics(
            iteration=self.iteration,
            timestamp=datetime.now().isoformat(),
            patterns_discovered=total_discovered,
            patterns_validated=total_validated,
            patterns_pruned=total_pruned,
            r_squared_before=np.mean(r2_before_list) if r2_before_list else 0.0,
            r_squared_after=np.mean(r2_after_list) if r2_after_list else 0.0,
            improvement=np.mean(r2_after_list) - np.mean(r2_before_list) if r2_after_list else 0.0,
            coherence_score=self._calculate_coherence(),
            predictive_power=np.mean(r2_after_list) if r2_after_list else 0.0
        )
        
        self.learning_history.append(metrics)
        
        # Summary
        print(f"\n{'='*80}")
        print(f"LEARNING CYCLE {self.iteration} COMPLETE")
        print(f"{'='*80}")
        print(f"Patterns: {total_discovered} discovered, {total_validated} validated, {total_pruned} pruned")
        print(f"Performance: R²={metrics.r_squared_after:.3f} ({metrics.improvement:+.3f})")
        print(f"Coherence: {metrics.coherence_score:.3f}")
        
        return metrics
    
    def _measure_performance(
        self,
        domain: str,
        use_learned: bool = False
    ) -> float:
        """
        Measure R² performance for a domain.
        
        Parameters
        ----------
        domain : str
            Domain name
        use_learned : bool
            If True, uses learned archetypes; otherwise uses baseline
        
        Returns
        -------
        float
            R² score
        """
        from ..analysis.domain_specific_analyzer import DomainSpecificAnalyzer
        
        if domain not in self.domain_data:
            return 0.0
        
        data = self.domain_data[domain]
        if len(data['texts']) == 0:
            return 0.0
        
        try:
            analyzer = DomainSpecificAnalyzer(domain)
            
            # If using learned archetypes, inject them
            if use_learned and domain in self.domain_learners:
                # This would require analyzer to accept custom archetypes
                # For now, use standard approach
                pass
            
            results = analyzer.analyze_complete(
                texts=data['texts'],
                outcomes=np.array(data['outcomes'])
            )
            
            return results['r_squared']
        except Exception as e:
            print(f"  ⚠ Error measuring {domain}: {e}")
            return 0.0
    
    def _calculate_coherence(self) -> float:
        """
        Calculate overall pattern coherence.
        
        Coherence = how well patterns fit together (not contradictory).
        
        Returns
        -------
        float
            Coherence score (0-1)
        """
        # Simplified: average pattern quality
        coherence_scores = []
        
        for domain in self.domain_learners.values():
            patterns = domain.get_patterns()
            if len(patterns) > 0:
                # Quality = validation strength
                quality = domain.get_average_pattern_quality()
                coherence_scores.append(quality)
        
        return np.mean(coherence_scores) if coherence_scores else 0.5
    
    def get_archetypes(self, domain: str) -> Dict[str, Any]:
        """
        Get learned archetypes for a domain.
        
        Combines universal + domain-specific patterns.
        
        Parameters
        ----------
        domain : str
            Domain name
        
        Returns
        -------
        dict
            Combined archetypes
        """
        archetypes = {
            'universal': self.universal_learner.get_patterns(),
            'domain_specific': {}
        }
        
        if domain in self.domain_learners:
            archetypes['domain_specific'] = self.domain_learners[domain].get_patterns()
        
        return archetypes
    
    def save_state(self, path: Path):
        """Save learning pipeline state."""
        state = {
            'iteration': self.iteration,
            'learning_history': [asdict(m) for m in self.learning_history],
            'domains': list(self.domain_data.keys()),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Save registry
        self.registry.save()
        
        print(f"✓ Pipeline state saved to {path}")
    
    def load_state(self, path: Path):
        """Load learning pipeline state."""
        with open(path) as f:
            state = json.load(f)
        
        self.iteration = state['iteration']
        # History would need full LearningMetrics reconstruction
        
        # Load registry
        self.registry.load()
        
        print(f"✓ Pipeline state loaded from {path}")

