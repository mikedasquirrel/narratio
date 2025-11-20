"""
Versioned Archetype Registry

Stores learned archetypes with version control, A/B testing, and performance tracking.

Author: Narrative Integration System
Date: November 2025
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class ArchetypeVersion:
    """Single version of an archetype."""
    version: str
    patterns: Dict[str, Any]
    performance: float
    created_at: str
    metadata: Dict[str, Any]


class VersionedArchetypeRegistry:
    """
    Registry with version control for learned archetypes.
    
    Features:
    - Save/load archetype versions
    - A/B test different versions
    - Track performance over time
    - Rollback to previous versions
    - Compare versions
    
    Parameters
    ----------
    registry_path : Path, optional
        Path to save registry
    """
    
    def __init__(self, registry_path: Optional[Path] = None):
        if registry_path is None:
            registry_path = Path.home() / '.narrative_optimization' / 'archetype_registry.json'
        
        self.registry_path = registry_path
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Structure: domain -> pattern_name -> [versions]
        self.universal_patterns = {}  # pattern_name -> [versions]
        self.domain_patterns = {}  # domain -> pattern_name -> [versions]
        
        # Performance tracking
        self.performance_history = defaultdict(list)  # pattern_name -> [performance_scores]
        
        # A/B testing
        self.ab_tests = {}  # test_name -> test_config
        
        # Load existing if available
        if self.registry_path.exists():
            self.load()
    
    def register_universal_patterns(
        self,
        patterns: Dict[str, Dict],
        version: str,
        performance_improvement: float,
        metadata: Optional[Dict] = None
    ):
        """
        Register universal patterns with version.
        
        Parameters
        ----------
        patterns : dict
            Pattern data
        version : str
            Version identifier
        performance_improvement : float
            Performance gain from this version
        metadata : dict, optional
            Additional metadata
        """
        for pattern_name, pattern_data in patterns.items():
            if pattern_name not in self.universal_patterns:
                self.universal_patterns[pattern_name] = []
            
            version_obj = ArchetypeVersion(
                version=version,
                patterns=pattern_data,
                performance=performance_improvement,
                created_at=datetime.now().isoformat(),
                metadata=metadata or {}
            )
            
            self.universal_patterns[pattern_name].append(version_obj)
            self.performance_history[pattern_name].append(performance_improvement)
    
    def register_domain_patterns(
        self,
        domain: str,
        patterns: Dict[str, Dict],
        version: str,
        performance_improvement: float,
        metadata: Optional[Dict] = None
    ):
        """
        Register domain-specific patterns with version.
        
        Parameters
        ----------
        domain : str
            Domain name
        patterns : dict
            Pattern data
        version : str
            Version identifier
        performance_improvement : float
            Performance gain
        metadata : dict, optional
            Additional metadata
        """
        if domain not in self.domain_patterns:
            self.domain_patterns[domain] = {}
        
        for pattern_name, pattern_data in patterns.items():
            if pattern_name not in self.domain_patterns[domain]:
                self.domain_patterns[domain][pattern_name] = []
            
            version_obj = ArchetypeVersion(
                version=version,
                patterns=pattern_data,
                performance=performance_improvement,
                created_at=datetime.now().isoformat(),
                metadata=metadata or {}
            )
            
            self.domain_patterns[domain][pattern_name].append(version_obj)
            
            full_name = f"{domain}::{pattern_name}"
            self.performance_history[full_name].append(performance_improvement)
    
    def get_latest_version(
        self,
        pattern_name: str,
        domain: Optional[str] = None
    ) -> Optional[ArchetypeVersion]:
        """Get latest version of a pattern."""
        if domain:
            # Domain-specific
            if domain in self.domain_patterns and pattern_name in self.domain_patterns[domain]:
                versions = self.domain_patterns[domain][pattern_name]
                return versions[-1] if versions else None
        else:
            # Universal
            if pattern_name in self.universal_patterns:
                versions = self.universal_patterns[pattern_name]
                return versions[-1] if versions else None
        
        return None
    
    def get_best_version(
        self,
        pattern_name: str,
        domain: Optional[str] = None
    ) -> Optional[ArchetypeVersion]:
        """Get best-performing version of a pattern."""
        if domain:
            if domain in self.domain_patterns and pattern_name in self.domain_patterns[domain]:
                versions = self.domain_patterns[domain][pattern_name]
            else:
                return None
        else:
            if pattern_name in self.universal_patterns:
                versions = self.universal_patterns[pattern_name]
            else:
                return None
        
        if not versions:
            return None
        
        # Return version with highest performance
        best = max(versions, key=lambda v: v.performance)
        return best
    
    def rollback_to_version(
        self,
        pattern_name: str,
        version: str,
        domain: Optional[str] = None
    ) -> bool:
        """
        Rollback to a specific version.
        
        Parameters
        ----------
        pattern_name : str
            Pattern name
        version : str
            Version to rollback to
        domain : str, optional
            Domain (if domain-specific)
        
        Returns
        -------
        bool
            Success
        """
        if domain:
            if domain not in self.domain_patterns or pattern_name not in self.domain_patterns[domain]:
                return False
            versions = self.domain_patterns[domain][pattern_name]
        else:
            if pattern_name not in self.universal_patterns:
                return False
            versions = self.universal_patterns[pattern_name]
        
        # Find target version
        target = None
        for v in versions:
            if v.version == version:
                target = v
                break
        
        if target is None:
            return False
        
        # Create new version that's a copy of target
        rollback_version = ArchetypeVersion(
            version=f"rollback_to_{version}",
            patterns=target.patterns,
            performance=target.performance,
            created_at=datetime.now().isoformat(),
            metadata={'rollback_from': versions[-1].version if versions else None}
        )
        
        # Append rollback version
        versions.append(rollback_version)
        
        return True
    
    def compare_versions(
        self,
        pattern_name: str,
        version1: str,
        version2: str,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare two versions.
        
        Returns
        -------
        dict
            Comparison results
        """
        if domain:
            if domain not in self.domain_patterns or pattern_name not in self.domain_patterns[domain]:
                return {}
            versions = self.domain_patterns[domain][pattern_name]
        else:
            if pattern_name not in self.universal_patterns:
                return {}
            versions = self.universal_patterns[pattern_name]
        
        v1 = next((v for v in versions if v.version == version1), None)
        v2 = next((v for v in versions if v.version == version2), None)
        
        if not v1 or not v2:
            return {}
        
        return {
            'version1': {
                'version': v1.version,
                'performance': v1.performance,
                'created_at': v1.created_at,
                'num_patterns': len(v1.patterns.get('keywords', v1.patterns.get('patterns', [])))
            },
            'version2': {
                'version': v2.version,
                'performance': v2.performance,
                'created_at': v2.created_at,
                'num_patterns': len(v2.patterns.get('keywords', v2.patterns.get('patterns', [])))
            },
            'performance_diff': v2.performance - v1.performance,
            'better': 'version2' if v2.performance > v1.performance else 'version1'
        }
    
    def create_ab_test(
        self,
        test_name: str,
        pattern_name: str,
        version_a: str,
        version_b: str,
        domain: Optional[str] = None
    ):
        """
        Create an A/B test between two versions.
        
        Parameters
        ----------
        test_name : str
            Test identifier
        pattern_name : str
            Pattern to test
        version_a : str
            First version
        version_b : str
            Second version
        domain : str, optional
            Domain (if domain-specific)
        """
        self.ab_tests[test_name] = {
            'pattern_name': pattern_name,
            'domain': domain,
            'version_a': version_a,
            'version_b': version_b,
            'created_at': datetime.now().isoformat(),
            'results': []
        }
    
    def record_ab_result(
        self,
        test_name: str,
        version_used: str,
        performance: float
    ):
        """Record result for an A/B test."""
        if test_name not in self.ab_tests:
            return
        
        self.ab_tests[test_name]['results'].append({
            'version': version_used,
            'performance': performance,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_ab_winner(self, test_name: str) -> Optional[str]:
        """Get winning version from A/B test."""
        if test_name not in self.ab_tests:
            return None
        
        test = self.ab_tests[test_name]
        results = test['results']
        
        if len(results) < 10:  # Need minimum samples
            return None
        
        # Calculate average performance for each version
        a_results = [r['performance'] for r in results if r['version'] == test['version_a']]
        b_results = [r['performance'] for r in results if r['version'] == test['version_b']]
        
        if not a_results or not b_results:
            return None
        
        avg_a = np.mean(a_results)
        avg_b = np.mean(b_results)
        
        return test['version_a'] if avg_a > avg_b else test['version_b']
    
    def get_performance_trend(self, pattern_name: str) -> List[float]:
        """Get performance trend for a pattern."""
        return self.performance_history.get(pattern_name, [])
    
    def save(self):
        """Save registry to disk."""
        data = {
            'universal_patterns': {
                name: [asdict(v) for v in versions]
                for name, versions in self.universal_patterns.items()
            },
            'domain_patterns': {
                domain: {
                    name: [asdict(v) for v in versions]
                    for name, versions in patterns.items()
                }
                for domain, patterns in self.domain_patterns.items()
            },
            'performance_history': dict(self.performance_history),
            'ab_tests': self.ab_tests,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(self.registry_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Registry saved to {self.registry_path}")
    
    def load(self):
        """Load registry from disk."""
        if not self.registry_path.exists():
            return
        
        with open(self.registry_path) as f:
            data = json.load(f)
        
        # Load universal patterns
        self.universal_patterns = {
            name: [ArchetypeVersion(**v) for v in versions]
            for name, versions in data.get('universal_patterns', {}).items()
        }
        
        # Load domain patterns
        self.domain_patterns = {
            domain: {
                name: [ArchetypeVersion(**v) for v in versions]
                for name, versions in patterns.items()
            }
            for domain, patterns in data.get('domain_patterns', {}).items()
        }
        
        # Load performance history
        self.performance_history = defaultdict(list, data.get('performance_history', {}))
        
        # Load A/B tests
        self.ab_tests = data.get('ab_tests', {})
        
        print(f"✓ Registry loaded from {self.registry_path}")

