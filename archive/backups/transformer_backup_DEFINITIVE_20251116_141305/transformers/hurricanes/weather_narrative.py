"""
Weather Narrative Transformer for Hurricanes

Analyzes hurricane-specific narrative features beyond just the name:
- Weather terminology and severity signals
- Geographic and temporal patterns
- Historical context and retired name associations
- Media framing and communication patterns

Author: Narrative Integration System
Date: November 2025
"""

import sys
from pathlib import Path
import numpy as np
from typing import List, Dict, Any
import re

# Add narrative_optimization to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ..base_transformer import FeatureNarrativeTransformer


class WeatherNarrativeTransformer(FeatureNarrativeTransformer):
    """
    Extract hurricane-specific narrative features.
    
    Focuses on domain-specific signals that affect perception
    beyond the name itself.
    """
    
    def __init__(self, include_temporal: bool = True,
                 include_geographic: bool = True):
        """
        Initialize weather narrative transformer.
        
        Parameters
        ----------
        include_temporal : bool
            Include seasonal/temporal features
        include_geographic : bool
            Include geographic/basin features
        """
        super().__init__(
            narrative_id="weather_narrative",
            description="Extracts hurricane-specific narrative features including "
                       "geographic, temporal, and historical context signals"
        )
        self.include_temporal = include_temporal
        self.include_geographic = include_geographic
        
        # Initialize severity terminology
        self.severity_terms = self._init_severity_vocabulary()
    
    def fit(self, X, y=None):
        """
        Learn narrative patterns from hurricane data.
        
        Parameters
        ----------
        X : array-like or list of dict
            Hurricane feature data including metadata
        y : array-like, optional
            Target outcomes
        
        Returns
        -------
        self
        """
        self._validate_input(X)
        
        # Extract corpus statistics
        years = []
        months = []
        categories = []
        basins = []
        retired_count = 0
        
        for hurricane in X:
            if isinstance(hurricane, dict):
                years.append(hurricane.get('year', 2000))
                months.append(hurricane.get('month', 8))
                categories.append(hurricane.get('category', 3))
                basins.append(hurricane.get('basin', 'Atlantic'))
                if hurricane.get('retired', False):
                    retired_count += 1
        
        self.metadata['corpus_stats'] = {
            'n_hurricanes': len(X),
            'year_range': (min(years) if years else 1950, max(years) if years else 2024),
            'category_distribution': {
                f'cat_{i}': categories.count(i) for i in range(1, 6)
            },
            'basin_distribution': {
                basin: basins.count(basin) for basin in set(basins)
            },
            'retired_pct': retired_count / len(X) if X else 0
        }
        
        # Store feature metadata
        self.metadata['n_features'] = self._calculate_n_features()
        self.metadata['feature_names'] = self._generate_feature_names()
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Transform hurricane data to narrative features.
        
        Parameters
        ----------
        X : array-like or list of dict
            Hurricane data
        
        Returns
        -------
        numpy.ndarray
            Feature matrix
        """
        self._validate_fitted()
        
        features = []
        for hurricane in X:
            feature_vec = self._extract_narrative_features(hurricane)
            features.append(feature_vec)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_narrative_features(self, hurricane: Dict) -> List[float]:
        """Extract narrative features from single hurricane."""
        features = []
        
        # 1. Temporal features (if enabled)
        if self.include_temporal:
            year = hurricane.get('year', 2000)
            month = hurricane.get('month', 8)
            
            # Year normalized (1950-2024 → 0-1)
            year_norm = (year - 1950) / 74
            features.append(year_norm)
            
            # Era indicators
            features.append(1.0 if year < 1970 else 0.0)  # Early era
            features.append(1.0 if 1970 <= year < 1990 else 0.0)  # Mid era
            features.append(1.0 if 1990 <= year < 2010 else 0.0)  # Modern era
            features.append(1.0 if year >= 2010 else 0.0)  # Recent era
            
            # Month/season features
            features.append(month / 12.0)  # Normalized month
            features.append(1.0 if month <= 7 else 0.0)  # Early season
            features.append(1.0 if 8 <= month <= 9 else 0.0)  # Peak season
            features.append(1.0 if month >= 10 else 0.0)  # Late season
        
        # 2. Geographic features (if enabled)
        if self.include_geographic:
            basin = hurricane.get('basin', 'Atlantic')
            location = hurricane.get('landfall_location', '')
            
            # Basin indicators
            features.append(1.0 if basin == 'Atlantic' else 0.0)
            features.append(1.0 if basin == 'Pacific' else 0.0)
            features.append(1.0 if basin == 'Gulf' else 0.0)
            
            # Region indicators (from landfall location)
            features.append(1.0 if 'Florida' in location else 0.0)
            features.append(1.0 if any(state in location for state in ['Louisiana', 'Texas', 'Mississippi', 'Alabama']) else 0.0)
            features.append(1.0 if any(state in location for state in ['Carolina', 'Virginia', 'Georgia']) else 0.0)
            features.append(1.0 if any(state in location for state in ['New York', 'New Jersey', 'Massachusetts']) else 0.0)
        
        # 3. Historical context features
        retired = hurricane.get('retired', False)
        features.append(1.0 if retired else 0.0)
        
        # Category-based narrative framing
        category = hurricane.get('category', 3)
        features.append(float(category) / 5.0)  # Normalized category
        
        # Category threshold indicators
        features.append(1.0 if category >= 3 else 0.0)  # Major hurricane
        features.append(1.0 if category >= 4 else 0.0)  # Extreme hurricane
        features.append(1.0 if category >= 5 else 0.0)  # Catastrophic
        
        # 4. Narrative intensity signals
        # These represent how the hurricane might be communicated
        features.append(self._calculate_media_intensity(hurricane))
        features.append(self._calculate_urgency_score(hurricane))
        
        return features
    
    def _calculate_media_intensity(self, hurricane: Dict) -> float:
        """
        Calculate likely media coverage intensity.
        
        Based on category, retirement status, and era.
        """
        category = hurricane.get('category', 3)
        retired = hurricane.get('retired', False)
        year = hurricane.get('year', 2000)
        
        # Base intensity from category
        intensity = category / 5.0
        
        # Retired storms get more coverage
        if retired:
            intensity += 0.3
        
        # Recent storms get more media attention
        if year >= 2000:
            intensity += 0.2
        
        return min(1.0, intensity)
    
    def _calculate_urgency_score(self, hurricane: Dict) -> float:
        """
        Calculate urgency score based on hurricane characteristics.
        
        Represents how urgently officials would communicate threat.
        """
        category = hurricane.get('category', 3)
        month = hurricane.get('month', 8)
        
        # Base urgency from category
        urgency = (category - 1) / 4.0  # 0 to 1 for cat 1-5
        
        # Peak season = more preparedness, slightly less panic
        if 8 <= month <= 9:
            urgency *= 0.95
        
        # Early/late season = less prepared, more urgent
        if month <= 7 or month >= 11:
            urgency *= 1.1
        
        return min(1.0, urgency)
    
    def _calculate_n_features(self) -> int:
        """Calculate total number of output features."""
        n = 0
        
        if self.include_temporal:
            n += 9  # year_norm + 4 era indicators + month_norm + 3 season indicators
        
        if self.include_geographic:
            n += 7  # 3 basin + 4 region indicators
        
        n += 7  # Historical context + category features
        n += 2  # Media intensity + urgency
        
        return n
    
    def _generate_feature_names(self) -> List[str]:
        """Generate feature names."""
        names = []
        
        if self.include_temporal:
            names.extend([
                'year_normalized',
                'era_early',
                'era_mid',
                'era_modern',
                'era_recent',
                'month_normalized',
                'season_early',
                'season_peak',
                'season_late'
            ])
        
        if self.include_geographic:
            names.extend([
                'basin_atlantic',
                'basin_pacific',
                'basin_gulf',
                'region_florida',
                'region_gulf_coast',
                'region_east_coast',
                'region_northeast'
            ])
        
        names.extend([
            'is_retired',
            'category_normalized',
            'is_major',
            'is_extreme',
            'is_catastrophic',
            'media_intensity',
            'urgency_score'
        ])
        
        return names
    
    def _init_severity_vocabulary(self) -> Dict[str, List[str]]:
        """Initialize severity-related terminology."""
        return {
            'extreme': ['catastrophic', 'devastating', 'extreme', 'historic', 
                       'unprecedented', 'deadly', 'killer'],
            'high': ['major', 'dangerous', 'severe', 'powerful', 'intense',
                    'destructive', 'violent'],
            'moderate': ['significant', 'considerable', 'notable', 'substantial'],
            'low': ['minimal', 'minor', 'weak', 'small']
        }
    
    def _generate_interpretation(self) -> str:
        """Generate interpretation of weather narrative patterns."""
        stats = self.metadata['corpus_stats']
        
        interpretation = f"Weather Narrative Analysis ({stats['n_hurricanes']} storms)\n\n"
        
        interpretation += f"Time Range: {stats['year_range'][0]}-{stats['year_range'][1]}\n\n"
        
        interpretation += "Category Distribution:\n"
        for cat, count in stats['category_distribution'].items():
            pct = count / stats['n_hurricanes'] * 100
            interpretation += f"  {cat}: {count} ({pct:.1f}%)\n"
        
        interpretation += f"\nRetired Names: {stats['retired_pct']*100:.1f}%\n"
        
        interpretation += "\nBasin Distribution:\n"
        for basin, count in stats['basin_distribution'].items():
            pct = count / stats['n_hurricanes'] * 100
            interpretation += f"  {basin}: {count} ({pct:.1f}%)\n"
        
        interpretation += "\nNarrative Context:\n"
        interpretation += "  Weather narrative features capture temporal, geographic, and\n"
        interpretation += "  historical context that influences hurricane perception beyond\n"
        interpretation += "  just the name. These features interact with nominative features\n"
        interpretation += "  to predict evacuation behavior and response effectiveness.\n"
        
        return interpretation
    
    def _validate_input(self, X):
        """Validate input format."""
        if X is None or len(X) == 0:
            raise ValueError("Input X cannot be None or empty")
        
        # Check if dict-like
        if len(X) > 0 and not isinstance(X[0], dict):
            raise ValueError("Input must be list of dictionaries with hurricane metadata")
        
        return True

