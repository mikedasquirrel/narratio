"""
Hurricane Severity Calculator

Normalizes and calculates severity metrics for hurricanes, enabling
separation of actual vs. perceived threat in analysis.

Key metrics:
- Saffir-Simpson category normalization
- Wind speed intensity index
- Pressure anomaly
- Duration-weighted impact
- Composite severity score
"""

from typing import Dict, Any, Optional
import math


class SeverityCalculator:
    """
    Calculates normalized severity metrics for hurricanes.
    
    Enables fair comparison across different storm characteristics
    and historical periods.
    """
    
    def __init__(self):
        """Initialize severity calculator with reference values."""
        # Saffir-Simpson scale thresholds (mph)
        self.category_thresholds = {
            1: (74, 95),
            2: (96, 110),
            3: (111, 129),
            4: (130, 156),
            5: (157, 200)
        }
        
        # Historical reference values for normalization
        self.reference_pressure = 1013  # mb (sea level standard)
        self.min_pressure_record = 882  # mb (Wilma 2005)
        self.max_wind_record = 190  # mph (sustained)
        
        # Weighting factors for composite score
        self.weights = {
            'wind': 0.40,
            'pressure': 0.30,
            'duration': 0.20,
            'size': 0.10
        }
    
    def normalize_severity(self, severity: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize severity metrics to 0-1 scales.
        
        Parameters
        ----------
        severity : dict
            Raw severity metrics
            
        Returns
        -------
        dict
            Normalized severity metrics
        """
        category = severity.get('category', 1)
        wind_speed = severity.get('max_wind_speed_mph', 74)
        pressure = severity.get('min_pressure_mb', 1000)
        duration = severity.get('duration_hours', 24)
        
        return {
            'category_normalized': self._normalize_category(category),
            'wind_normalized': self._normalize_wind(wind_speed),
            'pressure_normalized': self._normalize_pressure(pressure),
            'duration_normalized': self._normalize_duration(duration),
            'composite_severity': self._calculate_composite(
                wind_speed, pressure, duration
            )
        }
    
    def _normalize_category(self, category: int) -> float:
        """Normalize category to 0-1 scale."""
        return (category - 1) / 4.0  # Cat 1-5 → 0.0-1.0
    
    def _normalize_wind(self, wind_speed_mph: float) -> float:
        """Normalize wind speed to 0-1 scale."""
        # Use category 1 threshold as minimum
        min_wind = 74
        normalized = (wind_speed_mph - min_wind) / (self.max_wind_record - min_wind)
        return max(0.0, min(1.0, normalized))
    
    def _normalize_pressure(self, pressure_mb: float) -> float:
        """
        Normalize pressure to 0-1 scale.
        
        Lower pressure = higher intensity, so we invert the scale.
        """
        # Invert so lower pressure = higher score
        intensity = (self.reference_pressure - pressure_mb) / \
                   (self.reference_pressure - self.min_pressure_record)
        return max(0.0, min(1.0, intensity))
    
    def _normalize_duration(self, duration_hours: float) -> float:
        """Normalize duration to 0-1 scale."""
        # Use log scale since duration has high variance
        # Typical range: 6-200 hours
        if duration_hours <= 0:
            return 0.0
        
        log_duration = math.log(duration_hours + 1)
        log_max = math.log(200)  # ~200 hours as reference max
        
        normalized = log_duration / log_max
        return max(0.0, min(1.0, normalized))
    
    def _calculate_composite(self, wind_speed: float, pressure: float,
                            duration: float) -> float:
        """
        Calculate weighted composite severity score.
        
        Parameters
        ----------
        wind_speed : float
            Maximum sustained wind speed (mph)
        pressure : float
            Minimum central pressure (mb)
        duration : float
            Duration as major hurricane (hours)
        
        Returns
        -------
        float
            Composite severity (0-1)
        """
        wind_norm = self._normalize_wind(wind_speed)
        pressure_norm = self._normalize_pressure(pressure)
        duration_norm = self._normalize_duration(duration)
        
        # Weighted average
        composite = (
            self.weights['wind'] * wind_norm +
            self.weights['pressure'] * pressure_norm +
            self.weights['duration'] * duration_norm
        )
        
        return composite
    
    def calculate_intensity_index(self, severity: Dict[str, float]) -> float:
        """
        Calculate a single intensity index combining wind and pressure.
        
        This is useful for quick severity assessment.
        
        Parameters
        ----------
        severity : dict
            Raw severity metrics
        
        Returns
        -------
        float
            Intensity index (0-100)
        """
        wind = severity.get('max_wind_speed_mph', 74)
        pressure = severity.get('min_pressure_mb', 1000)
        
        # Simple formula combining both metrics
        wind_component = (wind - 74) / 126 * 50  # 0-50 from wind
        pressure_component = (1013 - pressure) / 131 * 50  # 0-50 from pressure
        
        index = wind_component + pressure_component
        return max(0.0, min(100.0, index))
    
    def classify_storm_strength(self, severity: Dict[str, float]) -> str:
        """
        Classify storm into descriptive strength categories.
        
        Parameters
        ----------
        severity : dict
            Raw severity metrics
        
        Returns
        -------
        str
            Strength classification
        """
        category = severity.get('category', 1)
        
        classifications = {
            1: 'Minimal',
            2: 'Moderate',
            3: 'Extensive',
            4: 'Extreme',
            5: 'Catastrophic'
        }
        
        return classifications.get(category, 'Unknown')
    
    def calculate_damage_potential(self, severity: Dict[str, float]) -> Dict[str, Any]:
        """
        Estimate damage potential based on severity metrics.
        
        Parameters
        ----------
        severity : dict
            Raw severity metrics
        
        Returns
        -------
        dict
            Damage potential estimates
        """
        category = severity.get('category', 1)
        wind_speed = severity.get('max_wind_speed_mph', 74)
        duration = severity.get('duration_hours', 24)
        
        # Damage scales exponentially with wind speed
        # Using standard wind damage relationship: damage ∝ v^3
        base_damage_factor = (wind_speed / 100) ** 3
        
        # Duration multiplier
        duration_factor = 1.0 + math.log(duration / 24 + 1) * 0.5
        
        # Combined damage potential
        damage_potential = base_damage_factor * duration_factor
        
        return {
            'relative_potential': damage_potential,
            'category_description': self.classify_storm_strength(severity),
            'expected_damage_class': self._classify_damage_class(damage_potential),
            'structural_threat': self._assess_structural_threat(category),
            'flooding_threat': self._assess_flooding_threat(wind_speed, duration)
        }
    
    def _classify_damage_class(self, potential: float) -> str:
        """Classify damage potential into discrete classes."""
        if potential < 1.0:
            return 'Minor'
        elif potential < 3.0:
            return 'Moderate'
        elif potential < 8.0:
            return 'Major'
        elif potential < 20.0:
            return 'Severe'
        else:
            return 'Catastrophic'
    
    def _assess_structural_threat(self, category: int) -> str:
        """Assess threat to structures based on category."""
        threats = {
            1: 'Minimal structural damage; primarily to mobile homes and weak structures',
            2: 'Moderate damage to roofs, doors, and windows; some trees downed',
            3: 'Extensive damage; structural failure of small buildings; many trees downed',
            4: 'Extreme damage; roof and wall failures; trees and power poles down',
            5: 'Catastrophic damage; complete roof and building failures; area uninhabitable'
        }
        return threats.get(category, 'Unknown')
    
    def _assess_flooding_threat(self, wind_speed: float, duration: float) -> str:
        """Assess flooding threat based on wind and duration."""
        # Longer duration = more rain = more flooding
        # Higher winds = higher storm surge
        
        surge_factor = wind_speed / 100
        rainfall_factor = duration / 24
        
        combined = surge_factor * rainfall_factor
        
        if combined < 1.5:
            return 'Minor coastal flooding expected'
        elif combined < 3.0:
            return 'Moderate flooding likely; storm surge 4-6 feet'
        elif combined < 5.0:
            return 'Major flooding expected; storm surge 6-12 feet'
        else:
            return 'Catastrophic flooding; storm surge > 12 feet; extensive inland flooding'
    
    def compare_severities(self, severity1: Dict[str, float],
                          severity2: Dict[str, float]) -> Dict[str, Any]:
        """
        Compare two hurricane severities.
        
        Parameters
        ----------
        severity1, severity2 : dict
            Severity metrics to compare
        
        Returns
        -------
        dict
            Comparison results
        """
        norm1 = self.normalize_severity(severity1)
        norm2 = self.normalize_severity(severity2)
        
        more_severe = 1 if norm1['composite_severity'] > norm2['composite_severity'] else 2
        
        return {
            'more_severe': f"Hurricane {more_severe}",
            'severity_difference': abs(norm1['composite_severity'] - norm2['composite_severity']),
            'wind_difference_mph': abs(
                severity1.get('max_wind_speed_mph', 0) - 
                severity2.get('max_wind_speed_mph', 0)
            ),
            'pressure_difference_mb': abs(
                severity1.get('min_pressure_mb', 1013) - 
                severity2.get('min_pressure_mb', 1013)
            ),
            'category_difference': abs(
                severity1.get('category', 1) - 
                severity2.get('category', 1)
            ),
            'severity1_composite': norm1['composite_severity'],
            'severity2_composite': norm2['composite_severity']
        }
    
    def get_severity_percentile(self, severity: Dict[str, float],
                               historical_data: list) -> float:
        """
        Calculate percentile ranking of storm severity.
        
        Parameters
        ----------
        severity : dict
            Storm severity to rank
        historical_data : list
            List of historical storm severities
        
        Returns
        -------
        float
            Percentile (0-100)
        """
        if not historical_data:
            return 50.0  # Default to median if no historical data
        
        composite = self.normalize_severity(severity)['composite_severity']
        
        # Count how many historical storms are less severe
        less_severe = sum(
            1 for hist in historical_data
            if self.normalize_severity(hist)['composite_severity'] < composite
        )
        
        percentile = (less_severe / len(historical_data)) * 100
        return percentile

