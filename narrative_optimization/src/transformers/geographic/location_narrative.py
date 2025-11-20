"""
Geographic Narrative Factor Transformer

Analyzes location-based story elements including historic venues,
travel narratives, climate contrasts, and regional rivalries.

This transformer identifies when geographic factors create narrative
significance that influences competitive outcomes.

Author: Narrative Enhancement System
Date: November 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime, timedelta
import math


class GeographicNarrativeTransformer(BaseEstimator, TransformerMixin):
    """
    Extract geographic narrative factor features.
    
    Philosophy:
    - Location carries narrative weight
    - Historic venues have special power
    - Travel creates story arcs
    - Climate differences affect perception
    - Regional identity drives competition
    
    Features (35 total):
    - Historic venue advantage (6)
    - Travel fatigue narratives (6)
    - Time zone drama effects (5)
    - Climate contrast stories (6)
    - Regional rivalry intensity (6)
    - Border battle dynamics (6)
    """
    
    def __init__(
        self,
        distance_weight: float = 0.7,
        venue_history_weight: float = 0.8,
        include_weather: bool = True
    ):
        """
        Initialize geographic narrative analyzer.
        
        Parameters
        ----------
        distance_weight : float
            How much travel distance matters
        venue_history_weight : float
            Weight for historic venue effects
        include_weather : bool
            Include weather/climate analysis
        """
        self.distance_weight = distance_weight
        self.venue_history_weight = venue_history_weight
        self.include_weather = include_weather
        
        # Historic venue classifications
        self.historic_venues = {
            'original_six': {
                'venues': ['Madison Square Garden', 'TD Garden', 'Bell Centre',
                          'Little Caesars Arena', 'Scotiabank Arena', 'United Center'],
                'power': 0.9
            },
            'classic': {
                'min_age': 50,
                'power': 0.8
            },
            'legendary': {
                'championships': 5,
                'power': 0.85
            },
            'new_cathedral': {
                'opened_after': 2010,
                'capacity': 18000,
                'power': 0.6
            }
        }
        
        # Regional classifications
        self.regions = {
            'northeast': ['BOS', 'NYR', 'NYI', 'NJD', 'PHI', 'PIT', 'BUF'],
            'atlantic': ['FLA', 'TBL', 'CAR', 'WSH'],
            'central': ['CHI', 'DET', 'STL', 'MIN', 'NSH', 'CBJ'],
            'mountain': ['COL', 'ARI', 'VGK', 'UTA'],
            'pacific': ['LAK', 'SJS', 'ANA', 'VAN', 'CGY', 'EDM', 'SEA'],
            'canadian': ['TOR', 'MTL', 'OTT', 'WPG', 'CGY', 'EDM', 'VAN']
        }
        
        # Travel impact thresholds
        self.travel_thresholds = {
            'short': 500,      # miles
            'medium': 1500,
            'long': 2500,
            'extreme': 3500
        }
        
    def fit(self, X, y=None):
        """
        Learn geographic patterns from training data.
        
        Parameters
        ----------
        X : list of dicts or DataFrame
            Historical game data with geographic info
        y : ignored
        
        Returns
        -------
        self
        """
        # Build venue power database
        self.venue_powers_ = self._build_venue_powers(X)
        
        # Learn travel impact patterns
        self.travel_impacts_ = self._learn_travel_impacts(X)
        
        # Analyze regional patterns
        self.regional_patterns_ = self._analyze_regional_patterns(X)
        
        return self
        
    def transform(self, X):
        """
        Extract geographic narrative features.
        
        Parameters
        ----------
        X : list of dicts or DataFrame
            Game data with location context
            
        Returns
        -------
        np.ndarray
            Geographic features (n_samples, 35)
        """
        # Handle single sample
        if isinstance(X, dict):
            X = [X]
            
        features = []
        for item in X:
            feature_vec = []
            
            # Historic venue advantage (6)
            venue_features = self._extract_venue_advantage(item)
            feature_vec.extend(venue_features)
            
            # Travel fatigue narratives (6)
            travel_features = self._extract_travel_narratives(item)
            feature_vec.extend(travel_features)
            
            # Time zone drama (5)
            timezone_features = self._extract_timezone_drama(item)
            feature_vec.extend(timezone_features)
            
            # Climate contrast (6)
            if self.include_weather:
                climate_features = self._extract_climate_contrast(item)
            else:
                climate_features = [0.0] * 6
            feature_vec.extend(climate_features)
            
            # Regional rivalry (6)
            regional_features = self._extract_regional_rivalry(item)
            feature_vec.extend(regional_features)
            
            # Border battle dynamics (6)
            border_features = self._extract_border_battles(item)
            feature_vec.extend(border_features)
            
            features.append(feature_vec)
            
        return np.array(features, dtype=np.float32)
        
    def _build_venue_powers(self, X):
        """Build database of venue narrative powers."""
        venue_powers = {}
        
        # Would analyze actual venue impacts
        # For now, use classifications
        default_venues = {
            'Madison Square Garden': 0.95,  # "The Garden"
            'Bell Centre': 0.90,            # Hockey cathedral
            'TD Garden': 0.85,
            'United Center': 0.85,
            'T-Mobile Arena': 0.75,         # New but loud
            'Climate Pledge Arena': 0.70    # New venue
        }
        
        return default_venues
        
    def _learn_travel_impacts(self, X):
        """Learn how travel affects performance."""
        return {
            'road_trip_game_3': -0.15,  # Third game of road trip
            'coast_to_coast': -0.20,    # Cross-continent travel
            'back_to_back_travel': -0.25,
            'homecoming': 0.10          # Return from long trip
        }
        
    def _analyze_regional_patterns(self, X):
        """Analyze regional rivalry patterns."""
        return {
            'intra_division': 0.2,
            'regional_rival': 0.3,
            'cross_conference': 0.0,
            'border_battle': 0.4
        }
        
    def _extract_venue_advantage(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract historic venue narrative advantage.
        
        Returns 6 features measuring venue power.
        """
        features = []
        
        venue = item.get('venue', '')
        venue_age = item.get('venue_age_years', 20)
        is_home = item.get('is_home', True)
        
        # Base venue power
        if venue in self.venue_powers_:
            base_power = self.venue_powers_[venue]
        else:
            # Calculate from characteristics
            base_power = 0.5
            if venue_age > 50:
                base_power += 0.2
            if item.get('venue_championships', 0) > 3:
                base_power += 0.15
                
        if is_home:
            features.append(base_power)
        else:
            features.append(base_power * 0.3)  # Visitors feel it too
            
        # Sellout streak pressure
        sellout_streak = item.get('sellout_streak', 0)
        if sellout_streak > 100:
            features.append(0.9)
        elif sellout_streak > 50:
            features.append(0.7)
        elif sellout_streak > 20:
            features.append(0.5)
        else:
            features.append(0.2)
            
        # Venue mystique (special atmosphere)
        mystique_factors = 0.0
        if item.get('venue_nickname', None):  # Has a nickname
            mystique_factors += 0.3
        if item.get('unique_features', False):  # Special characteristics
            mystique_factors += 0.3
        if item.get('loudness_rank', 20) <= 5:  # Top 5 loudest
            mystique_factors += 0.4
            
        features.append(min(1.0, mystique_factors))
        
        # First visit narrative
        first_visit = item.get('opponent_first_visit', False)
        games_at_venue = item.get('opponent_games_at_venue', 10)
        
        if first_visit:
            features.append(0.8)  # Intimidation factor
        elif games_at_venue < 5:
            features.append(0.5)
        else:
            features.append(0.1)
            
        # Playoff venue transformation
        if item.get('is_playoffs', False):
            regular_season_record = item.get('venue_regular_home_record', 0.5)
            playoff_record = item.get('venue_playoff_home_record', 0.5)
            
            transformation = playoff_record - regular_season_record
            features.append(np.tanh(transformation * 5))  # -1 to 1
        else:
            features.append(0.0)
            
        # Historic moment proximity
        days_since_historic = item.get('days_since_venue_historic_moment', 365)
        if days_since_historic < 30:
            features.append(0.8)  # Fresh in memory
        elif days_since_historic < 365:
            features.append(0.4)
        else:
            features.append(0.1)
            
        return features
        
    def _extract_travel_narratives(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract travel fatigue narrative features.
        
        Returns 6 features measuring travel story elements.
        """
        features = []
        
        # Distance traveled
        distance = item.get('travel_distance_miles', 0)
        
        if distance > self.travel_thresholds['extreme']:
            features.append(1.0)  # Extreme travel
        elif distance > self.travel_thresholds['long']:
            features.append(0.7)
        elif distance > self.travel_thresholds['medium']:
            features.append(0.4)
        elif distance > self.travel_thresholds['short']:
            features.append(0.2)
        else:
            features.append(0.0)
            
        # Road trip position
        road_trip_game = item.get('road_trip_game_number', 0)
        if road_trip_game >= 4:
            features.append(0.9)  # Deep into trip
        elif road_trip_game == 3:
            features.append(0.7)
        elif road_trip_game == 2:
            features.append(0.4)
        elif road_trip_game == 1:
            features.append(0.2)
        else:
            features.append(0.0)
            
        # Travel schedule density
        miles_last_week = item.get('miles_traveled_last_7_days', 0)
        if miles_last_week > 5000:
            features.append(0.9)  # Brutal travel
        elif miles_last_week > 3000:
            features.append(0.6)
        elif miles_last_week > 1500:
            features.append(0.3)
        else:
            features.append(0.0)
            
        # Circadian disruption
        time_zone_changes = item.get('time_zone_changes_last_week', 0)
        if time_zone_changes >= 6:  # Back and forth
            features.append(0.8)
        elif time_zone_changes >= 3:
            features.append(0.5)
        else:
            features.append(0.1)
            
        # Homecoming narrative
        days_away = item.get('consecutive_days_on_road', 0)
        next_game_home = item.get('next_game_is_home', False)
        
        if days_away > 10 and next_game_home:
            features.append(0.8)  # Almost home
        elif days_away > 7:
            features.append(0.5)  # Road weary
        else:
            features.append(0.0)
            
        # Travel advantage differential
        home_travel = item.get('home_team_miles_last_week', 0)
        away_travel = item.get('away_team_miles_last_week', 0)
        
        travel_diff = (away_travel - home_travel) / 1000.0  # Per 1000 miles
        features.append(np.tanh(travel_diff / 3.0))  # -1 to 1
        
        return features
        
    def _extract_timezone_drama(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract time zone drama effect features.
        
        Returns 5 features measuring temporal displacement.
        """
        features = []
        
        # Time zone difference
        tz_diff = abs(item.get('time_zone_difference', 0))
        
        if tz_diff >= 3:
            features.append(0.9)  # Coast to coast
        elif tz_diff == 2:
            features.append(0.5)
        elif tz_diff == 1:
            features.append(0.2)
        else:
            features.append(0.0)
            
        # Body clock game time
        home_body_time = item.get('home_team_body_clock_time', 19)  # 7pm
        away_body_time = item.get('away_team_body_clock_time', 19)
        
        # Early/late game effects
        for body_time in [home_body_time, away_body_time]:
            if body_time < 17:  # Before 5pm body time
                features.append(0.7)  # Afternoon game
            elif body_time > 22:  # After 10pm body time
                features.append(0.8)  # Late night game
            else:
                features.append(0.0)
                
        # West coast trip for east team
        if item.get('east_team_on_west_coast', False):
            consecutive_west_games = item.get('consecutive_west_coast_games', 1)
            if consecutive_west_games == 1:
                features.append(0.7)  # First game hardest
            else:
                features.append(0.3)  # Adjusting
        else:
            features.append(0.0)
            
        # Sunday afternoon special
        is_sunday_afternoon = (item.get('day_of_week') == 'Sunday' and 
                             item.get('game_time_hour', 19) < 17)
        if is_sunday_afternoon:
            features.append(0.6)  # Different rhythm
        else:
            features.append(0.0)
            
        return features
        
    def _extract_climate_contrast(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract climate contrast story features.
        
        Returns 6 features measuring weather narrative.
        """
        features = []
        
        # Temperature differential
        temp_diff = abs(item.get('city_temperature_difference', 0))
        
        if temp_diff > 40:  # Fahrenheit
            features.append(0.9)  # Extreme contrast
        elif temp_diff > 25:
            features.append(0.6)
        elif temp_diff > 15:
            features.append(0.3)
        else:
            features.append(0.0)
            
        # Warm weather team in cold
        if item.get('warm_team_in_cold', False):
            outside_temp = item.get('game_day_temperature', 32)
            if outside_temp < 20:
                features.append(0.8)  # Narrative active
            elif outside_temp < 32:
                features.append(0.5)
            else:
                features.append(0.2)
        else:
            features.append(0.0)
            
        # Weather event narrative
        weather_event = item.get('significant_weather_event', None)
        if weather_event == 'blizzard':
            features.append(1.0)  # Maximum drama
        elif weather_event == 'storm':
            features.append(0.7)
        elif weather_event == 'extreme_cold':
            features.append(0.5)
        else:
            features.append(0.0)
            
        # Outdoor game factors
        is_outdoor = item.get('is_outdoor_game', False)
        if is_outdoor:
            wind_speed = item.get('wind_speed_mph', 0)
            features.append(min(1.0, wind_speed / 20.0))
        else:
            features.append(0.0)
            
        # Altitude adjustment
        altitude_diff = item.get('altitude_difference_feet', 0)
        if altitude_diff > 5000:  # Denver effect
            features.append(0.8)
        elif altitude_diff > 3000:
            features.append(0.5)
        else:
            features.append(0.0)
            
        # Climate rivalry (sun belt vs snow belt)
        climate_rivalry = item.get('climate_contrast_rivalry', False)
        if climate_rivalry:
            features.append(0.7)
        else:
            features.append(0.0)
            
        return features
        
    def _extract_regional_rivalry(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract regional rivalry intensity features.
        
        Returns 6 features measuring geographic rivalry.
        """
        features = []
        
        # Get team regions
        home_team = item.get('home_team_code', '')
        away_team = item.get('away_team_code', '')
        
        # Same region intensity
        same_region = False
        for region, teams in self.regions.items():
            if home_team in teams and away_team in teams:
                same_region = True
                if region == 'canadian':
                    features.append(0.9)  # Canadian rivalries intense
                elif region == 'northeast':
                    features.append(0.8)  # Original six territory
                else:
                    features.append(0.6)
                break
                
        if not same_region:
            features.append(0.0)
            
        # Geographic proximity
        distance = item.get('cities_distance_miles', 1000)
        if distance < 100:
            features.append(1.0)  # Same metro area
        elif distance < 300:
            features.append(0.8)  # Driving distance
        elif distance < 500:
            features.append(0.5)
        else:
            features.append(0.0)
            
        # State/province rivalry
        same_state = item.get('same_state_province', False)
        if same_state:
            features.append(0.8)
        else:
            features.append(0.0)
            
        # Cultural rivalry markers
        cultural_rivalry_score = 0.0
        rivalry_factors = {
            'language_difference': 0.3,  # Montreal vs others
            'country_rivalry': 0.4,       # US vs Canada
            'expansion_vs_original': 0.2,
            'coast_rivalry': 0.3          # East vs West
        }
        
        for factor, weight in rivalry_factors.items():
            if item.get(factor, False):
                cultural_rivalry_score += weight
                
        features.append(min(1.0, cultural_rivalry_score))
        
        # Regional pride game
        is_regional_showcase = item.get('regional_tv_game', False)
        if is_regional_showcase:
            features.append(0.7)
        else:
            features.append(0.0)
            
        # Division rivalry history
        division_meetings = item.get('division_meetings_last_3_years', 0)
        playoff_meetings = item.get('playoff_meetings_last_5_years', 0)
        
        rivalry_heat = min(1.0, division_meetings / 50.0 + playoff_meetings * 0.2)
        features.append(rivalry_heat)
        
        return features
        
    def _extract_border_battles(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract border battle dynamics features.
        
        Returns 6 features measuring cross-border narratives.
        """
        features = []
        
        # US vs Canada game
        us_vs_canada = item.get('us_vs_canada_game', False)
        
        if us_vs_canada:
            features.append(0.7)
            
            # Olympics/international context
            days_from_international = item.get('days_from_international_tournament', 365)
            if days_from_international < 30:
                features.append(0.9)  # Fresh rivalry
            elif days_from_international < 180:
                features.append(0.5)
            else:
                features.append(0.2)
        else:
            features.extend([0.0, 0.0])
            
        # Border proximity game
        border_distance = min(
            item.get('home_city_to_border_miles', 1000),
            item.get('away_city_to_border_miles', 1000)
        )
        
        if border_distance < 100:
            features.append(0.8)  # True border cities
        elif border_distance < 300:
            features.append(0.4)
        else:
            features.append(0.0)
            
        # National broadcast emphasis
        is_national_broadcast = item.get('is_national_broadcast', False)
        broadcast_country = item.get('primary_broadcast_country', 'US')
        
        if is_national_broadcast and us_vs_canada:
            if broadcast_country == 'Canada':
                features.append(0.8)  # Hockey Night in Canada
            else:
                features.append(0.6)
        else:
            features.append(0.0)
            
        # Cross-border player narrative
        canadians_on_us_team = item.get('canadian_players_on_us_team', 0)
        americans_on_cdn_team = item.get('american_players_on_canadian_team', 0)
        
        crossover_factor = (canadians_on_us_team + americans_on_cdn_team) / 40.0
        features.append(min(1.0, crossover_factor))
        
        # Currency/economy narrative
        if us_vs_canada:
            exchange_rate = item.get('usd_cad_exchange_rate', 1.3)
            if exchange_rate > 1.4:
                features.append(0.6)  # Economic subplot
            else:
                features.append(0.2)
        else:
            features.append(0.0)
            
        return features
        
    def get_feature_names(self) -> List[str]:
        """Return feature names for interpretability."""
        names = []
        
        # Venue features
        names.extend([
            'venue_base_power',
            'venue_sellout_streak',
            'venue_mystique_factor',
            'venue_first_visit_effect',
            'venue_playoff_transformation',
            'venue_historic_proximity'
        ])
        
        # Travel features
        names.extend([
            'travel_distance_impact',
            'travel_road_trip_position',
            'travel_schedule_density',
            'travel_circadian_disruption',
            'travel_homecoming_narrative',
            'travel_advantage_differential'
        ])
        
        # Time zone features
        names.extend([
            'timezone_difference_effect',
            'timezone_home_body_clock',
            'timezone_away_body_clock',
            'timezone_west_coast_trip',
            'timezone_sunday_afternoon'
        ])
        
        # Climate features
        names.extend([
            'climate_temperature_differential',
            'climate_warm_team_in_cold',
            'climate_weather_event',
            'climate_outdoor_game_wind',
            'climate_altitude_adjustment',
            'climate_rivalry_narrative'
        ])
        
        # Regional rivalry features
        names.extend([
            'regional_same_region_intensity',
            'regional_geographic_proximity',
            'regional_state_province_rivalry',
            'regional_cultural_markers',
            'regional_pride_showcase',
            'regional_division_history'
        ])
        
        # Border battle features
        names.extend([
            'border_us_canada_game',
            'border_international_context',
            'border_proximity_factor',
            'border_national_broadcast',
            'border_crossover_players',
            'border_economic_subplot'
        ])
        
        return names
