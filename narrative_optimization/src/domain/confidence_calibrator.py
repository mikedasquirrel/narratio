"""
Confidence Calibration

Calibrates predictions based on domain expertise and training experience.

Philosophy: "Know what you know, and know what you don't know"
System should be honest about its confidence in each domain.
"""

from typing import Dict, Any
import numpy as np


class ConfidenceCalibrator:
    """
    Calibrates predictions based on domain familiarity.
    
    - High expertise → Trust prediction
    - Low expertise → Push towards uncertainty
    - No expertise → Cannot predict
    """
    
    def calibrate_prediction(self, raw_prediction: float, expertise_info: Dict) -> Dict[str, Any]:
        """
        Calibrate prediction based on domain expertise.
        
        Parameters
        ----------
        raw_prediction : float
            Uncalibrated prediction (0-1)
        expertise_info : dict
            Domain expertise information
        
        Returns
        -------
        calibrated : dict
            Calibrated prediction with confidence and reasoning
        """
        overall_confidence = expertise_info['overall_confidence']
        training_samples = expertise_info['total_training_samples']
        
        # Calibration strength based on sample size
        if training_samples > 10000:
            calibration_strength = 1.0  # Trust fully
        elif training_samples > 5000:
            calibration_strength = 0.9
        elif training_samples > 1000:
            calibration_strength = 0.75
        elif training_samples > 100:
            calibration_strength = 0.55
        else:
            calibration_strength = 0.3
        
        # Apply calibration: push towards 50% (uncertainty) when expertise is low
        deviation_from_50 = raw_prediction - 0.5
        calibrated_deviation = deviation_from_50 * calibration_strength * overall_confidence
        calibrated_prediction = 0.5 + calibrated_deviation
        
        # Confidence in the prediction
        prediction_confidence = overall_confidence * calibration_strength
        
        # Determine if we can actually make this prediction
        can_predict_reliably = prediction_confidence > 0.40
        
        return {
            'calibrated_prediction': float(calibrated_prediction),
            'raw_prediction': float(raw_prediction),
            'prediction_confidence': float(prediction_confidence),
            'calibration_applied': float(calibration_strength),
            'can_predict': can_predict_reliably,
            'confidence_level': self._confidence_to_level(prediction_confidence),
            'uncertainty': float(1 - prediction_confidence),
            'reasoning': self._generate_reasoning(
                training_samples,
                overall_confidence,
                calibration_strength,
                can_predict_reliably
            )
        }
    
    def _confidence_to_level(self, confidence: float) -> str:
        """Convert confidence score to level."""
        if confidence > 0.75:
            return "HIGH"
        elif confidence > 0.55:
            return "MODERATE"
        elif confidence > 0.35:
            return "LOW"
        else:
            return "VERY LOW"
    
    def _generate_reasoning(self, samples: int, conf: float, calib: float, can_predict: bool) -> str:
        """Generate explanation of confidence."""
        if can_predict:
            if samples > 10000:
                return f"High confidence: Based on {samples:,} training examples. System is well-trained in this domain."
            elif samples > 1000:
                return f"Moderate confidence: Based on {samples:,} training examples. System has good familiarity with this domain."
            else:
                return f"Low confidence: Based on {samples} training examples. Prediction is educated guess."
        else:
            return f"Cannot predict reliably: Only {samples} training examples. Insufficient data for this domain. System would be guessing."

