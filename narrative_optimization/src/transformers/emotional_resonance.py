"""
Emotional Resonance Transformer

Extracts emotional arc structure, complexity, resonance patterns, and dynamics.
Most universal transformer - applies to ALL narrative domains.

Core insight: Emotional journey is fundamental to storytelling and connection.
"""

import numpy as np
import re
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Dict, Any
from .utils.input_validation import ensure_string_list, ensure_string


class EmotionalResonanceTransformer(BaseEstimator, TransformerMixin):
    """
    Extract emotional resonance features from narrative text.
    
    Captures:
    1. Emotional Arc Structure - trajectory, peaks, baseline
    2. Emotional Complexity - mixed emotions, range, authenticity
    3. Emotional Resonance Patterns - universal emotions, empathy triggers
    4. Emotional Dynamics - pacing, contrast, volatility
    
    ~35 features total
    """
    
    def __init__(self):
        """Initialize emotion lexicons and patterns"""
        
        # Plutchik's 8 basic emotions + extensions
        self.emotion_lexicons = {
            'joy': ['happy', 'joy', 'delight', 'pleased', 'glad', 'cheerful', 'elated', 
                   'ecstatic', 'thrilled', 'excited', 'wonderful', 'fantastic', 'great',
                   'love', 'amazing', 'beautiful', 'perfect', 'brilliant', 'awesome'],
            
            'sadness': ['sad', 'unhappy', 'depressed', 'miserable', 'grief', 'sorrow',
                       'melancholy', 'despair', 'hopeless', 'devastated', 'heartbroken',
                       'tragic', 'mourn', 'loss', 'lonely', 'empty', 'disappointed'],
            
            'anger': ['angry', 'mad', 'furious', 'rage', 'outrage', 'hostile', 'bitter',
                     'resentful', 'irritated', 'frustrated', 'annoyed', 'hate', 'violent',
                     'aggressive', 'cruel', 'vicious', 'hostile'],
            
            'fear': ['afraid', 'scared', 'terrified', 'frightened', 'anxious', 'worried',
                    'nervous', 'panic', 'dread', 'horror', 'terror', 'alarm', 'threatened',
                    'insecure', 'vulnerable', 'paranoid', 'uneasy'],
            
            'trust': ['trust', 'faith', 'believe', 'confident', 'secure', 'safe', 'reliable',
                     'loyal', 'honest', 'genuine', 'authentic', 'dependable', 'certain'],
            
            'disgust': ['disgust', 'repulsed', 'revolted', 'sickened', 'appalled', 'horrified',
                       'nauseated', 'loathe', 'detest', 'abhor', 'vile', 'gross'],
            
            'surprise': ['surprised', 'shocked', 'astonished', 'amazed', 'stunned', 'startled',
                        'unexpected', 'sudden', 'astounded', 'bewildered', 'speechless'],
            
            'anticipation': ['hope', 'expect', 'anticipate', 'eager', 'looking forward', 
                            'await', 'excited', 'optimistic', 'aspire', 'dream', 'wish',
                            'desire', 'long', 'yearn', 'crave']
        }
        
        # Valence: positive vs negative
        self.positive_emotions = ['joy', 'trust', 'anticipation']
        self.negative_emotions = ['sadness', 'anger', 'fear', 'disgust']
        
        # Vulnerability markers
        self.vulnerability_markers = [
            'struggle', 'difficult', 'hard', 'challenge', 'fail', 'mistake', 'wrong',
            'hurt', 'pain', 'suffer', 'weak', 'vulnerable', 'insecure', 'doubt',
            'uncertain', 'confused', 'lost', 'broken', 'damaged', 'scared'
        ]
        
        # Catharsis/resolution markers
        self.catharsis_markers = [
            'finally', 'breakthrough', 'realize', 'understand', 'overcome', 'triumph',
            'succeed', 'achieve', 'accomplish', 'resolve', 'heal', 'recover',
            'transform', 'change', 'grow', 'learn', 'discover', 'accept'
        ]
        
        # Empathy triggers
        self.empathy_triggers = [
            'everyone', 'we all', 'you know', 'imagine', 'remember when', 'felt like',
            'we\'ve all', 'universal', 'human', 'relate', 'understand', 'share'
        ]
        
        # Emotional intensity modifiers
        self.intensifiers = [
            'very', 'extremely', 'incredibly', 'absolutely', 'completely', 'totally',
            'utterly', 'so', 'really', 'deeply', 'profoundly', 'intensely'
        ]
        
        self.diminishers = [
            'slightly', 'somewhat', 'kind of', 'sort of', 'a bit', 'a little',
            'barely', 'hardly', 'almost', 'nearly'
        ]
        
    def fit(self, X, y=None):
        """Fit transformer (no-op, lexicon-based)"""
        # Ensure X is list of strings
        X = ensure_string_list(X)
        
        return self
    
    def transform(self, X):
        """
        Transform texts into emotional resonance features.
        
        Parameters
        ----------
        X : array-like of strings
            Narrative texts
            
        Returns
        -------
        features : ndarray of shape (n_samples, 35)
            Emotional resonance features
        """
        features = []
        
        for text in X:
            text_lower = text.lower()
            words = re.findall(r'\b\w+\b', text_lower)
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            feat_dict = {}
            
            # === 1. EMOTIONAL ARC STRUCTURE (8 features) ===
            
            # Emotion counts per category
            emotion_counts = {}
            for emotion, lexicon in self.emotion_lexicons.items():
                count = sum(1 for word in words if word in lexicon)
                emotion_counts[emotion] = count
            
            total_emotion_words = sum(emotion_counts.values())
            
            # Valence (positive vs negative)
            positive_count = sum(emotion_counts[e] for e in self.positive_emotions)
            negative_count = sum(emotion_counts[e] for e in self.negative_emotions)
            
            feat_dict['emotional_valence'] = (positive_count - negative_count) / (total_emotion_words + 1)
            feat_dict['emotional_density'] = total_emotion_words / (len(words) + 1)
            
            # Emotional trajectory (first half vs second half)
            mid_point = len(words) // 2
            first_half_words = words[:mid_point]
            second_half_words = words[mid_point:]
            
            first_half_positive = sum(1 for w in first_half_words if any(w in self.emotion_lexicons[e] for e in self.positive_emotions))
            second_half_positive = sum(1 for w in second_half_words if any(w in self.emotion_lexicons[e] for e in self.positive_emotions))
            
            feat_dict['trajectory_rising'] = float(second_half_positive > first_half_positive)
            feat_dict['trajectory_magnitude'] = (second_half_positive - first_half_positive) / (len(words) / 2 + 1)
            
            # Peak emotion detection (maximum emotion in any sentence)
            sentence_emotions = []
            for sent in sentences:
                sent_words = re.findall(r'\b\w+\b', sent.lower())
                sent_emotion_count = sum(1 for w in sent_words if any(w in lex for lex in self.emotion_lexicons.values()))
                sentence_emotions.append(sent_emotion_count / (len(sent_words) + 1))
            
            feat_dict['peak_emotional_intensity'] = max(sentence_emotions) if sentence_emotions else 0.0
            feat_dict['baseline_emotional_tone'] = np.mean(sentence_emotions) if sentence_emotions else 0.0
            
            # Emotional arc shape (U-shape, inverse-U, etc.)
            if len(sentence_emotions) >= 3:
                third = len(sentence_emotions) // 3
                beginning = np.mean(sentence_emotions[:third])
                middle = np.mean(sentence_emotions[third:2*third])
                end = np.mean(sentence_emotions[2*third:])
                
                # U-shaped: low middle, high ends
                feat_dict['arc_u_shaped'] = float((beginning + end) / 2 > middle)
                # Inverse U: high middle, low ends
                feat_dict['arc_inverse_u'] = float(middle > (beginning + end) / 2)
            else:
                feat_dict['arc_u_shaped'] = 0.0
                feat_dict['arc_inverse_u'] = 0.0
            
            # === 2. EMOTIONAL COMPLEXITY (7 features) ===
            
            # Emotional diversity (how many different emotions present)
            emotions_present = sum(1 for count in emotion_counts.values() if count > 0)
            feat_dict['emotional_diversity'] = emotions_present / len(self.emotion_lexicons)
            
            # Mixed emotions (positive AND negative both present significantly)
            feat_dict['mixed_emotions'] = float(positive_count > 0 and negative_count > 0 and 
                                               abs(positive_count - negative_count) < (total_emotion_words / 3 + 1))
            
            # Emotional range (max - min across categories)
            if emotion_counts.values():
                max_emotion = max(emotion_counts.values())
                min_emotion = min(emotion_counts.values())
                feat_dict['emotional_range'] = (max_emotion - min_emotion) / (total_emotion_words + 1)
            else:
                feat_dict['emotional_range'] = 0.0
            
            # Vulnerability markers
            vulnerability_count = sum(1 for word in words if word in self.vulnerability_markers)
            feat_dict['vulnerability_density'] = vulnerability_count / (len(words) + 1)
            
            # Emotional authenticity (specific emotions vs. generic "feel")
            specific_emotions = total_emotion_words
            generic_feel = sum(1 for word in words if word in ['feel', 'felt', 'feeling', 'feels'])
            feat_dict['emotional_specificity'] = specific_emotions / (specific_emotions + generic_feel + 1)
            
            # Emotional coherence (consistency vs. whiplash)
            if len(sentence_emotions) > 1:
                emotion_changes = [abs(sentence_emotions[i] - sentence_emotions[i-1]) 
                                 for i in range(1, len(sentence_emotions))]
                feat_dict['emotional_coherence'] = 1.0 - (np.mean(emotion_changes) if emotion_changes else 0.0)
            else:
                feat_dict['emotional_coherence'] = 1.0
            
            # Ambivalence (contradictory emotions in close proximity)
            ambivalence_score = 0
            for i in range(len(sentences) - 1):
                sent1_words = set(re.findall(r'\b\w+\b', sentences[i].lower()))
                sent2_words = set(re.findall(r'\b\w+\b', sentences[i+1].lower()))
                
                sent1_pos = any(w in sent1_words for e in self.positive_emotions for w in self.emotion_lexicons[e])
                sent1_neg = any(w in sent1_words for e in self.negative_emotions for w in self.emotion_lexicons[e])
                sent2_pos = any(w in sent2_words for e in self.positive_emotions for w in self.emotion_lexicons[e])
                sent2_neg = any(w in sent2_words for e in self.negative_emotions for w in self.emotion_lexicons[e])
                
                if (sent1_pos and sent2_neg) or (sent1_neg and sent2_pos):
                    ambivalence_score += 1
            
            feat_dict['emotional_ambivalence'] = ambivalence_score / (len(sentences) + 1)
            
            # === 3. EMOTIONAL RESONANCE PATTERNS (10 features) ===
            
            # Universal emotion words (Plutchik's core)
            for emotion in self.emotion_lexicons.keys():
                count = emotion_counts[emotion]
                feat_dict[f'emotion_{emotion}_density'] = count / (len(words) + 1)
            
            # Empathy triggers
            empathy_count = sum(1 for phrase in self.empathy_triggers if phrase in text_lower)
            feat_dict['empathy_trigger_density'] = empathy_count / (len(sentences) + 1)
            
            # Catharsis/breakthrough moments
            catharsis_count = sum(1 for word in words if word in self.catharsis_markers)
            feat_dict['catharsis_markers'] = catharsis_count / (len(words) + 1)
            
            # === 4. EMOTIONAL DYNAMICS (10 features) ===
            
            # Emotional pacing (rate of emotional change)
            if len(sentence_emotions) > 1:
                emotion_velocity = [sentence_emotions[i] - sentence_emotions[i-1] 
                                  for i in range(1, len(sentence_emotions))]
                feat_dict['emotional_pacing_fast'] = float(np.std(emotion_velocity) > 0.1) if emotion_velocity else 0.0
                feat_dict['emotional_pacing_sustained'] = float(np.std(sentence_emotions) < 0.1) if sentence_emotions else 0.0
            else:
                feat_dict['emotional_pacing_fast'] = 0.0
                feat_dict['emotional_pacing_sustained'] = 0.0
            
            # Emotional contrast (highs vs lows)
            if sentence_emotions:
                feat_dict['emotional_contrast'] = max(sentence_emotions) - min(sentence_emotions)
            else:
                feat_dict['emotional_contrast'] = 0.0
            
            # Emotional volatility (rapid shifts)
            if len(sentence_emotions) > 2:
                volatility = np.std(sentence_emotions)
                feat_dict['emotional_volatility'] = volatility
            else:
                feat_dict['emotional_volatility'] = 0.0
            
            # Intensity modifiers
            intensifier_count = sum(1 for word in words if word in self.intensifiers)
            diminisher_count = sum(1 for word in words if word in self.diminishers)
            
            feat_dict['emotional_intensity_amplified'] = intensifier_count / (len(words) + 1)
            feat_dict['emotional_intensity_muted'] = diminisher_count / (len(words) + 1)
            
            # Emotional momentum (building vs. dissipating)
            if len(sentence_emotions) >= 3:
                # Check if emotions increase over time (momentum building)
                increasing = sum(1 for i in range(1, len(sentence_emotions)) 
                               if sentence_emotions[i] > sentence_emotions[i-1])
                feat_dict['emotional_momentum_building'] = increasing / (len(sentence_emotions) - 1)
            else:
                feat_dict['emotional_momentum_building'] = 0.5
            
            # Emotional resolution (ending on positive vs. negative)
            if sentence_emotions:
                final_third = sentence_emotions[len(sentence_emotions)*2//3:]
                feat_dict['emotional_resolution_positive'] = float(np.mean(final_third) > feat_dict['baseline_emotional_tone'])
            else:
                feat_dict['emotional_resolution_positive'] = 0.5
            
            # Emotional saturation (too much emotion = performative)
            feat_dict['emotional_saturation'] = float(feat_dict['emotional_density'] > 0.15)
            
            # Convert to feature vector
            features.append(list(feat_dict.values()))
        
        return np.array(features)
    
    def get_feature_names_out(self, input_features=None):
        """Get feature names for output"""
        feature_names = [
            # Arc structure (8)
            'emotional_valence',
            'emotional_density',
            'trajectory_rising',
            'trajectory_magnitude',
            'peak_emotional_intensity',
            'baseline_emotional_tone',
            'arc_u_shaped',
            'arc_inverse_u',
            
            # Complexity (7)
            'emotional_diversity',
            'mixed_emotions',
            'emotional_range',
            'vulnerability_density',
            'emotional_specificity',
            'emotional_coherence',
            'emotional_ambivalence',
            
            # Resonance patterns (10)
            'emotion_joy_density',
            'emotion_sadness_density',
            'emotion_anger_density',
            'emotion_fear_density',
            'emotion_trust_density',
            'emotion_disgust_density',
            'emotion_surprise_density',
            'emotion_anticipation_density',
            'empathy_trigger_density',
            'catharsis_markers',
            
            # Dynamics (10)
            'emotional_pacing_fast',
            'emotional_pacing_sustained',
            'emotional_contrast',
            'emotional_volatility',
            'emotional_intensity_amplified',
            'emotional_intensity_muted',
            'emotional_momentum_building',
            'emotional_resolution_positive',
            'emotional_saturation'
        ]
        
        return np.array([f'emotional_resonance_{name}' for name in feature_names])

