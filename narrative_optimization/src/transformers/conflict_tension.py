"""
Conflict/Tension Transformer

Extracts conflict structure, stakes, tension building, and resolution quality.
Core narrative structure - essential for movies, novels, competitive domains.

Core insight: Narratives need opposition, stakes, and escalation to engage.
"""

import numpy as np
import re
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from .utils.input_validation import ensure_string_list


class ConflictTensionTransformer(BaseEstimator, TransformerMixin):
    """
    Extract conflict and tension features from narrative text.
    
    Captures:
    1. Conflict Structure - protagonist vs antagonist, internal vs external
    2. Stakes & Urgency - consequences, time pressure, irreversibility
    3. Tension Building - escalation, obstacles, setbacks
    4. Resolution Quality - type, earned vs. deus ex machina
    
    ~28 features total
    """
    
    def __init__(self):
        """Initialize conflict and tension markers"""
        
        # Opposition/antagonist markers
        self.opposition_markers = [
            'against', 'versus', 'vs', 'enemy', 'opponent', 'rival', 'adversary',
            'antagonist', 'villain', 'threat', 'danger', 'attack', 'fight',
            'battle', 'war', 'conflict', 'confrontation', 'oppose', 'resist'
        ]
        
        # Internal conflict markers
        self.internal_conflict_markers = [
            'struggle', 'doubt', 'question', 'wonder', 'confused', 'torn',
            'conflicted', 'dilemma', 'choice', 'decision', 'uncertain',
            'afraid', 'worried', 'anxious', 'guilty', 'shame', 'regret'
        ]
        
        # External conflict markers
        self.external_conflict_markers = [
            'face', 'confront', 'encounter', 'challenge', 'obstacle', 'problem',
            'difficulty', 'barrier', 'opposition', 'resistance', 'force', 'power'
        ]
        
        # Stakes markers (consequences)
        self.high_stakes_markers = [
            'life', 'death', 'die', 'kill', 'survive', 'survival', 'destroy',
            'lose', 'loss', 'risk', 'danger', 'critical', 'crucial', 'vital',
            'essential', 'everything', 'fate', 'future', 'forever', 'never'
        ]
        
        # Urgency/time pressure
        self.urgency_markers = [
            'now', 'immediately', 'quickly', 'hurry', 'rush', 'urgent', 'deadline',
            'time', 'soon', 'fast', 'rapid', 'before', 'until', 'by', 'limited'
        ]
        
        # Irreversibility cues
        self.irreversibility_markers = [
            'forever', 'never', 'always', 'permanent', 'irreversible', 'final',
            'last', 'end', 'no turning back', 'point of no return', 'too late'
        ]
        
        # Escalation markers
        self.escalation_markers = [
            'worse', 'more', 'even', 'intensify', 'increase', 'grow', 'escalate',
            'worsen', 'deteriorate', 'spiral', 'compound', 'multiply'
        ]
        
        # Obstacle markers
        self.obstacle_markers = [
            'but', 'however', 'problem', 'issue', 'challenge', 'difficulty',
            'obstacle', 'barrier', 'block', 'prevent', 'stop', 'fail', 'unable'
        ]
        
        # Setback markers
        self.setback_markers = [
            'fail', 'failure', 'defeat', 'lose', 'loss', 'setback', 'mistake',
            'error', 'wrong', 'disaster', 'catastrophe', 'collapse', 'ruin'
        ]
        
        # Complication markers
        self.complication_markers = [
            'complicate', 'complex', 'difficult', 'worse', 'twist', 'turn',
            'unexpected', 'surprise', 'discover', 'reveal', 'realize'
        ]
        
        # Resolution markers
        self.resolution_markers = [
            'finally', 'end', 'resolve', 'solution', 'answer', 'overcome',
            'succeed', 'victory', 'win', 'triumph', 'defeat', 'conquer',
            'accomplish', 'achieve', 'complete', 'finish'
        ]
        
        # Transformation markers
        self.transformation_markers = [
            'change', 'transform', 'become', 'grow', 'learn', 'realize',
            'understand', 'discover', 'accept', 'embrace', 'evolve'
        ]
        
        # Deus ex machina markers (unearned resolution)
        self.deus_ex_machina_markers = [
            'suddenly', 'coincidentally', 'luckily', 'fortunately', 'happen to',
            'just then', 'out of nowhere', 'miraculously', 'unexpectedly'
        ]
        
    def fit(self, X, y=None):
        """Fit transformer"""
        return self
    
    def transform(self, X):
        """Transform texts into conflict/tension features"""
        features = []
        
        # Ensure X is list of strings
        X = ensure_string_list(X)
        
        for text in X:
            # Ensure text is string
            text = str(text) if not isinstance(text, str) else text
            text_lower = text.lower()
            words = re.findall(r'\b\w+\b', text_lower)
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            feat_dict = {}
            
            # === 1. CONFLICT STRUCTURE (8 features) ===
            
            # Opposition presence
            opposition_count = sum(1 for w in words if w in self.opposition_markers)
            feat_dict['opposition_density'] = opposition_count / (len(words) + 1)
            
            # Internal vs external conflict ratio
            internal_count = sum(1 for w in words if w in self.internal_conflict_markers)
            external_count = sum(1 for w in words if w in self.external_conflict_markers)
            
            feat_dict['internal_conflict_density'] = internal_count / (len(words) + 1)
            feat_dict['external_conflict_density'] = external_count / (len(words) + 1)
            feat_dict['conflict_type_ratio'] = external_count / (internal_count + 1)
            
            # Conflict introduction timing (early vs late)
            conflict_words = set(self.opposition_markers + self.internal_conflict_markers + self.external_conflict_markers)
            first_conflict_pos = None
            for i, word in enumerate(words):
                if word in conflict_words:
                    first_conflict_pos = i / len(words)
                    break
            feat_dict['conflict_early_introduction'] = 1.0 - (first_conflict_pos if first_conflict_pos else 1.0)
            
            # Multiple conflict threads
            conflict_sentence_count = sum(1 for sent in sentences 
                                         if any(w in sent.lower() for w in conflict_words))
            feat_dict['conflict_thread_density'] = conflict_sentence_count / (len(sentences) + 1)
            
            # Protagonist agency (action verbs in first person)
            action_verbs = ['fight', 'struggle', 'try', 'attempt', 'work', 'push', 'force', 'make']
            first_person = ['i', 'we']
            agency_count = sum(1 for i, w in enumerate(words[:-1]) 
                             if w in first_person and words[i+1] in action_verbs)
            feat_dict['protagonist_agency'] = agency_count / (len(sentences) + 1)
            
            # Antagonist presence
            antagonist_words = ['enemy', 'villain', 'antagonist', 'opponent', 'rival', 'foe']
            feat_dict['antagonist_explicit'] = float(any(w in words for w in antagonist_words))
            
            # === 2. STAKES & URGENCY (6 features) ===
            
            # Consequence magnitude
            high_stakes_count = sum(1 for w in words if w in self.high_stakes_markers)
            feat_dict['stakes_magnitude'] = high_stakes_count / (len(words) + 1)
            
            # Time pressure
            urgency_count = sum(1 for w in words if w in self.urgency_markers)
            feat_dict['time_pressure'] = urgency_count / (len(words) + 1)
            
            # Loss/gain framing
            loss_words = ['lose', 'loss', 'fail', 'defeat', 'destroy', 'end']
            gain_words = ['win', 'gain', 'achieve', 'succeed', 'victory', 'triumph']
            loss_count = sum(1 for w in words if w in loss_words)
            gain_count = sum(1 for w in words if w in gain_words)
            
            feat_dict['loss_frame_density'] = loss_count / (len(words) + 1)
            feat_dict['gain_frame_density'] = gain_count / (len(words) + 1)
            feat_dict['loss_gain_ratio'] = loss_count / (gain_count + 1)
            
            # Irreversibility cues
            irreversible_count = sum(1 for w in words if w in self.irreversibility_markers)
            feat_dict['irreversibility'] = irreversible_count / (len(words) + 1)
            
            # === 3. TENSION BUILDING (8 features) ===
            
            # Escalation pattern
            escalation_count = sum(1 for w in words if w in self.escalation_markers)
            feat_dict['escalation_density'] = escalation_count / (len(words) + 1)
            
            # Obstacle density
            obstacle_count = sum(1 for w in words if w in self.obstacle_markers)
            feat_dict['obstacle_density'] = obstacle_count / (len(sentences) + 1)
            
            # Setback frequency
            setback_count = sum(1 for w in words if w in self.setback_markers)
            feat_dict['setback_frequency'] = setback_count / (len(sentences) + 1)
            
            # Complication introduction
            complication_count = sum(1 for w in words if w in self.complication_markers)
            feat_dict['complication_density'] = complication_count / (len(sentences) + 1)
            
            # Tension arc (increasing obstacles over time)
            if len(sentences) >= 3:
                third = len(sentences) // 3
                early_obstacles = sum(1 for sent in sentences[:third] 
                                    if any(w in sent.lower() for w in self.obstacle_markers))
                late_obstacles = sum(1 for sent in sentences[2*third:] 
                                   if any(w in sent.lower() for w in self.obstacle_markers))
                feat_dict['tension_rising'] = float(late_obstacles > early_obstacles)
            else:
                feat_dict['tension_rising'] = 0.5
            
            # Exclamation density (emotional intensity)
            exclamation_count = text.count('!')
            feat_dict['emotional_intensity'] = exclamation_count / (len(sentences) + 1)
            
            # Sentence length decrease (urgency building)
            if len(sentences) > 2:
                early_avg_length = np.mean([len(s.split()) for s in sentences[:len(sentences)//2]])
                late_avg_length = np.mean([len(s.split()) for s in sentences[len(sentences)//2:]])
                feat_dict['pacing_acceleration'] = float(late_avg_length < early_avg_length)
            else:
                feat_dict['pacing_acceleration'] = 0.5
            
            # Negative momentum
            negative_words = ['bad', 'worse', 'worst', 'fail', 'wrong', 'terrible', 'awful']
            feat_dict['negative_momentum'] = sum(1 for w in words if w in negative_words) / (len(words) + 1)
            
            # === 4. RESOLUTION QUALITY (6 features) ===
            
            # Resolution presence
            resolution_count = sum(1 for w in words if w in self.resolution_markers)
            feat_dict['resolution_present'] = float(resolution_count > 0)
            feat_dict['resolution_density'] = resolution_count / (len(words) + 1)
            
            # Transformation through conflict
            transformation_count = sum(1 for w in words if w in self.transformation_markers)
            feat_dict['transformation_present'] = transformation_count / (len(words) + 1)
            
            # Earned vs deus ex machina
            deus_ex_count = sum(1 for w in words if w in self.deus_ex_machina_markers)
            feat_dict['deus_ex_machina_risk'] = deus_ex_count / (len(sentences) + 1)
            
            # Resolution completeness (resolution in final third)
            if len(sentences) >= 3:
                final_third = sentences[len(sentences)*2//3:]
                resolution_in_final = sum(1 for sent in final_third 
                                        if any(w in sent.lower() for w in self.resolution_markers))
                feat_dict['resolution_placement'] = resolution_in_final / (len(final_third) + 1)
            else:
                feat_dict['resolution_placement'] = 0.0
            
            # Win/lose/compromise resolution
            final_quarter = words[len(words)*3//4:] if len(words) > 4 else words
            win_in_end = sum(1 for w in final_quarter if w in gain_words)
            lose_in_end = sum(1 for w in final_quarter if w in loss_words)
            
            if win_in_end > lose_in_end:
                feat_dict['resolution_type_win'] = 1.0
            elif lose_in_end > win_in_end:
                feat_dict['resolution_type_win'] = 0.0
            else:
                feat_dict['resolution_type_win'] = 0.5  # Compromise
            
            features.append(list(feat_dict.values()))
        
        return np.array(features)
    
    def get_feature_names_out(self, input_features=None):
        """Get feature names"""
        names = [
            # Conflict structure (8)
            'opposition_density', 'internal_conflict_density', 'external_conflict_density',
            'conflict_type_ratio', 'conflict_early_introduction', 'conflict_thread_density',
            'protagonist_agency', 'antagonist_explicit',
            
            # Stakes & urgency (6)
            'stakes_magnitude', 'time_pressure', 'loss_frame_density',
            'gain_frame_density', 'loss_gain_ratio', 'irreversibility',
            
            # Tension building (8)
            'escalation_density', 'obstacle_density', 'setback_frequency',
            'complication_density', 'tension_rising', 'emotional_intensity',
            'pacing_acceleration', 'negative_momentum',
            
            # Resolution (6)
            'resolution_present', 'resolution_density', 'transformation_present',
            'deus_ex_machina_risk', 'resolution_placement', 'resolution_type_win'
        ]
        
        return np.array([f'conflict_tension_{n}' for n in names])

