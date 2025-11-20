"""
Suspense/Mystery Transformer

Extracts information control, curiosity gaps, mystery structure, and suspense mechanics.
Specialized for genres/domains where withholding and revealing information is core.

Core insight: Compelling narratives manage information flow to create curiosity and anticipation.
"""

import numpy as np
import re
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin


class SuspenseMysteryTransformer(BaseEstimator, TransformerMixin):
    """
    Extract suspense and mystery features from narrative text.
    
    Captures:
    1. Information Control - questions vs answers, withholding patterns
    2. Curiosity Gaps - unanswered questions, intrigue, anticipation
    3. Mystery Structure - red herrings, clues, foreshadowing
    4. Suspense Mechanics - dramatic irony, tension, uncertainty
    
    ~25 features total
    """
    
    def __init__(self):
        """Initialize suspense/mystery markers"""
        
        # Question markers
        self.question_words = [
            'who', 'what', 'where', 'when', 'why', 'how', 'which', 'whose',
            'wonder', 'curious', 'question', 'mystery', 'unknown', 'unclear'
        ]
        
        # Answer/revelation markers
        self.answer_markers = [
            'answer', 'explain', 'because', 'reason', 'revealed', 'discovered',
            'found', 'learned', 'understand', 'realized', 'turns out', 'actually'
        ]
        
        # Mystery setup markers
        self.mystery_setup_markers = [
            'mystery', 'secret', 'hidden', 'unknown', 'enigma', 'puzzle',
            'riddle', 'disappear', 'vanish', 'strange', 'unusual', 'odd',
            'curious', 'mysterious', 'cryptic', 'suspicious'
        ]
        
        # Clue/evidence markers
        self.clue_markers = [
            'clue', 'evidence', 'hint', 'sign', 'trace', 'track', 'lead',
            'discover', 'find', 'notice', 'observe', 'detect', 'uncover'
        ]
        
        # Red herring/misdirection markers
        self.misdirection_markers = [
            'but', 'however', 'actually', 'really', 'turns out', 'realized',
            'thought', 'assumed', 'believed', 'seemed', 'appeared', 'wrong'
        ]
        
        # Foreshadowing markers
        self.foreshadowing_markers = [
            'later', 'soon', 'eventually', 'would', 'little did', 'if only',
            'ominous', 'foreboding', 'hint', 'suggest', 'imply'
        ]
        
        # Intrigue generation
        self.intrigue_markers = [
            'intriguing', 'fascinating', 'compelling', 'curious', 'interesting',
            'strange', 'unusual', 'unexpected', 'surprising', 'shocking'
        ]
        
        # Cliffhanger markers
        self.cliffhanger_markers = [
            'suddenly', 'then', 'just then', 'at that moment', 'before',
            'too late', 'but', 'however', 'until', 'when'
        ]
        
        # Anticipation building
        self.anticipation_markers = [
            'about to', 'going to', 'will', 'soon', 'coming', 'approaching',
            'imminent', 'near', 'almost', 'ready', 'prepare', 'expect'
        ]
        
        # Dramatic irony (audience knows, character doesn't)
        self.dramatic_irony_markers = [
            'little did', 'if only', 'unaware', 'didn\'t know', 'didn\'t realize',
            'unbeknownst', 'meanwhile', 'unknown to', 'secretly', 'hidden'
        ]
        
        # Tension markers
        self.tension_markers = [
            'tense', 'nervous', 'worried', 'anxious', 'afraid', 'fear',
            'danger', 'threat', 'risk', 'careful', 'cautious', 'hesitate'
        ]
        
        # Uncertainty markers
        self.uncertainty_markers = [
            'uncertain', 'unsure', 'unclear', 'unknown', 'maybe', 'perhaps',
            'possibly', 'might', 'could', 'wonder', 'question', 'doubt'
        ]
        
    def fit(self, X, y=None):
        """Fit transformer"""
        return self
    
    def transform(self, X):
        """Transform texts into suspense/mystery features"""
        features = []
        
        for text in X:
            text_lower = text.lower()
            words = re.findall(r'\b\w+\b', text_lower)
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            feat_dict = {}
            
            # === 1. INFORMATION CONTROL (7 features) ===
            
            # Questions raised
            question_count = text.count('?')
            question_word_count = sum(1 for w in words if w in self.question_words)
            feat_dict['question_density'] = question_count / (len(sentences) + 1)
            feat_dict['question_word_density'] = question_word_count / (len(words) + 1)
            
            # Answers provided
            answer_count = sum(1 for w in words if w in self.answer_markers)
            feat_dict['answer_density'] = answer_count / (len(words) + 1)
            
            # Question-answer ratio (high = more questions than answers = mystery)
            feat_dict['question_answer_ratio'] = (question_count + question_word_count) / (answer_count + 1)
            
            # Information withholding
            withholding_words = ['secret', 'hidden', 'concealed', 'withheld', 'unrevealed']
            withholding_count = sum(1 for w in words if w in withholding_words)
            feat_dict['information_withholding'] = withholding_count / (len(words) + 1)
            
            # Mystery setup quality
            mystery_count = sum(1 for w in words if w in self.mystery_setup_markers)
            feat_dict['mystery_setup'] = mystery_count / (len(words) + 1)
            
            # Reveal timing (early vs late answers)
            if answer_count > 0:
                # Find position of last answer marker
                last_answer_pos = None
                for i, w in enumerate(reversed(words)):
                    if w in self.answer_markers:
                        last_answer_pos = (len(words) - i) / len(words)
                        break
                feat_dict['reveal_timing_late'] = last_answer_pos if last_answer_pos else 0.0
            else:
                feat_dict['reveal_timing_late'] = 0.0
            
            # === 2. CURIOSITY GAPS (6 features) ===
            
            # Unanswered question density
            feat_dict['unanswered_questions'] = feat_dict['question_density'] * feat_dict['question_answer_ratio']
            
            # Intrigue generation
            intrigue_count = sum(1 for w in words if w in self.intrigue_markers)
            feat_dict['intrigue_generation'] = intrigue_count / (len(words) + 1)
            
            # Cliffhanger usage (at sentence/section ends)
            if len(sentences) > 0:
                # Check if last sentence has cliffhanger markers
                last_sent = sentences[-1].lower()
                cliffhanger_in_end = any(m in last_sent for m in self.cliffhanger_markers)
                feat_dict['cliffhanger_ending'] = float(cliffhanger_in_end)
            else:
                feat_dict['cliffhanger_ending'] = 0.0
            
            # Anticipation building
            anticipation_count = sum(1 for w in words if w in self.anticipation_markers)
            feat_dict['anticipation_building'] = anticipation_count / (len(words) + 1)
            
            # Unresolved tension (tension markers without resolution)
            tension_count = sum(1 for w in words if w in self.tension_markers)
            resolution_words = ['resolved', 'solved', 'answered', 'explained', 'settled']
            resolution_count = sum(1 for w in words if w in resolution_words)
            feat_dict['unresolved_tension'] = tension_count / (resolution_count + 1)
            
            # Enigmatic language (ambiguous, cryptic)
            enigmatic_words = ['enigmatic', 'cryptic', 'ambiguous', 'elusive', 'mysterious']
            enigmatic_count = sum(1 for w in words if w in enigmatic_words)
            feat_dict['enigmatic_language'] = enigmatic_count / (len(words) + 1)
            
            # === 3. MYSTERY STRUCTURE (6 features) ===
            
            # Red herrings/misdirection
            misdirection_count = sum(1 for w in words if w in self.misdirection_markers)
            feat_dict['misdirection_density'] = misdirection_count / (len(sentences) + 1)
            
            # Clue planting
            clue_count = sum(1 for w in words if w in self.clue_markers)
            feat_dict['clue_density'] = clue_count / (len(sentences) + 1)
            
            # Foreshadowing
            foreshadow_count = sum(1 for w in words if w in self.foreshadowing_markers)
            feat_dict['foreshadowing'] = foreshadow_count / (len(words) + 1)
            
            # Puzzle complexity (multiple mysteries)
            mystery_threads = sum(1 for sent in sentences 
                                if any(w in sent.lower() for w in self.mystery_setup_markers))
            feat_dict['puzzle_complexity'] = mystery_threads / (len(sentences) + 1)
            
            # Revelation pattern (gradual vs sudden)
            if len(sentences) >= 3:
                third = len(sentences) // 3
                early_reveals = sum(1 for sent in sentences[:third] 
                                  if any(w in sent.lower() for w in self.answer_markers))
                late_reveals = sum(1 for sent in sentences[2*third:] 
                                 if any(w in sent.lower() for w in self.answer_markers))
                feat_dict['revelation_gradual'] = float(early_reveals > 0 and late_reveals > 0)
            else:
                feat_dict['revelation_gradual'] = 0.0
            
            # Mystery layering (nested mysteries)
            mystery_depth = max(mystery_threads, question_count)
            feat_dict['mystery_layering'] = mystery_depth / (len(sentences) + 1)
            
            # === 4. SUSPENSE MECHANICS (6 features) ===
            
            # Dramatic irony presence
            dramatic_irony_count = sum(1 for marker in self.dramatic_irony_markers if marker in text_lower)
            feat_dict['dramatic_irony'] = dramatic_irony_count / (len(sentences) + 1)
            
            # Tension without resolution
            feat_dict['sustained_tension'] = tension_count / (len(sentences) + 1)
            
            # Deadline pressure
            deadline_words = ['deadline', 'time', 'running out', 'too late', 'before', 'by', 'until']
            deadline_count = sum(1 for w in words if w in deadline_words)
            feat_dict['deadline_pressure'] = deadline_count / (len(words) + 1)
            
            # Uncertainty maintenance
            uncertainty_count = sum(1 for w in words if w in self.uncertainty_markers)
            feat_dict['uncertainty_maintenance'] = uncertainty_count / (len(words) + 1)
            
            # Sentence rhythm (short sentences = tension)
            if len(sentences) > 1:
                sentence_lengths = [len(s.split()) for s in sentences]
                avg_length = np.mean(sentence_lengths)
                feat_dict['tension_pacing'] = float(avg_length < 15)  # Short sentences = tension
            else:
                feat_dict['tension_pacing'] = 0.5
            
            # Ellipsis/suspension usage (trailing off...)
            ellipsis_count = text.count('...')
            feat_dict['narrative_suspension'] = ellipsis_count / (len(sentences) + 1)
            
            features.append(list(feat_dict.values()))
        
        return np.array(features)
    
    def get_feature_names_out(self, input_features=None):
        """Get feature names"""
        names = [
            # Information control (7)
            'question_density', 'question_word_density', 'answer_density',
            'question_answer_ratio', 'information_withholding', 'mystery_setup',
            'reveal_timing_late',
            
            # Curiosity gaps (6)
            'unanswered_questions', 'intrigue_generation', 'cliffhanger_ending',
            'anticipation_building', 'unresolved_tension', 'enigmatic_language',
            
            # Mystery structure (6)
            'misdirection_density', 'clue_density', 'foreshadowing',
            'puzzle_complexity', 'revelation_gradual', 'mystery_layering',
            
            # Suspense mechanics (6)
            'dramatic_irony', 'sustained_tension', 'deadline_pressure',
            'uncertainty_maintenance', 'tension_pacing', 'narrative_suspension'
        ]
        
        return np.array([f'suspense_mystery_{n}' for n in names])

