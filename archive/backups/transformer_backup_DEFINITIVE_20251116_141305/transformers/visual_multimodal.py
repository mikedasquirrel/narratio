"""
Visual/Multimodal Transformer

Extracts visual composition, image-text alignment, symbolic content, and modal integration.
Requires image processing infrastructure - enables entirely new domain categories.

Core insight: Visual narratives convey meaning through composition, color, symbols,
and their alignment (or tension) with textual narratives.

Note: This transformer can work in text-only mode (analyzing visual descriptions)
or with actual image inputs (requires PIL/CV2).
"""

import numpy as np
import re
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Dict, Any, Optional, Union


class VisualMultimodalTransformer(BaseEstimator, TransformerMixin):
    """
    Extract visual and multimodal features.
    
    Two modes:
    1. Text-only: Analyzes visual descriptions in text
    2. Image mode: Processes actual images (requires PIL)
    
    Captures:
    1. Visual Composition - color, balance, complexity
    2. Image-Text Alignment - consistency, complementarity
    3. Symbolic/Semantic Content - objects, scenes, emotions
    4. Multimodal Integration - reinforcement, synergy
    
    ~40+ features total
    """
    
    def __init__(self, mode='text_only', image_processor=None):
        """
        Initialize transformer
        
        Parameters
        ----------
        mode : str
            'text_only' - analyze visual descriptions in text
            'image' - process actual images (requires image_processor)
        image_processor : callable, optional
            Function that takes image path and returns features
        """
        self.mode = mode
        self.image_processor = image_processor
        
        # Color vocabulary
        self.color_words = {
            'warm': ['red', 'orange', 'yellow', 'gold', 'amber', 'crimson', 'scarlet'],
            'cool': ['blue', 'green', 'purple', 'cyan', 'teal', 'azure', 'violet'],
            'neutral': ['white', 'black', 'gray', 'grey', 'silver', 'beige', 'brown'],
            'bright': ['bright', 'vivid', 'brilliant', 'glowing', 'radiant', 'luminous'],
            'dark': ['dark', 'dim', 'shadowy', 'murky', 'gloomy', 'obscure']
        }
        
        # Visual composition markers
        self.composition_markers = [
            'center', 'centered', 'balanced', 'symmetrical', 'frame', 'framed',
            'foreground', 'background', 'perspective', 'angle', 'view',
            'composition', 'layout', 'arrangement', 'positioned'
        ]
        
        # Visual complexity markers
        self.complexity_markers = [
            'detailed', 'intricate', 'complex', 'elaborate', 'busy', 'crowded',
            'simple', 'minimal', 'clean', 'sparse', 'empty', 'plain'
        ]
        
        # Aesthetic style markers
        self.style_markers = {
            'modern': ['modern', 'contemporary', 'sleek', 'minimalist', 'clean'],
            'classic': ['classic', 'traditional', 'elegant', 'timeless', 'refined'],
            'artistic': ['artistic', 'creative', 'expressive', 'abstract', 'avant-garde'],
            'commercial': ['professional', 'polished', 'commercial', 'sleek', 'branded']
        }
        
        # Visual quality markers
        self.quality_markers = {
            'high': ['professional', 'high-quality', 'polished', 'crisp', 'sharp', 'clear'],
            'low': ['amateur', 'low-quality', 'blurry', 'grainy', 'poor', 'rough']
        }
        
        # Visual emotion markers
        self.visual_emotion_markers = {
            'positive': ['bright', 'light', 'warm', 'cheerful', 'uplifting', 'happy'],
            'negative': ['dark', 'gloomy', 'cold', 'harsh', 'sad', 'depressing']
        }
        
        # Object/scene markers (semantic content)
        self.scene_markers = {
            'indoor': ['room', 'house', 'building', 'office', 'interior', 'inside'],
            'outdoor': ['outside', 'landscape', 'nature', 'sky', 'outdoor', 'exterior'],
            'urban': ['city', 'street', 'building', 'urban', 'downtown', 'metropolitan'],
            'natural': ['nature', 'forest', 'mountain', 'ocean', 'landscape', 'wilderness']
        }
        
        # Symbol markers
        self.symbolic_markers = [
            'symbol', 'symbolic', 'represent', 'signify', 'metaphor', 'emblem',
            'icon', 'imagery', 'motif', 'allegory', 'meaning'
        ]
        
        # Modal alignment markers
        self.alignment_markers = {
            'reinforcement': ['shows', 'depicts', 'illustrates', 'demonstrates', 'reflects'],
            'contrast': ['however', 'but', 'while', 'whereas', 'despite', 'although'],
            'complement': ['also', 'additionally', 'furthermore', 'moreover', 'along with']
        }
        
    def fit(self, X, y=None):
        """Fit transformer"""
        return self
    
    def transform(self, X, images=None):
        """
        Transform texts (and optionally images) into multimodal features
        
        Parameters
        ----------
        X : array-like of strings
            Narrative texts
        images : array-like of image paths or arrays, optional
            Images to analyze (if mode='image')
            
        Returns
        -------
        features : ndarray
            Visual/multimodal features
        """
        features = []
        
        for i, text in enumerate(X):
            text_lower = text.lower()
            words = re.findall(r'\b\w+\b', text_lower)
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            feat_dict = {}
            
            # === 1. VISUAL COMPOSITION (12 features) ===
            
            # Color palette (from text description)
            color_counts = {}
            for category, color_list in self.color_words.items():
                count = sum(1 for w in words if w in color_list)
                color_counts[category] = count
            
            total_color_words = sum(color_counts.values())
            
            # Color palette features
            feat_dict['color_richness'] = total_color_words / (len(words) + 1)
            feat_dict['warm_color_dominance'] = color_counts['warm'] / (total_color_words + 1)
            feat_dict['cool_color_dominance'] = color_counts['cool'] / (total_color_words + 1)
            feat_dict['color_contrast'] = abs(color_counts['bright'] - color_counts['dark']) / (total_color_words + 1)
            
            # Composition mentions
            composition_count = sum(1 for w in words if w in self.composition_markers)
            feat_dict['composition_awareness'] = composition_count / (len(words) + 1)
            
            # Visual complexity
            complex_words = [w for w in words if w in self.complexity_markers[:6]]  # Complex side
            simple_words = [w for w in words if w in self.complexity_markers[6:]]  # Simple side
            feat_dict['visual_complexity'] = (len(complex_words) - len(simple_words)) / (len(words) + 1)
            
            # Aesthetic style
            style_scores = {}
            for style, markers in self.style_markers.items():
                count = sum(1 for w in words if w in markers)
                style_scores[style] = count
            
            dominant_style = max(style_scores, key=style_scores.get) if style_scores else 'modern'
            feat_dict['aesthetic_style_score'] = style_scores.get(dominant_style, 0) / (len(words) + 1)
            
            # Visual quality signals
            high_quality = sum(1 for w in words if w in self.quality_markers['high'])
            low_quality = sum(1 for w in words if w in self.quality_markers['low'])
            feat_dict['visual_quality_signal'] = (high_quality - low_quality) / (high_quality + low_quality + 1)
            
            # Visual description density
            visual_description_words = (list(self.composition_markers) + 
                                       [c for colors in self.color_words.values() for c in colors])
            visual_desc_count = sum(1 for w in words if w in visual_description_words)
            feat_dict['visual_description_density'] = visual_desc_count / (len(words) + 1)
            
            # Visual specificity (specific visual details)
            visual_detail_patterns = [r'shade of', r'tone of', r'hue', r'texture', r'pattern']
            visual_details = sum(len(re.findall(p, text_lower)) for p in visual_detail_patterns)
            feat_dict['visual_specificity'] = visual_details / (len(sentences) + 1)
            
            # Balance/symmetry mentions
            balance_words = ['balance', 'balanced', 'symmetry', 'symmetrical', 'centered', 'even']
            balance_count = sum(1 for w in words if w in balance_words)
            feat_dict['composition_balance'] = balance_count / (len(words) + 1)
            
            # === 2. IMAGE-TEXT ALIGNMENT (8 features) ===
            
            # Visual-verbal consistency (how much text describes visuals)
            feat_dict['visual_verbal_consistency'] = feat_dict['visual_description_density']
            
            # Complementarity (text adds to image or vice versa)
            complement_count = sum(1 for marker in self.alignment_markers['complement'] if marker in text_lower)
            feat_dict['modal_complementarity'] = complement_count / (len(sentences) + 1)
            
            # Redundancy (text just restates image)
            restate_markers = ['as shown', 'pictured', 'seen', 'visible', 'displayed']
            restate_count = sum(1 for marker in restate_markers if marker in text_lower)
            feat_dict['modal_redundancy'] = restate_count / (len(sentences) + 1)
            
            # Visual reference in text
            visual_ref_markers = ['see', 'look', 'notice', 'observe', 'watch', 'view', 'image', 'picture']
            visual_ref_count = sum(1 for w in words if w in visual_ref_markers)
            feat_dict['visual_reference_density'] = visual_ref_count / (len(words) + 1)
            
            # Professional vs amateur (quality signals)
            feat_dict['professional_quality'] = high_quality / (len(words) + 1)
            feat_dict['amateur_signals'] = low_quality / (len(words) + 1)
            
            # Visual narrative pacing (visual action words)
            visual_action = ['zoom', 'pan', 'focus', 'shift', 'transition', 'cut', 'fade']
            visual_action_count = sum(1 for w in words if w in visual_action)
            feat_dict['visual_pacing'] = visual_action_count / (len(sentences) + 1)
            
            # Ekphrasis (detailed visual description)
            ekphrasis_score = feat_dict['visual_specificity'] * feat_dict['visual_description_density']
            feat_dict['ekphrasis_quality'] = ekphrasis_score
            
            # === 3. SYMBOLIC/SEMANTIC CONTENT (10 features) ===
            
            # Object presence (concrete nouns)
            concrete_nouns = ['man', 'woman', 'child', 'house', 'car', 'tree', 'door', 
                            'hand', 'face', 'eye', 'table', 'chair', 'window', 'light']
            object_count = sum(1 for w in words if w in concrete_nouns)
            feat_dict['object_density'] = object_count / (len(words) + 1)
            
            # Scene understanding (setting markers)
            scene_counts = {}
            for scene_type, markers in self.scene_markers.items():
                count = sum(1 for w in words if w in markers)
                scene_counts[scene_type] = count
            
            total_scene_markers = sum(scene_counts.values())
            feat_dict['scene_specificity'] = total_scene_markers / (len(words) + 1)
            
            # Indoor vs outdoor
            if scene_counts['indoor'] + scene_counts['outdoor'] > 0:
                feat_dict['setting_indoor_ratio'] = scene_counts['indoor'] / (scene_counts['indoor'] + scene_counts['outdoor'])
            else:
                feat_dict['setting_indoor_ratio'] = 0.5
            
            # Urban vs natural
            if scene_counts['urban'] + scene_counts['natural'] > 0:
                feat_dict['setting_urban_ratio'] = scene_counts['urban'] / (scene_counts['urban'] + scene_counts['natural'])
            else:
                feat_dict['setting_urban_ratio'] = 0.5
            
            # Emotional tone of visuals (from descriptive words)
            positive_visual = sum(1 for w in words if w in self.visual_emotion_markers['positive'])
            negative_visual = sum(1 for w in words if w in self.visual_emotion_markers['negative'])
            feat_dict['visual_emotional_valence'] = (positive_visual - negative_visual) / (positive_visual + negative_visual + 1)
            
            # Symbolic language
            symbolic_count = sum(1 for w in words if w in self.symbolic_markers)
            feat_dict['symbolic_density'] = symbolic_count / (len(words) + 1)
            
            # Cultural symbols (flags, monuments, iconic objects)
            cultural_symbol_words = ['flag', 'monument', 'statue', 'landmark', 'icon', 'sacred']
            cultural_symbol_count = sum(1 for w in words if w in cultural_symbol_words)
            feat_dict['cultural_symbol_presence'] = cultural_symbol_count / (len(words) + 1)
            
            # Character presence (people in scene)
            character_markers = ['man', 'woman', 'person', 'people', 'character', 'figure', 'face']
            character_count = sum(1 for w in words if w in character_markers)
            feat_dict['character_presence'] = character_count / (len(words) + 1)
            
            # Action vs static scene
            action_verbs = ['run', 'jump', 'fight', 'chase', 'move', 'dance', 'fly']
            static_verbs = ['stand', 'sit', 'lie', 'rest', 'stay', 'remain', 'wait']
            action_count = sum(1 for w in words if w in action_verbs)
            static_count = sum(1 for w in words if w in static_verbs)
            feat_dict['visual_dynamism'] = action_count / (static_count + 1)
            
            # Scale markers (intimate vs epic)
            intimate_markers = ['close', 'near', 'intimate', 'personal', 'detail', 'micro']
            epic_markers = ['vast', 'enormous', 'epic', 'grand', 'massive', 'panoramic']
            intimate_count = sum(1 for w in words if w in intimate_markers)
            epic_count = sum(1 for w in words if w in epic_markers)
            feat_dict['visual_scale'] = (epic_count - intimate_count) / (epic_count + intimate_count + 1)
            
            # === 4. MULTIMODAL INTEGRATION (10 features) ===
            
            # Cross-modal reinforcement
            reinforcement_count = sum(1 for marker in self.alignment_markers['reinforcement'] if marker in text_lower)
            feat_dict['modal_reinforcement'] = reinforcement_count / (len(sentences) + 1)
            
            # Modal contrast (text vs image convey different things)
            contrast_count = sum(1 for marker in self.alignment_markers['contrast'] if marker in text_lower)
            feat_dict['modal_contrast'] = contrast_count / (len(sentences) + 1)
            
            # Modal dominance (which mode carries more information)
            text_info_density = len(words) / (len(sentences) + 1)
            visual_info_density = feat_dict['visual_description_density'] * 100
            
            if text_info_density + visual_info_density > 0:
                feat_dict['modal_dominance_text'] = text_info_density / (text_info_density + visual_info_density)
            else:
                feat_dict['modal_dominance_text'] = 0.5
            
            # Information distribution (balanced vs concentrated)
            feat_dict['information_distribution_balance'] = 1.0 - abs(feat_dict['modal_dominance_text'] - 0.5)
            
            # Synergy (modes work together)
            # High visual description + high narrative = synergy
            feat_dict['multimodal_synergy'] = feat_dict['visual_description_density'] * (len(words) / 100)
            feat_dict['multimodal_synergy'] = min(1.0, feat_dict['multimodal_synergy'])
            
            # Modal conflict (contradictory signals)
            # Positive text + negative visual or vice versa
            text_positive = sum(1 for w in words if w in ['good', 'great', 'excellent', 'wonderful'])
            text_negative = sum(1 for w in words if w in ['bad', 'terrible', 'awful', 'horrible'])
            
            text_valence = (text_positive - text_negative) / (text_positive + text_negative + 1)
            visual_valence = feat_dict['visual_emotional_valence']
            
            feat_dict['modal_conflict'] = abs(text_valence - visual_valence)
            
            # Cohesion (all modes tell coherent story)
            feat_dict['multimodal_cohesion'] = 1.0 - feat_dict['modal_conflict']
            
            # Integration sophistication (explicit linking)
            linking_words = ['as shown', 'pictured', 'illustrated', 'depicted', 'image shows']
            linking_count = sum(1 for phrase in linking_words if phrase in text_lower)
            feat_dict['modal_integration_explicit'] = linking_count / (len(sentences) + 1)
            
            # Visual narrative (story told through visuals)
            feat_dict['visual_narrative_strength'] = feat_dict['visual_description_density'] * feat_dict['visual_dynamism']
            
            # Accessibility (text describes visuals for those who can't see)
            accessibility_markers = ['appears', 'looks like', 'shows', 'features', 'contains']
            accessibility_count = sum(1 for w in words if w in accessibility_markers)
            feat_dict['visual_accessibility'] = accessibility_count / (len(sentences) + 1)
            
            features.append(list(feat_dict.values()))
        
        return np.array(features)
    
    def transform_with_images(self, texts, image_paths):
        """
        Transform with actual image inputs (requires image processor)
        
        Parameters
        ----------
        texts : array-like of strings
            Narrative texts
        image_paths : array-like of paths
            Paths to images
            
        Returns
        -------
        features : ndarray
            Combined text + image features
        """
        if self.mode != 'image' or self.image_processor is None:
            raise ValueError("Image mode requires image_processor to be set")
        
        # Get text-based features
        text_features = self.transform(texts)
        
        # Get image features
        image_features = []
        for img_path in image_paths:
            img_feats = self.image_processor(img_path)
            image_features.append(img_feats)
        
        image_features = np.array(image_features)
        
        # Combine
        combined = np.hstack([text_features, image_features])
        
        return combined
    
    def get_feature_names_out(self, input_features=None):
        """Get feature names"""
        names = [
            # Visual composition (12)
            'color_richness', 'warm_color_dominance', 'cool_color_dominance',
            'color_contrast', 'composition_awareness', 'visual_complexity',
            'aesthetic_style_score', 'visual_quality_signal', 'visual_description_density',
            'visual_specificity', 'composition_balance',
            
            # Image-text alignment (8)
            'visual_verbal_consistency', 'modal_complementarity', 'modal_redundancy',
            'visual_reference_density', 'professional_quality', 'amateur_signals',
            'visual_pacing', 'ekphrasis_quality',
            
            # Symbolic/semantic (10)
            'object_density', 'scene_specificity', 'setting_indoor_ratio',
            'setting_urban_ratio', 'visual_emotional_valence', 'symbolic_density',
            'cultural_symbol_presence', 'character_presence', 'visual_dynamism',
            'visual_scale',
            
            # Multimodal integration (10)
            'modal_reinforcement', 'modal_contrast', 'modal_dominance_text',
            'information_distribution_balance', 'multimodal_synergy', 'modal_conflict',
            'multimodal_cohesion', 'modal_integration_explicit', 'visual_narrative_strength',
            'visual_accessibility'
        ]
        
        return np.array([f'visual_multimodal_{n}' for n in names])

