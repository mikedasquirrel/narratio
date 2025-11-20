"""
Optics Transformer

Captures visual and aesthetic narrative elements including colors, 
appearance language, design terminology, and visual metaphors.

Research Foundation:
- Color psychology: Red increases perceived aggression/urgency
- Aesthetic fluency: Beautiful = good heuristic
- Visual language activates different neural pathways than abstract
- Design language signals quality/sophistication

Universal across domains:
- Sports: Team colors, uniform aesthetics
- Products: Design, appearance, visual appeal
- Brands: Visual identity, color schemes
- Profiles: Appearance descriptions (when present)
"""

from typing import List, Dict, Any, Set
import numpy as np
import re
from collections import Counter
from .base import NarrativeTransformer


class OpticsTransformer(NarrativeTransformer):
    """
    Analyzes visual and aesthetic narrative elements.
    
    Captures how things LOOK and the visual/aesthetic language used
    to describe them. Tests hypothesis that optics matter independent
    of functional features.
    
    Features extracted (15):
    - Color references (specific colors)
    - Color families (warm vs. cool)
    - Visual appearance words (sleek, rugged, elegant, bold)
    - Aesthetic markers (beautiful, stunning, plain, ugly)
    - Design terminology (modern, classic, vintage, contemporary)
    - Visual metaphors (shining, glowing, dark, bright)
    - Material references (metal, wood, glass, fabric)
    - Size/scale language (massive, tiny, compact, grand)
    - Visual clarity (clear, sharp, blurry, fuzzy)
    - Symmetry/balance language
    
    Parameters
    ----------
    track_color_psychology : bool
        Whether to compute psychological color associations
    """
    
    def __init__(self, track_color_psychology: bool = True):
        super().__init__(
            narrative_id="optics",
            description="Visual and aesthetic narrative elements: how things look and are described visually"
        )
        
        self.track_color_psychology = track_color_psychology
        
        # Color vocabularies
        self.colors = {
            'red': ['red', 'crimson', 'scarlet', 'ruby', 'cherry', 'burgundy', 'maroon'],
            'blue': ['blue', 'navy', 'azure', 'cobalt', 'cyan', 'sapphire', 'indigo'],
            'green': ['green', 'emerald', 'jade', 'lime', 'olive', 'forest', 'mint'],
            'yellow': ['yellow', 'gold', 'amber', 'lemon', 'golden', 'sunshine', 'mustard'],
            'orange': ['orange', 'tangerine', 'coral', 'peach', 'apricot', 'rust'],
            'purple': ['purple', 'violet', 'lavender', 'plum', 'magenta', 'lilac', 'mauve'],
            'black': ['black', 'ebony', 'jet', 'coal', 'obsidian', 'onyx', 'midnight'],
            'white': ['white', 'ivory', 'pearl', 'snow', 'alabaster', 'cream', 'vanilla'],
            'gray': ['gray', 'grey', 'silver', 'slate', 'ash', 'charcoal', 'pewter'],
            'brown': ['brown', 'tan', 'beige', 'khaki', 'bronze', 'copper', 'chocolate']
        }
        
        # Color psychology associations
        self.color_psychology = {
            'red': 'aggressive',      # Aggression, urgency, power, passion
            'blue': 'calm',           # Trust, calm, stability, authority
            'green': 'natural',       # Growth, nature, balance, harmony
            'yellow': 'energetic',    # Energy, optimism, attention, caution
            'orange': 'vibrant',      # Enthusiasm, creativity, vitality
            'purple': 'prestigious',  # Royalty, luxury, wisdom, creativity
            'black': 'powerful',      # Power, sophistication, elegance, mystery
            'white': 'pure',          # Purity, simplicity, cleanliness
            'gray': 'neutral',        # Neutrality, balance, sophistication
            'brown': 'earthy'         # Stability, reliability, earthiness
        }
        
        # Warm vs. cool colors
        self.warm_colors = {'red', 'orange', 'yellow', 'brown'}
        self.cool_colors = {'blue', 'green', 'purple', 'gray'}
        
        # Aesthetic appearance vocabulary
        self.appearance_positive = [
            'beautiful', 'stunning', 'gorgeous', 'elegant', 'sleek', 'stylish',
            'attractive', 'handsome', 'graceful', 'refined', 'polished', 'sophisticated',
            'striking', 'impressive', 'magnificent', 'exquisite', 'lovely', 'pretty',
            'chic', 'classy', 'tasteful', 'aesthetic'
        ]
        
        self.appearance_negative = [
            'ugly', 'plain', 'dull', 'bland', 'boring', 'unattractive', 'awkward',
            'clunky', 'crude', 'rough', 'unrefined', 'tacky', 'gaudy', 'garish',
            'hideous', 'unsightly', 'shabby'
        ]
        
        self.appearance_neutral = [
            'simple', 'minimal', 'basic', 'standard', 'ordinary', 'typical',
            'conventional', 'traditional', 'classic', 'timeless'
        ]
        
        # Design terminology
        self.design_modern = [
            'modern', 'contemporary', 'futuristic', 'cutting-edge', 'innovative',
            'progressive', 'forward-thinking', 'avant-garde', 'minimalist', 'streamlined'
        ]
        
        self.design_classic = [
            'classic', 'traditional', 'vintage', 'retro', 'timeless', 'heritage',
            'old-school', 'nostalgic', 'antique', 'historic'
        ]
        
        # Visual quality descriptors
        self.visual_descriptors = {
            'bold': ['bold', 'striking', 'dramatic', 'vivid', 'intense', 'strong'],
            'subtle': ['subtle', 'understated', 'muted', 'soft', 'gentle', 'delicate'],
            'bright': ['bright', 'brilliant', 'vibrant', 'luminous', 'radiant', 'glowing'],
            'dark': ['dark', 'dim', 'shadowy', 'murky', 'gloomy', 'somber'],
            'clear': ['clear', 'crisp', 'sharp', 'defined', 'distinct', 'precise'],
            'blurry': ['blurry', 'fuzzy', 'hazy', 'vague', 'indistinct', 'cloudy']
        }
        
        # Materials
        self.materials = {
            'metal': ['metal', 'steel', 'iron', 'aluminum', 'chrome', 'titanium', 'brass', 'copper'],
            'wood': ['wood', 'wooden', 'oak', 'pine', 'mahogany', 'cedar', 'bamboo'],
            'glass': ['glass', 'crystal', 'transparent', 'translucent'],
            'fabric': ['fabric', 'cloth', 'textile', 'leather', 'cotton', 'silk', 'velvet'],
            'stone': ['stone', 'marble', 'granite', 'concrete', 'brick', 'ceramic'],
            'plastic': ['plastic', 'synthetic', 'polymer', 'acrylic']
        }
        
        # Size/scale
        self.size_large = ['massive', 'huge', 'enormous', 'giant', 'grand', 'vast', 'colossal', 'monumental']
        self.size_small = ['tiny', 'small', 'compact', 'miniature', 'petite', 'diminutive', 'little']
        
        # Visual metaphors
        self.visual_metaphors = {
            'light': ['shining', 'glowing', 'radiant', 'luminous', 'sparkling', 'gleaming'],
            'dark': ['shadowy', 'murky', 'gloomy', 'dim', 'obscure'],
            'smooth': ['smooth', 'silky', 'sleek', 'polished', 'glossy'],
            'rough': ['rough', 'rugged', 'coarse', 'textured', 'grainy']
        }
    
    def fit(self, X, y=None):
        """
        Learn optics patterns from corpus.
        
        Parameters
        ----------
        X : list of str
            Text documents
        y : ignored
        
        Returns
        -------
        self
        """
        # Analyze corpus-level visual language
        all_colors = []
        all_aesthetics = []
        
        for text in X:
            text_lower = text.lower()
            
            # Track color usage
            for color_family, color_words in self.colors.items():
                for color_word in color_words:
                    if color_word in text_lower:
                        all_colors.append(color_family)
            
            # Track aesthetic language
            for word in self.appearance_positive + self.appearance_negative + self.appearance_neutral:
                if word in text_lower:
                    all_aesthetics.append(word)
        
        # Metadata
        if all_colors:
            self.metadata['color_distribution'] = dict(Counter(all_colors).most_common(10))
            self.metadata['most_common_color'] = Counter(all_colors).most_common(1)[0] if all_colors else None
        
        if all_aesthetics:
            self.metadata['aesthetic_words'] = dict(Counter(all_aesthetics).most_common(20))
        
        self.metadata['has_visual_language'] = len(all_colors) > 0 or len(all_aesthetics) > 0
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Transform documents to optics features.
        
        Parameters
        ----------
        X : list of str
            Documents to transform
        
        Returns
        -------
        features : array, shape (n_samples, 15)
            Optics feature matrix
        """
        self._validate_fitted()
        
        features_list = []
        
        for text in X:
            doc_features = self._extract_optics_features(text)
            features_list.append(doc_features)
        
        return np.array(features_list)
    
    def _extract_optics_features(self, text: str) -> np.ndarray:
        """Extract all 15 optics features from text."""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        word_count = len(words) if words else 1
        
        features = []
        
        # 1. Color density (total color references per 100 words)
        color_count = 0
        for color_family, color_words in self.colors.items():
            for color_word in color_words:
                color_count += text_lower.count(color_word)
        features.append(color_count / word_count * 100)
        
        # 2. Warm color ratio (warm / (warm + cool))
        warm_count = sum(
            sum(text_lower.count(cw) for cw in self.colors[color])
            for color in self.warm_colors
        )
        cool_count = sum(
            sum(text_lower.count(cw) for cw in self.colors[color])
            for color in self.cool_colors
        )
        total_temp_colors = warm_count + cool_count
        features.append(warm_count / total_temp_colors if total_temp_colors > 0 else 0.5)
        
        # 3. Aggressive colors (red, black) density
        aggressive_count = (
            sum(text_lower.count(w) for w in self.colors['red']) +
            sum(text_lower.count(w) for w in self.colors['black'])
        )
        features.append(aggressive_count / word_count * 100)
        
        # 4. Calm colors (blue, green) density
        calm_count = (
            sum(text_lower.count(w) for w in self.colors['blue']) +
            sum(text_lower.count(w) for w in self.colors['green'])
        )
        features.append(calm_count / word_count * 100)
        
        # 5. Aesthetic valence (positive - negative)
        pos_count = sum(text_lower.count(w) for w in self.appearance_positive)
        neg_count = sum(text_lower.count(w) for w in self.appearance_negative)
        features.append((pos_count - neg_count) / word_count * 100)
        
        # 6. Aesthetic intensity (total aesthetic references)
        aesthetic_total = pos_count + neg_count + sum(text_lower.count(w) for w in self.appearance_neutral)
        features.append(aesthetic_total / word_count * 100)
        
        # 7. Modern vs. classic design (ratio)
        modern_count = sum(text_lower.count(w) for w in self.design_modern)
        classic_count = sum(text_lower.count(w) for w in self.design_classic)
        total_design = modern_count + classic_count
        features.append(modern_count / total_design if total_design > 0 else 0.5)
        
        # 8. Bold vs. subtle (ratio)
        bold_count = sum(text_lower.count(w) for w in self.visual_descriptors['bold'])
        subtle_count = sum(text_lower.count(w) for w in self.visual_descriptors['subtle'])
        total_intensity = bold_count + subtle_count
        features.append(bold_count / total_intensity if total_intensity > 0 else 0.5)
        
        # 9. Bright vs. dark (ratio)
        bright_count = sum(text_lower.count(w) for w in self.visual_descriptors['bright'])
        dark_count = sum(text_lower.count(w) for w in self.visual_descriptors['dark'])
        total_luminance = bright_count + dark_count
        features.append(bright_count / total_luminance if total_luminance > 0 else 0.5)
        
        # 10. Visual clarity (clear vs. blurry)
        clear_count = sum(text_lower.count(w) for w in self.visual_descriptors['clear'])
        blurry_count = sum(text_lower.count(w) for w in self.visual_descriptors['blurry'])
        total_clarity = clear_count + blurry_count
        features.append(clear_count / total_clarity if total_clarity > 0 else 0.5)
        
        # 11. Material references density
        material_count = sum(
            sum(text_lower.count(w) for w in words)
            for material, words in self.materials.items()
        )
        features.append(material_count / word_count * 100)
        
        # 12. Size/scale language (large - small)
        large_count = sum(text_lower.count(w) for w in self.size_large)
        small_count = sum(text_lower.count(w) for w in self.size_small)
        features.append((large_count - small_count) / word_count * 100)
        
        # 13. Visual metaphor density
        metaphor_count = sum(
            sum(text_lower.count(w) for w in words)
            for category, words in self.visual_metaphors.items()
        )
        features.append(metaphor_count / word_count * 100)
        
        # 14. Smooth vs. rough visual texture
        smooth_count = sum(text_lower.count(w) for w in self.visual_metaphors['smooth'])
        rough_count = sum(text_lower.count(w) for w in self.visual_metaphors['rough'])
        total_texture = smooth_count + rough_count
        features.append(smooth_count / total_texture if total_texture > 0 else 0.5)
        
        # 15. Overall visual language density
        total_visual = (
            color_count + aesthetic_total + modern_count + classic_count +
            material_count + large_count + small_count + metaphor_count
        )
        features.append(total_visual / word_count * 100)
        
        return np.array(features)
    
    def get_feature_names(self) -> List[str]:
        """Return names of all features."""
        return [
            'color_density',
            'warm_color_ratio',
            'aggressive_colors',
            'calm_colors',
            'aesthetic_valence',
            'aesthetic_intensity',
            'modern_vs_classic',
            'bold_vs_subtle',
            'bright_vs_dark',
            'visual_clarity',
            'material_density',
            'size_scale',
            'visual_metaphor_density',
            'smooth_vs_rough',
            'overall_visual_density'
        ]
    
    def interpret_features(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Interpret optics features in plain English.
        
        Parameters
        ----------
        features : array, shape (15,)
            Feature vector for one document
        
        Returns
        -------
        interpretation : dict
            Plain English interpretation
        """
        names = self.get_feature_names()
        
        interpretation = {
            'summary': self._generate_summary(features),
            'features': {},
            'insights': []
        }
        
        # Interpret each feature
        for i, (name, value) in enumerate(zip(names, features)):
            interpretation['features'][name] = {
                'value': float(value),
                'description': self._describe_feature(name, value)
            }
        
        # Generate insights
        if features[0] > 2.0:  # High color density
            interpretation['insights'].append("Rich visual color language - emphasizes appearance")
        
        if features[4] > 1.0:  # Positive aesthetic valence
            interpretation['insights'].append("Predominantly positive aesthetic framing")
        elif features[4] < -1.0:
            interpretation['insights'].append("Negative aesthetic framing - emphasizes flaws")
        
        if features[6] > 0.7:  # Modern design
            interpretation['insights'].append("Modern/contemporary design framing")
        elif features[6] < 0.3:
            interpretation['insights'].append("Classic/traditional design framing")
        
        if features[14] > 5.0:  # Overall high visual density
            interpretation['insights'].append("HIGHLY visual narrative - appearance is central")
        elif features[14] < 1.0:
            interpretation['insights'].append("Minimal visual language - focuses on non-visual attributes")
        
        return interpretation
    
    def _generate_summary(self, features: np.ndarray) -> str:
        """Generate plain English summary of optics."""
        color_density = features[0]
        warm_ratio = features[1]
        aesthetic_valence = features[4]
        modern_ratio = features[6]
        visual_density = features[14]
        
        summary_parts = []
        
        # Overall visual language
        if visual_density > 5.0:
            summary_parts.append("Highly visual narrative")
        elif visual_density > 2.0:
            summary_parts.append("Moderately visual narrative")
        else:
            summary_parts.append("Minimal visual language")
        
        # Color usage
        if color_density > 2.0:
            if warm_ratio > 0.6:
                summary_parts.append("with warm, energetic color palette")
            elif warm_ratio < 0.4:
                summary_parts.append("with cool, calm color palette")
            else:
                summary_parts.append("with balanced color usage")
        
        # Aesthetic framing
        if aesthetic_valence > 1.0:
            summary_parts.append("emphasizing positive aesthetics")
        elif aesthetic_valence < -1.0:
            summary_parts.append("highlighting visual flaws")
        
        # Design framing
        if modern_ratio > 0.7:
            summary_parts.append("framed as modern/contemporary")
        elif modern_ratio < 0.3:
            summary_parts.append("framed as classic/traditional")
        
        return ", ".join(summary_parts) + "."
    
    def _describe_feature(self, name: str, value: float) -> str:
        """Describe what a feature value means."""
        descriptions = {
            'color_density': f"{'High' if value > 2 else 'Moderate' if value > 0.5 else 'Low'} color reference density",
            'warm_color_ratio': f"{'Warm' if value > 0.6 else 'Cool' if value < 0.4 else 'Balanced'} color temperature",
            'aggressive_colors': f"{'High' if value > 1 else 'Moderate' if value > 0.3 else 'Low'} aggressive color usage",
            'calm_colors': f"{'High' if value > 1 else 'Moderate' if value > 0.3 else 'Low'} calm color usage",
            'aesthetic_valence': f"{'Positive' if value > 0.5 else 'Negative' if value < -0.5 else 'Neutral'} aesthetic framing",
            'aesthetic_intensity': f"{'High' if value > 3 else 'Moderate' if value > 1 else 'Low'} aesthetic language intensity",
            'modern_vs_classic': f"{'Modern' if value > 0.6 else 'Classic' if value < 0.4 else 'Balanced'} design framing",
            'bold_vs_subtle': f"{'Bold' if value > 0.6 else 'Subtle' if value < 0.4 else 'Balanced'} visual intensity",
            'bright_vs_dark': f"{'Bright' if value > 0.6 else 'Dark' if value < 0.4 else 'Balanced'} luminance",
            'visual_clarity': f"{'Clear' if value > 0.6 else 'Blurry' if value < 0.4 else 'Mixed'} visual clarity",
            'material_density': f"{'High' if value > 2 else 'Moderate' if value > 0.5 else 'Low'} material references",
            'size_scale': f"{'Large-scale' if value > 0.5 else 'Small-scale' if value < -0.5 else 'Neutral'} size language",
            'visual_metaphor_density': f"{'High' if value > 2 else 'Moderate' if value > 0.5 else 'Low'} visual metaphors",
            'smooth_vs_rough': f"{'Smooth' if value > 0.6 else 'Rough' if value < 0.4 else 'Mixed'} texture language",
            'overall_visual_density': f"{'Very high' if value > 10 else 'High' if value > 5 else 'Moderate' if value > 2 else 'Low'} overall visual language"
        }
        
        return descriptions.get(name, f"Value: {value:.2f}")

