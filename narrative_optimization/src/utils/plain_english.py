"""
Plain English Conversion Utilities

Convert technical ML/narrative terms into clear, understandable language
for non-technical audiences.
"""

from typing import Dict, Any, List


class PlainEnglishExplainer:
    """
    Translate technical concepts into plain English.
    """
    
    def __init__(self):
        self.feature_explanations = self._build_feature_explanations()
        self.metric_explanations = self._build_metric_explanations()
    
    def _build_feature_explanations(self) -> Dict[str, Dict[str, str]]:
        """Define plain English explanations for all features."""
        return {
            # Ensemble features
            'ensemble_size': {
                'name': 'Number of Unique Elements',
                'simple': 'How many different words or concepts appear',
                'example': '"hiking, reading, travel" = 3 elements',
                'why': 'More elements = richer, more diverse narrative',
                'good_range': '10-50 elements'
            },
            'cooccurrence_density': {
                'name': 'How Connected Are Elements',
                'simple': 'Do elements appear together (forming relationships)',
                'example': '"hiking" and "nature" appearing together',
                'why': 'Connected elements = coherent story, not random words',
                'good_range': '0.3-0.7 (moderate connectivity)'
            },
            'ensemble_diversity': {
                'name': 'Variety of Elements',
                'simple': 'How spread out vs concentrated are the elements',
                'example': 'Many different topics vs repeating same things',
                'why': 'Diversity = multifaceted, interesting narrative',
                'good_range': '0.5-0.9 (diverse but not scattered)'
            },
            'network_centrality': {
                'name': 'Importance of Elements',
                'simple': 'Uses mainstream/central concepts vs niche ones',
                'example': 'Common interests like "travel" vs rare ones',
                'why': 'Central = accessible, peripheral = unique',
                'good_range': 'Context-dependent'
            },
            
            # Linguistic features
            'first_person_density': {
                'name': 'How Often "I/Me/My" Appears',
                'simple': 'Personal voice vs impersonal',
                'example': '"I love this" vs "This is good"',
                'why': 'Personal voice = ownership and engagement',
                'good_range': '0.10-0.30 (moderately personal)'
            },
            'future_orientation': {
                'name': 'Forward-Looking Language',
                'simple': 'Uses future tense and future plans',
                'example': '"I will do" vs "I did"',
                'why': 'Future focus = forward momentum and goals',
                'good_range': '0.05-0.20 (some future focus)'
            },
            'agency_score': {
                'name': 'Active vs Passive Voice',
                'simple': 'Taking action vs things happening to you',
                'example': '"I made it happen" vs "It happened to me"',
                'why': 'High agency = control and responsibility',
                'good_range': '0.6-0.9 (mostly active)'
            },
            'voice_consistency': {
                'name': 'Consistent Point of View',
                'simple': 'Sticks to one perspective throughout',
                'example': 'All "I" statements vs mixing "I", "you", "one"',
                'why': 'Consistency = coherent narrative voice',
                'good_range': '0.7-0.9 (mostly consistent)'
            },
            
            # Self-perception features
            'growth_mindset_score': {
                'name': 'Growth vs Fixed Mindset',
                'simple': 'Uses change/development language vs static',
                'example': '"I\'m becoming" vs "I am"',
                'why': 'Growth mindset predicts improvement and learning',
                'good_range': '0.6-0.9 (growth-oriented)'
            },
            'positive_attribution': {
                'name': 'Positive Self-Description',
                'simple': 'Describes self with positive traits',
                'example': '"I\'m creative, thoughtful" vs negative traits',
                'why': 'Positive self-concept affects confidence and outcomes',
                'good_range': '0.4-0.8 (balanced positivity)'
            },
            'identity_coherence': {
                'name': 'Consistent Self-Description',
                'simple': 'Describes self consistently throughout',
                'example': 'Same identity markers repeated vs contradictory',
                'why': 'Coherent identity = stable sense of self',
                'good_range': '0.6-0.9 (coherent)'
            },
            'self_focus_ratio': {
                'name': 'Individual vs Collective Focus',
                'simple': '"I" vs "we" language',
                'example': '"I want" vs "we want"',
                'why': 'Shows individual vs group orientation',
                'good_range': 'Context-dependent'
            },
            
            # Narrative potential features
            'possibility_language': {
                'name': 'Possibility and Potential Words',
                'simple': 'Uses "could", "might", "possible", "potential"',
                'example': '"I could achieve" vs "I will" or "I can\'t"',
                'why': 'Possibility language = open mindset',
                'good_range': '0.1-0.3 (some openness)'
            },
            'future_intentions': {
                'name': 'Stated Future Plans',
                'simple': 'Explicit intentions and goals',
                'example': '"I plan to", "I intend to", "I hope to"',
                'why': 'Intentions = commitment to future action',
                'good_range': '0.05-0.15'
            },
            'narrative_momentum': {
                'name': 'Forward vs Backward Movement',
                'simple': 'Moving forward ("progress", "advance") vs backward',
                'example': '"moving forward" vs "going back"',
                'why': 'Momentum = trajectory and direction',
                'good_range': 'Positive values (forward momentum)'
            }
        }
    
    def _build_metric_explanations(self) -> Dict[str, Dict[str, str]]:
        """Plain English for evaluation metrics."""
        return {
            'accuracy': {
                'plain': 'How often the model guesses correctly',
                'formula': '(Correct guesses) ÷ (Total guesses)',
                'example': '69 right out of 100 = 69% accuracy',
                'good_score': '> 70% (depends on task difficulty)',
                'interpretation': 'Higher = better predictions'
            },
            'f1_macro': {
                'plain': 'Balance between finding everything and being accurate, averaged fairly across all categories',
                'formula': 'Harmonic mean of precision and recall',
                'example': 'Good at finding all types, not just the common ones',
                'good_score': '> 0.70',
                'interpretation': 'Treats rare and common categories equally'
            },
            'precision': {
                'plain': 'When it says "yes", how often is it right',
                'formula': '(Correct positives) ÷ (All predicted positives)',
                'example': 'Of 100 predictions, 80 were actually correct = 80% precision',
                'good_score': '> 0.75',
                'interpretation': 'High precision = few false alarms'
            },
            'recall': {
                'plain': 'Of all the actual cases, how many did it find',
                'formula': '(Found cases) ÷ (Total actual cases)',
                'example': 'Found 70 out of 100 actual cases = 70% recall',
                'good_score': '> 0.70',
                'interpretation': 'High recall = doesn\'t miss many'
            },
            'f1': {
                'plain': 'Balance between precision and recall',
                'formula': '2 × (precision × recall) ÷ (precision + recall)',
                'example': 'Balances finding everything with being accurate',
                'good_score': '> 0.70',
                'interpretation': 'Single score combining precision and recall'
            }
        }
    
    def explain_feature(self, feature_key: str, value: float) -> str:
        """
        Generate plain English explanation of a feature value.
        
        Parameters
        ----------
        feature_key : str
            Feature identifier
        value : float
            Feature value
        
        Returns
        -------
        explanation : str
            Human-readable explanation
        """
        if feature_key not in self.feature_explanations:
            return f"{feature_key}: {value:.3f}"
        
        info = self.feature_explanations[feature_key]
        
        # Interpret the value
        if value > 0.7:
            level = "high"
        elif value > 0.4:
            level = "moderate"
        else:
            level = "low"
        
        explanation = (
            f"**{info['name']}**: {value:.3f} ({level})\n"
            f"  - What it means: {info['simple']}\n"
            f"  - Example: {info['example']}\n"
            f"  - Why it matters: {info['why']}\n"
            f"  - Typical range: {info['good_range']}"
        )
        
        return explanation
    
    def explain_metric(self, metric_name: str, value: float) -> str:
        """
        Explain a metric in plain English.
        
        Parameters
        ----------
        metric_name : str
            Metric name
        value : float
            Metric value
        
        Returns
        -------
        explanation : str
            Plain English explanation
        """
        if metric_name not in self.metric_explanations:
            return f"{metric_name}: {value:.3f}"
        
        info = self.metric_explanations[metric_name]
        
        explanation = (
            f"**{metric_name.replace('_', ' ').title()}**: {value:.1%}\n\n"
            f"**Plain English**: {info['plain']}\n\n"
            f"**How it's calculated**: {info['formula']}\n\n"
            f"**Example**: {info['example']}\n\n"
            f"**Good score**: {info['good_score']}\n\n"
            f"**What it means**: {info['interpretation']}"
        )
        
        return explanation
    
    def generate_narrative_interpretation(
        self,
        transformer_name: str,
        features: Dict[str, float]
    ) -> str:
        """
        Generate comprehensive plain English interpretation.
        
        Parameters
        ----------
        transformer_name : str
            Which transformer was used
        features : dict
            Feature name → value mappings
        
        Returns
        -------
        interpretation : str
            Comprehensive plain English explanation
        """
        interpretations = {
            'ensemble': self._interpret_ensemble,
            'linguistic': self._interpret_linguistic,
            'self_perception': self._interpret_self_perception,
            'potential': self._interpret_potential
        }
        
        if transformer_name in interpretations:
            return interpretations[transformer_name](features)
        else:
            return f"Analysis from {transformer_name} transformer."
    
    def _interpret_ensemble(self, features: Dict[str, float]) -> str:
        """Plain English for ensemble features."""
        size = features.get('ensemble_size', 0)
        density = features.get('cooccurrence_density', 0)
        diversity = features.get('diversity', 0)
        
        interpretation = "**Ensemble Analysis (How Elements Relate)**:\n\n"
        
        if size > 15:
            interpretation += f"• You have **{size:.0f} unique elements** - a rich, diverse narrative with many concepts.\n"
        elif size > 8:
            interpretation += f"• You have **{size:.0f} unique elements** - moderate diversity.\n"
        else:
            interpretation += f"• You have **{size:.0f} unique elements** - focused, narrow narrative.\n"
        
        if density > 0.5:
            interpretation += f"• **High connectivity** ({density:.2f}) - your elements form a tight network, appearing together frequently.\n"
        elif density > 0.3:
            interpretation += f"• **Moderate connectivity** ({density:.2f}) - elements have some relationships.\n"
        else:
            interpretation += f"• **Low connectivity** ({density:.2f}) - elements appear independently.\n"
        
        interpretation += "\n**In Plain Terms**: "
        if size > 12 and density > 0.4:
            interpretation += "You have a rich narrative with diverse elements that connect meaningfully."
        elif size < 8 and density < 0.3:
            interpretation += "Your narrative is focused and direct, with few interconnected elements."
        else:
            interpretation += "Your narrative balances variety with coherence."
        
        return interpretation
    
    def _interpret_linguistic(self, features: Dict[str, float]) -> str:
        """Plain English for linguistic features."""
        first_person = features.get('first_person_density', 0)
        future = features.get('future_orientation', 0)
        agency = features.get('agency_score', 0)
        
        interpretation = "**Linguistic Analysis (How You Tell Your Story)**:\n\n"
        
        # Voice
        if first_person > 0.15:
            interpretation += f"• **Very personal voice** ({first_person:.1%}) - you use 'I/me/my' frequently, making it personal.\n"
        elif first_person > 0.08:
            interpretation += f"• **Moderately personal** ({first_person:.1%}) - balanced between personal and impersonal.\n"
        else:
            interpretation += f"• **Impersonal voice** ({first_person:.1%}) - more detached, less personal.\n"
        
        # Future orientation
        if future > 0.15:
            interpretation += f"• **Highly future-focused** ({future:.1%}) - lots of 'will', 'going to', plans.\n"
        elif future > 0.05:
            interpretation += f"• **Some future focus** ({future:.1%}) - mentions future but not dominant.\n"
        else:
            interpretation += f"• **Present/past focused** ({future:.1%}) - little future-looking language.\n"
        
        # Agency
        if agency > 0.7:
            interpretation += f"• **High agency** ({agency:.2f}) - you take action ('I did', 'I made').\n"
        elif agency > 0.4:
            interpretation += f"• **Moderate agency** ({agency:.2f}) - balanced action-taking.\n"
        else:
            interpretation += f"• **Low agency** ({agency:.2f}) - more passive ('it happened', 'I was').\n"
        
        interpretation += "\n**In Plain Terms**: "
        if first_person > 0.12 and agency > 0.6:
            interpretation += "You write with a strong, personal, active voice - taking ownership and control."
        elif first_person < 0.08 and agency < 0.4:
            interpretation += "You write impersonally and passively - describing rather than claiming."
        else:
            interpretation += "You have a balanced communication style."
        
        return interpretation
    
    def _interpret_self_perception(self, features: Dict[str, float]) -> str:
        """Plain English for self-perception features."""
        growth = features.get('growth_mindset_score', 0)
        attribution = features.get('attribution_balance', 0)
        coherence = features.get('identity_coherence', 0)
        
        interpretation = "**Self-Perception Analysis (How You See Yourself)**:\n\n"
        
        # Growth mindset
        if growth > 0.7:
            interpretation += f"• **Strong growth mindset** ({growth:.2f}) - you use development language ('learning', 'becoming', 'growing').\n"
        elif growth > 0.4:
            interpretation += f"• **Moderate growth mindset** ({growth:.2f}) - some development focus.\n"
        else:
            interpretation += f"• **Fixed mindset** ({growth:.2f}) - static self-description ('I am', 'I've always been').\n"
        
        # Attribution
        if attribution > 0.3:
            interpretation += f"• **Positive self-view** ({attribution:+.2f}) - you describe yourself with positive traits.\n"
        elif attribution > -0.3:
            interpretation += f"• **Balanced self-view** ({attribution:+.2f}) - realistic mix of positive and negative.\n"
        else:
            interpretation += f"• **Negative self-view** ({attribution:+.2f}) - focus on negative traits.\n"
        
        # Coherence
        if coherence > 0.7:
            interpretation += f"• **Coherent identity** ({coherence:.2f}) - consistent self-description throughout.\n"
        elif coherence > 0.4:
            interpretation += f"• **Moderately coherent** ({coherence:.2f}) - some variability in self-description.\n"
        else:
            interpretation += f"• **Variable identity** ({coherence:.2f}) - inconsistent self-description.\n"
        
        interpretation += "\n**In Plain Terms**: "
        if growth > 0.6 and attribution > 0.2 and coherence > 0.6:
            interpretation += "You have a growth-oriented, positive, and stable sense of self."
        elif growth < 0.4 and attribution < 0:
            interpretation += "You describe yourself in fixed, negative terms."
        else:
            interpretation += "You have a developing sense of self."
        
        return interpretation
    
    def _interpret_potential(self, features: Dict[str, float]) -> str:
        """Plain English for narrative potential features."""
        future_orient = features.get('future_orientation_score', 0)
        possibility = features.get('possibility_score', 0)
        momentum = features.get('narrative_momentum', 0)
        
        interpretation = "**Narrative Potential (Future Possibilities)**:\n\n"
        
        if future_orient > 0.2:
            interpretation += f"• **Forward-looking** ({future_orient:.2f}) - focused on future possibilities.\n"
        elif future_orient > 0.1:
            interpretation += f"• **Some future focus** ({future_orient:.2f}).\n"
        else:
            interpretation += f"• **Present/past focused** ({future_orient:.2f}).\n"
        
        if possibility > 0.15:
            interpretation += f"• **High openness** ({possibility:.2f}) - uses possibility language ('could', 'might', 'possible').\n"
        elif possibility > 0.05:
            interpretation += f"• **Moderate openness** ({possibility:.2f}).\n"
        else:
            interpretation += f"• **Closed/certain** ({possibility:.2f}) - little possibility language.\n"
        
        if momentum > 0.1:
            interpretation += f"• **Forward momentum** ({momentum:+.2f}) - moving ahead.\n"
        elif momentum > -0.1:
            interpretation += f"• **Stable** ({momentum:+.2f}) - maintaining position.\n"
        else:
            interpretation += f"• **Backward movement** ({momentum:+.2f}).\n"
        
        interpretation += "\n**In Plain Terms**: "
        if future_orient > 0.15 and possibility > 0.1:
            interpretation += "You're future-oriented and open to possibilities - high narrative potential."
        elif future_orient < 0.05 and possibility < 0.05:
            interpretation += "You focus on present/past with little future exploration."
        else:
            interpretation += "You balance present focus with some future consideration."
        
        return interpretation


def create_feature_explanation_html(transformer_name: str, features: Dict[str, float]) -> str:
    """
    Generate HTML with plain English feature explanations.
    
    Parameters
    ----------
    transformer_name : str
        Transformer name
    features : dict
        Feature values
    
    Returns
    -------
    html : str
        Formatted HTML explanation
    """
    explainer = PlainEnglishExplainer()
    
    html = f"<div class='explanation-panel'>"
    html += f"<h3>{transformer_name.replace('_', ' ').title()} Features Explained</h3>"
    
    for feature_key, value in features.items():
        if feature_key in explainer.feature_explanations:
            info = explainer.feature_explanations[feature_key]
            
            # Determine level
            if value > 0.7:
                level = "high"
                level_class = "high"
            elif value > 0.4:
                level = "moderate"
                level_class = "medium"
            else:
                level = "low"
                level_class = "low"
            
            html += f"""
            <div class='feature-card'>
                <div class='feature-header'>
                    <span class='feature-name'>{info['name']}</span>
                    <span class='feature-value {level_class}'>{value:.3f} ({level})</span>
                </div>
                <p class='feature-simple'><strong>What it means:</strong> {info['simple']}</p>
                <p class='feature-example'><strong>Example:</strong> {info['example']}</p>
                <p class='feature-why'><strong>Why it matters:</strong> {info['why']}</p>
                <p class='feature-range'><strong>Good range:</strong> {info['good_range']}</p>
            </div>
            """
    
    html += "</div>"
    
    return html


if __name__ == '__main__':
    # Demo
    explainer = PlainEnglishExplainer()
    
    print("Plain English Explainer Demo\n")
    print("=" * 80)
    
    # Example features
    example_features = {
        'ensemble_size': 15,
        'cooccurrence_density': 0.45,
        'first_person_density': 0.18,
        'agency_score': 0.82,
        'growth_mindset_score': 0.73
    }
    
    print("\nExample Features with Plain English:\n")
    for key, value in example_features.items():
        print(explainer.explain_feature(key, value))
        print()

