"""
Universal Narrative Analyzer - "Everything is a Book"

Analyzes ANY domain as a narrative structure, testing which narrative elements
drive outcomes without assuming what works where.

Core principle: Every domain is like a book, but domains differ in which
narrative elements (names, ensemble, style, arc, content) matter most.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


class UniversalNarrativeAnalyzer:
    """
    Analyzes ANY domain as a narrative structure.
    
    Principle: Everything is like a book - but which elements drive THIS story?
    
    Book Elements:
    - Names/Identity (Nominative)
    - Character Development (Self-Perception, Potential)
    - Ensemble/Cast (Ensemble, Relational)
    - Communication Style (Linguistic)
    - Plot/Content (Statistical)
    - Arc/Context (Temporal, Contextual)
    """
    
    def __init__(self):
        # Import transformers
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        
        from src.transformers.statistical import StatisticalTransformer
        from src.transformers.ensemble import EnsembleNarrativeTransformer
        from src.transformers.linguistic_advanced import LinguisticPatternsTransformer
        from src.transformers.self_perception import SelfPerceptionTransformer
        from src.transformers.narrative_potential import NarrativePotentialTransformer
        from src.transformers.relational import RelationalValueTransformer
        from src.transformers.nominative import NominativeAnalysisTransformer
        
        self.narrative_elements = {
            'content': StatisticalTransformer(max_features=300),
            'names': NominativeAnalysisTransformer(),
            'development_self': SelfPerceptionTransformer(),
            'development_potential': NarrativePotentialTransformer(),
            'ensemble_network': EnsembleNarrativeTransformer(n_top_terms=30),
            'ensemble_relational': RelationalValueTransformer(n_features=50),
            'style': LinguisticTransformer()
        }
        
        self.book_element_groups = {
            'plot': ['content'],
            'character': ['names', 'development_self', 'development_potential'],
            'ensemble': ['ensemble_network', 'ensemble_relational'],
            'style': ['style']
        }
    
    def analyze_as_narrative(
        self,
        texts: List[str],
        outcomes: np.ndarray,
        book_analogy: str = None,
        sub_domain_labels: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze domain as if it's a book.
        
        Parameters
        ----------
        texts : list of str
            Text samples
        outcomes : array
            Target variable
        book_analogy : str, optional
            Hint about narrative type ("plot-driven", "character-driven", etc.)
        sub_domain_labels : list, optional
            Sub-domain labels for each sample (e.g., genre)
        
        Returns
        -------
        analysis : dict
            Complete narrative analysis including:
            - Which elements matter (element_importance)
            - Book narrative type (narrative_classification)
            - Sub-domain patterns (if labels provided)
            - Confidence in findings
        """
        print("\n" + "="*70)
        print("UNIVERSAL NARRATIVE ANALYSIS")
        print("'Everything is a Book' - Discovering What Matters Here")
        print("="*70 + "\n")
        
        analysis = {
            'overall': {},
            'element_importance': {},
            'narrative_type': '',
            'sub_domains': {},
            'book_analogy': book_analogy
        }
        
        # Test each narrative element
        print("Testing All Narrative Elements...")
        print("-"*70)
        
        for element_name, transformer in self.narrative_elements.items():
            print(f"\n[{element_name}]", end=" ", flush=True)
            
            try:
                # Fit and evaluate
                transformer.fit(texts)
                
                # Quick 3-fold CV
                from sklearn.linear_model import LogisticRegression
                from sklearn.pipeline import Pipeline
                
                pipe = Pipeline([
                    ('transformer', transformer),
                    ('scaler', StandardScaler(with_mean=False)),
                    ('classifier', LogisticRegression(max_iter=500))
                ])
                
                scores = cross_val_score(pipe, texts, outcomes, cv=3, scoring='accuracy')
                
                mean_score = scores.mean()
                std_score = scores.std()
                
                analysis['element_importance'][element_name] = {
                    'accuracy': float(mean_score),
                    'std': float(std_score),
                    'scores': scores.tolist()
                }
                
                print(f"{mean_score:.3f} ± {std_score:.3f}", flush=True)
                
            except Exception as e:
                print(f"Error: {e}", flush=True)
                analysis['element_importance'][element_name] = {'error': str(e)}
        
        # Classify narrative type
        print("\n" + "-"*70)
        analysis['narrative_type'] = self._classify_narrative_type(
            analysis['element_importance']
        )
        print(f"\nNarrative Classification: {analysis['narrative_type']}")
        
        # Calculate domain parameter α
        analysis['overall']['alpha'] = self._calculate_alpha(analysis['element_importance'])
        print(f"Domain Parameter α: {analysis['overall']['alpha']:.3f}")
        
        # Test sub-domains if provided
        if sub_domain_labels is not None:
            print("\n" + "="*70)
            print("SUB-DOMAIN ANALYSIS")
            print("="*70)
            
            analysis['sub_domains'] = self._analyze_sub_domains(
                texts, outcomes, sub_domain_labels
            )
        
        # Generate interpretation
        analysis['interpretation'] = self._generate_book_interpretation(analysis)
        
        print("\n" + "="*70)
        print("INTERPRETATION")
        print("="*70)
        print(analysis['interpretation'])
        print()
        
        return analysis
    
    def _classify_narrative_type(self, element_scores: Dict) -> str:
        """Classify what kind of 'book' this domain is."""
        # Get best performer
        valid_scores = {k: v['accuracy'] for k, v in element_scores.items() 
                       if 'accuracy' in v}
        
        if not valid_scores:
            return "Unknown (insufficient data)"
        
        best_element = max(valid_scores.items(), key=lambda x: x[1])
        best_name, best_score = best_element
        
        # Check if multiple are competitive (within 5%)
        competitive = [name for name, score in valid_scores.items() 
                      if score >= best_score - 0.05]
        
        # Classify by dominant element
        if best_name == 'content':
            if len(competitive) == 1:
                return "Plot-Driven Narrative (content dominates, like a thriller)"
            else:
                return "Plot-Focused Hybrid (content primary, others support)"
        
        elif best_name in ['names']:
            return "Identity-Driven Narrative (names matter, like combat sports)"
        
        elif best_name in ['development_self', 'development_potential']:
            return "Character-Driven Narrative (growth/arc matters, like bildungsroman)"
        
        elif best_name in ['ensemble_network', 'ensemble_relational']:
            return "Ensemble-Driven Narrative (relationships matter, like multi-POV novel)"
        
        elif best_name == 'style':
            return "Style-Driven Narrative (voice/emotion matters, like literary fiction)"
        
        # If multiple competitive
        if len(competitive) > 2:
            return "Multi-Dimensional Narrative (several elements matter equally)"
        
        return f"Hybrid Narrative (primary: {best_name}, secondary: {competitive[1:]})"
    
    def _calculate_alpha(self, element_scores: Dict) -> float:
        """
        Calculate domain parameter α.
        
        α = content_performance / (content + best_narrative)
        
        α ≈ 1: Content-pure
        α ≈ 0.5: Hybrid
        α ≈ 0: Narrative-pure
        """
        content_score = element_scores.get('content', {}).get('accuracy', 0.5)
        
        # Get best narrative score
        narrative_scores = [
            v.get('accuracy', 0) 
            for k, v in element_scores.items() 
            if k != 'content' and 'accuracy' in v
        ]
        
        if not narrative_scores:
            return 1.0  # Default to content if no narrative scores
        
        best_narrative = max(narrative_scores)
        
        alpha = content_score / (content_score + best_narrative)
        
        return alpha
    
    def _analyze_sub_domains(
        self,
        texts: List[str],
        outcomes: np.ndarray,
        labels: List[str]
    ) -> Dict[str, Any]:
        """Test sub-domains separately (like book genres)."""
        sub_domain_results = {}
        unique_labels = list(set(labels))
        
        print(f"\nTesting {len(unique_labels)} sub-domains...")
        
        for label in unique_labels:
            # Get samples for this sub-domain
            indices = [i for i, l in enumerate(labels) if l == label]
            
            if len(indices) < 30:
                print(f"\n  {label}: Too few samples (n={len(indices)}), skipping")
                continue
            
            sub_texts = [texts[i] for i in indices]
            sub_outcomes = outcomes[indices]
            
            print(f"\n  {label} (n={len(indices)}): ", end="", flush=True)
            
            # Quick test with content and best narrative
            try:
                content_score = self._quick_test(
                    sub_texts, sub_outcomes, self.narrative_elements['content']
                )
                
                ensemble_score = self._quick_test(
                    sub_texts, sub_outcomes, self.narrative_elements['ensemble_network']
                )
                
                linguistic_score = self._quick_test(
                    sub_texts, sub_outcomes, self.narrative_elements['style']
                )
                
                # Calculate α for this sub-domain
                best_narrative = max(ensemble_score, linguistic_score)
                alpha_sub = content_score / (content_score + best_narrative)
                
                sub_domain_results[label] = {
                    'n': len(indices),
                    'content': float(content_score),
                    'best_narrative': float(best_narrative),
                    'alpha': float(alpha_sub),
                    'type': self._infer_type(alpha_sub)
                }
                
                print(f"α={alpha_sub:.2f} ({sub_domain_results[label]['type']})")
                
            except Exception as e:
                print(f"Error: {e}")
                sub_domain_results[label] = {'error': str(e)}
        
        # Test heterogeneity
        if len(sub_domain_results) > 1:
            alphas = [v['alpha'] for v in sub_domain_results.values() if 'alpha' in v]
            if len(alphas) > 1:
                heterogeneity = np.std(alphas)
                sub_domain_results['heterogeneity'] = {
                    'alpha_std': float(heterogeneity),
                    'interpretation': 'High' if heterogeneity > 0.15 else 'Moderate' if heterogeneity > 0.08 else 'Low'
                }
                print(f"\n  Heterogeneity: σ(α) = {heterogeneity:.3f} ({sub_domain_results['heterogeneity']['interpretation']})")
        
        return sub_domain_results
    
    def _quick_test(self, X, y, transformer) -> float:
        """Quick 2-fold test."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        
        pipe = Pipeline([
            ('transformer', transformer),
            ('scaler', StandardScaler(with_mean=False)),
            ('classifier', LogisticRegression(max_iter=300))
        ])
        
        scores = cross_val_score(pipe, X, y, cv=2, scoring='accuracy')
        return scores.mean()
    
    def _infer_type(self, alpha: float) -> str:
        """Infer narrative type from α."""
        if alpha > 0.7:
            return "content-heavy"
        elif alpha > 0.5:
            return "balanced hybrid"
        elif alpha > 0.3:
            return "narrative-leaning"
        else:
            return "narrative-driven"
    
    def _generate_book_interpretation(self, analysis: Dict) -> str:
        """Generate book-analogy interpretation."""
        narrative_type = analysis['narrative_type']
        alpha = analysis['overall']['alpha']
        
        interpretation = f"DOMAIN AS NARRATIVE (Book Analogy):\n\n"
        interpretation += f"Type: {narrative_type}\n"
        interpretation += f"Domain Parameter α: {alpha:.3f}\n\n"
        
        # Explain what this means
        if alpha > 0.7:
            interpretation += "Like a PLOT-DRIVEN book (thriller, action):\n"
            interpretation += "- What happens (content) matters most\n"
            interpretation += "- Characters, style are secondary\n"
            interpretation += "- Statistical methods optimal\n"
        elif alpha > 0.5:
            interpretation += "Like a BALANCED NARRATIVE (literary fiction):\n"
            interpretation += "- Plot AND character/style matter\n"
            interpretation += "- Multiple elements contribute\n"
            interpretation += "- Combined methods optimal\n"
        else:
            interpretation += "Like a CHARACTER/RELATIONSHIP-DRIVEN book (literary, romance):\n"
            interpretation += "- Who they are (identity, growth) matters most\n"
            interpretation += "- How they relate (ensemble) crucial\n"
            interpretation += "- Narrative methods optimal\n"
        
        # Add element rankings
        interpretation += "\nElement Importance (Ranked):\n"
        element_scores = analysis['element_importance']
        ranked = sorted(
            [(k, v.get('accuracy', 0)) for k, v in element_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        for i, (element, score) in enumerate(ranked[:5], 1):
            interpretation += f"  {i}. {element}: {score:.1%}\n"
        
        # Sub-domain insights
        if analysis['sub_domains']:
            interpretation += "\nSub-Domain Patterns (Like Different Genres):\n"
            for sub, data in list(analysis['sub_domains'].items())[:5]:
                if 'alpha' in data:
                    interpretation += f"  {sub}: α={data['alpha']:.2f} ({data['type']})\n"
        
        return interpretation


class NarrativeBookMapper:
    """
    Maps domains to book archetypes for intuitive understanding.
    """
    
    def __init__(self):
        self.book_archetypes = {
            'plot_driven': {
                'alpha_range': (0.7, 1.0),
                'examples': ['Thriller', 'Action', 'Mystery'],
                'optimal_method': 'Statistical',
                'real_domains': ['News', 'Technical Documentation']
            },
            'character_driven': {
                'alpha_range': (0.0, 0.3),
                'examples': ['Character Study', 'Bildungsroman', 'Memoir'],
                'optimal_method': 'Nominative + Self-Perception + Potential',
                'real_domains': ['Wellness Journals', 'Personal Profiles']
            },
            'ensemble_driven': {
                'alpha_range': (0.2, 0.4),
                'examples': ['Multi-POV', 'Ensemble Cast', 'Network Novel'],
                'optimal_method': 'Ensemble + Relational',
                'real_domains': ['Relationships', 'Team Dynamics']
            },
            'style_driven': {
                'alpha_range': (0.4, 0.6),
                'examples': ['Literary Fiction', 'Poetry', 'Experimental'],
                'optimal_method': 'Linguistic',
                'real_domains': ['Reviews', 'Opinion Pieces']
            },
            'hybrid': {
                'alpha_range': (0.45, 0.65),
                'examples': ['Commercial Fiction', 'Genre Blends'],
                'optimal_method': 'Combined (weighted)',
                'real_domains': ['Crypto', 'Product Descriptions']
            }
        }
    
    def map_to_book_type(self, alpha: float, element_scores: Dict) -> Dict[str, Any]:
        """Map domain to book archetype based on α and element patterns."""
        # Find matching archetype
        for archetype, properties in self.book_archetypes.items():
            alpha_min, alpha_max = properties['alpha_range']
            if alpha_min <= alpha <= alpha_max:
                return {
                    'archetype': archetype,
                    'book_examples': properties['examples'],
                    'optimal_method': properties['optimal_method'],
                    'similar_domains': properties['real_domains'],
                    'alpha': float(alpha)
                }
        
        return {'archetype': 'uncategorized', 'alpha': float(alpha)}
    
    def explain_as_book(self, domain_name: str, analysis: Dict) -> str:
        """Explain domain analysis using book analogy."""
        alpha = analysis['overall'].get('alpha', 0.5)
        mapping = self.map_to_book_type(alpha, analysis['element_importance'])
        
        explanation = f"BOOK ANALOGY FOR {domain_name.upper()}\n\n"
        
        explanation += f"This domain is like: {mapping.get('archetype', 'unknown').replace('_', ' ').title()}\n"
        explanation += f"Book examples: {', '.join(mapping.get('book_examples', []))}\n"
        explanation += f"Similar real domains: {', '.join(mapping.get('similar_domains', []))}\n\n"
        
        explanation += "What This Means:\n"
        
        archetype = mapping.get('archetype')
        if archetype == 'plot_driven':
            explanation += "- Like a thriller: WHAT HAPPENS matters most\n"
            explanation += "- Content/plot drives the story\n"
            explanation += "- Characters/style are secondary\n"
        elif archetype == 'character_driven':
            explanation += "- Like a character study: WHO THEY ARE matters most\n"
            explanation += "- Names, identity, growth drive the story\n"
            explanation += "- Plot is vehicle for character\n"
        elif archetype == 'ensemble_driven':
            explanation += "- Like a multi-POV novel: HOW THEY RELATE matters most\n"
            explanation += "- Connections, dynamics drive the story\n"
            explanation += "- Individual elements less important than relationships\n"
        elif archetype == 'style_driven':
            explanation += "- Like literary fiction: HOW IT'S TOLD matters most\n"
            explanation += "- Voice, emotion, style drive experience\n"
            explanation += "- Content is vehicle for expression\n"
        elif archetype == 'hybrid':
            explanation += "- Like commercial fiction: MULTIPLE ELEMENTS matter\n"
            explanation += "- Plot, character, and style all contribute\n"
            explanation += "- Balance is key\n"
        
        explanation += f"\nOptimal Approach: {mapping.get('optimal_method', 'Combined')}\n"
        
        return explanation


if __name__ == '__main__':
    print("Universal Narrative Analyzer")
    print("Principle: Everything is like a book")
    print("Question: Which narrative elements drive THIS story?")
    print("\nReady to analyze any domain without assumptions.")

