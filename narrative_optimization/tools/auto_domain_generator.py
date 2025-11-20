"""
Automatic Domain Generator

Automatically generates domain configuration by:
1. Analyzing data characteristics
2. Finding similar domains
3. Transferring patterns
4. Generating archetype transformer
5. Creating config entry

Author: Narrative Integration System
Date: November 2025
"""

import sys
from pathlib import Path
import json
import numpy as np
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DataLoader
from src.learning import MetaLearner, UniversalArchetypeLearner
from src.config import DomainConfig


class AutoDomainGenerator:
    """
    Automatically generates complete domain setup.
    
    Given only data, this will:
    - Estimate domain π (narrativity)
    - Find structurally similar domains
    - Transfer patterns
    - Generate archetype transformer code
    - Create config entry
    - Generate README
    """
    
    def __init__(self):
        self.meta_learner = MetaLearner()
        self.loader = DataLoader()
        
    def generate_domain(
        self,
        domain_name: str,
        data_path: Path,
        manual_pi: Optional[float] = None
    ) -> Dict:
        """
        Automatically generate complete domain setup.
        
        Parameters
        ----------
        domain_name : str
            Domain name
        data_path : Path
            Path to data file
        manual_pi : float, optional
            Manual π override (if None, auto-estimates)
        
        Returns
        -------
        dict
            Generated configuration
        """
        print(f"\n{'='*80}")
        print(f"AUTO-GENERATING DOMAIN: {domain_name.upper()}")
        print(f"{'='*80}\n")
        
        # Load data
        print("[1/7] Loading data...")
        data = self.loader.load(data_path)
        
        if not self.loader.validate_data(data):
            raise ValueError("Invalid data format")
        
        print(f"  ✓ Loaded {len(data['texts'])} samples")
        
        # Estimate π
        print("\n[2/7] Estimating narrativity (π)...")
        if manual_pi:
            pi = manual_pi
            print(f"  Using manual π: {pi:.3f}")
        else:
            pi = self._estimate_pi(data)
            print(f"  ✓ Estimated π: {pi:.3f}")
        
        # Characterize domain
        print("\n[3/7] Characterizing domain...")
        characteristics = self._characterize_domain(data, pi)
        print(f"  Type: {characteristics['type']}")
        print(f"  Structure: {characteristics['structure']}")
        
        # Find similar domains
        print("\n[4/7] Finding similar domains...")
        similar = self._find_similar(domain_name, characteristics)
        print(f"  ✓ Similar to: {', '.join([d for d, _ in similar[:3]])}")
        
        # Discover patterns
        print("\n[5/7] Discovering patterns...")
        patterns = self._discover_patterns(data['texts'], data['outcomes'])
        print(f"  ✓ Discovered {len(patterns)} patterns")
        
        # Generate code
        print("\n[6/7] Generating code...")
        transformer_code = self._generate_transformer_code(domain_name, patterns, similar)
        config_entry = self._generate_config_entry(domain_name, patterns, pi, characteristics)
        readme = self._generate_readme(domain_name, patterns, similar, characteristics)
        
        # Save files
        print("\n[7/7] Saving generated files...")
        output_dir = Path(__file__).parent.parent / 'narrative_optimization' / 'domains' / domain_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save transformer
        transformer_path = Path(__file__).parent.parent / 'src' / 'transformers' / 'archetypes' / f'{domain_name}_archetype.py'
        with open(transformer_path, 'w') as f:
            f.write(transformer_code)
        print(f"  ✓ Transformer: {transformer_path.name}")
        
        # Save config (append to file)
        config_path = Path(__file__).parent.parent / 'src' / 'config' / 'domain_archetypes.py'
        with open(config_path, 'a') as f:
            f.write(f"\n\n{config_entry}")
        print(f"  ✓ Config entry added")
        
        # Save README
        readme_path = output_dir / 'README.md'
        with open(readme_path, 'w') as f:
            f.write(readme)
        print(f"  ✓ README: {readme_path}")
        
        print(f"\n{'='*80}")
        print(f"AUTO-GENERATION COMPLETE: {domain_name.upper()}")
        print(f"{'='*80}")
        
        return {
            'domain_name': domain_name,
            'pi': pi,
            'characteristics': characteristics,
            'similar_domains': similar,
            'patterns': patterns,
            'files_created': [str(transformer_path), str(readme_path)]
        }
    
    def _estimate_pi(self, data: Dict) -> float:
        """
        Auto-estimate π from data characteristics.
        
        Based on:
        - Text complexity
        - Outcome variance
        - Entity diversity
        """
        texts = data['texts']
        outcomes = data['outcomes']
        
        # Text complexity (avg words per text)
        avg_words = np.mean([len(str(text).split()) for text in texts])
        text_complexity = min(1.0, avg_words / 50.0)  # Normalize to [0, 1]
        
        # Outcome variance
        outcome_variance = np.var(outcomes) / (0.25)  # Max var for binary = 0.25
        outcome_variance = min(1.0, outcome_variance)
        
        # Entity diversity (unique capitalized words)
        all_caps = set()
        for text in texts:
            words = str(text).split()
            caps = [w for w in words if w and w[0].isupper()]
            all_caps.update(caps)
        
        entity_diversity = min(1.0, len(all_caps) / 50.0)
        
        # Combine
        pi_estimate = 0.4 * text_complexity + 0.3 * outcome_variance + 0.3 * entity_diversity
        
        # Clamp to reasonable range
        pi_estimate = max(0.1, min(0.95, pi_estimate))
        
        return round(pi_estimate, 2)
    
    def _characterize_domain(self, data: Dict, pi: float) -> Dict:
        """Characterize domain type and structure."""
        texts = data['texts']
        
        # Detect domain type
        domain_type = 'unknown'
        
        # Check for sports keywords
        sports_keywords = ['player', 'game', 'match', 'tournament', 'team', 'score']
        if any(any(kw in str(text).lower() for kw in sports_keywords) for text in texts[:10]):
            domain_type = 'sport'
        
        # Check for business keywords
        business_keywords = ['company', 'startup', 'market', 'revenue', 'funding']
        if any(any(kw in str(text).lower() for kw in business_keywords) for text in texts[:10]):
            domain_type = 'business'
        
        # Check for entertainment keywords
        entertainment_keywords = ['movie', 'film', 'actor', 'director', 'award']
        if any(any(kw in str(text).lower() for kw in entertainment_keywords) for text in texts[:10]):
            domain_type = 'entertainment'
        
        # Determine structure
        if pi > 0.7:
            structure = 'high_narrative'
        elif pi > 0.4:
            structure = 'moderate_narrative'
        else:
            structure = 'low_narrative'
        
        return {
            'type': domain_type,
            'structure': structure,
            'pi': pi
        }
    
    def _find_similar(self, domain_name: str, characteristics: Dict) -> List[tuple]:
        """Find similar domains."""
        # Simplified: based on type
        domain_type = characteristics['type']
        pi = characteristics['pi']
        
        similar = []
        
        if domain_type == 'sport':
            if pi > 0.7:
                similar = [('tennis', 0.8), ('golf', 0.75)]
            else:
                similar = [('nba', 0.6), ('nfl', 0.55)]
        elif domain_type == 'business':
            similar = [('startups', 0.7), ('crypto', 0.5)]
        elif domain_type == 'entertainment':
            similar = [('oscars', 0.8), ('movies', 0.6)]
        
        return similar
    
    def _discover_patterns(self, texts: List[str], outcomes: np.ndarray) -> Dict:
        """Discover initial patterns."""
        learner = UniversalArchetypeLearner()
        patterns = learner.discover_patterns(texts, outcomes, n_patterns=5)
        
        return patterns
    
    def _generate_transformer_code(
        self,
        domain_name: str,
        patterns: Dict,
        similar_domains: List[tuple]
    ) -> str:
        """Generate archetype transformer code."""
        class_name = ''.join(word.capitalize() for word in domain_name.split('_'))
        
        code = f'''"""
{class_name} Archetype Transformer

Auto-generated domain-specific archetype transformer.

Author: Auto-generated
Date: November 2025
"""

import numpy as np
from typing import List

from ..domain_archetype import DomainArchetypeTransformer
from ...config.domain_config import DomainConfig


class {class_name}ArchetypeTransformer(DomainArchetypeTransformer):
    """
    {class_name}-specific Ξ measurement.
    
    Auto-generated from data analysis.
    Similar to: {', '.join([d for d, _ in similar_domains[:3]])}
    """
    
    def __init__(self):
        config = DomainConfig('{domain_name}')
        super().__init__(config)
        
        # Auto-discovered patterns
'''
        
        # Add pattern lists
        for pattern_name, pattern_data in list(patterns.items())[:3]:
            keywords = pattern_data.get('keywords', pattern_data.get('patterns', []))
            code += f"        self.{pattern_name} = {keywords}\n"
        
        code += '''
    
    def _extract_archetype_features(self, X) -> np.ndarray:
        """Extract archetype features with domain context."""
        base_features = super()._extract_archetype_features(X)
        
        # Context-specific boosts (auto-generated)
        enhanced_features = []
        for i, text in enumerate(X):
            boost = 1.0
            
            # Apply domain-specific boosts here
            # (customize based on discovered patterns)
            
            enhanced = base_features[i] * boost
            enhanced_features.append(enhanced)
        
        return np.array(enhanced_features)
'''
        
        return code
    
    def _generate_config_entry(
        self,
        domain_name: str,
        patterns: Dict,
        pi: float,
        characteristics: Dict
    ) -> str:
        """Generate config entry."""
        # Extract pattern names and keywords
        pattern_dict = {}
        for pattern_name, pattern_data in list(patterns.items())[:5]:
            keywords = pattern_data.get('keywords', pattern_data.get('patterns', []))[:5]
            clean_name = pattern_name.replace('universal_', '').replace(f'{domain_name}_', '')
            pattern_dict[clean_name] = keywords
        
        config = f"    '{domain_name}': {{\n"
        config += f"        'archetype_patterns': {{\n"
        
        for i, (name, keywords) in enumerate(pattern_dict.items()):
            config += f"            '{name}': {keywords}"
            if i < len(pattern_dict) - 1:
                config += ","
            config += "\n"
        
        config += f"        }},\n"
        config += f"        'nominative_richness_requirement': 20,\n"
        config += f"        'archetype_weights': {{\n"
        
        # Equal weights for simplicity
        weight = 1.0 / len(pattern_dict)
        for i, name in enumerate(pattern_dict.keys()):
            config += f"            '{name}': {weight:.2f}"
            if i < len(pattern_dict) - 1:
                config += ","
            config += "\n"
        
        config += f"        }},\n"
        config += f"        'pi': {pi},\n"
        config += f"        'theta_range': (0.40, 0.50),\n"
        config += f"        'lambda_range': (0.50, 0.60)\n"
        config += f"    }}"
        
        return config
    
    def _generate_readme(
        self,
        domain_name: str,
        patterns: Dict,
        similar_domains: List[tuple],
        characteristics: Dict
    ) -> str:
        """Generate README."""
        readme = f"# {domain_name.title()} Domain Analysis\n\n"
        readme += f"**Auto-generated**: {Path(__file__).stat().st_mtime}\n\n"
        readme += f"**π (Narrativity)**: {characteristics['pi']:.3f}\n"
        readme += f"**Type**: {characteristics['type']}\n"
        readme += f"**Structure**: {characteristics['structure']}\n\n"
        readme += "---\n\n"
        
        readme += "## Similar Domains\n\n"
        for domain, similarity in similar_domains[:3]:
            readme += f"- {domain}: {similarity:.0%} similar\n"
        readme += "\n"
        
        readme += "## Discovered Patterns\n\n"
        for pattern_name, pattern_data in list(patterns.items())[:5]:
            freq = pattern_data.get('frequency', 0.0)
            readme += f"- **{pattern_name}**: {freq:.1%} frequency\n"
        readme += "\n"
        
        readme += "## Next Steps\n\n"
        readme += "1. Review discovered patterns\n"
        readme += "2. Refine archetype transformer if needed\n"
        readme += "3. Run validation\n"
        readme += "4. Integrate into main system\n"
        
        return readme


def auto_generate_domain(domain_name: str, data_path: str, pi: Optional[float] = None):
    """
    CLI interface for auto-generation.
    
    Usage:
    >>> auto_generate_domain('chess', 'data/domains/chess.json')
    >>> auto_generate_domain('poker', 'data/domains/poker.json', pi=0.65)
    """
    generator = AutoDomainGenerator()
    
    result = generator.generate_domain(
        domain_name,
        Path(data_path),
        manual_pi=pi
    )
    
    print(f"\n✓ Domain '{domain_name}' auto-generated successfully!")
    print(f"\nGenerated files:")
    for file_path in result['files_created']:
        print(f"  - {file_path}")
    
    print(f"\nNext: Run analysis with MASTER_INTEGRATION.py")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Auto-generate domain configuration')
    parser.add_argument('domain', help='Domain name')
    parser.add_argument('data', help='Path to data file')
    parser.add_argument('--pi', type=float, help='Manual π (if not provided, auto-estimates)')
    
    args = parser.parse_args()
    
    auto_generate_domain(args.domain, args.data, args.pi)

