"""
Transformer Library

Central registry and intelligent selection of transformers.

Provides:
- Organized access to all 25+ transformers
- Category-based selection
- Domain-appropriate transformer sets
- Smart pipeline building
- п-based selection with domain-type logic
- Rationale generation for selections
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import sys
from pathlib import Path

# Import DomainConfig for type hints
sys.path.insert(0, str(Path(__file__).parent.parent / 'pipelines'))
try:
    from pipelines.domain_config import DomainConfig, DomainType
except ImportError:
    DomainConfig = None
    DomainType = None


@dataclass
class TransformerInfo:
    """Metadata about a transformer"""
    name: str
    category: str
    feature_count: int
    best_for_alpha: Tuple[float, float]  # (min, max) alpha range
    best_for_domains: List[str]
    requires_embeddings: bool = False
    requires_llm: bool = False
    computational_cost: str = 'low'  # 'low', 'medium', 'high'
    
    def rationale_for_narrativity(self, п: float) -> str:
        """
        Generate rationale for why this transformer is appropriate for given п.
        
        Parameters
        ----------
        п : float
            Domain narrativity (0-1)
        
        Returns
        -------
        rationale : str
            Explanation of why this transformer fits the domain
        """
        # Base rationale by category
        category_rationale = {
            'core': "Foundational features applicable across all domains",
            'semantic': "Intelligent embedding-based understanding of meaning and emotion",
            'structural': "Plot structure, tension, pacing - narrative architecture",
            'credibility': "Authenticity, expertise, trust markers",
            'contextual': "Cultural fit, temporal positioning, zeitgeist alignment",
            'nominative': "Name-based features - phonetics, semantics, cultural associations",
            'specialized': "Domain-specific or multimodal features",
            'statistical': "Baseline statistical features for comparison"
        }
        
        base = category_rationale.get(self.category, "Domain-appropriate features")
        
        # п-specific rationale
        if п < 0.3:
            narrativity_note = "Constrained domain - focuses on objective, measurable patterns"
        elif п > 0.7:
            narrativity_note = "Open domain - captures subjective, character-driven nuances"
        else:
            narrativity_note = "Mixed domain - balances objective and subjective features"
        
        # Combine
        return f"{base}. {narrativity_note}"


class TransformerLibrary:
    """
    Organized registry of all narrative transformers.
    
    Categories:
    - core: 6 foundational transformers (always recommended)
    - semantic: Embedding-based (emotions, culture, etc.)
    - structural: Narrative structure (conflict, suspense, framing)
    - credibility: Truth and expertise (authenticity, authority)
    - contextual: Time and context (temporal, cultural fit)
    - nominative: Name-based analysis (phonetic, social status)
    - specialized: Cross-domain tools (audio, visual, etc.)
    """
    
    def __init__(self):
        """Initialize transformer registry"""
        self._load_transformer_catalog()
    
    def _load_transformer_catalog(self):
        """Load all available transformers with metadata"""
        self.transformers = {
            # === CORE (6) - Always recommended ===
            'nominative': TransformerInfo(
                name='NominativeAnalysisTransformer',
                category='core',
                feature_count=51,
                best_for_alpha=(0.0, 0.4),
                best_for_domains=['character-driven', 'profiles', 'products'],
                computational_cost='medium'
            ),
            'self_perception': TransformerInfo(
                name='SelfPerceptionTransformer',
                category='core',
                feature_count=21,
                best_for_alpha=(0.0, 0.4),
                best_for_domains=['profiles', 'personal', 'therapeutic'],
                computational_cost='low'
            ),
            'narrative_potential': TransformerInfo(
                name='NarrativePotentialTransformer',
                category='core',
                feature_count=35,
                best_for_alpha=(0.0, 0.6),
                best_for_domains=['growth', 'futures', 'development'],
                computational_cost='low'
            ),
            'linguistic': TransformerInfo(
                name='LinguisticPatternsTransformer',
                category='core',
                feature_count=36,
                best_for_alpha=(0.0, 0.7),
                best_for_domains=['all'],
                computational_cost='medium'
            ),
            'ensemble': TransformerInfo(
                name='EnsembleNarrativeTransformer',
                category='core',
                feature_count=25,
                best_for_alpha=(0.2, 0.6),
                best_for_domains=['multi-character', 'teams', 'relationships'],
                computational_cost='medium'
            ),
            'relational': TransformerInfo(
                name='RelationalValueTransformer',
                category='core',
                feature_count=17,
                best_for_alpha=(0.0, 0.5),
                best_for_domains=['relationships', 'complementarity'],
                computational_cost='low'
            ),
            'awareness_amplification': TransformerInfo(
                name='AwarenessAmplificationTransformer',
                category='core',
                feature_count=15,
                best_for_alpha=(0.0, 0.8),
                best_for_domains=['all'],
                computational_cost='low'
            ),
            
            # === SEMANTIC (Intelligent, embedding-based) ===
            'emotional_semantic': TransformerInfo(
                name='EmotionalSemanticTransformer',
                category='semantic',
                feature_count=34,
                best_for_alpha=(0.0, 0.6),
                best_for_domains=['all'],
                requires_embeddings=True,
                computational_cost='medium'
            ),
            
            # === STRUCTURAL ===
            'conflict': TransformerInfo(
                name='ConflictTensionTransformer',
                category='structural',
                feature_count=28,
                best_for_alpha=(0.0, 0.5),
                best_for_domains=['movies', 'stories', 'competitive'],
                computational_cost='low'
            ),
            'suspense': TransformerInfo(
                name='SuspenseMysteryTransformer',
                category='structural',
                feature_count=25,
                best_for_alpha=(0.0, 0.5),
                best_for_domains=['thrillers', 'mysteries', 'marketing'],
                computational_cost='low'
            ),
            'framing': TransformerInfo(
                name='FramingTransformer',
                category='structural',
                feature_count=24,
                best_for_alpha=(0.0, 0.7),
                best_for_domains=['all'],
                computational_cost='low'
            ),
            
            # === CREDIBILITY ===
            'authenticity': TransformerInfo(
                name='AuthenticityTransformer',
                category='credibility',
                feature_count=30,
                best_for_alpha=(0.0, 0.4),
                best_for_domains=['startups', 'dating', 'grants', 'reviews'],
                computational_cost='low'
            ),
            'expertise': TransformerInfo(
                name='ExpertiseAuthorityTransformer',
                category='credibility',
                feature_count=32,
                best_for_alpha=(0.3, 0.8),
                best_for_domains=['academic', 'professional', 'technical'],
                computational_cost='low'
            ),
            
            # === STATISTICAL (baseline) ===
            'statistical': TransformerInfo(
                name='StatisticalTransformer',
                category='statistical',
                feature_count=200,
                best_for_alpha=(0.5, 1.0),
                best_for_domains=['all'],
                computational_cost='low'
            ),
        }
        
        # Add all others with defaults...
    
    def get_for_domain(
        self,
        alpha: float,
        narrativity: float,
        domain_type: Optional[str] = None,
        max_transformers: int = 12,
        require_core: bool = True
    ) -> List[str]:
        """
        Intelligently select transformers for domain.
        
        Parameters
        ----------
        alpha : float
            Domain's alpha (0=narrative, 1=statistical)
        narrativity : float
            Domain's narrativity (openness)
        domain_type : str, optional
            Domain type hint ('movies', 'startups', 'profiles', etc.)
        max_transformers : int
            Maximum transformers to include
        require_core : bool
            Always include core 6
            
        Returns
        -------
        transformer_names : list of str
            Selected transformer keys
        """
        selected = []
        
        # Always include core if required
        if require_core:
            selected.extend(['nominative', 'self_perception', 'narrative_potential',
                           'linguistic', 'ensemble', 'relational'])
        
        # Add semantic (intelligent) transformers for narrative-heavy domains
        if alpha < 0.5:
            selected.append('emotional_semantic')
        
        # Add structural for plot-driven
        if domain_type in ['movies', 'stories', 'novels']:
            selected.extend(['conflict', 'suspense'])
        
        # Add credibility for high-stakes
        if domain_type in ['startups', 'grants', 'dating', 'professional']:
            selected.extend(['authenticity', 'expertise'])
        
        # Always add statistical as baseline
        if 'statistical' not in selected:
            selected.append('statistical')
        
        # Trim to max
        selected = selected[:max_transformers]
        
        return selected
    
    def get_by_category(self, category: str) -> List[str]:
        """Get all transformers in a category"""
        return [
            key for key, info in self.transformers.items()
            if info.category == category
        ]
    
    def get_all_core(self) -> List[str]:
        """Get 6 core transformers"""
        return self.get_by_category('core')
    
    def get_intelligent_transformers(self) -> List[str]:
        """Get embedding-based transformers"""
        return [
            key for key, info in self.transformers.items()
            if info.requires_embeddings
        ]
    
    def estimate_compute_cost(self, transformer_names: List[str]) -> str:
        """Estimate computational cost of transformer set"""
        costs = [self.transformers[name].computational_cost 
                for name in transformer_names if name in self.transformers]
        
        high_count = costs.count('high')
        medium_count = costs.count('medium')
        
        if high_count > 2:
            return 'high'
        elif high_count > 0 or medium_count > 5:
            return 'medium'
        else:
            return 'low'
    
    def get_info(self, transformer_name: str) -> TransformerInfo:
        """Get metadata for a transformer"""
        return self.transformers.get(transformer_name)
    
    def get_for_narrativity(
        self,
        п: float,
        target_feature_count: int = 300,
        include_statistical: bool = True
    ) -> Tuple[List[str], int]:
        """
        Select transformers based on narrativity (п).
        
        KEY METHOD: Implements theoretical principle that п determines
        which features from ж matter for computing ю.
        
        Parameters
        ----------
        п : float
            Domain narrativity [0, 1]
        target_feature_count : int
            Target number of features (theory says 40-100, we use 200-400)
        include_statistical : bool
            Always include statistical baseline
            
        Returns
        -------
        transformer_names : list of str
            Selected transformer keys
        actual_feature_count : int
            Actual feature count from selection
        """
        
        if п < 0.3:
            # CONSTRAINED/PLOT-DRIVEN
            # Focus: Content, stats, what happened
            priority = [
                ('statistical', 'Plot/content features essential'),
                ('quantitative', 'Numerical patterns'),
                ('linguistic', 'How it\'s told'),
                ('ensemble', 'Cast/element patterns'),
                ('temporal_evolution', 'Time patterns'),
                ('conflict', 'Plot structure'),
                ('suspense', 'Information flow')
            ]
            
        elif п > 0.7:
            # OPEN/CHARACTER-DRIVEN
            # Focus: Identity, character, who they are
            priority = [
                ('nominative', 'Names and categorization'),
                ('self_perception', 'Identity and agency'),
                ('narrative_potential', 'Growth and possibility'),
                ('emotional_semantic', 'Emotional resonance'),
                ('authenticity', 'Truth and uniqueness'),
                ('phonetic', 'Name sounds and patterns'),
                ('social_status', 'Status signals'),
                ('linguistic', 'Voice and style')
            ]
            
        else:
            # BALANCED (0.3 ≤ п ≤ 0.7)
            # Mix of both - discover optimal α empirically
            priority = [
                ('nominative', 'Identity foundation'),
                ('self_perception', 'Character depth'),
                ('linguistic', 'Voice patterns'),
                ('ensemble', 'Network effects'),
                ('emotional_semantic', 'Emotional journey'),
                ('statistical', 'Content baseline'),
                ('conflict', 'Narrative structure'),
                ('cultural_context', 'Contextual fit'),
                ('narrative_potential', 'Growth arc')
            ]
        
        # Select transformers until reaching target
        selected = []
        feature_count = 0
        rationales = {}
        
        for trans_name, rationale in priority:
            if trans_name in self.transformers:
                info = self.transformers[trans_name]
                
                # Check if adding this would exceed target
                if feature_count + info.feature_count <= target_feature_count * 1.2:
                    selected.append(trans_name)
                    rationales[trans_name] = rationale
                    feature_count += info.feature_count
                    
                    # Stop if we've reached target
                    if feature_count >= target_feature_count * 0.9:
                        break
        
        # Always add statistical if not included and requested
        if include_statistical and 'statistical' not in selected:
            selected.append('statistical')
            feature_count += self.transformers['statistical'].feature_count
        
        print(f"\nп-GUIDED TRANSFORMER SELECTION (п={п:.2f})")
        print("="*60)
        print(f"Selected {len(selected)} transformers ({feature_count} features):")
        for trans_name in selected:
            info = self.transformers[trans_name]
            rat = rationales.get(trans_name, info.category)
            print(f"  • {trans_name:25s} ({info.feature_count:3d} feat) - {rat}")
        print(f"\nTheory target: 40-100 features")
        print(f"Expanded target: {target_feature_count} features")
        print(f"Actual: {feature_count} features")
        
        return selected, feature_count
    
    def print_catalog(self):
        """Print organized catalog of all transformers"""
        print("TRANSFORMER CATALOG")
        print("="*80)
        
        categories = set(info.category for info in self.transformers.values())
        
        for category in sorted(categories):
            print(f"\n{category.upper()}:")
            cat_transformers = self.get_by_category(category)
            for trans_name in cat_transformers:
                info = self.transformers[trans_name]
                print(f"  • {trans_name:25s} ({info.feature_count:3d} features) - {info.name}")
    
    def select_for_config(
        self,
        config: 'DomainConfig',
        target_feature_count: int = 300,
        require_core: bool = True
    ) -> Tuple[List[str], Dict[str, str], int]:
        """
        Intelligently select transformers based on domain configuration.
        
        This is the main method that combines п-based selection with domain-type
        augmentation and custom transformers.
        
        Parameters
        ----------
        config : DomainConfig
            Complete domain configuration
        target_feature_count : int
            Target number of features
        require_core : bool
            Always include core 6 transformers
            
        Returns
        -------
        transformer_names : list of str
            Selected transformer keys
        rationales : dict
            Mapping from transformer name to rationale string
        feature_count : int
            Total feature count from selection
        """
        п = config.pi
        domain_type = config.type
        
        # Start with п-based selection
        selected, feature_count = self.get_for_narrativity(
            п=п,
            target_feature_count=target_feature_count,
            include_statistical=True
        )
        
        # Get rationales from п-based selection
        rationales = self._generate_rationales_for_selection(selected, п, domain_type)
        
        # Add domain-type specific transformers
        domain_specific = self._get_domain_type_transformers(domain_type)
        for trans_name in domain_specific:
            if trans_name not in selected:
                if trans_name in self.transformers:
                    info = self.transformers[trans_name]
                    # Check if we have room
                    if feature_count + info.feature_count <= target_feature_count * 1.3:
                        selected.append(trans_name)
                        feature_count += info.feature_count
                        rationales[trans_name] = f"Domain-type specific ({domain_type.value})"
        
        # Add custom augmentation transformers
        for trans_name in config.transformer_augmentation:
            if trans_name not in selected:
                if trans_name in self.transformers:
                    info = self.transformers[trans_name]
                    selected.append(trans_name)
                    feature_count += info.feature_count
                    rationales[trans_name] = "Custom augmentation (from config)"
                else:
                    # Warn about unknown transformer
                    print(f"⚠ Warning: Unknown transformer '{trans_name}' in augmentation")
        
        # Ensure core transformers if required
        if require_core:
            core_transformers = self.get_all_core()
            for trans_name in core_transformers:
                if trans_name not in selected:
                    selected.insert(0, trans_name)  # Add at beginning
                    if trans_name in self.transformers:
                        info = self.transformers[trans_name]
                        feature_count += info.feature_count
                        rationales[trans_name] = "Core transformer (always included)"
        
        return selected, rationales, feature_count
    
    def _get_domain_type_transformers(self, domain_type: 'DomainType') -> List[str]:
        """
        Get domain-type specific transformers.
        
        Parameters
        ----------
        domain_type : DomainType
            Domain type classification
            
        Returns
        -------
        transformers : list of str
            Transformer keys for this domain type
        """
        if domain_type is None:
            return []
        
        type_mapping = {
            DomainType.SPORTS: ['ensemble', 'conflict'],  # Team dynamics, competition
            DomainType.SPORTS_INDIVIDUAL: ['self_perception', 'narrative_potential'],  # Individual focus
            DomainType.SPORTS_TEAM: ['ensemble', 'relational'],  # Team dynamics
            DomainType.ENTERTAINMENT: ['conflict', 'suspense', 'framing'],  # Story structure
            DomainType.NOMINATIVE: ['nominative', 'phonetic', 'social_status'],  # Name-focused
            DomainType.BUSINESS: ['authenticity', 'expertise', 'narrative_potential'],  # Credibility
            DomainType.MEDICAL: ['authenticity', 'expertise'],  # Trust markers
        }
        
        return type_mapping.get(domain_type, [])
    
    def _generate_rationales_for_selection(
        self,
        transformer_names: List[str],
        п: float,
        domain_type: Optional['DomainType'] = None
    ) -> Dict[str, str]:
        """
        Generate detailed rationales for transformer selection.
        
        Parameters
        ----------
        transformer_names : list of str
            Selected transformer keys
        п : float
            Domain narrativity
        domain_type : DomainType, optional
            Domain type classification
            
        Returns
        -------
        rationales : dict
            Mapping from transformer name to rationale
        """
        rationales = {}
        
        for trans_name in transformer_names:
            if trans_name not in self.transformers:
                continue
            
            info = self.transformers[trans_name]
            
            # Base rationale from transformer info
            base_rationale = info.rationale_for_narrativity(п)
            
            # Add domain-type specific note
            if domain_type:
                type_note = self._get_domain_type_rationale(trans_name, domain_type)
                if type_note:
                    base_rationale += f" {type_note}"
            
            rationales[trans_name] = base_rationale
        
        return rationales
    
    def _get_domain_type_rationale(
        self,
        transformer_name: str,
        domain_type: 'DomainType'
    ) -> Optional[str]:
        """Get domain-type specific rationale note"""
        type_notes = {
            DomainType.SPORTS: {
                'ensemble': "Critical for team dynamics and chemistry",
                'conflict': "Competitive tension drives outcomes",
            },
            DomainType.ENTERTAINMENT: {
                'conflict': "Story structure essential for engagement",
                'suspense': "Narrative tension predicts audience response",
                'framing': "How story is presented matters",
            },
            DomainType.NOMINATIVE: {
                'nominative': "Names are the primary signal",
                'phonetic': "Sound patterns carry meaning",
            },
            DomainType.BUSINESS: {
                'authenticity': "Trust and credibility are critical",
                'narrative_potential': "Growth story drives investment",
            },
        }
        
        domain_notes = type_notes.get(domain_type, {})
        return domain_notes.get(transformer_name)
    
    def generate_selection_report(
        self,
        transformer_names: List[str],
        rationales: Dict[str, str],
        feature_count: int,
        п: float,
        domain_type: Optional['DomainType'] = None
    ) -> str:
        """
        Generate human-readable report of transformer selection.
        
        Parameters
        ----------
        transformer_names : list of str
            Selected transformers
        rationales : dict
            Rationales for each transformer
        feature_count : int
            Total feature count
        п : float
            Domain narrativity
        domain_type : DomainType, optional
            Domain type
            
        Returns
        -------
        report : str
            Formatted report string
        """
        lines = [
            "=" * 80,
            "TRANSFORMER SELECTION REPORT",
            "=" * 80,
            f"\nNarrativity (п): {п:.3f}",
        ]
        
        if domain_type:
            lines.append(f"Domain Type: {domain_type.value}")
        
        lines.extend([
            f"Selected Transformers: {len(transformer_names)}",
            f"Total Features: {feature_count}",
            "\nSelection Details:",
            "-" * 80
        ])
        
        # Group by category
        by_category = {}
        for trans_name in transformer_names:
            if trans_name in self.transformers:
                info = self.transformers[trans_name]
                category = info.category
                if category not in by_category:
                    by_category[category] = []
                by_category[category].append((trans_name, info, rationales.get(trans_name, "")))
        
        for category in sorted(by_category.keys()):
            lines.append(f"\n{category.upper()}:")
            for trans_name, info, rationale in by_category[category]:
                lines.append(f"  • {trans_name:25s} ({info.feature_count:3d} features)")
                if rationale:
                    lines.append(f"    {rationale}")
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)

