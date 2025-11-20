"""
Pipeline Composer

Intelligently assembles complete narrative analysis pipelines from domain configurations.

Assembles: Data Loader → Transformer Suite → Analyzer → Validator → Reporter
"""

from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import sys
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.pipelines.domain_config import DomainConfig, DomainType, OutcomeType
    from src.transformers.transformer_library import TransformerLibrary
    from src.pipelines.narrative_pipeline import NarrativePipeline
    from src.analysis.universal_analyzer import UniversalDomainAnalyzer
    from src.analysis.cross_domain_validator import CrossDomainValidator
    from src.analysis.quality_aggregator import QualityAggregator
    from src.analysis.perspective_weights import NarrativePerspective
except ImportError:
    # Fallback for different import paths
    from pipelines.domain_config import DomainConfig, DomainType, OutcomeType
    from transformers.transformer_library import TransformerLibrary
    from pipelines.narrative_pipeline import NarrativePipeline
    from analysis.universal_analyzer import UniversalDomainAnalyzer
    from analysis.cross_domain_validator import CrossDomainValidator
    from analysis.quality_aggregator import QualityAggregator
    from analysis.perspective_weights import NarrativePerspective


class PipelineComposer:
    """
    Composes complete narrative analysis pipelines from domain configurations.
    
    This is the main entry point for creating pipelines. It handles:
    - Loading domain configuration
    - Selecting transformers intelligently
    - Instantiating transformers
    - Assembling pipeline stages
    - Setting up caching and parallelization
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize pipeline composer.
        
        Parameters
        ----------
        project_root : Path, optional
            Root directory of project. If None, auto-detects.
        """
        if project_root is None:
            # Auto-detect project root (go up from src/pipelines)
            project_root = Path(__file__).parent.parent.parent.parent
        
        self.project_root = Path(project_root)
        self.transformer_library = TransformerLibrary()
        self.validator = CrossDomainValidator()
    
    def compose_pipeline(
        self,
        config: DomainConfig,
        data_path: Optional[Path] = None,
        target_feature_count: int = 300,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Compose complete pipeline from domain configuration.
        
        Parameters
        ----------
        config : DomainConfig
            Domain configuration
        data_path : Path, optional
            Path to data file. If None, looks in standard location.
        target_feature_count : int
            Target number of features to extract
        use_cache : bool
            Enable caching for transformers
            
        Returns
        -------
        pipeline_info : dict
            Complete pipeline information including:
            - config: Domain configuration
            - transformers: Selected transformer names
            - transformer_instances: Instantiated transformers
            - rationales: Selection rationales
            - feature_count: Expected feature count
            - analyzer: UniversalDomainAnalyzer instance
            - validator: CrossDomainValidator instance
        """
        print("=" * 80)
        print(f"COMPOSING PIPELINE: {config.domain.upper()}")
        print("=" * 80)
        print(f"\n{config.get_summary()}")
        
        # Step 1: Select transformers
        print("\n" + "-" * 80)
        print("STEP 1: Selecting Transformers")
        print("-" * 80)
        print("NOTE: Core transformers (nominative, self_perception, narrative_potential,")
        print("      linguistic, ensemble, relational) are available to ALL domains.")
        print("      Domain types add ADDITIONAL transformers.")
        
        transformer_names, rationales, feature_count = self.transformer_library.select_for_config(
            config=config,
            target_feature_count=target_feature_count,
            require_core=True  # Core transformers always included
        )
        
        print(self.transformer_library.generate_selection_report(
            transformer_names, rationales, feature_count, config.pi, config.type
        ))
        
        # Step 2: Instantiate transformers
        print("\n" + "-" * 80)
        print("STEP 2: Instantiating Transformers")
        print("-" * 80)
        
        transformer_instances = self._instantiate_transformers(transformer_names)
        
        print(f"✓ Instantiated {len(transformer_instances)} transformers")
        
        # Step 3: Create analyzer
        print("\n" + "-" * 80)
        print("STEP 3: Creating Analyzer")
        print("-" * 80)
        
        analyzer = UniversalDomainAnalyzer(
            domain_name=config.domain,
            narrativity=config.pi
        )
        
        # Create quality aggregator for multi-perspective/multi-method analysis
        quality_aggregator = QualityAggregator(config.pi)
        
        # Get domain type preferences if available
        try:
            try:
                from src.pipelines.domain_types import get_domain_type_class
            except ImportError:
                from pipelines.domain_types import get_domain_type_class
            domain_type_class = get_domain_type_class(config.type)
            if domain_type_class:
                domain_type_instance = domain_type_class(config)
                # Override config with domain type preferences if not explicitly set
                if config.perspectives == ['director', 'audience', 'critic']:  # Default
                    config.perspectives = domain_type_instance.get_perspective_preferences()
                if config.quality_methods == ['weighted_mean', 'ensemble']:  # Default
                    config.quality_methods = domain_type_instance.get_quality_method_preferences()
                if config.scales == ['micro', 'meso', 'macro']:  # Default
                    config.scales = domain_type_instance.get_scale_preferences()
        except:
            pass  # Use config defaults
        
        print(f"✓ Created UniversalDomainAnalyzer for {config.domain}")
        print(f"  Narrativity (п): {config.pi:.3f}")
        print(f"✓ Created QualityAggregator for multi-dimensional analysis")
        print(f"  Perspectives: {', '.join(config.perspectives)}")
        print(f"  Methods: {', '.join(config.quality_methods)}")
        print(f"  Scales: {', '.join(config.scales)}")
        
        # Step 4: Create validator
        print("\n" + "-" * 80)
        print("STEP 4: Creating Validator")
        print("-" * 80)
        
        print(f"✓ Using CrossDomainValidator")
        
        # Step 5: Create narrative pipeline wrapper
        print("\n" + "-" * 80)
        print("STEP 5: Assembling Pipeline")
        print("-" * 80)
        
        # Create FeatureUnion to combine transformers in parallel
        # This is CRITICAL: transformers must run in parallel and concatenate features,
        # not sequentially where each output becomes the next input
        print("Creating FeatureUnion to combine transformers in parallel...")
        
        transformer_list = []
        for trans_name, transformer in zip(transformer_names, transformer_instances):
            if transformer is not None:
                transformer_list.append((trans_name, transformer))
        
        # Build pipeline with FeatureUnion
        if len(transformer_list) > 0:
            feature_union = FeatureUnion(transformer_list)
            pipeline = Pipeline([('features', feature_union)])
        else:
            raise ValueError("No valid transformers available")
        
        print(f"✓ Pipeline assembled with {len(transformer_list)} transformers in parallel")
        print(f"  Total features: {feature_count}")
        print(f"  Caching: {'disabled' if not use_cache else 'enabled'}")
        
        # Compile pipeline info
        pipeline_info = {
            'config': config,
            'transformers': transformer_names,
            'transformer_instances': transformer_instances,
            'rationales': rationales,
            'feature_count': feature_count,
            'analyzer': analyzer,
            'quality_aggregator': quality_aggregator,
            'validator': self.validator,
            'pipeline': pipeline,
            'data_path': data_path
        }
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPOSITION COMPLETE")
        print("=" * 80)
        
        return pipeline_info
    
    def _instantiate_transformers(self, transformer_names: List[str]) -> List[Any]:
        """
        Instantiate transformer objects from names.
        
        Parameters
        ----------
        transformer_names : list of str
            Transformer keys
            
        Returns
        -------
        transformers : list
            Instantiated transformer objects
        """
        transformers = []
        
        # Import transformer classes dynamically
        try:
            from src.transformers import (
                NominativeAnalysisTransformer,
                SelfPerceptionTransformer,
                NarrativePotentialTransformer,
                LinguisticPatternsTransformer,
                EnsembleNarrativeTransformer,
                RelationalValueTransformer,
                ConflictTensionTransformer,
                SuspenseMysteryTransformer,
                FramingTransformer,
                AuthenticityTransformer,
                ExpertiseAuthorityTransformer,
                StatisticalTransformer,
                EmotionalSemanticTransformer,
            )
        except ImportError:
            # Fallback: try importing from different locations
            try:
                from transformers import (
                    NominativeAnalysisTransformer,
                    SelfPerceptionTransformer,
                    NarrativePotentialTransformer,
                    LinguisticPatternsTransformer,
                    EnsembleNarrativeTransformer,
                    RelationalValueTransformer,
                    ConflictTensionTransformer,
                    SuspenseMysteryTransformer,
                    FramingTransformer,
                    AuthenticityTransformer,
                    ExpertiseAuthorityTransformer,
                    StatisticalTransformer,
                )
            except ImportError:
                print("⚠ Warning: Could not import transformers. Using placeholders.")
                return [None] * len(transformer_names)
        
        # Mapping from names to classes
        transformer_map = {
            'nominative': NominativeAnalysisTransformer,
            'self_perception': SelfPerceptionTransformer,
            'narrative_potential': NarrativePotentialTransformer,
            'linguistic': LinguisticPatternsTransformer,
            'ensemble': EnsembleNarrativeTransformer,
            'relational': RelationalValueTransformer,
            'conflict': ConflictTensionTransformer,
            'suspense': SuspenseMysteryTransformer,
            'framing': FramingTransformer,
            'authenticity': AuthenticityTransformer,
            'expertise': ExpertiseAuthorityTransformer,
            'statistical': StatisticalTransformer,
            'emotional_semantic': EmotionalSemanticTransformer if 'EmotionalSemanticTransformer' in globals() else None,
        }
        
        for trans_name in transformer_names:
            if trans_name in transformer_map:
                transformer_class = transformer_map[trans_name]
                if transformer_class is not None:
                    try:
                        # Instantiate with default parameters
                        if trans_name == 'ensemble':
                            transformer = transformer_class(n_top_terms=20)
                        elif trans_name == 'statistical':
                            transformer = transformer_class(max_features=200)
                        else:
                            transformer = transformer_class()
                        transformers.append(transformer)
                    except Exception as e:
                        print(f"⚠ Warning: Could not instantiate {trans_name}: {e}")
                        transformers.append(None)
                else:
                    print(f"⚠ Warning: Transformer {trans_name} not available")
                    transformers.append(None)
            else:
                print(f"⚠ Warning: Unknown transformer {trans_name}")
                transformers.append(None)
        
        return transformers
    
    def load_data(
        self,
        config: DomainConfig,
        data_path: Optional[Path] = None
    ) -> Tuple[List[str], np.ndarray, List[str], Optional[np.ndarray]]:
        """
        Load data according to domain configuration schema.
        
        Parameters
        ----------
        config : DomainConfig
            Domain configuration
        data_path : Path, optional
            Path to data file. If None, looks in standard location.
            
        Returns
        -------
        texts : list of str
            Narrative texts extracted from text_fields
        outcomes : ndarray
            Outcome values
        names : list of str
            Organism names (if available)
        context_features : ndarray, optional
            Context features (if available)
        """
        # Determine data path
        if data_path is None:
            # Look in standard location: data/domains/{domain}/
            data_path = self.project_root / 'data' / 'domains' / config.domain
            
            # Try to find data file
            json_files = list(data_path.glob('*.json'))
            csv_files = list(data_path.glob('*.csv'))
            
            if json_files:
                data_path = json_files[0]
            elif csv_files:
                data_path = csv_files[0]
            else:
                raise FileNotFoundError(
                    f"Could not find data file for domain {config.domain}. "
                    f"Looked in {data_path}"
                )
        
        data_path = Path(data_path)
        
        print(f"\nLoading data from: {data_path}")
        
        # Load data based on file type
        if data_path.suffix == '.json':
            import json
            import os
            file_size_mb = os.path.getsize(data_path) / (1024 * 1024)
            print(f"Loading JSON data ({file_size_mb:.1f} MB, this may take a moment)...")
            
            # For large files, use streaming or sample during load
            max_samples = config.sample_size if config.sample_size else 5000  # Reduced default
            try:
                if file_size_mb > 50:  # Large file - sample during load
                    print(f"⚠ Large file detected, sampling {max_samples} records during load...")
                    import random
                    random.seed(42)
                    with open(data_path, 'r', encoding='utf-8') as f:
                        full_data = json.load(f)
                        if isinstance(full_data, list) and len(full_data) > max_samples:
                            data = random.sample(full_data, max_samples)
                        else:
                            data = full_data
                    print(f"✓ Loaded and sampled to {len(data) if isinstance(data, list) else 'dict'} records")
                else:
                    with open(data_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    print(f"✓ Loaded {len(data) if isinstance(data, list) else 'dict'} records from JSON")
                    
                    # Sample if still too large
                    if isinstance(data, list) and len(data) > max_samples:
                        print(f"⚠ Dataset has {len(data)} records, sampling {max_samples} for analysis")
                        import random
                        random.seed(42)
                        data = random.sample(data, max_samples)
                        print(f"✓ Sampled to {len(data)} records")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON file {data_path}: {e}")
            except Exception as e:
                raise ValueError(f"Error loading {data_path}: {e}")
        elif data_path.suffix == '.csv':
            import pandas as pd
            print(f"Loading CSV data...")
            df = pd.read_csv(data_path)
            data = df.to_dict('records')
            print(f"✓ Loaded {len(data)} records from CSV")
            
            # Sample if too large
            max_samples = config.sample_size if config.sample_size else 10000
            if len(data) > max_samples:
                print(f"⚠ Dataset has {len(data)} records, sampling {max_samples} for analysis")
                data = data[:max_samples]
                print(f"✓ Sampled to {len(data)} records")
        else:
            raise ValueError(f"Unsupported file type: {data_path.suffix}")
        
        # Extract fields according to schema
        texts = []
        outcomes = []
        names = []
        context_features_list = []
        
        # Domain-specific adapters
        if config.domain == 'tennis':
            # Tennis data has complex structure - extract narrative and compute outcome from betting odds
            for record in data:
                # Extract narrative
                if 'narrative' in record and record['narrative']:
                    texts.append(str(record['narrative']))
                else:
                    texts.append('')
                
                # Compute outcome from betting odds (favorite wins = 1)
                if 'betting_odds' in record:
                    p1_odds = record['betting_odds'].get('player1_odds', 2.0)
                    p2_odds = record['betting_odds'].get('player2_odds', 2.0)
                    p1_favorite = p1_odds < p2_odds
                    # Outcome: 1 if favorite (player1) won, 0 if underdog won
                    # Note: In original data, player1 is always winner, so we use favorite logic
                    outcomes.append(1 if p1_favorite else 0)
                elif 'player1_wins' in record:
                    outcomes.append(int(record['player1_wins']))
                else:
                    # Default: assume player1 wins (original data structure)
                    outcomes.append(1)
                
                # Extract name
                if 'player1' in record and 'name' in record['player1']:
                    names.append(f"{record['player1']['name']}_vs_{record['player2'].get('name', 'Unknown')}")
                else:
                    names.append(f"match_{len(names)}")
        elif config.domain == 'golf':
            # Golf data has narrative and won_tournament fields
            for record in data:
                # Extract narrative - check multiple possible fields
                narrative = ''
                if 'narrative' in record and record['narrative']:
                    narrative = str(record['narrative'])
                elif 'enhanced_narrative' in record and record['enhanced_narrative']:
                    narrative = str(record['enhanced_narrative'])
                elif 'text' in record and record['text']:
                    narrative = str(record['text'])
                else:
                    # Build narrative from available fields
                    parts = []
                    if 'player_name' in record:
                        parts.append(f"Player {record['player_name']}")
                    if 'tournament_name' in record:
                        parts.append(f"at {record['tournament_name']}")
                    if 'course_name' in record:
                        parts.append(f"on {record['course_name']}")
                    if 'year' in record:
                        parts.append(f"in {record['year']}")
                    narrative = '. '.join(parts) + '.' if parts else ''
                
                texts.append(narrative)
                
                # Extract outcome - check multiple fields
                outcome = 0
                if 'won_tournament' in record:
                    outcome = int(record['won_tournament'])
                elif 'outcome' in record:
                    outcome = int(record['outcome'])
                elif 'finish_position' in record:
                    # Position 1 = winner
                    outcome = 1 if record['finish_position'] == 1 else 0
                outcomes.append(outcome)
                
                # Extract name
                if 'player_name' in record and 'tournament_name' in record:
                    names.append(f"{record['player_name']}_{record['tournament_name']}")
                elif 'player_name' in record:
                    names.append(str(record['player_name']))
                elif 'tournament_name' in record:
                    names.append(f"tournament_{record['tournament_name']}")
                else:
                    names.append(f"golf_{len(names)}")
        else:
            # Standard extraction for other domains
            for record in data:
                # Extract text fields (combine multiple text fields)
                text_parts = []
                for field in config.data.text_fields:
                    if field in record and record[field]:
                        text_parts.append(str(record[field]))
                
                if text_parts:
                    texts.append(' '.join(text_parts))
                else:
                    texts.append('')  # Empty text if no fields found
                
                # Extract outcome
                if config.data.outcome_field in record:
                    outcomes.append(record[config.data.outcome_field])
                else:
                    raise ValueError(
                        f"Outcome field '{config.data.outcome_field}' not found in data"
                    )
            
                # Extract name if available
                if config.data.name_field and config.data.name_field in record:
                    names.append(str(record[config.data.name_field]))
                else:
                    names.append(f"entity_{len(names)}")
            
                # Extract context features if available
                if config.data.context_fields:
                    context_row = []
                    for field in config.data.context_fields:
                        if field in record:
                            value = record[field]
                            # Convert to numeric if possible
                            try:
                                context_row.append(float(value))
                            except (ValueError, TypeError):
                                # Use hash for categorical
                                context_row.append(hash(str(value)) % 1000 / 1000.0)
                        else:
                            context_row.append(0.0)
                    context_features_list.append(context_row)
        
        # Convert to numpy arrays
        outcomes = np.array(outcomes)
        
        # Convert outcomes based on type
        if config.outcome_type == OutcomeType.BINARY:
            # Convert to binary if needed
            if outcomes.dtype != bool and outcomes.dtype != int:
                # Try to infer binary
                unique_values = np.unique(outcomes)
                if len(unique_values) == 2:
                    outcomes = (outcomes == unique_values[1]).astype(int)
                else:
                    # Use median split
                    median = np.median(outcomes)
                    outcomes = (outcomes > median).astype(int)
        
        context_features = None
        if context_features_list:
            context_features = np.array(context_features_list)
        
        # Ensure all texts are strings (not bytes or None)
        texts = [str(t) if t is not None else '' for t in texts]
        # Ensure texts are not empty (at least have some content)
        texts = [t if t.strip() else 'placeholder text' for t in texts]
        
        print(f"✓ Loaded {len(texts)} records")
        print(f"  Texts: {len(texts)} (all strings, min length: {min(len(t) for t in texts)})")
        print(f"  Outcomes: {outcomes.shape} ({config.outcome_type.value})")
        print(f"  Names: {len(names)}")
        if context_features is not None:
            print(f"  Context features: {context_features.shape}")
        
        return texts, outcomes, names, context_features
    
    def run_pipeline(
        self,
        config: DomainConfig,
        data_path: Optional[Path] = None,
        target_feature_count: int = 300,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Compose and run complete pipeline.
        
        This is the main entry point that does everything:
        1. Compose pipeline
        2. Load data
        3. Extract features
        4. Run analysis
        5. Validate results
        
        Parameters
        ----------
        config : DomainConfig
            Domain configuration
        data_path : Path, optional
            Path to data file
        target_feature_count : int
            Target feature count
        use_cache : bool
            Enable caching
            
        Returns
        -------
        results : dict
            Complete analysis results including:
            - pipeline_info: Pipeline composition info
            - data: Loaded data
            - features: Extracted features
            - analysis: Analysis results
            - validation: Validation results
        """
        import time
        total_start = time.time()
        
        # Initialize timing variables (in case of early exceptions)
        compose_elapsed = 0.0
        load_elapsed = 0.0
        fit_elapsed = 0.0
        transform_elapsed = 0.0
        analysis_elapsed = 0.0
        comprehensive_elapsed = 0.0
        validation_elapsed = 0.0
        
        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION STARTED")
        print("=" * 80)
        print(f"Domain: {config.domain}")
        print(f"Target features: {target_feature_count}")
        print(f"Cache: {'enabled' if use_cache else 'disabled'}")
        print("=" * 80)
        
        # Compose pipeline
        print("\n[PHASE 1] Composing pipeline...", flush=True)
        compose_start = time.time()
        pipeline_info = self.compose_pipeline(
            config, data_path, target_feature_count, use_cache
        )
        compose_elapsed = time.time() - compose_start
        print(f"\n✓ Pipeline composition completed in {compose_elapsed:.1f}s", flush=True)
        
        # Load data
        print("\n[PHASE 2] Loading data...", flush=True)
        load_start = time.time()
        texts, outcomes, names, context_features = self.load_data(config, data_path)
        load_elapsed = time.time() - load_start
        print(f"✓ Data loading completed in {load_elapsed:.1f}s", flush=True)
        
        # Extract features
        print("\n" + "=" * 80)
        print("EXTRACTING FEATURES")
        print("=" * 80)
        
        # Fit and transform with pipeline
        pipeline = pipeline_info['pipeline']
        
        # Fit transformers
        import time
        import numpy as np
        
        print("\n" + "-" * 80)
        print("FITTING TRANSFORMERS")
        print("-" * 80)
        
        # Ensure texts is a list (not numpy array) for sklearn pipeline
        if isinstance(texts, np.ndarray):
            texts = [str(x) if not isinstance(x, str) else x for x in texts]
        texts_list = list(texts) if not isinstance(texts, list) else texts
        
        # Get the FeatureUnion from the pipeline
        feature_union = pipeline.named_steps['features']
        n_transformers = len(feature_union.transformer_list)
        print(f"  Processing {len(texts_list)} documents through {n_transformers} transformers...", flush=True)
        print(f"  Transformer names: {[name for name, _ in feature_union.transformer_list]}", flush=True)
        print(f"  Starting fit process (transformers run in PARALLEL)...", flush=True)
        
        # Fit the pipeline (this will fit all transformers in sequence)
        fit_start = time.time()
        try:
            pipeline.fit(texts_list)
            fit_elapsed = time.time() - fit_start
            print(f"\n  ✓ All transformers fitted in {fit_elapsed:.1f}s total", flush=True)
        except Exception as e:
            fit_elapsed = time.time() - fit_start
            print(f"\n  ✗ Pipeline fitting FAILED after {fit_elapsed:.1f}s: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise
        
        # Transform to features
        print("\n" + "-" * 80)
        print("TRANSFORMING TO FEATURES")
        print("-" * 80)
        
        print(f"  Starting transform process...", flush=True)
        transform_start = time.time()
        try:
            features = pipeline.transform(texts_list)
            transform_elapsed = time.time() - transform_start
            print(f"\n  ✓ All transformers transformed in {transform_elapsed:.1f}s total", flush=True)
            print(f"  ✓ Final feature matrix: {features.shape}", flush=True)
        except Exception as e:
            transform_elapsed = time.time() - transform_start
            print(f"\n  ✗ Pipeline transformation FAILED after {transform_elapsed:.1f}s: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise
        
        # Get feature names from FeatureUnion
        print("\n  Extracting feature names...", flush=True)
        feature_names = []
        feature_union = pipeline.named_steps['features']
        for trans_name, transformer in feature_union.transformer_list:
            if transformer is not None and hasattr(transformer, 'get_feature_names_out'):
                try:
                    trans_features = transformer.get_feature_names_out()
                    feature_names.extend([f"{trans_name}_{f}" for f in trans_features])
                except Exception as e:
                    print(f"    Warning: Could not get feature names from {trans_name}: {e}", flush=True)
        
        if not feature_names:
            feature_names = [f"feature_{i}" for i in range(features.shape[1])]
        
        print(f"  ✓ Extracted {len(feature_names)} feature names", flush=True)
        
        # Run analysis
        print("\n" + "=" * 80)
        print("RUNNING ANALYSIS")
        print("=" * 80)
        
        analyzer = pipeline_info['analyzer']
        
        # Option 1: Standard analysis (single ю)
        print("\n  Running standard analysis (analyze_complete)...", flush=True)
        analysis_start = time.time()
        try:
            analysis_results = analyzer.analyze_complete(
                texts=texts,
                outcomes=outcomes,
                names=names,
                genome=features,
                feature_names=feature_names,
                masses=None,
                baseline_features=context_features
            )
            analysis_elapsed = time.time() - analysis_start
            print(f"  ✓ Standard analysis completed in {analysis_elapsed:.1f}s", flush=True)
        except Exception as e:
            analysis_elapsed = time.time() - analysis_start
            print(f"  ✗ Standard analysis FAILED after {analysis_elapsed:.1f}s: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise
        
        # Option 2: Multi-perspective/multi-method analysis (enhanced)
        print("\n" + "-" * 80)
        print("MULTI-DIMENSIONAL ANALYSIS")
        print("-" * 80)
        
        quality_aggregator = pipeline_info['quality_aggregator']
        config = pipeline_info['config']
        
        # Convert perspective strings to enums
        print(f"  Computing comprehensive ю...", flush=True)
        print(f"    Perspectives: {config.perspectives}", flush=True)
        print(f"    Methods: {config.quality_methods}", flush=True)
        print(f"    Scales: {config.scales}", flush=True)
        
        comprehensive_start = time.time()
        try:
            perspective_enums = [NarrativePerspective(p) for p in config.perspectives]
            
            comprehensive_results = quality_aggregator.compute_comprehensive_ю(
                genome=features,
                feature_names=feature_names,
                outcomes=outcomes,
                perspectives=perspective_enums,
                methods=config.quality_methods,
                scales=config.scales,
                context_features=context_features,
                aggregation_method=config.aggregation_method
            )
            
            comprehensive_elapsed = time.time() - comprehensive_start
            print(f"  ✓ Comprehensive analysis completed in {comprehensive_elapsed:.1f}s", flush=True)
            print(comprehensive_results['summary'])
        except Exception as e:
            comprehensive_elapsed = time.time() - comprehensive_start
            print(f"  ✗ Comprehensive analysis FAILED after {comprehensive_elapsed:.1f}s: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise
        
        # Add comprehensive results to analysis
        analysis_results['comprehensive_ю'] = comprehensive_results
        
        # Validate
        print("\n" + "=" * 80)
        print("VALIDATING RESULTS")
        print("=" * 80)
        
        print("  Running validation...", flush=True)
        validation_start = time.time()
        try:
            validation_result = pipeline_info['validator'].validate_domain(
                domain_name=config.domain,
                narrativity=config.pi,
                correlation=analysis_results['r_narrative'],
                coupling=0.5,  # Default coupling, can be overridden
                transformer_info=pipeline_info['rationales']
            )
            validation_elapsed = time.time() - validation_start
            print(f"  ✓ Validation completed in {validation_elapsed:.1f}s", flush=True)
            print(validation_result)
        except Exception as e:
            validation_elapsed = time.time() - validation_start
            print(f"  ✗ Validation FAILED after {validation_elapsed:.1f}s: {e}", flush=True)
            import traceback
            traceback.print_exc()
            # Don't raise - validation failure shouldn't stop the pipeline
            validation_result = {"error": str(e)}
        
        # Compile complete results
        total_elapsed = time.time() - total_start
        
        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION COMPLETED")
        print("=" * 80)
        print(f"Total execution time: {total_elapsed:.1f}s")
        print(f"  - Composition: {compose_elapsed:.1f}s")
        print(f"  - Data loading: {load_elapsed:.1f}s")
        print(f"  - Feature extraction: {fit_elapsed + transform_elapsed:.1f}s")
        print(f"  - Analysis: {analysis_elapsed + comprehensive_elapsed:.1f}s")
        print(f"  - Validation: {validation_elapsed:.1f}s")
        print("=" * 80)
        
        results = {
            'pipeline_info': pipeline_info,
            'data': {
                'texts': texts,
                'outcomes': outcomes,
                'names': names,
                'context_features': context_features
            },
            'features': {
                'genome': features,
                'feature_names': feature_names,
                'shape': features.shape,
                'n_features': features.shape[1] if len(features.shape) > 1 else 0
            },
            'analysis': analysis_results,
            'validation': validation_result,
            'timing': {
                'total': total_elapsed,
                'composition': compose_elapsed,
                'data_loading': load_elapsed,
                'fitting': fit_elapsed,
                'transformation': transform_elapsed,
                'analysis': analysis_elapsed,
                'comprehensive_analysis': comprehensive_elapsed,
                'validation': validation_elapsed
            }
        }
        
        return results

