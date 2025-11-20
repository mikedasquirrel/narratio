#!/usr/bin/env python3
"""
Domain Onboarding Wizard

Interactive wizard that guides through domain setup and generates all files.
Reduces domain onboarding from 4 weeks to 2-3 days.
"""

import sys
from pathlib import Path
import json
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipelines.domain_config import (
    DomainConfig, DomainType, OutcomeType,
    NarrativityComponents, DataSchema
)
from src.pipelines.domain_types import get_domain_type_class


class DomainOnboardingWizard:
    """Interactive wizard for domain onboarding"""
    
    def __init__(self, project_root: Path = None):
        """Initialize wizard"""
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent.parent
        
        self.project_root = Path(project_root)
        self.config = None
    
    def run(self):
        """Run the complete onboarding wizard"""
        print("\n" + "🎯" * 40)
        print("\n  NARRATIVE OPTIMIZATION - DOMAIN ONBOARDING WIZARD")
        print("  Automated Workflow for New Domain Integration")
        print("\n" + "🎯" * 40 + "\n")
        
        # Step 1: Domain Information
        self.step1_domain_information()
        
        # Step 2: Narrativity Assessment
        self.step2_narrativity_assessment()
        
        # Step 3: Data Schema
        self.step3_data_schema()
        
        # Step 4: Transformer Selection (auto)
        self.step4_transformer_selection()
        
        # Step 5: Generate Files
        self.step5_generate_files()
        
        # Step 6: Load and Validate Data
        self.step6_load_and_validate()
        
        # Step 7: Run Initial Analysis
        self.step7_run_initial_analysis()
        
        print("\n" + "🎉" * 40)
        print("\n  ONBOARDING COMPLETE")
        print("\n" + "🎉" * 40)
        print("\nNext steps:")
        print(f"1. Review validation report: {self.get_domain_dir() / 'VALIDATION_REPORT.md'}")
        print(f"2. Run comprehensive tests: pytest {self.get_domain_dir() / 'tests'}/")
        print(f"3. View results: http://localhost:5738/{self.config.domain}")
        print()
    
    def step1_domain_information(self):
        """Step 1: Collect basic domain information"""
        print("=" * 80)
        print("STEP 1/7: Domain Information")
        print("=" * 80)
        
        domain_name = input("\n> Domain name: ").strip()
        if not domain_name:
            domain_name = "new_domain"
        
        print("\n> Domain type:")
        print("  [1] sports")
        print("  [2] sports_individual")
        print("  [3] sports_team")
        print("  [4] entertainment")
        print("  [5] nominative")
        print("  [6] business")
        print("  [7] medical")
        print("  [8] hybrid")
        
        type_choice = input("> Select (1-8) [default: 8]: ").strip() or "8"
        type_map = {
            "1": DomainType.SPORTS,
            "2": DomainType.SPORTS_INDIVIDUAL,
            "3": DomainType.SPORTS_TEAM,
            "4": DomainType.ENTERTAINMENT,
            "5": DomainType.NOMINATIVE,
            "6": DomainType.BUSINESS,
            "7": DomainType.MEDICAL,
            "8": DomainType.HYBRID,
        }
        domain_type = type_map.get(type_choice, DomainType.HYBRID)
        
        print("\n> Outcome type:")
        print("  [1] binary (win/loss, success/failure)")
        print("  [2] continuous (score, rating, price)")
        print("  [3] ranked (ordinal ranking)")
        
        outcome_choice = input("> Select (1-3) [default: 2]: ").strip() or "2"
        outcome_map = {
            "1": OutcomeType.BINARY,
            "2": OutcomeType.CONTINUOUS,
            "3": OutcomeType.RANKED,
        }
        outcome_type = outcome_map.get(outcome_choice, OutcomeType.CONTINUOUS)
        
        # Store for later
        self.domain_name = domain_name
        self.domain_type = domain_type
        self.outcome_type = outcome_type
        
        print(f"\n✓ Domain: {domain_name}")
        print(f"✓ Type: {domain_type.value}")
        print(f"✓ Outcome Type: {outcome_type.value}")
    
    def step2_narrativity_assessment(self):
        """Step 2: Assess narrativity components"""
        print("\n" + "=" * 80)
        print("STEP 2/7: Narrativity Assessment")
        print("=" * 80)
        print("\nRate each component from 0.0 to 1.0:")
        print("  (0.0 = completely constrained, 1.0 = completely open)")
        
        structural = float(input("> Structural freedom (how many narrative paths possible?) [0.5]: ") or "0.5")
        temporal = float(input("> Temporal openness (does it unfold over time?) [0.5]: ") or "0.5")
        agency = float(input("> Volitional agency (do actors have choice?) [0.5]: ") or "0.5")
        interpretive = float(input("> Interpretive multiplicity (is judgment subjective?) [0.5]: ") or "0.5")
        format_flex = float(input("> Format flexibility (how flexible is the medium?) [0.5]: ") or "0.5")
        
        narrativity = NarrativityComponents(
            structural=structural,
            temporal=temporal,
            agency=agency,
            interpretive=interpretive,
            format=format_flex
        )
        
        pi = narrativity.calculate_pi()
        
        print(f"\n✓ Calculated п = {pi:.3f}")
        print("  Components:")
        print(f"    Structural: {structural:.2f}")
        print(f"    Temporal: {temporal:.2f}")
        print(f"    Agency: {agency:.2f}")
        print(f"    Interpretive: {interpretive:.2f}")
        print(f"    Format: {format_flex:.2f}")
        
        self.narrativity = narrativity
        self.pi = pi
    
    def step3_data_schema(self):
        """Step 3: Define data schema"""
        print("\n" + "=" * 80)
        print("STEP 3/7: Data Schema")
        print("=" * 80)
        
        text_fields_input = input("\n> Text fields (comma-separated, e.g., 'plot,description'): ").strip()
        if text_fields_input:
            text_fields = [f.strip() for f in text_fields_input.split(',')]
        else:
            text_fields = ['text']
        
        outcome_field = input("> Outcome field name [outcome]: ").strip() or "outcome"
        
        context_fields_input = input("> Context fields (optional, comma-separated, e.g., 'genre,budget'): ").strip()
        context_fields = None
        if context_fields_input:
            context_fields = [f.strip() for f in context_fields_input.split(',')]
        
        name_field = input("> Name field (optional, e.g., 'name', 'title'): ").strip() or None
        
        schema = DataSchema(
            text_fields=text_fields,
            outcome_field=outcome_field,
            context_fields=context_fields,
            name_field=name_field
        )
        
        schema.validate()
        
        print(f"\n✓ Schema validated")
        print(f"  Text fields: {', '.join(text_fields)}")
        print(f"  Outcome field: {outcome_field}")
        if context_fields:
            print(f"  Context fields: {', '.join(context_fields)}")
        if name_field:
            print(f"  Name field: {name_field}")
        
        self.data_schema = schema
    
    def step4_transformer_selection(self):
        """Step 4: Transformer selection (automated)"""
        print("\n" + "=" * 80)
        print("STEP 4/7: Transformer Selection")
        print("=" * 80)
        print("\nNOTE: Transformers are automatically selected based on:")
        print(f"  - Narrativity (п = {self.pi:.3f})")
        print(f"  - Domain type ({self.domain_type.value})")
        print("  - Core transformers (always included)")
        
        # Get domain type preferences
        domain_type_class = get_domain_type_class(self.domain_type)
        if domain_type_class:
            temp_config = DomainConfig(
                domain=self.domain_name,
                type=self.domain_type,
                narrativity=self.narrativity,
                data=self.data_schema,
                outcome_type=self.outcome_type
            )
            domain_type_instance = domain_type_class(temp_config)
            perspectives = domain_type_instance.get_perspective_preferences()
            methods = domain_type_instance.get_quality_method_preferences()
            scales = domain_type_instance.get_scale_preferences()
        else:
            perspectives = ['director', 'audience', 'critic']
            methods = ['weighted_mean', 'ensemble']
            scales = ['micro', 'meso', 'macro']
        
        print(f"\n✓ Selected perspectives: {', '.join(perspectives)}")
        print(f"✓ Selected methods: {', '.join(methods)}")
        print(f"✓ Selected scales: {', '.join(scales)}")
        
        custom_aug = input("\n> Add custom transformers? (comma-separated, or Enter to skip): ").strip()
        transformer_augmentation = []
        if custom_aug:
            transformer_augmentation = [t.strip() for t in custom_aug.split(',')]
        
        self.perspectives = perspectives
        self.quality_methods = methods
        self.scales = scales
        self.transformer_augmentation = transformer_augmentation
    
    def step5_generate_files(self):
        """Step 5: Generate all necessary files"""
        print("\n" + "=" * 80)
        print("STEP 5/7: Generate Files")
        print("=" * 80)
        
        # Create domain config
        self.config = DomainConfig(
            domain=self.domain_name,
            type=self.domain_type,
            narrativity=self.narrativity,
            data=self.data_schema,
            outcome_type=self.outcome_type,
            transformer_augmentation=self.transformer_augmentation,
            perspectives=self.perspectives,
            quality_methods=self.quality_methods,
            scales=self.scales
        )
        
        domain_dir = self.get_domain_dir()
        domain_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = domain_dir / 'config.yaml'
        self.config.to_yaml(config_path)
        print(f"✓ Created: {config_path.relative_to(self.project_root)}")
        
        # Generate data loader template
        self._generate_data_loader(domain_dir)
        
        # Generate test file
        self._generate_test_file(domain_dir)
        
        # Generate Flask route
        self._generate_flask_route(domain_dir)
        
        # Generate template HTML
        self._generate_template_html(domain_dir)
        
        print(f"\n✓ All files generated in: {domain_dir.relative_to(self.project_root)}")
    
    def step6_load_and_validate(self):
        """Step 6: Load and validate data"""
        print("\n" + "=" * 80)
        print("STEP 6/7: Load and Validate Data")
        print("=" * 80)
        
        data_path_input = input("\n> Data file path (JSON or CSV, or Enter to skip): ").strip()
        
        if not data_path_input:
            print("⚠ Skipping data validation (no file provided)")
            return
        
        data_path = Path(data_path_input)
        if not data_path.is_absolute():
            data_path = self.project_root / data_path
        
        if not data_path.exists():
            print(f"⚠ File not found: {data_path}")
            return
        
        # Try to load data
        try:
            from src.pipelines.pipeline_composer import PipelineComposer
            composer = PipelineComposer(self.project_root)
            
            texts, outcomes, names, context_features = composer.load_data(self.config, data_path)
            
            print(f"\n✓ Loaded {len(texts)} records")
            print(f"  Texts: {len(texts)}")
            print(f"  Outcomes: {outcomes.shape} ({self.outcome_type.value})")
            print(f"  Names: {len(names)}")
            if context_features is not None:
                print(f"  Context features: {context_features.shape}")
            
            self.data_loaded = True
            self.data_path = data_path
            
        except Exception as e:
            print(f"⚠ Error loading data: {e}")
            self.data_loaded = False
    
    def step7_run_initial_analysis(self):
        """Step 7: Run initial analysis"""
        print("\n" + "=" * 80)
        print("STEP 7/7: Run Initial Analysis")
        print("=" * 80)
        
        if not hasattr(self, 'data_loaded') or not self.data_loaded:
            print("\n⚠ Skipping analysis (no data loaded)")
            print("  Run analysis later with:")
            print(f"    python -m src.pipelines.pipeline_composer {self.get_domain_dir() / 'config.yaml'}")
            return
        
        run_analysis = input("\n> Run initial analysis now? [y/N]: ").strip().lower()
        
        if run_analysis != 'y':
            print("⚠ Skipping analysis")
            return
        
        try:
            from src.pipelines.pipeline_composer import PipelineComposer
            composer = PipelineComposer(self.project_root)
            
            results = composer.run_pipeline(
                self.config,
                data_path=self.data_path,
                target_feature_count=300,
                use_cache=False
            )
            
            # Save results
            results_path = self.get_domain_dir() / f"{self.domain_name}_results.json"
            self._save_results(results, results_path)
            
            print(f"\n✓ Analysis complete")
            print(f"  Results saved: {results_path.relative_to(self.project_root)}")
            
            # Generate validation report
            self._generate_validation_report(results)
            
        except Exception as e:
            print(f"⚠ Error running analysis: {e}")
            import traceback
            traceback.print_exc()
    
    def get_domain_dir(self) -> Path:
        """Get domain directory path"""
        return self.project_root / 'narrative_optimization' / 'domains' / self.domain_name
    
    def _generate_data_loader(self, domain_dir: Path):
        """Generate data loader template"""
        loader_path = domain_dir / 'data_loader.py'
        
        template = f'''"""
{self.domain_name.title()} Data Loader

Loads and processes {self.domain_name} domain data.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np


class {self.domain_name.title().replace('_', '')}DataLoader:
    """Load and process {self.domain_name} data"""
    
    def __init__(self, data_dir: Optional[str] = None):
        """Initialize data loader"""
        if data_dir is None:
            data_dir = Path(__file__).parent / 'data'
        else:
            data_dir = Path(data_dir)
        
        self.data_dir = Path(data_dir)
    
    def load_data(self) -> List[Dict]:
        """
        Load {self.domain_name} dataset.
        
        Returns
        -------
        list of dict
            Domain records
        """
        # TODO: Implement data loading
        # Look for JSON or CSV files in self.data_dir
        
        json_files = list(self.data_dir.glob('*.json'))
        csv_files = list(self.data_dir.glob('*.csv'))
        
        if json_files:
            with open(json_files[0], 'r') as f:
                return json.load(f)
        elif csv_files:
            df = pd.read_csv(csv_files[0])
            return df.to_dict('records')
        else:
            raise FileNotFoundError(f"No data files found in {{self.data_dir}}")
    
    def validate_data(self, data: List[Dict]) -> bool:
        """Validate data structure"""
        # TODO: Implement validation
        return True
'''
        
        loader_path.write_text(template)
        print(f"✓ Created: {loader_path.relative_to(self.project_root)}")
    
    def _generate_test_file(self, domain_dir: Path):
        """Generate test file"""
        tests_dir = domain_dir / 'tests'
        tests_dir.mkdir(exist_ok=True)
        
        test_path = tests_dir / f'test_{self.domain_name}.py'
        
        template = f'''"""
Tests for {self.domain_name} domain
"""

import pytest
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipelines.domain_config import DomainConfig
from src.pipelines.pipeline_composer import PipelineComposer


class Test{self.domain_name.title().replace('_', '')}:
    """Test {self.domain_name} domain"""
    
    def test_config_loadable(self):
        """Test that config can be loaded"""
        config_path = Path(__file__).parent.parent / 'config.yaml'
        if config_path.exists():
            config = DomainConfig.from_yaml(config_path)
            assert config.domain == '{self.domain_name}'
    
    def test_pipeline_composable(self):
        """Test that pipeline can be composed"""
        config_path = Path(__file__).parent.parent / 'config.yaml'
        if config_path.exists():
            config = DomainConfig.from_yaml(config_path)
            composer = PipelineComposer(project_root)
            pipeline_info = composer.compose_pipeline(config, target_feature_count=200)
            assert pipeline_info['config'] == config
'''
        
        test_path.write_text(template)
        print(f"✓ Created: {test_path.relative_to(self.project_root)}")
    
    def _generate_flask_route(self, domain_dir: Path):
        """Generate Flask route"""
        routes_dir = self.project_root / 'routes'
        routes_dir.mkdir(exist_ok=True)
        
        route_path = routes_dir / f'{self.domain_name}.py'
        
        template = f'''"""
{self.domain_name.title()} Domain Route

Flask route for {self.domain_name} domain dashboard.
"""

from flask import Blueprint, render_template, jsonify
from pathlib import Path
import json
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

{self.domain_name}_bp = Blueprint('{self.domain_name}', __name__)


@{self.domain_name}_bp.route('/{self.domain_name}')
def {self.domain_name}_dashboard():
    """{self.domain_name.title()} domain dashboard"""
    return render_template('{self.domain_name}_dashboard.html')


@{self.domain_name}_bp.route('/{self.domain_name}-results')
def {self.domain_name}_results():
    """{self.domain_name.title()} results page"""
    return render_template('{self.domain_name}_results.html')


@{self.domain_name}_bp.route('/api/{self.domain_name}/results')
def {self.domain_name}_results_api():
    """API endpoint for {self.domain_name} results"""
    results_path = project_root / 'narrative_optimization' / 'domains' / '{self.domain_name}' / '{self.domain_name}_results.json'
    
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
        return jsonify(results)
    else:
        return jsonify({{'error': 'Results not found'}}), 404
'''
        
        route_path.write_text(template)
        print(f"✓ Created: {route_path.relative_to(self.project_root)}")
    
    def _generate_template_html(self, domain_dir: Path):
        """Generate template HTML"""
        templates_dir = self.project_root / 'templates'
        templates_dir.mkdir(exist_ok=True)
        
        dashboard_path = templates_dir / f'{self.domain_name}_dashboard.html'
        
        template = f'''<!DOCTYPE html>
<html>
<head>
    <title>{self.domain_name.title()} - Narrative Optimization</title>
    <link rel="stylesheet" href="{{{{ url_for('static', filename='css/style.css') }}}}">
</head>
<body>
    <div class="container">
        <h1>{self.domain_name.title()} Domain Analysis</h1>
        
        <div id="dashboard-content">
            <p>Domain dashboard for {self.domain_name}.</p>
            <p>Results will be displayed here once analysis is complete.</p>
        </div>
    </div>
</body>
</html>
'''
        
        dashboard_path.write_text(template)
        print(f"✓ Created: {dashboard_path.relative_to(self.project_root)}")
    
    def _save_results(self, results: Dict, results_path: Path):
        """Save results to JSON"""
        # Convert numpy arrays to lists
        def make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            return obj
        
        serializable_results = make_serializable(results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def _generate_validation_report(self, results: Dict):
        """Generate validation report"""
        report_path = self.get_domain_dir() / 'VALIDATION_REPORT.md'
        
        # Extract key metrics
        analysis = results.get('analysis', {})
        r_narrative = analysis.get('r_narrative', 0)
        Д = analysis.get('Д', 0)
        pi = self.pi
        efficiency = Д / pi if pi > 0 else 0
        
        report = f'''# {self.domain_name.title()} - Validation Report

**Domain**: {self.domain_name}  
**Date**: {Path(__file__).stat().st_mtime if Path(__file__).exists() else 'Generated'}  
**Narrativity (п)**: {pi:.3f}

## Hypothesis

**Presumption**: Narrative laws should apply to {self.domain_name}

**Test**: Д/п > 0.5 (narrative efficiency threshold)

## Results

- **Narrativity (п)**: {pi:.3f}
- **Correlation (r)**: {r_narrative:.3f}
- **Bridge (Д)**: {Д:.3f}
- **Efficiency (Д/п)**: {efficiency:.3f}

## Validation Result

**Efficiency Test**: Д/п = {efficiency:.3f}

**Result**: {'✓ PASS' if efficiency > 0.5 else '❌ FAIL'}

## Interpretation

{'Narrative quality influences outcomes in this domain.' if efficiency > 0.5 else 'Reality constraints dominate in this domain.'}

## Next Steps

1. Review detailed results in `{self.domain_name}_results.json`
2. Run comprehensive tests: `pytest {self.domain_name}/tests/`
3. View dashboard: http://localhost:5738/{self.domain_name}
'''
        
        report_path.write_text(report)
        print(f"✓ Created: {report_path.relative_to(self.project_root)}")


if __name__ == '__main__':
    wizard = DomainOnboardingWizard()
    wizard.run()

