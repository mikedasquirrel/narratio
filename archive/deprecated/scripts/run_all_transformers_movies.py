"""
Run All Transformers on Merged Movie Dataset

Tests all applicable narrative transformers on comprehensive movie data.
Handles sparse features gracefully - each transformer runs on movies with required fields.

Progress tracking:
- Console output with timestamps
- JSON progress file (real-time)
- Detailed log file
- Result summary

Author: Narrative Optimization Framework
Date: November 16, 2025
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# Add narrative_optimization to path
sys.path.insert(0, str(Path(__file__).parent / 'narrative_optimization' / 'src'))

# Import all applicable transformers
from transformers.nominative import NominativeAnalysisTransformer
from transformers.self_perception import SelfPerceptionTransformer
from transformers.narrative_potential import NarrativePotentialTransformer
from transformers.linguistic_advanced import LinguisticPatternsTransformer
from transformers.relational import RelationalValueTransformer
from transformers.ensemble import EnsembleNarrativeTransformer
from transformers.statistical import StatisticalTransformer
from transformers.phonetic import PhoneticTransformer
from transformers.social_status import SocialStatusTransformer
from transformers.universal_nominative import UniversalNominativeTransformer
from transformers.hierarchical_nominative import HierarchicalNominativeTransformer
from transformers.nominative_richness import NominativeRichnessTransformer
from transformers.emotional_resonance import EmotionalResonanceTransformer
from transformers.authenticity import AuthenticityTransformer
from transformers.conflict_tension import ConflictTensionTransformer
from transformers.expertise_authority import ExpertiseAuthorityTransformer
from transformers.cultural_context import CulturalContextTransformer
from transformers.suspense_mystery import SuspenseMysteryTransformer
from transformers.optics import OpticsTransformer
from transformers.framing import FramingTransformer
from transformers.information_theory import InformationTheoryTransformer
from transformers.namespace_ecology import NamespaceEcologyTransformer
from transformers.anticipatory_commitment import AnticipatoryCommunicationTransformer
from transformers.cognitive_fluency import CognitiveFluencyTransformer
from transformers.quantitative import QuantitativeTransformer
from transformers.discoverability import DiscoverabilityTransformer

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# ============================================================================
# PROGRESS TRACKING INFRASTRUCTURE
# ============================================================================

progress_log = {
    "start_time": datetime.now().isoformat(),
    "dataset": "movies_merged_complete",
    "total_transformers": 0,
    "total_movies": 0,
    "completed_transformers": [],
    "current_transformer": None,
    "errors": [],
    "status": "initializing"
}

LOG_FILE = Path('movie_transformer_progress.log')
PROGRESS_FILE = Path('movie_transformer_progress.json')
RESULTS_FILE = Path('movie_transformer_results.json')

def log_progress(message, level="INFO", transformer_name=None):
    """Log to console and file simultaneously"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted = f"[{timestamp}] {level:8} | {message}"
    print(formatted)
    
    with open(LOG_FILE, 'a') as f:
        f.write(formatted + '\n')
    
    if transformer_name:
        progress_log["current_transformer"] = transformer_name
    
    # Update progress JSON
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress_log, f, indent=2)

def log_error(message, transformer_name):
    """Log error"""
    log_progress(f"ERROR: {message}", level="ERROR", transformer_name=transformer_name)
    progress_log["errors"].append({
        "transformer": transformer_name,
        "error": message,
        "timestamp": datetime.now().isoformat()
    })

# Initialize log files
LOG_FILE.write_text("")  # Clear log
log_progress("="*80, level="INIT")
log_progress("MOVIE TRANSFORMER ANALYSIS - ALL TRANSFORMERS", level="INIT")
log_progress("="*80, level="INIT")
log_progress("", level="INIT")

# ============================================================================
# LOAD DATA
# ============================================================================

log_progress("Loading merged movie dataset...")
data_path = Path('data/domains/movies_merged_complete.json')

if not data_path.exists():
    log_error("Merged dataset not found! Run scripts/merge_movie_datasets.py first", "SYSTEM")
    sys.exit(1)

with open(data_path) as f:
    data = json.load(f)

all_movies = data['movies']
metadata = data['metadata']

log_progress(f"✓ Loaded {len(all_movies):,} movies")
log_progress(f"  Source: {metadata['sources']}")
log_progress(f"  Statistics:")
for key, val in metadata['statistics'].items():
    log_progress(f"    {key}: {val}")
log_progress("")

progress_log["total_movies"] = len(all_movies)
progress_log["status"] = "loaded"

# ============================================================================
# DATA PREPARATION FUNCTIONS
# ============================================================================

def filter_movies_by_requirements(movies: List[Dict], requirements: List[str]) -> List[Dict]:
    """Filter movies that have all required fields"""
    valid = []
    for movie in movies:
        if all(movie.get(req) for req in requirements):
            valid.append(movie)
    return valid

def extract_features(movies: List[Dict], data_type: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Extract features based on data type required by transformer
    
    Args:
        movies: List of movie dicts
        data_type: 'text', 'numerical', 'metadata', or 'hybrid'
    
    Returns:
        X: Feature DataFrame or Series
        y: Target array (ratings or box office)
    """
    if data_type == 'text':
        # Text transformers use full_narrative
        X = pd.Series([m['full_narrative'] for m in movies])
    
    elif data_type == 'numerical':
        # Numerical transformers use structured features
        features = []
        for m in movies:
            features.append({
                'runtime': m.get('runtime', 0) or 0,
                'num_actors': m.get('num_actors', 0),
                'num_genres': len(m.get('genres', [])),
                'plot_length': m.get('plot_length', 0),
                'year': m.get('year', 2000) or 2000,
                'has_plot': 1.0 if m.get('has_plot') else 0.0,
                'has_cast': 1.0 if m.get('has_cast') else 0.0,
                'cast_diversity': m.get('cast_diversity', 0) or 0,
            })
        X = pd.DataFrame(features)
    
    elif data_type == 'metadata':
        # Metadata transformers use title, genres, actors
        X = pd.DataFrame([{
            'title': m.get('title', ''),
            'genres': ', '.join(m.get('genres', [])),
            'actors': ', '.join(m.get('actors', [])[:10]),  # Top 10 actors
            'year': m.get('year', 2000)
        } for m in movies])
    
    elif data_type == 'hybrid':
        # Hybrid transformers get both text and numerical
        text = [m['full_narrative'] for m in movies]
        features = []
        for i, m in enumerate(movies):
            features.append({
                'narrative': text[i],
                'runtime': m.get('runtime', 0) or 0,
                'num_actors': m.get('num_actors', 0),
                'num_genres': len(m.get('genres', [])),
                'year': m.get('year', 2000) or 2000,
            })
        X = pd.DataFrame(features)
    
    else:
        raise ValueError(f"Unknown data_type: {data_type}")
    
    # Extract target (prefer ratings, fallback to box office)
    y = []
    for m in movies:
        if m.get('avg_rating') and m.get('num_ratings', 0) > 5:
            y.append(m['avg_rating'])
        elif m.get('box_office_revenue'):
            # Normalize box office to similar scale as ratings
            y.append(np.log10(m['box_office_revenue'] + 1) / 2)
        else:
            y.append(3.0)  # Neutral default
    
    return X, np.array(y)

def evaluate_transformer(X_train, y_train, X_test, y_test) -> Dict:
    """Evaluate transformer features with simple regression model"""
    try:
        # Handle invalid values
        if not np.all(np.isfinite(X_train)):
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.all(np.isfinite(X_test)):
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Fit Ridge regression
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        
        # Predict
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Metrics
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        return {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse
        }
    except Exception as e:
        return {
            'train_r2': 0.0,
            'test_r2': 0.0,
            'train_rmse': 999.0,
            'test_rmse': 999.0,
            'error': str(e)
        }

# ============================================================================
# TRANSFORMER CONFIGURATION
# ============================================================================

transformers_config = [
    # Text-based transformers (require plot_summary)
    ("Linguistic Patterns", LinguisticPatternsTransformer(), 'text', ['has_plot'], "NLP"),
    ("Nominative Analysis", NominativeAnalysisTransformer(), 'text', ['has_plot'], "NLP"),
    ("Emotional Resonance", EmotionalResonanceTransformer(), 'text', ['has_plot'], "NLP"),
    ("Conflict & Tension", ConflictTensionTransformer(), 'text', ['has_plot'], "NLP"),
    ("Suspense & Mystery", SuspenseMysteryTransformer(), 'text', ['has_plot'], "NLP"),
    ("Narrative Potential", NarrativePotentialTransformer(), 'text', ['has_plot'], "NLP"),
    ("Authenticity", AuthenticityTransformer(), 'text', ['has_plot'], "NLP"),
    ("Cultural Context", CulturalContextTransformer(), 'text', ['has_plot'], "NLP"),
    ("Phonetic Analysis", PhoneticTransformer(), 'text', ['title'], "NLP"),
    ("Information Theory", InformationTheoryTransformer(), 'text', ['has_plot'], "NLP"),
    
    # Structured data transformers
    ("Statistical", StatisticalTransformer(), 'numerical', [], "Statistical"),
    ("Quantitative", QuantitativeTransformer(), 'numerical', [], "Statistical"),
    ("Social Status", SocialStatusTransformer(), 'metadata', ['has_cast'], "Social"),
    ("Optics", OpticsTransformer(), 'metadata', [], "Observability"),
    ("Framing", FramingTransformer(), 'metadata', [], "Framing"),
    ("Discoverability", DiscoverabilityTransformer(), 'metadata', [], "Discovery"),
    
    # Hybrid transformers (use multiple data types)
    # ("Ensemble Narrative", EnsembleNarrativeTransformer(), 'hybrid', ['has_plot'], "Ensemble"),  # SKIP: TF-IDF issues
    # ("Universal Nominative", UniversalNominativeTransformer(), 'text', ['title'], "Nominative"),  # SKIP: Too slow even with sample limit
    ("Nominative Richness", NominativeRichnessTransformer(), 'text', ['has_plot', 'has_cast'], "Nominative"),
    ("Hierarchical Nominative", HierarchicalNominativeTransformer(), 'text', ['has_plot'], "Nominative"),
    ("Cognitive Fluency", CognitiveFluencyTransformer(), 'text', ['title'], "Cognitive"),
    ("Self-Perception", SelfPerceptionTransformer(), 'text', ['has_plot'], "Identity"),
    ("Expertise & Authority", ExpertiseAuthorityTransformer(), 'metadata', ['has_cast'], "Authority"),
    ("Namespace Ecology", NamespaceEcologyTransformer(), 'text', ['has_plot'], "Ecology"),
    ("Relational Value", RelationalValueTransformer(), 'metadata', ['has_cast'], "Relational"),
]

progress_log["total_transformers"] = len(transformers_config)
log_progress(f"Configured {len(transformers_config)} transformers")
log_progress("")

# ============================================================================
# RUN TRANSFORMERS
# ============================================================================

log_progress("="*80)
log_progress("BEGINNING TRANSFORMER ANALYSIS")
log_progress("="*80)
log_progress("")

progress_log["status"] = "running"
all_results = []

for idx, (name, transformer, data_type, requirements, category) in enumerate(transformers_config, 1):
    log_progress(f"{'='*80}", transformer_name=name)
    log_progress(f"Transformer {idx}/{len(transformers_config)}: {name}", transformer_name=name)
    log_progress(f"Category: {category} | Data Type: {data_type} | Requirements: {requirements or 'None'}", transformer_name=name)
    
    try:
        start_time = time.time()
        
        # Filter movies with required fields
        if requirements:
            valid_movies = filter_movies_by_requirements(all_movies, requirements)
        else:
            valid_movies = all_movies
        
        log_progress(f"  Valid movies: {len(valid_movies):,} / {len(all_movies):,} ({len(valid_movies)/len(all_movies)*100:.1f}%)")
        
        if len(valid_movies) < 100:
            log_error(f"Too few valid movies ({len(valid_movies)}), skipping", name)
            continue
        
        # Limit samples for performance (max 10K movies per transformer)
        MAX_SAMPLES = 10000
        if len(valid_movies) > MAX_SAMPLES:
            log_progress(f"  Limiting to {MAX_SAMPLES:,} samples (from {len(valid_movies):,}) for performance")
            from sklearn.model_selection import train_test_split
            valid_movies, _ = train_test_split(valid_movies, train_size=MAX_SAMPLES, random_state=42)
        
        # Split train/test (80/20)
        train_movies, test_movies = train_test_split(valid_movies, test_size=0.2, random_state=42)
        log_progress(f"  Train: {len(train_movies):,} | Test: {len(test_movies):,}")
        
        # Extract features
        log_progress(f"  Extracting {data_type} features...")
        X_train, y_train = extract_features(train_movies, data_type)
        X_test, y_test = extract_features(test_movies, data_type)
        log_progress(f"  ✓ Features extracted (train samples: {len(X_train)})")
        
        # Fit transformer
        log_progress(f"  Fitting {name}...")
        X_train_transformed = transformer.fit_transform(X_train, y_train)
        
        # Handle output format (may be array or DataFrame)
        if hasattr(X_train_transformed, 'shape'):
            n_features = X_train_transformed.shape[1] if len(X_train_transformed.shape) > 1 else 1
        else:
            n_features = len(X_train_transformed[0]) if len(X_train_transformed) > 0 else 0
        
        log_progress(f"  ✓ Generated {n_features} features")
        
        # Transform test set
        log_progress(f"  Transforming test set...")
        X_test_transformed = transformer.transform(X_test)
        log_progress(f"  ✓ Test set transformed")
        
        # Evaluate
        log_progress(f"  Evaluating predictive power...")
        metrics = evaluate_transformer(X_train_transformed, y_train, X_test_transformed, y_test)
        
        elapsed = time.time() - start_time
        
        log_progress(f"  ✓ COMPLETE")
        log_progress(f"    Train R²: {metrics['train_r2']:.4f} | Test R²: {metrics['test_r2']:.4f}")
        log_progress(f"    Train RMSE: {metrics['train_rmse']:.4f} | Test RMSE: {metrics['test_rmse']:.4f}")
        log_progress(f"    Time: {elapsed:.1f}s")
        
        result = {
            'name': name,
            'category': category,
            'data_type': data_type,
            'requirements': requirements,
            'valid_movies': len(valid_movies),
            'train_size': len(train_movies),
            'test_size': len(test_movies),
            'features_generated': n_features,
            'train_r2': metrics['train_r2'],
            'test_r2': metrics['test_r2'],
            'train_rmse': metrics['train_rmse'],
            'test_rmse': metrics['test_rmse'],
            'time_seconds': elapsed,
            'success': True
        }
        
        all_results.append(result)
        progress_log["completed_transformers"].append(name)
        
        log_progress("")
        
    except Exception as e:
        elapsed = time.time() - start_time
        log_error(f"{str(e)}", name)
        
        result = {
            'name': name,
            'category': category,
            'data_type': data_type,
            'error': str(e),
            'time_seconds': elapsed,
            'success': False
        }
        all_results.append(result)
        log_progress("")
        continue

# ============================================================================
# RESULTS SUMMARY
# ============================================================================

progress_log["status"] = "complete"

log_progress("="*80)
log_progress("ANALYSIS COMPLETE")
log_progress("="*80)
log_progress("")

# Filter successful results
successful = [r for r in all_results if r.get('success')]
failed = [r for r in all_results if not r.get('success')]

log_progress(f"Transformers completed: {len(successful)}/{len(transformers_config)}")
log_progress(f"Transformers failed: {len(failed)}")
log_progress("")

if successful:
    # Sort by test R²
    successful_sorted = sorted(successful, key=lambda x: x['test_r2'], reverse=True)
    
    log_progress("="*80)
    log_progress("TOP TRANSFORMERS (by Test R²)")
    log_progress("="*80)
    log_progress("")
    log_progress(f"{'Rank':<5} {'Transformer':<35} {'Category':<15} {'Test R²':<10} {'Features':<10} {'Coverage':<10}")
    log_progress("-"*95)
    
    for rank, result in enumerate(successful_sorted[:10], 1):
        coverage = f"{result['valid_movies']/len(all_movies)*100:.1f}%"
        log_progress(
            f"{rank:<5} {result['name']:<35} {result['category']:<15} "
            f"{result['test_r2']:<10.4f} {result['features_generated']:<10} {coverage:<10}"
        )
    
    log_progress("")
    
    # Category analysis
    log_progress("CATEGORY PERFORMANCE")
    log_progress("-"*50)
    by_category = {}
    for r in successful:
        cat = r['category']
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r['test_r2'])
    
    for cat, scores in sorted(by_category.items(), key=lambda x: np.mean(x[1]), reverse=True):
        log_progress(f"  {cat:<20} Avg R²: {np.mean(scores):.4f} (n={len(scores)})")
    
    log_progress("")

# Save results
output = {
    'metadata': {
        'created_at': datetime.now().isoformat(),
        'dataset': 'movies_merged_complete',
        'total_movies': len(all_movies),
        'total_transformers': len(transformers_config),
        'successful_transformers': len(successful),
        'failed_transformers': len(failed),
        'execution_time': (datetime.now() - datetime.fromisoformat(progress_log['start_time'])).total_seconds()
    },
    'results': all_results,
    'top_10': successful_sorted[:10] if successful else [],
    'category_summary': {
        cat: {
            'count': len(scores),
            'avg_r2': float(np.mean(scores)),
            'max_r2': float(np.max(scores)),
            'min_r2': float(np.min(scores))
        }
        for cat, scores in by_category.items()
    } if successful else {},
    'errors': progress_log['errors']
}

with open(RESULTS_FILE, 'w') as f:
    json.dump(output, f, indent=2)

log_progress(f"✓ Results saved to {RESULTS_FILE}")
log_progress("")
log_progress("="*80)
log_progress("ALL TRANSFORMERS COMPLETE")
log_progress("="*80)

