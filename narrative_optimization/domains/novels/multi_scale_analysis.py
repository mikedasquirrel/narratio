"""
Multi-Scale Analysis

Analyzes narrative features across four scales:
- Nano: Individual sentences/phrases
- Micro: Paragraphs, scenes
- Meso: Chapters, sections
- Macro: Full book, author corpus
"""

import json
import numpy as np
from pathlib import Path
import sys
from typing import List, Dict, Any
import re

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from narrative_optimization.domains.novels.data_loader import NovelsDataLoader
from narrative_optimization.src.transformers.nominative import NominativeAnalysisTransformer
from narrative_optimization.src.transformers.phonetic import PhoneticTransformer
from narrative_optimization.src.transformers.ensemble import EnsembleNarrativeTransformer
from narrative_optimization.src.transformers.statistical import StatisticalTransformer

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')


def segment_text(text: str, scale: str) -> List[str]:
    """
    Segment text into scale-appropriate units.
    
    Parameters
    ----------
    text : str
        Full text
    scale : str
        'nano', 'micro', 'meso', or 'macro'
    
    Returns
    -------
    segments : list of str
        Text segments for the scale
    """
    if scale == 'nano':
        # Sentences
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    elif scale == 'micro':
        # Paragraphs (split by double newlines or long pauses)
        paragraphs = re.split(r'\n\n+', text)
        if len(paragraphs) == 1:
            # Fallback: split by sentences, group into ~3 sentence chunks
            sentences = re.split(r'[.!?]+', text)
            chunks = []
            current_chunk = []
            for s in sentences:
                s = s.strip()
                if s:
                    current_chunk.append(s)
                    if len(current_chunk) >= 3:
                        chunks.append(' '.join(current_chunk))
                        current_chunk = []
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            return chunks
        return [p.strip() for p in paragraphs if len(p.strip()) > 50]
    
    elif scale == 'meso':
        # Sections (split by headings or large breaks)
        sections = re.split(r'\n{3,}|#{2,}', text)
        if len(sections) == 1:
            # Fallback: split into ~500 word chunks
            words = text.split()
            chunks = []
            current_chunk = []
            for word in words:
                current_chunk.append(word)
                if len(current_chunk) >= 500:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            return chunks
        return [s.strip() for s in sections if len(s.strip()) > 200]
    
    elif scale == 'macro':
        # Full text
        return [text]
    
    else:
        raise ValueError(f"Unknown scale: {scale}")


def analyze_scale(
    texts: List[str],
    outcomes: np.ndarray,
    scale: str,
    transformer_name: str,
    transformer
) -> Dict[str, Any]:
    """
    Analyze features at a specific scale.
    
    Parameters
    ----------
    texts : list of str
        Full texts
    outcomes : np.ndarray
        Target outcomes
    scale : str
        Scale name
    transformer_name : str
        Name of transformer
    transformer
        Transformer instance
    
    Returns
    -------
    results : dict
        Scale analysis results
    """
    print(f"  Analyzing {scale} scale with {transformer_name}...", end=' ', flush=True)
    
    # Segment texts
    all_segments = []
    segment_to_text_idx = []
    
    for i, text in enumerate(texts):
        segments = segment_text(text, scale)
        all_segments.extend(segments)
        segment_to_text_idx.extend([i] * len(segments))
    
    if not all_segments:
        print("✗ No segments")
        return None
    
    # Extract features from segments
    try:
        if hasattr(transformer, 'fit_transform'):
            segment_features = transformer.fit_transform(all_segments)
        else:
            transformer.fit(all_segments)
            segment_features = transformer.transform(all_segments)
        
        if hasattr(segment_features, 'toarray'):
            segment_features = segment_features.toarray()
        elif isinstance(segment_features, np.ndarray):
            if segment_features.ndim == 1:
                segment_features = segment_features.reshape(-1, 1)
        
        # Aggregate features back to text level
        # Average features across segments for each text
        text_features = []
        for i in range(len(texts)):
            segment_indices = [j for j, text_idx in enumerate(segment_to_text_idx) if text_idx == i]
            if segment_indices:
                text_feature = np.mean(segment_features[segment_indices], axis=0)
            else:
                text_feature = np.zeros(segment_features.shape[1])
            text_features.append(text_feature)
        
        text_features = np.array(text_features)
        
        # Evaluate predictive power
        if text_features.shape[1] > 0:
            model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            model.fit(text_features, outcomes)
            predictions = model.predict(text_features)
            r2 = r2_score(outcomes, predictions)
            
            print(f"✓ R²: {r2:.4f} ({text_features.shape[1]} features)")
            
            return {
                'scale': scale,
                'transformer': transformer_name,
                'n_segments': len(all_segments),
                'n_features': text_features.shape[1],
                'r2': float(r2),
                'mean_features': float(np.mean(text_features)),
                'std_features': float(np.std(text_features))
            }
        else:
            print("✗ No features")
            return None
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def main():
    """Run multi-scale analysis."""
    print("="*80)
    print("MULTI-SCALE ANALYSIS")
    print("="*80)
    print("\nAnalyzing features across nano/micro/meso/macro scales")
    
    # Load data
    print("\n[1/5] Loading data...")
    loader = NovelsDataLoader()
    novels = loader.load_full_dataset()
    
    texts = [n['full_narrative'] for n in novels]
    outcomes = np.array([n['success_score'] for n in novels])
    
    print(f"✓ Loaded {len(novels)} novels")
    
    # Select key transformers for multi-scale analysis
    print("\n[2/5] Initializing transformers...")
    transformers = [
        ('nominative', NominativeAnalysisTransformer()),
        ('phonetic', PhoneticTransformer()),
        ('ensemble', EnsembleNarrativeTransformer()),
        ('statistical', StatisticalTransformer(max_features=50))
    ]
    
    scales = ['nano', 'micro', 'meso', 'macro']
    
    # Analyze each transformer at each scale
    print("\n[3/5] Analyzing scales...")
    results = []
    
    for trans_name, transformer in transformers:
        print(f"\n{trans_name}:")
        for scale in scales:
            result = analyze_scale(texts, outcomes, scale, trans_name, transformer)
            if result:
                results.append(result)
    
    # Aggregate results
    print("\n[4/5] Aggregating results...")
    
    # Group by transformer
    by_transformer = {}
    for result in results:
        trans = result['transformer']
        if trans not in by_transformer:
            by_transformer[trans] = {}
        by_transformer[trans][result['scale']] = result
    
    # Group by scale
    by_scale = {}
    for result in results:
        scale = result['scale']
        if scale not in by_scale:
            by_scale[scale] = []
        by_scale[scale].append(result)
    
    # Print summary
    print("\nScale Performance Summary:")
    print("\nBy Scale:")
    for scale in scales:
        scale_results = by_scale.get(scale, [])
        if scale_results:
            avg_r2 = np.mean([r['r2'] for r in scale_results])
            print(f"  {scale:6s}: Avg R² = {avg_r2:.4f} ({len(scale_results)} transformers)")
    
    print("\nBy Transformer:")
    for trans_name in by_transformer:
        trans_results = by_transformer[trans_name]
        print(f"  {trans_name}:")
        for scale in scales:
            if scale in trans_results:
                r2 = trans_results[scale]['r2']
                print(f"    {scale:6s}: R² = {r2:.4f}")
    
    # Save results
    print("\n[5/5] Saving results...")
    output_path = Path(__file__).parent / 'multi_scale_analysis.json'
    
    output_data = {
        'results': results,
        'by_transformer': {
            trans: {
                scale: {
                    'r2': float(data['r2']),
                    'n_features': data['n_features'],
                    'n_segments': data['n_segments']
                }
                for scale, data in scales_dict.items()
            }
            for trans, scales_dict in by_transformer.items()
        },
        'by_scale': {
            scale: [
                {
                    'transformer': r['transformer'],
                    'r2': float(r['r2']),
                    'n_features': r['n_features']
                }
                for r in results_list
            ]
            for scale, results_list in by_scale.items()
        },
        'summary': {
            'best_scale': max(by_scale.items(), key=lambda x: np.mean([r['r2'] for r in x[1]]) if x[1] else 0)[0],
            'best_transformer': max(by_transformer.items(), key=lambda x: np.mean([r['r2'] for r in x[1].values()]) if x[1] else 0)[0]
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Saved results to {output_path}")
    print("\n" + "="*80)
    print("Multi-Scale Analysis Complete")
    print("="*80)


if __name__ == '__main__':
    main()

