#!/usr/bin/env python3
"""View movie transformer results"""
import json

data = json.load(open('movie_transformer_results.json'))

print('='*70)
print('MOVIE TRANSFORMER ANALYSIS - FINAL RESULTS')
print('='*70)
print()
print(f'Dataset: {data["metadata"]["total_movies"]:,} movies')
print(f'Successful: {data["metadata"]["successful_transformers"]}/{data["metadata"]["total_transformers"]}')
print(f'Execution time: {data["metadata"]["execution_time"]:.1f} seconds ({data["metadata"]["execution_time"]/60:.1f} minutes)')
print()

print('='*70)
print('TOP 10 TRANSFORMERS (by Test R²)')
print('='*70)
print()
print(f"{'Rank':<5} {'Transformer':<35} {'Test R²':<10} {'Features':<10} {'Time(s)':<10}")
print('-'*70)
for i, t in enumerate(data['top_10'], 1):
    print(f'{i:<5} {t["name"]:<35} {t["test_r2"]:>8.4f}  {t["features_generated"]:>8}  {t["time_seconds"]:>8.1f}')

print()
print('='*70)
print('CATEGORY PERFORMANCE')
print('='*70)
print()
for cat, stats in sorted(data['category_summary'].items(), key=lambda x: x[1]['avg_r2'], reverse=True):
    print(f'{cat:<20} Count: {stats["count"]:>2} | Avg R²: {stats["avg_r2"]:>7.4f} | Max R²: {stats["max_r2"]:>7.4f}')

print()
print('='*70)
print('DATASET COVERAGE')
print('='*70)
print()
successful = [r for r in data['results'] if r.get('success')]
print(f"Total movies analyzed: {data['metadata']['total_movies']:,}")
print(f"  Movies with plots: ~6,051 (7.4%)")
print(f"  Movies with cast: ~5,987 (7.3%)")
print(f"  Movies with ratings: ~5,236 (6.4%)")
print()
print('='*70)

