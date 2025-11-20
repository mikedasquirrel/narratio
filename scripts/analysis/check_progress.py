#!/usr/bin/env python3
"""Quick progress checker"""
import json
import time
from pathlib import Path

progress_file = Path('movie_transformer_progress.json')

if not progress_file.exists():
    print("Progress file not found - analysis may not have started yet")
    exit(1)

with open(progress_file) as f:
    data = json.load(f)

print("=" * 60)
print("MOVIE TRANSFORMER PROGRESS")
print("=" * 60)
print()
print(f"Status: {data['status'].upper()}")
print(f"Progress: {len(data['completed_transformers'])}/{data['total_transformers']} ({len(data['completed_transformers'])/data['total_transformers']*100:.0f}%)")
print(f"Current: {data.get('current_transformer', 'None')}")
print(f"Errors: {len(data['errors'])}")
print()

if data['completed_transformers']:
    print("Recently completed:")
    for t in data['completed_transformers'][-5:]:
        print(f"  ✓ {t}")
    print()

if data['errors']:
    print("Errors:")
    for err in data['errors']:
        print(f"  ✗ {err['transformer']}: {err['error'][:100]}")
    print()

if data['status'] == 'complete':
    print("✓ ANALYSIS COMPLETE!")
    print()
    print("View results:")
    print("  cat movie_transformer_results.json | python3 -m json.tool | less")

