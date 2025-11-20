#!/usr/bin/env python3
"""
Test imports to find mutex deadlock
"""
import os
import sys

print("Step 1: Setting environment variables...")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
print("✓ Environment variables set")

print("\nStep 2: Testing basic imports...")
from pathlib import Path
print("✓ Path imported")

print("\nStep 3: Testing domain_registry import...")
sys.path.insert(0, 'narrative_optimization')
from domain_registry import DOMAINS
print(f"✓ domain_registry imported - {len(DOMAINS)} domains")

print("\nStep 4: Testing transformer_selector import...")
print("   (This may take 10-30 seconds on first import)")
try:
    from transformers.transformer_selector import TransformerSelector
    print("✓ TransformerSelector imported")
except Exception as e:
    print(f"✗ TransformerSelector failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 5: Testing universal_domain_processor import...")
try:
    from universal_domain_processor import UniversalDomainProcessor
    print("✓ UniversalDomainProcessor imported")
except Exception as e:
    print(f"✗ UniversalDomainProcessor failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓✓✓ ALL IMPORTS SUCCESSFUL - MUTEX FIX WORKS! ✓✓✓")


