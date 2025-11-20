#!/usr/bin/env python3
"""Test full processor import"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print("Step 1: Environment set")
print("Step 2: Importing domain registry...")

import sys
sys.path.insert(0, 'narrative_optimization')
from domain_registry import DOMAINS
print(f"✓ {len(DOMAINS)} domains")

print("Step 3: Importing UniversalDomainProcessor...")
try:
    from universal_domain_processor import UniversalDomainProcessor
    print("✓ UniversalDomainProcessor imported!")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Step 4: Creating processor instance...")
try:
    processor = UniversalDomainProcessor()
    print("✓ Processor created!")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓✓✓ ALL TESTS PASSED - READY TO RUN! ✓✓✓")

