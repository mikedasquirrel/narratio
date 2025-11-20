#!/usr/bin/env python3
"""Minimal test - just import domain processor without using it"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print("Environment set")

import sys
sys.path.insert(0, 'narrative_optimization')

print("Importing domain_registry...")
from domain_registry import DOMAINS
print(f"✓ {len(DOMAINS)} domains")

print("Importing UniversalDomainProcessor...")
print("  (This is where it hangs if mutex issue persists)")

# Try importing with a timeout approach
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Import took too long - likely mutex deadlock")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(10)  # 10 second timeout

try:
    from universal_domain_processor import UniversalDomainProcessor
    signal.alarm(0)  # Cancel timeout
    print("✓ SUCCESS - UniversalDomainProcessor imported!")
    print("✓✓✓ MUTEX FIX WORKS! ✓✓✓")
except TimeoutError:
    print("✗ TIMEOUT - Still hitting mutex deadlock")
    print("  The import is hanging, likely TensorFlow mutex issue")
    sys.exit(1)
except Exception as e:
    signal.alarm(0)
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

