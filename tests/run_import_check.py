#!/usr/bin/env python3
"""Import diagnostics with progress logging."""
import os
import sys
import time
from datetime import datetime

REQUIRED_ENV = [
    'TF_CPP_MIN_LOG_LEVEL',
    'CUDA_VISIBLE_DEVICES',
    'TF_ENABLE_ONEDNN_OPTS',
    'OMP_NUM_THREADS',
    'TOKENIZERS_PARALLELISM'
]

print("[diag] Checking TensorFlow env vars...")
missing = [var for var in REQUIRED_ENV if var not in os.environ]
if missing:
    print(f"[diag][WARN] Missing env vars: {missing}")
else:
    print("[diag] All TensorFlow env vars present")

sys.path.insert(0, 'narrative_optimization')

steps = [
    ("domain_registry", "from domain_registry import DOMAINS"),
    ("universal_domain_processor", "from universal_domain_processor import UniversalDomainProcessor"),
]

for name, code in steps:
    start = time.time()
    print(f"[diag] Importing {name}...")
    eval(compile(code, '<diag>', 'exec'))
    duration = time.time() - start
    print(f"[diag] ✓ {name} imported in {duration:.2f}s")

print("[diag] Creating UniversalDomainProcessor...")
from universal_domain_processor import UniversalDomainProcessor
proc = UniversalDomainProcessor(use_transformers=False)
print("[diag] ✓ Processor created")

print("[diag] SUCCESS - no mutex deadlock detected")
