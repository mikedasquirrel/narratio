#!/usr/bin/env python3
"""Quick test - minimal imports"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print("Environment set")
print("Testing domain registry...")

import sys
sys.path.insert(0, 'narrative_optimization')
from domain_registry import DOMAINS
print(f"✓ {len(DOMAINS)} domains loaded")

print("SUCCESS - No mutex deadlock!")

