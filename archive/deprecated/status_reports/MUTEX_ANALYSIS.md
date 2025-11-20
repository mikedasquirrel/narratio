# TensorFlow Mutex Deadlock Analysis

## Problem
`[mutex.cc : 452] RAW: Lock blocking` - TensorFlow mutex deadlock on macOS

## Root Cause
TensorFlow is trying to initialize Metal GPU support before our environment variables are set.

## Import Chain Causing Issue

```
test_processor.py
  ↓ (sets env vars)
  ↓ import universal_domain_processor
    ↓ (sets env vars again)
    ↓ from pipelines.feature_extraction_pipeline import FeatureExtractionPipeline
      ↓ (sets env vars again)
      ↓ from transformers import * (COMMENTED OUT - good!)
      ↓ BUT transformers/__init__.py still loaded when we call importlib.import_module('transformers')
        ↓ from .universal_themes import UniversalThemesTransformer
          ↓ from sentence_transformers import SentenceTransformer
            ↓ import torch
              ↓ import tensorflow (indirectly)
                ↓ TF tries to init Metal GPU
                  ↓ MUTEX DEADLOCK
```

## Why Environment Variables Don't Help

Environment variables are set AFTER Python starts importing modules. By the time we set them, TensorFlow's C++ code has already started initializing.

## Solutions

### Option 1: Set env vars BEFORE Python starts (BEST)
```bash
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=-1
export TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
python3 app.py
```

### Option 2: Lazy imports everywhere
Don't import FeatureExtractionPipeline until it's actually used.

### Option 3: Create separate process for transformers
Run transformer processing in subprocess with env vars set before process starts.

### Option 4: Don't use transformers that need TensorFlow
Use only lightweight transformers for web interface.

