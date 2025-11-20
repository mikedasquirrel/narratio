"""
API Configuration for AI-Enhanced Transformers

IMPORTANT: In production, use environment variables instead of hardcoding keys.
This file is for development/testing purposes.
"""

import os

# OpenAI API Configuration
OPENAI_API_KEY = os.environ.get(
    'OPENAI_API_KEY',
    os.environ.get('OPENAI_API_KEY', 'your-api-key-here')
)

# Semantic Analysis Settings
SEMANTIC_ANALYSIS_MODE = 'hybrid'  # 'keyword', 'ai', or 'hybrid'
SEMANTIC_CACHE_ENABLED = True
EMBEDDING_MODEL = 'text-embedding-3-small'  # Efficient and cost-effective
BATCH_SIZE = 20

