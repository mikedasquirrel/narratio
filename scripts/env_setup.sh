#!/bin/bash
# Environment setup for Narrative Optimization - prevents TensorFlow mutex deadlocks

export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-3}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:--1}
export TF_ENABLE_ONEDNN_OPTS=${TF_ENABLE_ONEDNN_OPTS:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

if [[ -n "$1" ]]; then
  echo "[env_setup] Running command: $@"
  exec "$@"
else
  echo "[env_setup] Environment variables configured. Run your python command now."
fi
