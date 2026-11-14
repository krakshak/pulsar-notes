#!/usr/bin/env bash
set -euo pipefail

# HuggingFace / MLX / Python offline flags
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export DO_NOT_TRACK=1
export ANONYMIZED_TELEMETRY=False

# Ollama strict loopback, zero prompt logging, and memory optimizations
export OLLAMA_HOST="127.0.0.1:11434"
export OLLAMA_ORIGINS="http://127.0.0.1:8000"
export OLLAMA_NOHISTORY=1
export OLLAMA_DEBUG=0
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE="q8_0"

# Target port and host
export HOST="127.0.0.1"
export PORT="8000"

# Ensure Ollama is running on loopback only
if ! curl -s http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  if command -v ollama >/dev/null 2>&1; then
    echo "Starting local Ollama daemon on loopback..."
    ollama serve >/dev/null 2>&1 &
  fi
fi

# Run FastAPI backend via virtual environment if present
if [ -f ".venv/bin/uvicorn" ]; then
  exec .venv/bin/uvicorn backend.main:app --host 127.0.0.1 --port 8000 --no-access-log
else
  exec uvicorn backend.main:app --host 127.0.0.1 --port 8000 --no-access-log
fi

