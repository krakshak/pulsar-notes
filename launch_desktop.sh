#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

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

if ! /usr/bin/curl -s http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  if command -v ollama >/dev/null 2>&1; then
    nohup ollama serve >/dev/null 2>&1 &
  fi
fi

cd /Users/kunchum/study/local_meeting_assistant
exec .venv/bin/python3 desktop_wrapper.py
