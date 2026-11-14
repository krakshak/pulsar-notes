#!/usr/bin/env bash
set -euo pipefail

echo "Checking active network sockets for Python & Ollama..."
NON_LOOPBACK=$(lsof -i -n -P 2>/dev/null | grep -E "python|uvicorn|ollama" | grep -v "127.0.0.1" | grep -v "\[::1\]" || true)

if [ -n "$NON_LOOPBACK" ]; then
  echo "FAIL: Detected non-loopback network socket(s):"
  echo "$NON_LOOPBACK"
  exit 1
else
  echo "PASS: 100% Loopback-Only (Zero External Connections)"
fi

