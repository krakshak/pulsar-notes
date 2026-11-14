"""Offline & Security Configuration for Zero-Trust Local Meeting Assistant.

Enforces offline mode, zero telemetry, local loopback bindings, and generates
a per-boot 256-bit session token to prevent cross-origin/local network hijacking.
"""

import os
import secrets
from pathlib import Path

# Required offline & anti-telemetry environment variables
MANDATORY_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "DO_NOT_TRACK": "1",
    "ANONYMIZED_TELEMETRY": "False",
    "OLLAMA_NOHISTORY": "1",
    "OLLAMA_DEBUG": "0",
}

RECOMMENDED_OLLAMA_ENV = {
    "OLLAMA_HOST": "127.0.0.1:11434",
    "OLLAMA_ORIGINS": "http://127.0.0.1:8000",
    "OLLAMA_FLASH_ATTENTION": "1",
    "OLLAMA_KV_CACHE_TYPE": "q8_0",
}


def enforce_offline_env() -> None:
    """Sets and enforces mandatory offline & privacy environment variables in os.environ."""
    for key, expected_value in MANDATORY_OFFLINE_ENV.items():
        current_value = os.environ.get(key)
        if current_value is None:
            os.environ[key] = expected_value
        elif current_value.lower() != expected_value.lower():
            # Override insecure override to guarantee zero leakage
            os.environ[key] = expected_value

    for key, val in RECOMMENDED_OLLAMA_ENV.items():
        if key not in os.environ:
            os.environ[key] = val


def validate_offline_env() -> bool:
    """Validates that all mandatory offline environment variables are strictly set."""
    for key, expected_value in MANDATORY_OFFLINE_ENV.items():
        current_value = os.environ.get(key)
        if current_value is None or current_value.lower() != expected_value.lower():
            return False
    return True


# Run enforcement immediately on module import
enforce_offline_env()
assert validate_offline_env(), "Failed to enforce mandatory offline environment variables."

# Generate per-boot 256-bit (32 bytes) URL-safe cryptographic session token
SESSION_TOKEN: str = secrets.token_urlsafe(32)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "local_meetings.db"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# Networking & Service Bindings
HOST = "127.0.0.1"
PORT = 8000
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# AI Model Stack
DEFAULT_ASR_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
FALLBACK_ASR_MODEL = "mlx-community/whisper-large-v3-turbo"
SUPPORTED_ASR_MODELS = [
    "mlx-community/parakeet-tdt-0.6b-v3",
    "mlx-community/parakeet-tdt-0.6b-v2",
    "mlx-community/whisper-large-v3-turbo",
]

DEFAULT_LLM_MODEL = "qwen2.5:14b"
FALLBACK_LLM_MODEL = "llama3.1:8b"
SUPPORTED_LLM_MODELS = [
    "qwen2.5:14b",
    "qwen3:14b",
    "gemma3:12b",
    "qwen3:8b",
    "llama3.1:8b",
]

# Audio Sampling Configuration
AUDIO_TARGET_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1  # Mono
AUDIO_WINDOW_SECONDS = 5.0
AUDIO_OVERLAP_SECONDS = 0.5

