# Zero-Trust Local Meeting Assistant

A 100% offline, privacy-first meeting transcription and summarization assistant built for **Apple Silicon (macOS)**. All audio capture, MLX speech-to-text, Ollama LLM summarization, and SQLite storage stay strictly on your machine with zero cloud APIs or external network calls.

## Key Features

- **100% Air-Gapped Inference:** Enforces `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `OLLAMA_NOHISTORY=1`, and loopback-only binding (`127.0.0.1:8000` & `127.0.0.1:11434`).
- **Dual-Stream Audio Capture (`[Me]` vs `[Others]`):** Captures your microphone and system audio (via BlackHole 2ch) in RAM, resamples 48 kHz stereo to 16 kHz mono, and immediately zeroes consumed PCM buffers.
- **On-Device MLX Transcription:** Runs `mlx-community/parakeet-tdt-0.6b-v3` on Apple Silicon GPU/Neural Engine and clears MLX Metal cache on stop.
- **Slack AI–Style Structured Notes:** Generates timestamp-prefixed meeting titles (`YYYY-MM-DD HH:MM — <Title>`), executive summaries, topic-grouped key takeaways (`TopicTakeaway`), and owner-attributed action items via local Ollama (`keep_alive=0`, `num_ctx=32768`).
- **Purge-on-Commit & Secure Delete:** Raw transcript segments live only in RAM and are purged once structured notes commit to SQLite (with `PRAGMA secure_delete = ON` + `wal_checkpoint(TRUNCATE)` on deletion).
- **Zero-CDN Local Web UI & Native macOS Window:** Served locally with strict Content-Security-Policy (CSP), per-boot 256-bit `X-Local-Session-Token`, and optional `PyWebView` desktop wrapper (`desktop_wrapper.py`).

## Project Structure

```text
local_meeting_assistant/
├── architecture/              # Architecture specs & phased implementation plan
├── assets/                    # App icons (app_icon.png, app_icon.icns)
├── backend/                   # FastAPI server, audio DSP, MLX ASR, Ollama summarizer, SQLite DAL
├── frontend/                  # Zero-CDN HTML/CSS/JS workspace + vendored local icons
├── tests/                     # Phase 1–7 automated unit, security, & air-gap E2E tests
├── .github/workflows/ci.yml   # GitHub Actions CI pipeline
├── desktop_wrapper.py         # Native macOS PyWebView desktop launcher
├── launch_desktop.sh          # Desktop .app launcher script
├── run_local.sh               # Local FastAPI + Ollama startup script
├── verify_offline.sh          # Loopback-only socket verification audit
├── pyproject.toml             # Python project & pytest/ruff configuration
└── requirements.txt           # Pinned dependency minimums
```

## Prerequisites

- **macOS on Apple Silicon** (M1/M2/M3/M4/M5)
- **Python 3.11+**
- **[Ollama](https://ollama.com/)** running locally with a pulled model (e.g., `qwen2.5:14b`, `llama3.1:8b`, or `llama3.2:latest`)
- **[BlackHole 2ch](https://existential.audio/blackhole/)** (optional, for capturing system audio `[Others]`)
- Pre-cached HuggingFace ASR model (`mlx-community/parakeet-tdt-0.6b-v3`)

## Quick Start

```bash
# 1. Create & activate virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Launch in browser (http://127.0.0.1:8000)
./run_local.sh

# OR launch as a native macOS desktop window
./launch_desktop.sh
```

## Testing & Verification

```bash
# Run the full Phase 1–7 test suite
.venv/bin/pytest tests/ -v

# Verify zero non-loopback network sockets
./verify_offline.sh
```

## Documentation

- [System Architecture](architecture/local_meeting_assistant_architecture.md)
- [Implementation Plan](architecture/implementation_plan.md)
- [Security Policy](SECURITY.md)
- [Contributing Guide](CONTRIBUTING.md)
