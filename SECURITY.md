# Security Policy

## Threat Model & Guarantees

Pulsar Notes is designed for confidential and private meetings:

1. **Zero Network Exfiltration:**
   - Offline flags (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `DO_NOT_TRACK=1`, `OLLAMA_NOHISTORY=1`) are enforced at startup in `backend/config.py`.
   - Frontend assets (`frontend/`) contain zero external `https://` or `//` URLs and enforce a strict `Content-Security-Policy` blocking external `connect-src`.
2. **Loopback-Only Binding & Anti-Hijacking:**
   - FastAPI binds strictly to `127.0.0.1:8000` with `TrustedHostMiddleware(allowed_hosts=["127.0.0.1"])`.
   - All `/api/*` and `/ws/live` routes require a per-boot 256-bit `X-Local-Session-Token` (compared via `secrets.compare_digest`) and `Origin` header validation.
3. **Ephemeral RAM & Secure Disk Scrubbing:**
   - Audio PCM buffers are zeroed immediately after processing.
   - Raw transcript segments stay strictly in RAM (`StreamingTranscriber.transcript_segments`) and are scrubbed via **Purge-on-Commit** or on window close (`purge_ephemeral_buffers`).
   - SQLite uses `PRAGMA secure_delete = ON` and `PRAGMA wal_checkpoint(TRUNCATE)` on meeting deletion.

## Reporting a Vulnerability

If you discover a security issue, please open a private security advisory on GitHub or contact the repository maintainer directly rather than posting exploit details publicly.
