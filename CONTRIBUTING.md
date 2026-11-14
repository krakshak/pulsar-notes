# Contributing

Thank you for contributing to **Local Meeting Assistant**! Please follow the SDLC workflow and security invariants below.

## Development Workflow

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feat/<short-description>
   ```
2. Keep all changes **100% offline-capable**:
   - Never introduce external CDNs, fonts, analytics, or telemetry in `frontend/` or `backend/`.
   - Never persist raw `.wav`/`.pcm`/`.txt` transcripts to disk.
   - Preserve `X-Local-Session-Token`, `Origin` validation, `TrustedHostMiddleware`, and strict CSP headers.
3. Run the automated test suite before opening a Pull Request:
   ```bash
   .venv/bin/pytest tests/ -v
   ./verify_offline.sh
   ```
4. Use clear, imperative commit messages (e.g., `feat(summarizer): add topic-grouped takeaways`).
