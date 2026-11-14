"""FastAPI Server, Anti-Hijacking Security Middleware & WebSocket State Machine.

Features:
- Strictly bound to 127.0.0.1:8000 with access logging disabled.
- Enforces TrustedHostMiddleware(allowed_hosts=["127.0.0.1"]).
- Enforces per-boot 256-bit X-Local-Session-Token and Origin checks on /api/* and /ws/live.
- Strict Content-Security-Policy (CSP) header blocking all external connections.
- Lifecycle State Machine: IDLE -> RECORDING -> SUMMARIZING -> IDLE (or SUMMARIZATION_FAILED -> Retry/Discard).
- Registers atexit / SIGINT / SIGTERM cleanup hooks to scrub RAM buffers on shutdown.
"""

import atexit
import asyncio
from contextlib import asynccontextmanager
import os
import secrets
import signal
import sys
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from backend import config, database
from backend.audio_engine import AudioEngine, list_input_devices
from backend.config import (
    DEFAULT_ASR_MODEL,
    DEFAULT_LLM_MODEL,
    FRONTEND_DIR,
    HOST,
    PORT,
)
from backend.schemas import (
    ActionItemCreate,
    ActionItemDB,
    ActionItemUpdate,
    HealthResponse,
    MeetingResponse,
    MeetingRetryRequest,
    MeetingStartRequest,
    MeetingStopRequest,
    MeetingUpdateRequest,
)
from backend.summarizer import SummarizationError, check_ollama_health, summarize_and_commit
from backend.transcriber import StreamingTranscriber

ALLOWED_ORIGINS = {f"http://{HOST}:{PORT}"}
CSP_POLICY = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline'; "
    f"connect-src 'self' http://{HOST}:{PORT} ws://{HOST}:{PORT}; "
    "img-src 'self' data:;"
)


class SessionStateManager:
    """Thread-safe lifecycle state machine for live meeting capture and summarization."""

    def __init__(self) -> None:
        self.state: str = "IDLE"  # IDLE | RECORDING | SUMMARIZING | SUMMARIZATION_FAILED
        self.started_at: Optional[float] = None
        self.duration_seconds: int = 0
        self.asr_model: str = DEFAULT_ASR_MODEL
        self.llm_model: str = DEFAULT_LLM_MODEL
        self.last_error: Optional[str] = None
        self.audio_engine: Optional[AudioEngine] = None
        self.transcriber: Optional[StreamingTranscriber] = None

    def elapsed_seconds(self) -> int:
        if self.state == "RECORDING" and self.started_at is not None:
            return int(time.time() - self.started_at)
        return self.duration_seconds

    def scrub_all(self) -> None:
        """Immediately stops streams, unloads MLX weights, and zeroes all RAM buffers."""
        if self.audio_engine is not None:
            try:
                self.audio_engine.stop()
            except Exception:
                pass
            self.audio_engine = None

        if self.transcriber is not None:
            try:
                self.transcriber.unload_asr_model()
                self.transcriber.purge_transcript()
            except Exception:
                pass
            self.transcriber = None

        self.state = "IDLE"
        self.started_at = None
        self.duration_seconds = 0
        self.last_error = None


session = SessionStateManager()


def purge_ephemeral_buffers() -> None:
    """Cleanup hook for atexit, SIGINT, SIGTERM, and PyWebView window close."""
    session.scrub_all()


def _install_signal_handlers() -> None:
    """Registers SIGINT and SIGTERM hooks that scrub RAM buffers before exiting."""
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handler = signal.getsignal(sig)

        def _handler(signum, frame, prev=previous_handler):
            purge_ephemeral_buffers()
            if callable(prev) and prev not in (signal.SIG_DFL, signal.SIG_IGN):
                prev(signum, frame)
            elif prev == signal.SIG_DFL:
                raise SystemExit(0)

        try:
            signal.signal(sig, _handler)
        except ValueError:
            # signal.signal only works in main thread
            pass


atexit.register(purge_ephemeral_buffers)
_install_signal_handlers()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes SQLite DB on startup and scrubs RAM buffers on shutdown."""
    database.init_db()
    yield
    purge_ephemeral_buffers()


app = FastAPI(
    title="Zero-Trust Local Meeting Assistant",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)

# 1. Enforce strict Host header validation (blocks DNS rebinding)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=[HOST])


# 2. Enforce Session Token, Origin validation, and strict CSP headers
@app.middleware("http")
async def zero_trust_security_middleware(request: Request, call_next) -> Response:
    path = request.url.path

    if path.startswith("/api/"):
        # Validate per-boot X-Local-Session-Token
        provided_token = request.headers.get("X-Local-Session-Token", "")
        if not secrets.compare_digest(provided_token, config.SESSION_TOKEN):
            return JSONResponse(
                status_code=403,
                content={"detail": "Forbidden: Missing or invalid X-Local-Session-Token"},
            )

        # Validate Origin on state-changing mutations (or whenever Origin is sent)
        origin = request.headers.get("origin")
        if origin is not None and origin not in ALLOWED_ORIGINS:
            return JSONResponse(
                status_code=403,
                content={"detail": "Forbidden: Cross-origin API access blocked"},
            )

    response = await call_next(request)
    response.headers["Content-Security-Policy"] = CSP_POLICY
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    return response


# Mount local static assets if frontend/ exists
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ==============================================================================
# Root Frontend Shell (with Session Token Injection)
# ==============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_index() -> HTMLResponse:
    """Serves the local frontend HTML with the per-boot SESSION_TOKEN injected."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        html = index_path.read_text(encoding="utf-8")
        html = html.replace("__LOCAL_SESSION_TOKEN__", config.SESSION_TOKEN)
    else:
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="local-session-token" content="{config.SESSION_TOKEN}">
  <title>Local Meeting Notes</title>
</head>
<body>
  <h1>Zero-Trust Local Meeting Assistant</h1>
  <p>Backend API active on {HOST}:{PORT}. Frontend UI will be served in Phase 6.</p>
</body>
</html>"""
    return HTMLResponse(content=html, headers={"Content-Security-Policy": CSP_POLICY})


# ==============================================================================
# 12 REST API Endpoints
# ==============================================================================

@app.get("/api/health", response_model=HealthResponse)
async def get_health() -> HealthResponse:
    """1. Returns session state, local Ollama status, and macOS audio input devices."""
    ollama_info = await run_in_threadpool(check_ollama_health)
    devices = await run_in_threadpool(list_input_devices)
    return HealthResponse(
        status="ok",
        session_state=session.state,
        ollama_reachable=ollama_info["reachable"],
        available_models=ollama_info["models"],
        audio_devices=devices,
    )


@app.get("/api/meetings")
async def list_meetings(
    search: Optional[str] = Query(default=None),
    filter: str = Query(default="all"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """2. Returns saved meetings ordered by created_at DESC with optional search/filter."""
    filter_open = filter == "open_tasks"
    return await run_in_threadpool(
        database.list_meetings,
        search=search,
        filter_open_tasks=filter_open,
        limit=limit,
        offset=offset,
    )


@app.get("/api/meetings/{meeting_id}", response_model=MeetingResponse)
async def get_meeting(meeting_id: int):
    """3. Returns full details for a single meeting."""
    meeting = await run_in_threadpool(database.get_meeting, meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return meeting


@app.patch("/api/meetings/{meeting_id}", response_model=MeetingResponse)
async def update_meeting(meeting_id: int, payload: MeetingUpdateRequest):
    """4. Updates title, summary, or takeaways inline."""
    updated = await run_in_threadpool(
        database.update_meeting,
        meeting_id=meeting_id,
        title=payload.title,
        summary=payload.summary,
        takeaways=payload.takeaways,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return updated


@app.post("/api/meetings/{meeting_id}/action-items", response_model=ActionItemDB)
async def add_action_item(meeting_id: int, payload: ActionItemCreate):
    """5. Adds a new action item to a meeting."""
    created = await run_in_threadpool(
        database.create_action_item,
        meeting_id=meeting_id,
        owner=payload.owner,
        task=payload.task,
        due_date=payload.due_date,
    )
    if not created:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return created


@app.patch("/api/action-items/{item_id}", response_model=ActionItemDB)
async def update_action_item(item_id: str, payload: ActionItemUpdate):
    """6. Toggles done or edits owner, task, or due_date on an action item."""
    updated = await run_in_threadpool(
        database.update_action_item,
        item_id=item_id,
        done=payload.done,
        owner=payload.owner,
        task=payload.task,
        due_date=payload.due_date,
        position=payload.position,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Action item not found")
    return updated


@app.delete("/api/action-items/{item_id}")
async def delete_action_item(item_id: str):
    """7. Deletes a single action item."""
    deleted = await run_in_threadpool(database.delete_action_item, item_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Action item not found")
    return {"status": "deleted", "id": item_id}


@app.delete("/api/meetings/{meeting_id}")
async def delete_meeting(meeting_id: int):
    """8. Deletes a meeting record and runs PRAGMA wal_checkpoint(TRUNCATE)."""
    deleted = await run_in_threadpool(database.delete_meeting, meeting_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return {"status": "deleted", "id": meeting_id}


@app.post("/api/meeting/start")
async def start_meeting(payload: Optional[MeetingStartRequest] = None):
    """9. Starts dual-stream audio capture & parakeet-mlx streaming transcription."""
    if session.state != "IDLE":
        raise HTTPException(
            status_code=409,
            detail=f"Cannot start meeting while session is in state '{session.state}'",
        )

    req = payload or MeetingStartRequest()

    # Verify Ollama is reachable before starting
    ollama_info = await run_in_threadpool(check_ollama_health)
    if not ollama_info["reachable"]:
        raise HTTPException(
            status_code=503,
            detail="Local Ollama server is not reachable on 127.0.0.1:11434. Start Ollama before recording.",
        )

    asr_model = req.asr_model or DEFAULT_ASR_MODEL
    transcriber = StreamingTranscriber(model_name=asr_model)
    await run_in_threadpool(transcriber.load_model)

    engine = AudioEngine()
    transcriber.start_worker(engine.audio_queue)
    engine.start(mic_device_id=req.mic_device_id, system_device_id=req.system_device_id)

    session.asr_model = asr_model
    session.transcriber = transcriber
    session.audio_engine = engine
    session.started_at = time.time()
    session.duration_seconds = 0
    session.last_error = None
    session.state = "RECORDING"

    return {"status": "recording", "session_state": session.state, "asr_model": asr_model}


@app.post("/api/meeting/stop")
async def stop_meeting(payload: Optional[MeetingStopRequest] = None):
    """10. Stops recording, unloads ASR, summarizes via Ollama, and commits to SQLite."""
    if session.state != "RECORDING":
        raise HTTPException(
            status_code=409,
            detail=f"Cannot stop meeting while session is in state '{session.state}'",
        )

    req = payload or MeetingStopRequest()
    llm_model = req.llm_model or DEFAULT_LLM_MODEL
    session.llm_model = llm_model

    if session.started_at is not None:
        session.duration_seconds = max(0, int(time.time() - session.started_at))

    # Stop audio streams immediately and scrub raw PCM buffers
    if session.audio_engine is not None:
        session.audio_engine.stop()
        session.audio_engine = None

    session.state = "SUMMARIZING"

    try:
        saved_meeting = await run_in_threadpool(
            summarize_and_commit,
            transcriber=session.transcriber,
            llm_model=llm_model,
            asr_model=session.asr_model,
            duration_seconds=session.duration_seconds,
        )
    except SummarizationError as exc:
        # Retain transcriber & RAM transcript segments for user retry!
        session.state = "SUMMARIZATION_FAILED"
        session.last_error = str(exc)
        raise HTTPException(
            status_code=502,
            detail=f"Summarization failed (transcript safely retained in RAM): {exc}",
        )
    except Exception as exc:
        session.state = "SUMMARIZATION_FAILED"
        session.last_error = str(exc)
        raise HTTPException(status_code=500, detail=str(exc))

    # Commit & purge succeeded
    session.transcriber = None
    session.started_at = None
    session.duration_seconds = 0
    session.last_error = None
    session.state = "IDLE"

    return saved_meeting


@app.post("/api/meeting/retry-summarize")
async def retry_summarize(payload: Optional[MeetingRetryRequest] = None):
    """11. Re-runs summarization on the retained in-memory transcript after a failure."""
    if session.state != "SUMMARIZATION_FAILED" or session.transcriber is None:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot retry summarization in state '{session.state}'",
        )

    req = payload or MeetingRetryRequest()
    llm_model = req.llm_model or session.llm_model
    session.llm_model = llm_model
    session.state = "SUMMARIZING"

    try:
        saved_meeting = await run_in_threadpool(
            summarize_and_commit,
            transcriber=session.transcriber,
            llm_model=llm_model,
            asr_model=session.asr_model,
            duration_seconds=session.duration_seconds,
        )
    except SummarizationError as exc:
        session.state = "SUMMARIZATION_FAILED"
        session.last_error = str(exc)
        raise HTTPException(
            status_code=502,
            detail=f"Retry summarization failed (transcript retained in RAM): {exc}",
        )

    session.transcriber = None
    session.started_at = None
    session.duration_seconds = 0
    session.last_error = None
    session.state = "IDLE"

    return saved_meeting


@app.post("/api/meeting/discard")
async def discard_meeting():
    """12. Stops recording or clears failed state and scrubs all RAM buffers without saving."""
    session.scrub_all()
    return {"status": "discarded", "session_state": session.state}


# ==============================================================================
# Protected WebSocket Streamer (WS /ws/live?token=...)
# ==============================================================================

@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket, token: Optional[str] = Query(default=None)):
    """Streams live RMS meters, word/token counts, and latest transcript snippet."""
    # Validate Origin before accepting WebSocket handshake
    origin = websocket.headers.get("origin")
    if origin is not None and origin not in ALLOWED_ORIGINS:
        await websocket.close(code=1008)
        return

    # Validate per-boot session token before accepting WebSocket handshake
    if not token or not secrets.compare_digest(token, config.SESSION_TOKEN):
        await websocket.close(code=1008)
        return

    await websocket.accept()

    try:
        while True:
            mic_rms, sys_rms = (0.0, 0.0)
            if session.audio_engine is not None:
                mic_rms, sys_rms = session.audio_engine.get_levels()

            stats = {"word_count_in_ram": 0, "estimated_tokens": 0, "latest_snippet": ""}
            if session.transcriber is not None:
                stats = session.transcriber.get_stats()

            await websocket.send_json(
                {
                    "status": session.state.lower(),
                    "elapsed_seconds": session.elapsed_seconds(),
                    "mic_rms": round(mic_rms, 4),
                    "system_rms": round(sys_rms, 4),
                    "word_count_in_ram": stats["word_count_in_ram"],
                    "estimated_tokens": stats["estimated_tokens"],
                    "latest_snippet": stats["latest_snippet"],
                }
            )
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
