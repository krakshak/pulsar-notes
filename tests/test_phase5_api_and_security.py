"""Automated Test Suite for Phase 5: FastAPI Security, Anti-Hijacking & WebSocket State Machine.

Tests:
1. test_rejects_missing_or_invalid_session_token: Verifies GET /api/meetings, POST /api/meeting/start,
   and WS /ws/live reject missing or invalid SESSION_TOKEN with 403.
2. test_rejects_cross_origin_websocket_hijack: Verifies WS /ws/live with Origin: https://evil.example.com
   is immediately rejected.
3. test_csp_headers_present: Verifies GET / returns strict Content-Security-Policy header and SESSION_TOKEN.
4. test_state_machine_conflict_and_retry_flow: Verifies double-start returns 409 Conflict and
   RECORDING -> SUMMARIZATION_FAILED -> retry-summarize -> IDLE works end-to-end.
"""

from pathlib import Path
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
import pytest

from backend import config, database, main
from backend.summarizer import SummarizationError


@pytest.fixture
def isolated_client(tmp_path: Path, monkeypatch):
    """Provides a TestClient bound to http://127.0.0.1:8000 with an isolated temp DB."""
    test_db = tmp_path / "test_api.db"
    monkeypatch.setattr(config, "DB_PATH", test_db)
    monkeypatch.setattr(database, "DB_PATH", test_db)
    database.init_db(test_db)
    main.session.scrub_all()

    with TestClient(main.app, base_url="http://127.0.0.1:8000") as client:
        yield client

    main.session.scrub_all()


def test_rejects_missing_or_invalid_session_token(isolated_client: TestClient):
    """Verifies GET /api/meetings, POST /api/meeting/start, and WS /ws/live return 403 without valid token."""
    # 1. Missing token on GET /api/meetings
    resp = isolated_client.get("/api/meetings")
    assert resp.status_code == 403

    # 2. Invalid token on POST /api/meeting/start
    resp = isolated_client.post(
        "/api/meeting/start",
        headers={"X-Local-Session-Token": "invalid-token-123"},
    )
    assert resp.status_code == 403

    # 3. Missing/invalid token on WS /ws/live
    with pytest.raises(WebSocketDisconnect):
        with isolated_client.websocket_connect("ws://127.0.0.1:8000/ws/live?token=wrong-token"):
            pass

    # 4. Valid token succeeds on GET /api/meetings
    valid_resp = isolated_client.get(
        "/api/meetings",
        headers={"X-Local-Session-Token": config.SESSION_TOKEN},
    )
    assert valid_resp.status_code == 200
    assert valid_resp.json() == []


def test_rejects_cross_origin_websocket_hijack(isolated_client: TestClient):
    """Attempts to connect to WS /ws/live with Origin: https://evil.example.com and confirms rejection."""
    with pytest.raises(WebSocketDisconnect):
        with isolated_client.websocket_connect(
            f"ws://127.0.0.1:8000/ws/live?token={config.SESSION_TOKEN}",
            headers={"Origin": "https://evil.example.com"},
        ):
            pass

    # Also verify cross-origin HTTP API requests are rejected with 403
    resp = isolated_client.post(
        "/api/meeting/discard",
        headers={
            "X-Local-Session-Token": config.SESSION_TOKEN,
            "Origin": "https://evil.example.com",
        },
    )
    assert resp.status_code == 403

    # Valid local origin + valid token connects cleanly on WS /ws/live
    with isolated_client.websocket_connect(
        f"ws://127.0.0.1:8000/ws/live?token={config.SESSION_TOKEN}",
        headers={"Origin": "http://127.0.0.1:8000"},
    ) as ws:
        msg = ws.receive_json()
        assert msg["status"] == "idle"
        assert "mic_rms" in msg
        assert "system_rms" in msg


def test_csp_headers_present(isolated_client: TestClient):
    """Verifies GET / returns strict Content-Security-Policy header and injects SESSION_TOKEN."""
    resp = isolated_client.get("/")
    assert resp.status_code == 200
    csp = resp.headers.get("content-security-policy", "")
    assert "default-src 'self'" in csp
    assert "script-src 'self'" in csp
    assert "connect-src 'self' http://127.0.0.1:8000 ws://127.0.0.1:8000" in csp
    assert config.SESSION_TOKEN in resp.text


def test_state_machine_conflict_and_retry_flow(isolated_client: TestClient, monkeypatch):
    """Verifies double-start returns 409 Conflict, and RECORDING -> SUMMARIZATION_FAILED -> retry -> IDLE."""
    headers = {
        "X-Local-Session-Token": config.SESSION_TOKEN,
        "Origin": "http://127.0.0.1:8000",
    }

    # Stub Ollama health, AudioEngine, and StreamingTranscriber so test needs no mic/GPU/Ollama
    monkeypatch.setattr(main, "check_ollama_health", lambda: {"reachable": True, "models": ["qwen2.5:14b"]})

    class FakeAudioEngine:
        def __init__(self):
            self.audio_queue = None
        def start(self, **kwargs):
            return True
        def stop(self):
            pass
        def get_levels(self):
            return (0.12, 0.34)

    class FakeTranscriber:
        def __init__(self, model_name=""):
            self.model_name = model_name
            self.transcript_segments = ["[Me] We need to ship the release by Friday."]
        def load_model(self):
            pass
        def start_worker(self, q):
            pass
        def unload_asr_model(self):
            return 0.0
        def purge_transcript(self):
            self.transcript_segments.clear()
        def get_transcript(self):
            return list(self.transcript_segments)
        def get_stats(self):
            return {"word_count_in_ram": 8, "estimated_tokens": 11, "latest_snippet": self.transcript_segments[0]}

    monkeypatch.setattr(main, "AudioEngine", FakeAudioEngine)
    monkeypatch.setattr(main, "StreamingTranscriber", FakeTranscriber)

    # 1. Start meeting -> RECORDING
    r1 = isolated_client.post("/api/meeting/start", headers=headers)
    assert r1.status_code == 200
    assert r1.json()["session_state"] == "RECORDING"

    # 2. Calling start again while RECORDING -> 409 Conflict
    r_conflict = isolated_client.post("/api/meeting/start", headers=headers)
    assert r_conflict.status_code == 409

    # 3. Stop meeting with simulated Ollama failure -> SUMMARIZATION_FAILED (transcript retained)
    def fail_summarize(**kwargs):
        raise SummarizationError("Ollama timeout")

    monkeypatch.setattr(main, "summarize_and_commit", fail_summarize)

    r_stop = isolated_client.post("/api/meeting/stop", json={"llm_model": "qwen2.5:14b"}, headers=headers)
    assert r_stop.status_code == 502
    assert main.session.state == "SUMMARIZATION_FAILED"
    assert main.session.transcriber is not None
    assert len(main.session.transcriber.get_transcript()) == 1

    # 4. Retry summarize with fallback model -> commits to DB and transitions to IDLE
    def succeed_summarize(transcriber, llm_model, asr_model, duration_seconds, **kwargs):
        saved = database.create_meeting(
            title="Release Planning Sync",
            summary="Agreed to ship the release by Friday.",
            takeaways=["Release scheduled for Friday"],
            asr_model=asr_model,
            llm_model=llm_model,
            action_items=[{"owner": "Me", "task": "Ship the release", "due_date": "Friday"}],
            duration_seconds=duration_seconds,
        )
        transcriber.purge_transcript()
        return saved

    monkeypatch.setattr(main, "summarize_and_commit", succeed_summarize)

    r_retry = isolated_client.post(
        "/api/meeting/retry-summarize",
        json={"llm_model": "llama3.1:8b"},
        headers=headers,
    )
    assert r_retry.status_code == 200
    saved_data = r_retry.json()
    assert saved_data["title"] == "Release Planning Sync"
    assert saved_data["llm_model"] == "llama3.1:8b"
    assert main.session.state == "IDLE"

    # Verify meeting is persisted in DB
    meetings_resp = isolated_client.get("/api/meetings", headers={"X-Local-Session-Token": config.SESSION_TOKEN})
    assert meetings_resp.status_code == 200
    assert len(meetings_resp.json()) == 1
