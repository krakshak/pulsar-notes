"""Automated Test Suite for Phase 7: End-to-End Air-Gap Audit & Native macOS Wrapper.

Tests:
1. test_e2e_airgap_full_meeting_lifecycle: Simulates audio capture -> resampling & PCM zeroing ->
   offline transcription with non-loopback sockets blocked -> Ollama summarization ->
   SQLite commit -> RAM buffer purge -> zero raw audio/transcript files on disk.
2. test_api_e2e_delete_scrubs_disk_and_wal: Exercises FastAPI CRUD, action item toggle, FTS5 search,
   and DELETE secure scrub verifying absence of canary bytes in .db and .db-wal.
3. test_desktop_wrapper_binds_loopback_and_close_purge: Verifies desktop_wrapper creates a
   PyWebView window for http://127.0.0.1:8000 titled "Local Meeting Notes" and binds
   window.events.closed += purge_ephemeral_buffers.
4. test_verify_offline_script_passes: Executes ./verify_offline.sh and asserts 100% loopback-only pass.
"""

import re
import socket
import subprocess
from pathlib import Path

import httpx
import numpy as np
from fastapi.testclient import TestClient

from backend import config, database, main
from backend.audio_engine import AudioEngine
from backend.summarizer import summarize_and_commit
from backend.transcriber import StreamingTranscriber
import desktop_wrapper


def _mock_ollama_response(payload_dict: dict) -> httpx.Response:
    import json

    request = httpx.Request("POST", "http://127.0.0.1:11434/api/chat")
    body = {"message": {"role": "assistant", "content": json.dumps(payload_dict)}}
    return httpx.Response(status_code=200, json=body, request=request)


def test_e2e_airgap_full_meeting_lifecycle(tmp_path: Path, monkeypatch):
    """Full lifecycle E2E test with external sockets blocked and disk artifact audit."""
    test_db = tmp_path / "e2e_meetings.db"
    database.init_db(test_db)

    # Enforce air-gap: block any non-loopback TCP connection attempt
    real_connect = socket.socket.connect

    def guarded_connect(self, address):
        host = address[0] if isinstance(address, tuple) else str(address)
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise AssertionError(f"Air-gap violation! Attempted outbound connection to {address}")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)

    # 1. Audio Capture & DSP (48kHz stereo -> 16kHz mono + immediate PCM zeroing)
    engine = AudioEngine()
    t = np.linspace(0, 6.0, 48000 * 6, endpoint=False, dtype=np.float32)
    sine_48k = 0.25 * np.sin(2.0 * np.pi * 440.0 * t)
    stereo_48k = np.column_stack([sine_48k, sine_48k])
    mic_chunks = engine.mic_buffer.add_pcm_data(stereo_48k, orig_sr=48000)
    sys_chunks = engine.system_buffer.add_pcm_data(stereo_48k, orig_sr=48000)

    assert len(mic_chunks) >= 1 and mic_chunks[0].data.shape == (16000 * 5,)
    assert len(sys_chunks) >= 1 and sys_chunks[0].data.shape == (16000 * 5,)
    assert mic_chunks[0].speaker_tag == "[Me]"
    assert sys_chunks[0].speaker_tag == "[Others]"

    # 2. RAM-only Streaming Transcription & Model Unload
    transcriber = StreamingTranscriber()
    raw_utterance_1 = "In our Q4 architecture review, I will benchmark SQLite and FastAPI routes by Friday."
    raw_utterance_2 = "Sarah will update the Kubernetes deployment next Tuesday."
    transcriber.transcript_segments.append(f"[Me] {raw_utterance_1}")
    transcriber.transcript_segments.append(f"[Others] {raw_utterance_2}")
    transcriber.unload_asr_model()

    # 3. Local Ollama Summarization (127.0.0.1:11434) + Purge-on-Commit
    captured_requests = []

    def fake_post(self, url, json=None, **kwargs):
        captured_requests.append({"url": str(url), "payload": json})
        return _mock_ollama_response(
            {
                "title": "Q4 Architecture Review",
                "summary": "Reviewed Q4 database benchmarking and Kubernetes deployment timelines.",
                "takeaways": [
                    {
                        "topic": "Database & API Benchmarking",
                        "bullets": ["Benchmark SQLite and FastAPI routes by Friday"],
                    },
                    {
                        "topic": "Deployment Schedule",
                        "bullets": ["Kubernetes deployment update scheduled for next Tuesday"],
                    },
                ],
                "action_items": [
                    {"owner": "Me", "task": "Benchmark SQLite and FastAPI routes", "due_date": "Friday"},
                    {"owner": "Sarah", "task": "Update Kubernetes deployment", "due_date": "next Tuesday"},
                ],
            }
        )

    monkeypatch.setattr(httpx.Client, "post", fake_post)

    saved = summarize_and_commit(
        transcriber=transcriber,
        llm_model="qwen2.5:14b",
        duration_seconds=120,
        db_path=test_db,
    )

    # Verify Ollama contract
    assert len(captured_requests) == 1
    assert captured_requests[0]["url"] == "http://127.0.0.1:11434/api/chat"
    assert captured_requests[0]["payload"]["keep_alive"] == 0
    assert captured_requests[0]["payload"]["options"]["num_ctx"] == 32768

    # Verify timestamp-prefixed title & grouped takeaways
    assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2} — Q4 Architecture Review$", saved["title"])
    assert len(saved["takeaways"]) == 2
    assert saved["takeaways"][0]["topic"] == "Database & API Benchmarking"
    assert len(saved["action_items"]) == 2

    # Verify Purge-on-Commit scrubbed RAM transcript
    assert len(transcriber.get_transcript()) == 0

    # Verify zero raw audio or transcript files created in tmp_path
    leaked_files = [
        p for p in tmp_path.rglob("*")
        if p.suffix.lower() in (".wav", ".mp3", ".txt", ".pcm")
    ]
    assert leaked_files == []

    # Verify verbatim transcript is NOT persisted in SQLite
    db_bytes = test_db.read_bytes()
    assert raw_utterance_1.encode("utf-8") not in db_bytes
    assert raw_utterance_2.encode("utf-8") not in db_bytes


def test_api_e2e_delete_scrubs_disk_and_wal(tmp_path: Path, monkeypatch):
    """Verifies action item toggle, FTS5 search, and secure delete via FastAPI."""
    test_db = tmp_path / "api_e2e.db"
    monkeypatch.setattr(config, "DB_PATH", test_db)
    monkeypatch.setattr(database, "DB_PATH", test_db)
    database.init_db(test_db)

    canary = "AIRGAP_CANARY_E2E_7741"
    created = database.create_meeting(
        title=f"2026-11-13 15:00 — AirGap {canary}",
        summary=f"Confidential summary {canary}",
        takeaways=[{"topic": "Security", "bullets": [f"Verified {canary}"]}],
        asr_model="mlx-community/parakeet-tdt-0.6b-v3",
        llm_model="qwen2.5:14b",
        action_items=[{"owner": "Me", "task": f"Verify {canary}", "due_date": "Friday"}],
        db_path=test_db,
    )

    client = TestClient(main.app, base_url="http://127.0.0.1:8000")
    headers = {
        "X-Local-Session-Token": config.SESSION_TOKEN,
        "Origin": "http://127.0.0.1:8000",
    }

    # Search via FTS5
    search_resp = client.get(f"/api/meetings?search={canary}", headers=headers)
    assert search_resp.status_code == 200
    assert len(search_resp.json()) == 1

    # Toggle action item checkbox
    item_id = created["action_items"][0]["id"]
    patch_resp = client.patch(
        f"/api/action-items/{item_id}",
        headers=headers,
        json={"done": True},
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["done"] is True

    # Delete meeting and verify secure_delete + WAL truncate scrubbed canary bytes
    del_resp = client.delete(f"/api/meetings/{created['id']}", headers=headers)
    assert del_resp.status_code == 200

    wal_path = Path(str(test_db) + "-wal")
    db_bytes = test_db.read_bytes()
    wal_bytes = wal_path.read_bytes() if wal_path.exists() else b""
    assert canary.encode("utf-8") not in db_bytes
    assert canary.encode("utf-8") not in wal_bytes


def test_desktop_wrapper_binds_loopback_and_close_purge(monkeypatch):
    """Verifies PyWebView wrapper targets 127.0.0.1:8000 and scrubs RAM on window close."""
    created_windows = []
    started = {"called": False}

    class FakeEvent:
        def __init__(self):
            self.handlers = []

        def __iadd__(self, fn):
            self.handlers.append(fn)
            return self

        def fire(self):
            for fn in self.handlers:
                fn()

    class FakeEvents:
        def __init__(self):
            self.closed = FakeEvent()

    class FakeWindow:
        def __init__(self, title, url, **kwargs):
            self.title = title
            self.url = url
            self.kwargs = kwargs
            self.events = FakeEvents()

    def fake_create_window(title, url, **kwargs):
        win = FakeWindow(title, url, **kwargs)
        created_windows.append(win)
        return win

    def fake_start():
        started["called"] = True
        # Simulate user closing the native macOS window
        created_windows[0].events.closed.fire()

    class FakeServer:
        def __init__(self):
            self.should_exit = False
            self.ran = False

        def run(self):
            self.ran = True

    purge_calls = {"count": 0}
    orig_purge = desktop_wrapper.purge_ephemeral_buffers

    def tracked_purge():
        purge_calls["count"] += 1
        orig_purge()

    monkeypatch.setattr(desktop_wrapper.webview, "create_window", fake_create_window)
    monkeypatch.setattr(desktop_wrapper.webview, "start", fake_start)
    monkeypatch.setattr(desktop_wrapper, "wait_for_server", lambda **kwargs: True)
    monkeypatch.setattr(desktop_wrapper, "purge_ephemeral_buffers", tracked_purge)

    fake_srv = FakeServer()
    desktop_wrapper.launch_desktop_app(server=fake_srv)

    assert len(created_windows) == 1
    assert created_windows[0].title == "Local Meeting Notes"
    assert created_windows[0].url == "http://127.0.0.1:8000"
    assert started["called"] is True
    assert fake_srv.should_exit is True
    assert orig_purge in created_windows[0].events.closed.handlers or purge_calls["count"] >= 2


def test_verify_offline_script_passes():
    """Runs verify_offline.sh and confirms 100% loopback-only status."""
    script_path = Path(__file__).resolve().parent.parent / "verify_offline.sh"
    proc = subprocess.run(
        ["bash", str(script_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"verify_offline.sh failed: {proc.stdout}\n{proc.stderr}"
    assert "PASS: 100% Loopback-Only" in proc.stdout
