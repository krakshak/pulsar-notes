"""Automated Test Suite for Phase 4: Ollama Summarizer, Long-Meeting Map-Reduce & Purge-on-Commit.

Tests:
1. test_ollama_payload_contract: Verifies /api/chat payload includes num_ctx=32768, keep_alive=0,
   temperature=0.0, <transcript> tags, and MeetingNotesOutput.model_json_schema().
2. test_long_transcript_triggers_map_reduce: Generates a >12,000 token transcript (16,000+ tokens)
   and verifies chunked map-reduce executes and preserves early and late items.
3. test_purge_on_commit_retains_ram_on_ollama_failure: Simulates Ollama 500/timeout failure,
   asserts transcript_segments are preserved in RAM, then verifies retry commits to SQLite and purges RAM.
"""

import json
import re
from pathlib import Path
import httpx
import pytest

from backend import database
from backend.schemas import MeetingNotesOutput
from backend.summarizer import (
    MAP_REDUCE_TOKEN_THRESHOLD,
    OllamaSummarizer,
    SummarizationError,
    estimate_tokens,
    summarize_and_commit,
)
from backend.transcriber import StreamingTranscriber


def _mock_ollama_response(notes: dict, status_code: int = 200) -> httpx.Response:
    """Helper to construct an httpx.Response mimicking Ollama /api/chat."""
    request = httpx.Request("POST", "http://127.0.0.1:11434/api/chat")
    if status_code != 200:
        return httpx.Response(status_code=status_code, text="Internal Server Error", request=request)
    body = {"message": {"role": "assistant", "content": json.dumps(notes)}}
    return httpx.Response(status_code=200, json=body, request=request)


def test_ollama_payload_contract(monkeypatch):
    """Verifies requests to 127.0.0.1:11434/api/chat strictly satisfy the contract."""
    captured_requests = []

    sample_notes = {
        "title": "Q4 Infrastructure Sync",
        "summary": "Reviewed deployment timeline and offline verification status.",
        "takeaways": [
            {
                "topic": "Infrastructure & Offline Mode",
                "bullets": ["SQLite WAL mode enabled", "Offline mode verified"],
            }
        ],
        "action_items": [
            {"owner": "Me", "task": "Send benchmark report", "due_date": "Friday"},
        ],
    }

    def fake_post(self, url, json=None, **kwargs):
        captured_requests.append({"url": str(url), "payload": json})
        return _mock_ollama_response(sample_notes)

    monkeypatch.setattr(httpx.Client, "post", fake_post)

    summarizer = OllamaSummarizer()
    segments = [
        "[Me] I will send the benchmark report by Friday.",
        "[Others] Sounds great, SQLite WAL mode is looking solid.",
    ]
    result = summarizer.summarize(segments, model="qwen2.5:14b")

    assert isinstance(result, MeetingNotesOutput)
    assert result.title == "Q4 Infrastructure Sync"
    assert len(captured_requests) == 1

    req = captured_requests[0]
    assert req["url"] == "http://127.0.0.1:11434/api/chat"
    payload = req["payload"]
    assert payload["model"] == "qwen2.5:14b"
    assert payload["stream"] is False
    assert payload["keep_alive"] == 0
    assert payload["options"]["temperature"] == 0.0
    assert payload["options"]["num_ctx"] == 32768
    assert payload["format"] == MeetingNotesOutput.model_json_schema()

    user_msg = payload["messages"][1]["content"]
    assert "<transcript>" in user_msg
    assert "</transcript>" in user_msg
    assert "[Me] I will send the benchmark report by Friday." in user_msg


def test_long_transcript_triggers_map_reduce(monkeypatch):
    """Generates a 16,000+ token transcript with minute-2 decision & minute-58 task and verifies map-reduce."""
    # Build >16,000 tokens of transcript
    segments = [
        "[Others] Minute 2 Decision: Approved migrating the analytics service to Rust.",
    ]
    # ~16,000 tokens of filler discussion (~100 words per line * 125 lines = ~12,500 words = ~16,600 tokens)
    filler_sentence = " ".join(["discussion"] * 100)
    for i in range(125):
        speaker = "[Me]" if i % 2 == 0 else "[Others]"
        segments.append(f"{speaker} Segment {i}: {filler_sentence}")

    segments.append(
        "[Me] Minute 58 Action Item: I will publish the Rust migration RFC by next Monday."
    )

    assert estimate_tokens(segments) > MAP_REDUCE_TOKEN_THRESHOLD

    captured_payloads = []

    def fake_post(self, url, json=None, **kwargs):
        captured_payloads.append(json)
        user_text = json["messages"][1]["content"]

        # Map step responses preserve specific items found in each chunk
        if "Minute 2 Decision" in user_text:
            return _mock_ollama_response({
                "title": "Part 1 Notes",
                "summary": "Early meeting discussion.",
                "takeaways": [
                    {
                        "topic": "Architecture Decisions",
                        "bullets": ["Approved migrating the analytics service to Rust"],
                    }
                ],
                "action_items": [],
            })
        if "Minute 58 Action Item" in user_text:
            return _mock_ollama_response({
                "title": "Part 2 Notes",
                "summary": "Closing action items.",
                "takeaways": [
                    {
                        "topic": "Next Steps",
                        "bullets": ["RFC scheduled for Monday"],
                    }
                ],
                "action_items": [{"owner": "Me", "task": "Publish the Rust migration RFC", "due_date": "next Monday"}],
            })
        # Reduce step (receives consolidated JSON containing both)
        if "Approved migrating the analytics service to Rust" in user_text and "Publish the Rust migration RFC" in user_text:
            return _mock_ollama_response({
                "title": "Analytics Migration Architecture Review",
                "summary": "Approved analytics migration to Rust and assigned the RFC deliverable.",
                "takeaways": [
                    {
                        "topic": "Architecture Decisions",
                        "bullets": ["Approved migrating the analytics service to Rust"],
                    }
                ],
                "action_items": [{"owner": "Me", "task": "Publish the Rust migration RFC", "due_date": "next Monday"}],
            })
        return _mock_ollama_response({
            "title": "Intermediate Part",
            "summary": "General discussion.",
            "takeaways": [
                {
                    "topic": "General Discussion",
                    "bullets": ["Continued technical review"],
                }
            ],
            "action_items": [],
        })

    monkeypatch.setattr(httpx.Client, "post", fake_post)

    summarizer = OllamaSummarizer()
    result = summarizer.summarize(segments, model="qwen2.5:14b")

    # Should have called Ollama multiple times (N map chunks + 1 reduce call)
    assert len(captured_payloads) >= 3
    assert any(
        "Approved migrating the analytics service to Rust" in t.bullets
        for t in result.takeaways
    )
    assert any(
        item.owner == "Me" and "Rust migration RFC" in item.task
        for item in result.action_items
    )


def test_purge_on_commit_retains_ram_on_ollama_failure(tmp_path: Path, monkeypatch):
    """Simulates Ollama 500 error (retains RAM), then successful retry with llama3.1:8b (commits & purges)."""
    test_db = tmp_path / "test_purge_on_commit.db"
    database.init_db(test_db)

    transcriber = StreamingTranscriber()
    transcriber.transcript_segments = [
        "[Me] I will finalize the Phase 4 test suite today.",
        "[Others] Great, let's verify retry logic handles Ollama timeouts safely.",
    ]

    # 1. Simulate Ollama 500 error
    def fail_post(self, url, json=None, **kwargs):
        return _mock_ollama_response({}, status_code=500)

    monkeypatch.setattr(httpx.Client, "post", fail_post)

    with pytest.raises(SummarizationError):
        summarize_and_commit(
            transcriber=transcriber,
            llm_model="qwen2.5:14b",
            duration_seconds=120,
            db_path=test_db,
        )

    # CRITICAL: Transcript MUST remain intact in RAM on failure!
    assert len(transcriber.get_transcript()) == 2
    assert len(database.list_meetings(db_path=test_db)) == 0

    # 2. Simulate successful retry using fallback model llama3.1:8b
    def success_post(self, url, json=None, **kwargs):
        assert json["model"] == "llama3.1:8b"
        return _mock_ollama_response({
            "title": "Phase 4 Reliability Review",
            "summary": "Verified retry handling when Ollama fails.",
            "takeaways": [
                {
                    "topic": "Reliability",
                    "bullets": ["RAM transcripts are preserved on LLM failure"],
                }
            ],
            "action_items": [
                {"owner": "Me", "task": "Finalize the Phase 4 test suite", "due_date": "today"},
            ],
        })

    monkeypatch.setattr(httpx.Client, "post", success_post)

    saved = summarize_and_commit(
        transcriber=transcriber,
        llm_model="llama3.1:8b",
        duration_seconds=120,
        db_path=test_db,
    )

    # Verify DB commit succeeded AND RAM transcript buffer is now purged
    assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2} — Phase 4 Reliability Review$", saved["title"])
    assert saved["takeaways"] == [
        {"topic": "Reliability", "bullets": ["RAM transcripts are preserved on LLM failure"]}
    ]
    assert saved["llm_model"] == "llama3.1:8b"
    assert len(saved["action_items"]) == 1
    assert saved["action_items"][0]["owner"] == "Me"
    assert len(transcriber.get_transcript()) == 0
    assert len(database.list_meetings(db_path=test_db)) == 1

