"""Automated Test Suite for Phase 1: Config, Offline Guardrails & SQLite Database Layer.

Tests:
1. test_offline_env_assertions: Validates offline environment enforcement and session token generation.
2. test_sqlite_pragmas: Asserts journal_mode=WAL, secure_delete=ON, foreign_keys=ON.
3. test_meeting_and_action_items_crud: Validates meeting/action-item creation, checkbox toggling, and ordering.
4. test_fts5_sanitized_search: Validates FTS5 search handles special characters and returns results.
5. test_secure_delete_and_wal_truncate: Confirms deleted canary strings are wiped from both .db and -wal files.
"""

import os
from pathlib import Path
import pytest

from backend import config
from backend import database
from backend.schemas import ExtractedActionItem, MeetingNotesOutput


def test_offline_env_assertions():
    """Verifies config.py sets/validates all offline flags and generates a 32-byte URL-safe token."""
    # Verify mandatory offline environment variables
    assert config.validate_offline_env() is True
    for key, expected_value in config.MANDATORY_OFFLINE_ENV.items():
        assert os.environ.get(key) is not None
        assert os.environ.get(key).lower() == expected_value.lower()

    # Verify session token is non-empty, URL-safe, and derived from 32 bytes (length >= 40)
    assert isinstance(config.SESSION_TOKEN, str)
    assert len(config.SESSION_TOKEN) >= 40
    # Ensure characters are strictly URL-safe base64 characters
    import re
    assert re.match(r"^[A-Za-z0-9_-]+$", config.SESSION_TOKEN) is not None

    # Verify supported LLM and ASR models are configured for the UI selector
    assert config.DEFAULT_LLM_MODEL in config.SUPPORTED_LLM_MODELS
    assert config.FALLBACK_LLM_MODEL in config.SUPPORTED_LLM_MODELS
    for model_name in ["qwen2.5:14b", "qwen3:14b", "gemma3:12b", "qwen3:8b", "llama3.1:8b"]:
        assert model_name in config.SUPPORTED_LLM_MODELS


def test_sqlite_pragmas(tmp_path: Path):
    """Queries PRAGMA journal_mode, PRAGMA secure_delete, and PRAGMA foreign_keys."""
    test_db = tmp_path / "test_pragmas.db"
    database.init_db(test_db)

    conn = database.get_db_connection(test_db)
    try:
        journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        secure_delete = conn.execute("PRAGMA secure_delete;").fetchone()[0]
        foreign_keys = conn.execute("PRAGMA foreign_keys;").fetchone()[0]

        assert journal_mode.lower() == "wal"
        assert secure_delete == 1
        assert foreign_keys == 1
    finally:
        conn.close()


def test_meeting_and_action_items_crud(tmp_path: Path):
    """Inserts a meeting with 3 action items, toggles done, tests ordering & open-task filtering."""
    test_db = tmp_path / "test_crud.db"
    database.init_db(test_db)

    # 1. Create older meeting
    m1 = database.create_meeting(
        title="Weekly Architecture Sync",
        summary="Discussion on database indexing and local model memory budget.",
        takeaways=["Enable SQLite WAL mode", "Target <2GB ASR memory footprint"],
        asr_model="mlx-community/parakeet-tdt-0.6b-v3",
        llm_model="qwen2.5:14b",
        created_at="2026-11-01T10:00:00+00:00",
        action_items=[
            {"owner": "Me", "task": "Benchmark SQLite FTS5 queries", "due_date": "Friday"},
        ],
        db_path=test_db,
    )

    # 2. Create newer meeting with 3 action items
    m2 = database.create_meeting(
        title="Q4 Product Roadmap Review",
        summary="Release readiness and milestone tracking review.",
        takeaways=["Finalize UI components", "Complete offline verification suite"],
        asr_model="mlx-community/parakeet-tdt-0.6b-v3",
        llm_model="qwen2.5:14b",
        created_at="2026-11-08T15:00:00+00:00",
        action_items=[
            {"owner": "Me", "task": "Send updated benchmark report", "due_date": "Friday"},
            {"owner": "Sarah", "task": "Schedule security review", "due_date": "Next Tuesday"},
            {"owner": "John", "task": "Verify local bundle hashes", "due_date": "End of Month"},
        ],
        db_path=test_db,
    )

    # Verify chronological ordering (created_at DESC)
    meetings = database.list_meetings(db_path=test_db)
    assert len(meetings) == 2
    assert meetings[0]["id"] == m2["id"]  # Newer meeting first
    assert meetings[1]["id"] == m1["id"]

    # Verify action items count
    m2_details = database.get_meeting(m2["id"], db_path=test_db)
    assert m2_details is not None
    assert len(m2_details["action_items"]) == 3
    assert all(not item["done"] for item in m2_details["action_items"])

    # Toggle 'done' on the second action item (Sarah's task)
    sarah_item_id = m2_details["action_items"][1]["id"]
    updated_item = database.update_action_item(sarah_item_id, done=True, db_path=test_db)
    assert updated_item is not None
    assert updated_item["done"] is True

    # Check open tasks for m2 (now 2 open, 1 done)
    refreshed_m2 = database.get_meeting(m2["id"], db_path=test_db)
    assert refreshed_m2 is not None
    completed = [i for i in refreshed_m2["action_items"] if i["done"]]
    open_items = [i for i in refreshed_m2["action_items"] if not i["done"]]
    assert len(completed) == 1
    assert len(open_items) == 2
    assert completed[0]["owner"] == "Sarah"

    # Mark remaining items in m1 as done
    m1_details = database.get_meeting(m1["id"], db_path=test_db)
    assert m1_details is not None
    database.update_action_item(m1_details["action_items"][0]["id"], done=True, db_path=test_db)

    # Filter meetings with open tasks: only m2 should be returned
    open_meetings = database.list_meetings(filter_open_tasks=True, db_path=test_db)
    assert len(open_meetings) == 1
    assert open_meetings[0]["id"] == m2["id"]


def test_fts5_sanitized_search(tmp_path: Path):
    """Searches with special characters ('Q4 & Roadmap (release)*') and verifies robust FTS5 execution."""
    test_db = tmp_path / "test_fts5.db"
    database.init_db(test_db)

    database.create_meeting(
        title="Q4 & Roadmap (release) Session",
        summary="Meeting regarding Q4 release requirements and deployment procedures.",
        takeaways=["Pass the offline verification check", "Store zero data in cloud"],
        asr_model="mlx-community/parakeet-tdt-0.6b-v3",
        llm_model="qwen2.5:14b",
        action_items=[{"owner": "Me", "task": "Check deployment checklists"}],
        db_path=test_db,
    )

    database.create_meeting(
        title="Unrelated Engineering Sync",
        summary="Frontend widget updates and styling polish.",
        takeaways=["Refactor CSS variables"],
        asr_model="mlx-community/parakeet-tdt-0.6b-v3",
        llm_model="qwen2.5:14b",
        db_path=test_db,
    )

    # Test raw query with special FTS5 operators that would crash an unsanitized query
    special_query = '"Q4 & Roadmap (release)"*'
    sanitized = database.sanitize_fts5_query(special_query)
    assert '"Q4"*' in sanitized
    assert '"Roadmap"*' in sanitized
    assert '"release"*' in sanitized

    results = database.list_meetings(search=special_query, db_path=test_db)
    assert len(results) == 1
    assert "Roadmap" in results[0]["title"]

    # Test search finding text in action item task
    action_item_search = database.list_meetings(search="deployment checklists", db_path=test_db)
    assert len(action_item_search) == 1
    assert action_item_search[0]["title"] == "Q4 & Roadmap (release) Session"


def test_secure_delete_and_wal_truncate(tmp_path: Path):
    """Inserts a canary string, deletes it, and verifies absence from raw .db and -wal disk bytes."""
    test_db = tmp_path / "test_secure_delete.db"
    database.init_db(test_db)

    canary = "SECRET_CANARY_99281_TOP_SECRET"
    meeting = database.create_meeting(
        title=f"Confidential Meeting {canary}",
        summary=f"This summary contains sensitive meeting data: {canary}",
        takeaways=[f"Takeaway with canary: {canary}"],
        asr_model="mlx-community/parakeet-tdt-0.6b-v3",
        llm_model="qwen2.5:14b",
        action_items=[{"owner": "Me", "task": f"Destroy {canary}"}],
        db_path=test_db,
    )
    meeting_id = meeting["id"]

    # Read disk before deletion to verify canary is physically present
    db_bytes_before = test_db.read_bytes()
    wal_path = Path(str(test_db) + "-wal")
    wal_bytes_before = wal_path.read_bytes() if wal_path.exists() else b""
    assert (canary.encode("utf-8") in db_bytes_before) or (canary.encode("utf-8") in wal_bytes_before)

    # Delete meeting (which executes secure_delete + PRAGMA wal_checkpoint(TRUNCATE))
    deleted = database.delete_meeting(meeting_id, db_path=test_db)
    assert deleted is True

    # Read raw disk bytes of .db and .db-wal after deletion
    db_bytes_after = test_db.read_bytes()
    wal_bytes_after = wal_path.read_bytes() if wal_path.exists() else b""

    # Assert canary is completely purged from both disk files
    canary_bytes = canary.encode("utf-8")
    assert canary_bytes not in db_bytes_after, "Canary still found in raw .db bytes after secure delete!"
    assert canary_bytes not in wal_bytes_after, "Canary still found in raw .db-wal bytes after wal_checkpoint(TRUNCATE)!"


def test_pydantic_schemas_validation():
    """Validates MeetingNotesOutput and ExtractedActionItem schemas."""
    schema = MeetingNotesOutput.model_json_schema()
    assert "properties" in schema
    assert "title" in schema["properties"]
    assert "summary" in schema["properties"]
    assert "takeaways" in schema["properties"]
    assert "action_items" in schema["properties"]

    notes = MeetingNotesOutput(
        title="Product Strategy Session",
        summary="Executive summary of the meeting.",
        takeaways=[
            {
                "topic": "Key Decisions",
                "bullets": ["Key takeaway 1", "Key takeaway 2"],
            }
        ],
        action_items=[
            ExtractedActionItem(owner="Me", task="Draft memo", due_date="Friday"),
            ExtractedActionItem(owner="David", task="Review design spec"),
        ],
    )
    assert notes.title == "Product Strategy Session"
    assert notes.takeaways[0].topic == "Key Decisions"
    assert len(notes.takeaways[0].bullets) == 2
    assert len(notes.action_items) == 2
    assert notes.action_items[0].owner == "Me"
    assert notes.action_items[1].due_date is None

