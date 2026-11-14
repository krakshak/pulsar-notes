"""SQLite Database Layer with WAL Mode, FTS5 Search, and Zero-Leakage Secure Delete.

Enforces:
- PRAGMA journal_mode = WAL
- PRAGMA secure_delete = ON (zeros freed pages immediately on disk)
- PRAGMA foreign_keys = ON
- Normalized action_items with UUID keys
- FTS5 virtual table with auto-sync triggers and query sanitization
- PRAGMA wal_checkpoint(TRUNCATE) on meeting deletion
"""

import json
import re
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from backend.config import DB_PATH


def get_db_path(db_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolves database path, defaulting to configured DB_PATH."""
    if db_path is None:
        return DB_PATH
    return Path(db_path)


def get_db_connection(db_path: Optional[Union[str, Path]] = None) -> sqlite3.Connection:
    """Creates and configures an SQLite connection with WAL and privacy pragmas."""
    target_path = get_db_path(db_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(target_path))
    conn.row_factory = sqlite3.Row

    # Enforce SQLite pragmas for performance and privacy
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA temp_store = MEMORY;")
    conn.execute("PRAGMA mmap_size = 268435456;")  # 256MB memory-mapped I/O
    conn.execute("PRAGMA secure_delete = ON;")     # Overwrite deleted records with zeros
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA user_version = 1;")

    return conn


def init_db(db_path: Optional[Union[str, Path]] = None) -> None:
    """Initializes tables, indices, FTS5 search virtual table, and triggers."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()

        # 1. Main meetings table (No raw transcripts or audio stored)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS meetings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                duration_seconds INTEGER NOT NULL DEFAULT 0,
                summary TEXT NOT NULL,
                takeaways_json TEXT NOT NULL,
                asr_model TEXT NOT NULL,
                llm_model TEXT NOT NULL
            );
            """
        )

        # 2. Normalized action_items table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS action_items (
                id TEXT PRIMARY KEY,
                meeting_id INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
                owner TEXT NOT NULL,
                task TEXT NOT NULL,
                due_date TEXT,
                done INTEGER NOT NULL DEFAULT 0,
                position INTEGER NOT NULL DEFAULT 0
            );
            """
        )

        # 3. Indices for chronological sorting and open-task lookups
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_meetings_created_at ON meetings(created_at DESC);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_action_items_meeting ON action_items(meeting_id, position);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_action_items_open ON action_items(done, meeting_id);"
        )

        # 4. FTS5 Virtual Table for full-text search
        cursor.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS meetings_fts USING fts5(
                title,
                summary,
                takeaways_json,
                content='meetings',
                content_rowid='id'
            );
            """
        )

        # 5. Triggers to synchronize FTS5 index automatically
        cursor.execute(
            """
            CREATE TRIGGER IF NOT EXISTS meetings_ai AFTER INSERT ON meetings BEGIN
                INSERT INTO meetings_fts(rowid, title, summary, takeaways_json)
                VALUES (new.id, new.title, new.summary, new.takeaways_json);
            END;
            """
        )

        cursor.execute(
            """
            CREATE TRIGGER IF NOT EXISTS meetings_ad AFTER DELETE ON meetings BEGIN
                INSERT INTO meetings_fts(meetings_fts, rowid, title, summary, takeaways_json)
                VALUES ('delete', old.id, old.title, old.summary, old.takeaways_json);
            END;
            """
        )

        cursor.execute(
            """
            CREATE TRIGGER IF NOT EXISTS meetings_au AFTER UPDATE ON meetings BEGIN
                INSERT INTO meetings_fts(meetings_fts, rowid, title, summary, takeaways_json)
                VALUES ('delete', old.id, old.title, old.summary, old.takeaways_json);
                INSERT INTO meetings_fts(rowid, title, summary, takeaways_json)
                VALUES (new.id, new.title, new.summary, new.takeaways_json);
            END;
            """
        )

        conn.commit()


def sanitize_fts5_query(query: str) -> str:
    """Sanitizes user input into safe FTS5 prefix-matching tokens.

    Strips dangerous FTS5 operators and wraps tokens in double quotes, e.g.
    'Q4 & Roadmap (release)' -> '"Q4"* AND "Roadmap"* AND "release"*'
    """
    if not query:
        return ""
    # Extract alphanumeric and hyphenated tokens
    tokens = re.findall(r"[a-zA-Z0-9_\-]+", query)
    if not tokens:
        return ""
    return " AND ".join(f'"{token}"*' for token in tokens)


# ==============================================================================
# Meeting CRUD Operations
# ==============================================================================

def create_meeting(
    title: str,
    summary: str,
    takeaways: List[str],
    asr_model: str,
    llm_model: str,
    action_items: Optional[List[Dict[str, Any]]] = None,
    duration_seconds: int = 0,
    created_at: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Persists a new meeting and its normalized action items."""
    now_iso = datetime.now().astimezone().isoformat()
    meeting_created_at = created_at or now_iso
    meeting_updated_at = now_iso
    takeaways_json = json.dumps(takeaways or [])

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO meetings (title, created_at, updated_at, duration_seconds, summary, takeaways_json, asr_model, llm_model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                title,
                meeting_created_at,
                meeting_updated_at,
                duration_seconds,
                summary,
                takeaways_json,
                asr_model,
                llm_model,
            ),
        )
        meeting_id = cursor.lastrowid
        assert meeting_id is not None

        persisted_items = []
        if action_items:
            for idx, item in enumerate(action_items):
                item_id = str(uuid.uuid4())
                owner = item.get("owner", "Me") if isinstance(item, dict) else getattr(item, "owner", "Me")
                task = item.get("task", "") if isinstance(item, dict) else getattr(item, "task", "")
                due_date = item.get("due_date") if isinstance(item, dict) else getattr(item, "due_date", None)
                done_val = 1 if (item.get("done") if isinstance(item, dict) else getattr(item, "done", False)) else 0

                cursor.execute(
                    """
                    INSERT INTO action_items (id, meeting_id, owner, task, due_date, done, position)
                    VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    (item_id, meeting_id, owner, task, due_date, done_val, idx),
                )
                persisted_items.append(
                    {
                        "id": item_id,
                        "meeting_id": meeting_id,
                        "owner": owner,
                        "task": task,
                        "due_date": due_date,
                        "done": bool(done_val),
                        "position": idx,
                    }
                )

        conn.commit()

    return {
        "id": meeting_id,
        "title": title,
        "created_at": meeting_created_at,
        "updated_at": meeting_updated_at,
        "duration_seconds": duration_seconds,
        "summary": summary,
        "takeaways": takeaways or [],
        "asr_model": asr_model,
        "llm_model": llm_model,
        "action_items": persisted_items,
    }


def get_meeting(
    meeting_id: int,
    db_path: Optional[Union[str, Path]] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieves a single meeting with all nested action items."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM meetings WHERE id = ?;", (meeting_id,))
        row = cursor.fetchone()
        if not row:
            return None

        cursor.execute(
            "SELECT * FROM action_items WHERE meeting_id = ? ORDER BY position ASC, rowid ASC;",
            (meeting_id,),
        )
        item_rows = cursor.fetchall()

    action_items = [
        {
            "id": r["id"],
            "meeting_id": r["meeting_id"],
            "owner": r["owner"],
            "task": r["task"],
            "due_date": r["due_date"],
            "done": bool(r["done"]),
            "position": r["position"],
        }
        for r in item_rows
    ]

    return {
        "id": row["id"],
        "title": row["title"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "duration_seconds": row["duration_seconds"],
        "summary": row["summary"],
        "takeaways": json.loads(row["takeaways_json"]),
        "asr_model": row["asr_model"],
        "llm_model": row["llm_model"],
        "action_items": action_items,
    }


def list_meetings(
    search: Optional[str] = None,
    filter_open_tasks: bool = False,
    limit: int = 50,
    offset: int = 0,
    db_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Lists meetings ordered chronologically (created_at DESC) with optional search & filtering."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()

        where_clauses = []
        params: List[Any] = []

        if filter_open_tasks:
            where_clauses.append(
                "EXISTS (SELECT 1 FROM action_items a WHERE a.meeting_id = m.id AND a.done = 0)"
            )

        if search and search.strip():
            raw_search = search.strip()
            fts_query = sanitize_fts5_query(raw_search)
            like_pattern = f"%{raw_search}%"

            if fts_query:
                where_clauses.append(
                    """(
                        m.id IN (SELECT rowid FROM meetings_fts WHERE meetings_fts MATCH ?)
                        OR EXISTS (
                            SELECT 1 FROM action_items a
                            WHERE a.meeting_id = m.id AND (a.task LIKE ? OR a.owner LIKE ?)
                        )
                    )"""
                )
                params.extend([fts_query, like_pattern, like_pattern])
            else:
                where_clauses.append(
                    """(
                        m.title LIKE ? OR m.summary LIKE ?
                        OR EXISTS (
                            SELECT 1 FROM action_items a
                            WHERE a.meeting_id = m.id AND (a.task LIKE ? OR a.owner LIKE ?)
                        )
                    )"""
                )
                params.extend([like_pattern, like_pattern, like_pattern, like_pattern])

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        query_sql = f"""
            SELECT m.* FROM meetings m
            {where_sql}
            ORDER BY m.created_at DESC
            LIMIT ? OFFSET ?;
        """
        params.extend([limit, offset])

        cursor.execute(query_sql, params)
        meeting_rows = cursor.fetchall()

        if not meeting_rows:
            return []

        # Batch-fetch action items for returned meetings
        meeting_ids = [r["id"] for r in meeting_rows]
        placeholders = ",".join("?" * len(meeting_ids))
        cursor.execute(
            f"""
            SELECT * FROM action_items
            WHERE meeting_id IN ({placeholders})
            ORDER BY meeting_id, position ASC, rowid ASC;
            """,
            meeting_ids,
        )
        action_item_rows = cursor.fetchall()

    items_by_meeting: Dict[int, List[Dict[str, Any]]] = {mid: [] for mid in meeting_ids}
    for r in action_item_rows:
        items_by_meeting[r["meeting_id"]].append(
            {
                "id": r["id"],
                "meeting_id": r["meeting_id"],
                "owner": r["owner"],
                "task": r["task"],
                "due_date": r["due_date"],
                "done": bool(r["done"]),
                "position": r["position"],
            }
        )

    results = []
    for r in meeting_rows:
        results.append(
            {
                "id": r["id"],
                "title": r["title"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "duration_seconds": r["duration_seconds"],
                "summary": r["summary"],
                "takeaways": json.loads(r["takeaways_json"]),
                "asr_model": r["asr_model"],
                "llm_model": r["llm_model"],
                "action_items": items_by_meeting.get(r["id"], []),
            }
        )
    return results


def update_meeting(
    meeting_id: int,
    title: Optional[str] = None,
    summary: Optional[str] = None,
    takeaways: Optional[List[str]] = None,
    duration_seconds: Optional[int] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> Optional[Dict[str, Any]]:
    """Updates editable meeting fields and touches updated_at."""
    updates = []
    params: List[Any] = []

    if title is not None:
        updates.append("title = ?")
        params.append(title)
    if summary is not None:
        updates.append("summary = ?")
        params.append(summary)
    if takeaways is not None:
        updates.append("takeaways_json = ?")
        params.append(json.dumps(takeaways))
    if duration_seconds is not None:
        updates.append("duration_seconds = ?")
        params.append(duration_seconds)

    if not updates:
        return get_meeting(meeting_id, db_path=db_path)

    now_iso = datetime.now().astimezone().isoformat()
    updates.append("updated_at = ?")
    params.append(now_iso)
    params.append(meeting_id)

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"UPDATE meetings SET {', '.join(updates)} WHERE id = ?;",
            params,
        )
        conn.commit()

    return get_meeting(meeting_id, db_path=db_path)


def delete_meeting(
    meeting_id: int,
    db_path: Optional[Union[str, Path]] = None,
) -> bool:
    """Deletes a meeting and immediately runs PRAGMA wal_checkpoint(TRUNCATE)."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM meetings WHERE id = ?;", (meeting_id,))
        rows_deleted = cursor.rowcount
        conn.commit()

        # Zero and truncate WAL file so deleted text is wiped from disk
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")

    return rows_deleted > 0


# ==============================================================================
# Action Item CRUD Operations
# ==============================================================================

def create_action_item(
    meeting_id: int,
    owner: str,
    task: str,
    due_date: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> Optional[Dict[str, Any]]:
    """Adds a new action item to a meeting."""
    item_id = str(uuid.uuid4())
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        # Verify meeting exists
        cursor.execute("SELECT id FROM meetings WHERE id = ?;", (meeting_id,))
        if not cursor.fetchone():
            return None

        cursor.execute(
            "SELECT COALESCE(MAX(position), -1) + 1 FROM action_items WHERE meeting_id = ?;",
            (meeting_id,),
        )
        pos = cursor.fetchone()[0]

        cursor.execute(
            """
            INSERT INTO action_items (id, meeting_id, owner, task, due_date, done, position)
            VALUES (?, ?, ?, ?, ?, 0, ?);
            """,
            (item_id, meeting_id, owner, task, due_date, pos),
        )
        conn.commit()

    return {
        "id": item_id,
        "meeting_id": meeting_id,
        "owner": owner,
        "task": task,
        "due_date": due_date,
        "done": False,
        "position": pos,
    }


def update_action_item(
    item_id: str,
    done: Optional[bool] = None,
    owner: Optional[str] = None,
    task: Optional[str] = None,
    due_date: Optional[str] = None,
    position: Optional[int] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> Optional[Dict[str, Any]]:
    """Updates action item fields (e.g. checkbox toggle, editing text)."""
    updates = []
    params: List[Any] = []

    if done is not None:
        updates.append("done = ?")
        params.append(1 if done else 0)
    if owner is not None:
        updates.append("owner = ?")
        params.append(owner)
    if task is not None:
        updates.append("task = ?")
        params.append(task)
    if due_date is not None:
        updates.append("due_date = ?")
        params.append(due_date)
    if position is not None:
        updates.append("position = ?")
        params.append(position)

    if not updates:
        return get_action_item(item_id, db_path=db_path)

    params.append(item_id)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"UPDATE action_items SET {', '.join(updates)} WHERE id = ?;",
            params,
        )
        conn.commit()

    return get_action_item(item_id, db_path=db_path)


def get_action_item(
    item_id: str,
    db_path: Optional[Union[str, Path]] = None,
) -> Optional[Dict[str, Any]]:
    """Fetches a single action item by ID."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM action_items WHERE id = ?;", (item_id,))
        r = cursor.fetchone()
        if not r:
            return None
        return {
            "id": r["id"],
            "meeting_id": r["meeting_id"],
            "owner": r["owner"],
            "task": r["task"],
            "due_date": r["due_date"],
            "done": bool(r["done"]),
            "position": r["position"],
        }


def delete_action_item(
    item_id: str,
    db_path: Optional[Union[str, Path]] = None,
) -> bool:
    """Deletes an action item by ID."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM action_items WHERE id = ?;", (item_id,))
        rows = cursor.rowcount
        conn.commit()
    return rows > 0


def get_open_action_items(
    db_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Returns all open action items (done=0) across all meetings."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT a.*, m.title as meeting_title, m.created_at as meeting_created_at
            FROM action_items a
            JOIN meetings m ON a.meeting_id = m.id
            WHERE a.done = 0
            ORDER BY m.created_at DESC, a.position ASC;
            """
        )
        rows = cursor.fetchall()

    return [
        {
            "id": r["id"],
            "meeting_id": r["meeting_id"],
            "meeting_title": r["meeting_title"],
            "meeting_created_at": r["meeting_created_at"],
            "owner": r["owner"],
            "task": r["task"],
            "due_date": r["due_date"],
            "done": bool(r["done"]),
            "position": r["position"],
        }
        for r in rows
    ]

