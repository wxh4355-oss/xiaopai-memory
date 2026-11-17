import sqlite3
import threading
from contextlib import contextmanager
from functools import wraps
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Union

from .config import AppPaths
from .models import MemoryCandidate


def _locked_method(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return guarded


class _LockedCursor:
    def __init__(self, cursor, lock):
        self._cursor, self._lock = cursor, lock
    def fetchone(self):
        with self._lock: return self._cursor.fetchone()
    def fetchall(self):
        with self._lock: return self._cursor.fetchall()
    def __iter__(self): return self
    def __next__(self):
        with self._lock: return next(self._cursor)
    def __getattr__(self, name): return getattr(self._cursor, name)


class _LockedConnection:
    """Serialize every operation on one cross-thread SQLite connection."""
    def __init__(self, connection, lock): self._connection, self._lock = connection, lock
    def execute(self, *args, **kwargs):
        with self._lock: return _LockedCursor(self._connection.execute(*args, **kwargs), self._lock)
    def executemany(self, *args, **kwargs):
        with self._lock: return _LockedCursor(self._connection.executemany(*args, **kwargs), self._lock)
    def executescript(self, *args, **kwargs):
        with self._lock: return _LockedCursor(self._connection.executescript(*args, **kwargs), self._lock)
    def commit(self):
        with self._lock: return self._connection.commit()
    def rollback(self):
        with self._lock: return self._connection.rollback()
    def close(self):
        with self._lock: return self._connection.close()
    def __getattr__(self, name): return getattr(self._connection, name)


@dataclass(frozen=True)
class MessageRow:
    id: int
    session_id: str
    role: str
    content: str
    incomplete: bool
    created_at: str


@dataclass(frozen=True)
class MemoryRow:
    id: int
    category: str
    content: str
    source: str
    sensitivity: str
    active: bool
    created_at: str
    updated_at: str
    last_used_at: Optional[str]
    user_id: str = "user_001"
    subject: Optional[str] = None
    predicate: Optional[str] = None
    value: Optional[str] = None
    source_message_id: Optional[int] = None
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    status: str = "active"
    importance: float = 0.5
    scene: str = "default"
    temporary: bool = False


@dataclass(frozen=True)
class IndexTask:
    id: int
    memory_id: int
    operation: str
    attempts: int


@dataclass(frozen=True)
class EventRow:
    id: int
    title: str
    due_at: str
    important: bool
    active: bool
    created_at: str


# Schema 2 turns `memories` into a versioned fact table: every row can name the
# subject and predicate it is about, point at the message that produced it, and
# carry an explicit validity window plus a lifecycle status.
_MEMORY_V2_COLUMNS = (
    ("user_id", "TEXT NOT NULL DEFAULT 'user_001'"),
    ("subject", "TEXT"),
    ("predicate", "TEXT"),
    ("value", "TEXT"),
    ("source_message_id", "INTEGER"),
    ("valid_from", "TEXT"),
    ("valid_to", "TEXT"),
    ("status", "TEXT NOT NULL DEFAULT 'active'"),
    ("importance", "REAL NOT NULL DEFAULT 0.5"),
)


def _migrate_to_v2(connection) -> None:
    """Add structured fact columns and the durable index-work queue."""
    present = {row["name"] for row in connection.execute("PRAGMA table_info(memories)")}
    for name, declaration in _MEMORY_V2_COLUMNS:
        if name not in present:
            connection.execute(
                "ALTER TABLE memories ADD COLUMN {} {}".format(name, declaration)
            )
    connection.execute("UPDATE memories SET value = content WHERE value IS NULL")
    connection.execute(
        "UPDATE memories SET valid_from = created_at WHERE valid_from IS NULL"
    )
    connection.execute(
        "UPDATE memories SET status = CASE WHEN active = 1 THEN 'active' ELSE 'deleted' END"
    )
    connection.executescript(
        """
        CREATE INDEX IF NOT EXISTS memories_fact_idx
        ON memories(user_id, subject, predicate, status);

        CREATE TABLE IF NOT EXISTS memory_index_tasks (
            id INTEGER PRIMARY KEY,
            memory_id INTEGER NOT NULL,
            operation TEXT NOT NULL,
            created_at TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            completed_at TEXT
        );

        CREATE INDEX IF NOT EXISTS memory_index_tasks_pending_idx
        ON memory_index_tasks(completed_at, id);

        CREATE TABLE IF NOT EXISTS memory_index_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )


def _migrate_to_v3(connection) -> None:
    connection.execute("ALTER TABLE memories ADD COLUMN scene TEXT NOT NULL DEFAULT 'default'")
    connection.execute("ALTER TABLE memories ADD COLUMN temporary INTEGER NOT NULL DEFAULT 0")
    connection.execute("""
        CREATE TABLE memory_feedback (
            id INTEGER PRIMARY KEY,
            user_id TEXT NOT NULL,
            memory_id INTEGER NOT NULL REFERENCES memories(id),
            recall_id TEXT NOT NULL,
            scene TEXT NOT NULL,
            action TEXT NOT NULL,
            replacement_id INTEGER REFERENCES memories(id),
            created_at TEXT NOT NULL,
            UNIQUE(user_id, memory_id, recall_id, action)
        )
    """)


_MIGRATIONS = {2: _migrate_to_v2, 3: _migrate_to_v3}


class Storage:
    """Explicit SQLite repository for Puppy-owned local data.

    SQLite is the authoritative store: every memory row, its version history
    and its pending index work live here.  The vector index is a rebuildable
    cache that may lag behind or be thrown away at any time.
    """

    _SCHEMA_VERSION = 3

    def __init__(self, paths: Union[AppPaths, Path, str], *, now=None):
        database_file = paths.database_file if isinstance(paths, AppPaths) else Path(paths)
        database_file.parent.mkdir(parents=True, exist_ok=True)
        # UI work is intentionally dispatched off Tk's main thread. Access remains
        # serialized by the application, while this permits that worker boundary.
        self._lock = threading.RLock()
        self.clock = now or (lambda: datetime.now(timezone.utc).astimezone())
        raw_connection = sqlite3.connect(str(database_file), check_same_thread=False)
        raw_connection.row_factory = sqlite3.Row
        self.connection = _LockedConnection(raw_connection, self._lock)
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self._migrate()

    @_locked_method
    def close(self) -> None:
        self.connection.close()

    @contextmanager
    def locked(self):
        with self._lock:
            yield self.connection

    @contextmanager
    def transaction(self, *, immediate=False):
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            try:
                yield self.connection
                self.connection.commit()
            except BaseException:
                self.connection.rollback()
                raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _now(self) -> str:
        return self.clock().isoformat()

    @staticmethod
    def memory_key(content: str) -> str:
        """Comparison key for spotting the same fact written twice."""
        return " ".join(str(content or "").split()).casefold()

    def _active_memory_id(self, user_id: str, category: str, content: str,
                          value: Optional[str] = None, *, scene="default",
                          valid_to=None, subject=None, predicate=None) -> Optional[int]:
        """Find an identical fact that is already active for this user."""
        key = self.memory_key(value or content)
        for row in self.connection.execute(
            "SELECT id, content, value FROM memories WHERE active = 1"
            " AND user_id = ? AND category = ? AND scene = ? AND valid_to IS ?"
            " AND subject IS ? AND predicate IS ?",
            (user_id, category, scene, valid_to, subject, predicate),
        ):
            if self.memory_key(row["value"] or row["content"]) == key:
                return int(row["id"])
        return None

    @_locked_method
    def _migrate(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_meta (
                version INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                incomplete INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS messages_created_at_idx
            ON messages(created_at, id);

            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                content TEXT NOT NULL,
                first_message_id INTEGER NOT NULL,
                last_message_id INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY,
                category TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                sensitivity TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_used_at TEXT
            );

            CREATE TABLE IF NOT EXISTS memory_history (
                id INTEGER PRIMARY KEY,
                memory_id INTEGER NOT NULL REFERENCES memories(id),
                replacement_id INTEGER REFERENCES memories(id),
                action TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                due_at TEXT NOT NULL,
                important INTEGER NOT NULL DEFAULT 0,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS events_due_at_idx
            ON events(active, due_at);

            CREATE TABLE IF NOT EXISTS scheduler_log (
                id INTEGER PRIMARY KEY,
                kind TEXT NOT NULL,
                event_id INTEGER REFERENCES events(id),
                created_at TEXT NOT NULL
            );
            """
        )
        row = self.connection.execute("SELECT MAX(version) FROM schema_meta").fetchone()
        current_version = row[0]
        if current_version is None:
            self.connection.execute("INSERT INTO schema_meta(version) VALUES (1)")
            current_version = 1
        elif current_version > self._SCHEMA_VERSION:
            raise RuntimeError("Database schema is newer than this application")
        while current_version < self._SCHEMA_VERSION:
            target = current_version + 1
            migration = _MIGRATIONS.get(target)
            if migration is None:
                raise RuntimeError("Missing migration to schema version {}".format(target))
            if target == 2:
                migration(self.connection)
                self.connection.execute("UPDATE schema_meta SET version = ?", (target,))
                self.connection.commit()
            else:
                with self.transaction(immediate=True):
                    # Another desktop process may have finished migrating first.
                    if self.connection.execute("SELECT MAX(version) FROM schema_meta").fetchone()[0] < target:
                        migration(self.connection)
                        self.connection.execute("UPDATE schema_meta SET version = ?", (target,))
            current_version = target
        self.connection.commit()

    @_locked_method
    def add_message(
        self,
        role: str,
        content: str,
        session_id: str = "default",
        incomplete: bool = False,
        created_at: Optional[str] = None,
    ) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO messages(session_id, role, content, incomplete, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, role, content, int(incomplete), created_at or self._now()),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    @_locked_method
    def recent_messages(self, limit: int = 20, session_id: Optional[str] = None) -> List[MessageRow]:
        if limit < 0:
            raise ValueError("limit must not be negative")
        session_clause = "WHERE session_id = ?" if session_id is not None else ""
        parameters = (session_id, limit) if session_id is not None else (limit,)
        rows = self.connection.execute(
            """
            SELECT id, session_id, role, content, incomplete, created_at
            FROM (
                SELECT id, session_id, role, content, incomplete, created_at
                FROM messages {session_clause}
                ORDER BY created_at DESC, id DESC LIMIT ?
            )
            ORDER BY created_at, id
            """.format(session_clause=session_clause),
            parameters,
        ).fetchall()
        return [
            MessageRow(
                id=row["id"],
                session_id=row["session_id"],
                role=row["role"],
                content=row["content"],
                incomplete=bool(row["incomplete"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    @_locked_method
    def save_memory(
        self,
        category: Union[str, MemoryCandidate],
        content: Optional[str] = None,
        source: Optional[str] = None,
        sensitivity: Optional[str] = None,
        created_at: Optional[str] = None,
        user_id: str = "user_001",
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        value: Optional[str] = None,
        source_message_id: Optional[int] = None,
        valid_from: Optional[str] = None,
        importance: float = 0.5,
        *,
        scene: str = "default",
        valid_to: Optional[str] = None,
        commit: bool = True,
    ) -> int:
        if isinstance(category, MemoryCandidate):
            candidate = category
            category = candidate.category
            content = candidate.content
            source = candidate.source
            sensitivity = candidate.sensitivity.value
            subject = subject if subject is not None else getattr(candidate, "subject", None)
            predicate = predicate if predicate is not None else getattr(candidate, "predicate", None)
            value = value if value is not None else getattr(candidate, "value", None)
            source_message_id = (source_message_id if source_message_id is not None
                                 else getattr(candidate, "source_message_id", None))
            importance = getattr(candidate, "importance", importance)
            user_id = getattr(candidate, "user_id", None) or user_id
            scene = candidate.scene
            valid_to = candidate.valid_to
        if content is None or source is None or sensitivity is None:
            raise ValueError("memory content, source, and sensitivity are required")
        # Saying the same thing twice must not create a second row: the repeated
        # sentence would otherwise be injected twice and slowly crowd out the
        # token budget.  Genuine corrections are a separate decision.
        existing = self._active_memory_id(
            user_id, category, content, value, scene=scene, valid_to=valid_to,
            subject=subject, predicate=predicate)
        if existing is not None:
            return existing
        timestamp = created_at or self._now()
        cursor = self.connection.execute(
            """
            INSERT INTO memories(category, content, source, sensitivity, active, status,
                                 user_id, subject, predicate, value, source_message_id,
                                 valid_from, importance, created_at, updated_at,
                                 scene, valid_to, temporary)
            VALUES (?, ?, ?, ?, 1, 'active', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (category, content, source, sensitivity, user_id, subject, predicate,
             value if value is not None else content, source_message_id,
             valid_from or timestamp, float(importance), timestamp, timestamp,
             scene, valid_to, int(valid_to is not None)),
        )
        memory_id = int(cursor.lastrowid)
        self._enqueue_index_task(memory_id, "upsert", timestamp)
        if commit:
            self.connection.commit()
        return memory_id

    @_locked_method
    def message_by_id(self, message_id: int) -> Optional[MessageRow]:
        row = self.connection.execute(
            "SELECT id, session_id, role, content, incomplete, created_at"
            " FROM messages WHERE id = ?", (int(message_id),)
        ).fetchone()
        if row is None:
            return None
        return MessageRow(
            id=row["id"], session_id=row["session_id"], role=row["role"],
            content=row["content"], incomplete=bool(row["incomplete"]),
            created_at=row["created_at"],
        )

    @_locked_method
    def list_memories(self, active_only: bool = True) -> List[MemoryRow]:
        sql = """
            SELECT id, category, content, source, sensitivity, active,
                   created_at, updated_at, last_used_at, user_id, subject,
                   predicate, value, source_message_id, valid_from, valid_to,
                   status, importance, scene, temporary
            FROM memories
        """
        parameters = ()
        if active_only:
            sql += " WHERE active = ?"
            parameters = (1,)
        sql += " ORDER BY id"
        rows = self.connection.execute(sql, parameters).fetchall()
        return [
            MemoryRow(
                id=row["id"], category=row["category"], content=row["content"],
                source=row["source"], sensitivity=row["sensitivity"],
                active=bool(row["active"]), created_at=row["created_at"],
                updated_at=row["updated_at"], last_used_at=row["last_used_at"],
                user_id=row["user_id"], subject=row["subject"],
                predicate=row["predicate"], value=row["value"],
                source_message_id=row["source_message_id"],
                valid_from=row["valid_from"], valid_to=row["valid_to"],
                status=row["status"], importance=float(row["importance"] or 0.5),
                scene=row["scene"], temporary=bool(row["temporary"]),
            )
            for row in rows
        ]

    @_locked_method
    def memory_by_id(self, memory_id: int) -> Optional[MemoryRow]:
        row = next(
            (item for item in self.list_memories(active_only=False) if item.id == memory_id),
            None,
        )
        return row

    @_locked_method
    def memory_history(self, memory_id: int) -> list:
        rows = self.connection.execute(
            "SELECT memory_id, replacement_id, action, created_at FROM memory_history"
            " WHERE memory_id = ? OR replacement_id = ? ORDER BY id",
            (memory_id, memory_id),
        )
        return [dict(row) for row in rows]

    @_locked_method
    def record_memory_feedback(self, user_id, memory_id, recall_id, scene, *, replacement_id=None):
        action = "confirmed_correction" if replacement_id is not None else "not_applicable"
        self.connection.execute(
            "INSERT OR IGNORE INTO memory_feedback"
            " (user_id, memory_id, recall_id, scene, action, replacement_id, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, memory_id, recall_id, scene, action, replacement_id, self._now()),
        )
        return self.memory_feedback(user_id, memory_id, recall_id, action)

    @_locked_method
    def memory_feedback(self, user_id, memory_id, recall_id, action):
        row = self.connection.execute(
            "SELECT * FROM memory_feedback"
            " WHERE user_id = ? AND memory_id = ? AND recall_id = ? AND action = ?",
            (user_id, memory_id, recall_id, action),
        ).fetchone()
        return dict(row) if row is not None else None

    @_locked_method
    def active_facts(self, user_id: str = "user_001",
                     subject: Optional[str] = None,
                     predicate: Optional[str] = None) -> List[MemoryRow]:
        """Active rows for one fact key; the basis of ADD/UPDATE/NOOP."""
        clauses, parameters = ["user_id = ?", "active = 1"], [user_id]
        if subject is not None:
            clauses.append("subject = ?")
            parameters.append(subject)
        if predicate is not None:
            clauses.append("predicate = ?")
            parameters.append(predicate)
        rows = self.connection.execute(
            "SELECT id FROM memories WHERE {} ORDER BY id".format(" AND ".join(clauses)),
            tuple(parameters),
        ).fetchall()
        ids = {row["id"] for row in rows}
        return [item for item in self.list_memories(active_only=True) if item.id in ids]

    @_locked_method
    def supersede_memory(self, memory_id: int, replacement_id: Optional[int] = None,
                         when: Optional[str] = None, action: str = "updated", *,
                         commit: bool = True) -> None:
        """Retire an old version without deleting its evidence."""
        if commit:
            with self.transaction(immediate=True):
                self.supersede_memory(memory_id, replacement_id, when, action, commit=False)
            return
        timestamp = when or self._now()
        changed = self.connection.execute(
            "UPDATE memories SET active = 0, status = 'superseded', valid_to = ?,"
            " updated_at = ? WHERE id = ? AND active = 1",
            (timestamp, timestamp, memory_id),
        ).rowcount
        if not changed:
            return
        self.connection.execute(
            "INSERT INTO memory_history(memory_id, replacement_id, action, created_at)"
            " VALUES (?, ?, ?, ?)",
            (memory_id, replacement_id, action, timestamp),
        )
        self._enqueue_index_task(memory_id, "remove", timestamp)

    def _enqueue_index_task(self, memory_id: int, operation: str, timestamp: str) -> None:
        """Record index work in the same transaction as the authoritative write."""
        self.connection.execute(
            "INSERT INTO memory_index_tasks(memory_id, operation, created_at)"
            " VALUES (?, ?, ?)",
            (memory_id, operation, timestamp),
        )

    @_locked_method
    def replace_memory(
        self,
        memory_id: int,
        content: str,
        source: str,
        sensitivity: str,
        created_at: Optional[str] = None,
    ) -> int:
        timestamp = created_at or self._now()
        with self.transaction(immediate=True):
            old = self.connection.execute(
                "SELECT * FROM memories WHERE id = ? AND active = 1", (memory_id,)
            ).fetchone()
            if old is None:
                raise KeyError(memory_id)
            self.connection.execute(
                "UPDATE memories SET active = 0, status = 'superseded', valid_to = ?,"
                " updated_at = ? WHERE id = ?",
                (timestamp, timestamp, memory_id),
            )
            self._enqueue_index_task(memory_id, "remove", timestamp)
            cursor = self.connection.execute(
                """
                INSERT INTO memories(category, content, source, sensitivity, active, status,
                                     user_id, subject, predicate, value, source_message_id,
                                     valid_from, importance, created_at, updated_at,
                                     scene, valid_to, temporary)
                VALUES (?, ?, ?, ?, 1, 'active', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (old["category"], content, source, sensitivity, old["user_id"],
                 old["subject"], old["predicate"], content, old["source_message_id"],
                 timestamp, old["importance"], timestamp, timestamp,
                 old["scene"], old["valid_to"], old["temporary"]),
            )
            replacement_id = int(cursor.lastrowid)
            self._enqueue_index_task(replacement_id, "upsert", timestamp)
            self.connection.execute(
                """
                INSERT INTO memory_history(memory_id, replacement_id, action, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (memory_id, replacement_id, "replaced", timestamp),
            )
            return replacement_id

    @_locked_method
    def delete_memory(self, memory_id: int) -> None:
        timestamp = self._now()
        cursor = self.connection.execute(
            "UPDATE memories SET active = 0, status = 'deleted', valid_to = ?,"
            " updated_at = ? WHERE id = ? AND active = 1",
            (timestamp, timestamp, memory_id),
        )
        if cursor.rowcount:
            self._enqueue_index_task(memory_id, "remove", timestamp)
            self.connection.execute(
                "INSERT INTO memory_history(memory_id, action, created_at) VALUES (?, ?, ?)",
                (memory_id, "deleted", timestamp),
            )
        self.connection.commit()

    @_locked_method
    def clear_memories(self) -> int:
        timestamp = self._now()
        ids = [row[0] for row in self.connection.execute("SELECT id FROM memories WHERE active = 1")]
        self.connection.execute(
            "UPDATE memories SET active = 0, status = 'deleted', valid_to = ?,"
            " updated_at = ? WHERE active = 1",
            (timestamp, timestamp),
        )
        self.connection.executemany(
            "INSERT INTO memory_history(memory_id, action, created_at) VALUES (?, 'deleted', ?)",
            [(memory_id, timestamp) for memory_id in ids],
        )
        for memory_id in ids:
            self._enqueue_index_task(memory_id, "remove", timestamp)
        self.connection.commit()
        return len(ids)

    @_locked_method
    def add_event(
        self,
        title: str,
        due_at: str,
        important: bool = False,
        created_at: Optional[str] = None,
    ) -> int:
        cursor = self.connection.execute(
            "INSERT INTO events(title, due_at, important, created_at) VALUES (?, ?, ?, ?)",
            (title, due_at, int(important), created_at or self._now()),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    @_locked_method
    def due_events(self, now: Optional[str] = None) -> List[EventRow]:
        timestamp = now or self._now()
        rows = self.connection.execute(
            """
            SELECT id, title, due_at, important, active, created_at
            FROM events
            WHERE active = 1 AND datetime(due_at) <= datetime(?)
            ORDER BY datetime(due_at), id
            """,
            (timestamp,),
        ).fetchall()
        return [
            EventRow(
                id=row["id"], title=row["title"], due_at=row["due_at"],
                important=bool(row["important"]), active=bool(row["active"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    @_locked_method
    def prune_raw_messages(self, now: Optional[str] = None, retention_days: int = 30) -> int:
        raise RuntimeError("raw messages must be summarized before deletion")

    @_locked_method
    def list_events(self, active_only: bool = True) -> List[EventRow]:
        clause = "WHERE active = 1" if active_only else ""
        rows = self.connection.execute("SELECT id,title,due_at,important,active,created_at FROM events {} ORDER BY due_at,id".format(clause)).fetchall()
        return [EventRow(r["id"],r["title"],r["due_at"],bool(r["important"]),bool(r["active"]),r["created_at"]) for r in rows]

    @_locked_method
    def update_event(self, event_id: int, title: str, due_at: str, important: bool) -> None:
        cursor=self.connection.execute("UPDATE events SET title=?,due_at=?,important=? WHERE id=? AND active=1",(title,due_at,int(important),event_id)); self.connection.commit()
        if not cursor.rowcount: raise KeyError(event_id)

    @_locked_method
    def delete_event(self, event_id: int) -> None:
        self.connection.execute("UPDATE events SET active=0 WHERE id=?",(event_id,)); self.connection.commit()

    @_locked_method
    def _delete_messages_by_ids(self, message_ids) -> int:
        """Internal primitive for deletion after a summary is stored in a transaction."""
        ids = tuple(int(message_id) for message_id in message_ids)
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        cursor = self.connection.execute(
            "DELETE FROM messages WHERE id IN ({})".format(placeholders), ids
        )
        return cursor.rowcount

    @_locked_method
    def pending_index_tasks(self, limit: int = 50) -> List[IndexTask]:
        rows = self.connection.execute(
            """
            SELECT id, memory_id, operation, attempts
            FROM memory_index_tasks
            WHERE completed_at IS NULL
            ORDER BY id
            LIMIT ?
            """,
            (max(0, int(limit)),),
        ).fetchall()
        return [IndexTask(row["id"], row["memory_id"], row["operation"], row["attempts"])
                for row in rows]

    @_locked_method
    def complete_index_tasks(self, task_ids) -> int:
        ids = tuple(int(item) for item in task_ids)
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        cursor = self.connection.execute(
            "UPDATE memory_index_tasks SET completed_at = ?, last_error = NULL"
            " WHERE id IN ({})".format(placeholders),
            (self._now(),) + ids,
        )
        self.connection.commit()
        return cursor.rowcount

    @_locked_method
    def fail_index_task(self, task_id: int, error: str) -> None:
        self.connection.execute(
            "UPDATE memory_index_tasks SET attempts = attempts + 1, last_error = ?"
            " WHERE id = ?",
            (str(error)[:300], int(task_id)),
        )
        self.connection.commit()

    @_locked_method
    def touch_memory_usage(self, memory_ids, when: Optional[str] = None) -> None:
        ids = tuple(int(item) for item in memory_ids)
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        self.connection.execute(
            "UPDATE memories SET last_used_at = ? WHERE id IN ({})".format(placeholders),
            (when or self._now(),) + ids,
        )
        self.connection.commit()

    @_locked_method
    def index_meta(self, key: str) -> Optional[str]:
        row = self.connection.execute(
            "SELECT value FROM memory_index_meta WHERE key = ?", (key,)
        ).fetchone()
        return None if row is None else row["value"]

    @_locked_method
    def set_index_meta(self, key: str, value: str) -> None:
        self.connection.execute(
            "INSERT INTO memory_index_meta(key, value) VALUES (?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(value)),
        )
        self.connection.commit()
