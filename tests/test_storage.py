import sqlite3
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from puppy_agent.config import AppPaths
from puppy_agent.models import MemoryCandidate, Sensitivity
from puppy_agent.storage import Storage


V1_SCHEMA = """
CREATE TABLE schema_meta (version INTEGER NOT NULL);
CREATE TABLE messages (
    id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL,
    content TEXT NOT NULL, incomplete INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL);
CREATE TABLE summaries (
    id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, content TEXT NOT NULL,
    first_message_id INTEGER NOT NULL, last_message_id INTEGER NOT NULL,
    created_at TEXT NOT NULL);
CREATE TABLE memories (
    id INTEGER PRIMARY KEY, category TEXT NOT NULL, content TEXT NOT NULL,
    source TEXT NOT NULL, sensitivity TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL, last_used_at TEXT);
CREATE TABLE memory_history (
    id INTEGER PRIMARY KEY,
    memory_id INTEGER NOT NULL REFERENCES memories(id),
    replacement_id INTEGER REFERENCES memories(id),
    action TEXT NOT NULL, created_at TEXT NOT NULL);
"""


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.paths = AppPaths(Path(self.temporary_directory.name))

    def tearDown(self):
        self.temporary_directory.cleanup()

    def make_storage(self):
        return Storage(self.paths)

    def test_schema_records_migration_and_enables_sqlite_safety_modes(self):
        storage = self.make_storage()
        self.assertEqual("wal", storage.connection.execute("PRAGMA journal_mode").fetchone()[0])
        self.assertEqual(1, storage.connection.execute("PRAGMA foreign_keys").fetchone()[0])
        tables = {
            row[0]
            for row in storage.connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        self.assertTrue(
            {"schema_meta", "messages", "summaries", "memories", "memory_history", "events", "scheduler_log"}.issubset(tables)
        )
        self.assertEqual(3, storage.connection.execute("SELECT version FROM schema_meta").fetchone()[0])
        storage.close()

    def test_v1_database_is_upgraded_in_place_with_backfilled_facts(self):
        connection = sqlite3.connect(str(self.paths.database_file))
        connection.executescript(V1_SCHEMA)
        connection.execute("INSERT INTO schema_meta(version) VALUES (1)")
        connection.execute(
            "INSERT INTO memories(category, content, source, sensitivity, active,"
            " created_at, updated_at) VALUES ('preference','喜欢草莓','auto','low',1,"
            "'2026-07-15T09:00:00+08:00','2026-07-15T09:00:00+08:00')"
        )
        connection.execute(
            "INSERT INTO memories(category, content, source, sensitivity, active,"
            " created_at, updated_at) VALUES ('hobby','旧爱好','auto','low',0,"
            "'2026-07-15T09:00:00+08:00','2026-07-15T09:00:00+08:00')"
        )
        connection.commit()
        connection.close()

        storage = self.make_storage()
        self.assertEqual(3, storage.connection.execute("SELECT version FROM schema_meta").fetchone()[0])
        active, retired = storage.list_memories(active_only=False)
        self.assertEqual("喜欢草莓", active.value)
        self.assertEqual("active", active.status)
        self.assertEqual("2026-07-15T09:00:00+08:00", active.valid_from)
        self.assertEqual("user_001", active.user_id)
        self.assertEqual("deleted", retired.status)
        tables = {
            row[0] for row in storage.connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        self.assertTrue({"memory_index_tasks", "memory_index_meta"}.issubset(tables))
        storage.close()

    def test_structured_facts_round_trip_and_queue_index_work(self):
        storage = self.make_storage()
        memory_id = storage.save_memory(
            "preference", "喜欢无糖拿铁", "auto", "low",
            subject="用户", predicate="饮品偏好", value="无糖拿铁",
            source_message_id=None, importance=0.8,
        )
        row = storage.list_memories()[0]
        self.assertEqual("用户", row.subject)
        self.assertEqual("饮品偏好", row.predicate)
        self.assertEqual("无糖拿铁", row.value)
        self.assertEqual(0.8, row.importance)
        self.assertEqual("active", row.status)
        self.assertEqual([(memory_id, "upsert")],
                         [(task.memory_id, task.operation) for task in storage.pending_index_tasks()])
        storage.complete_index_tasks([task.id for task in storage.pending_index_tasks()])
        self.assertEqual([], storage.pending_index_tasks())
        self.assertEqual(memory_id, storage.save_memory(
            "preference", "喜欢无糖拿铁", "auto", "low",
            subject="用户", predicate="饮品偏好", value="无糖拿铁"))
        storage.close()

    def test_supersede_keeps_the_old_version_and_queues_removal(self):
        storage = self.make_storage()
        old = storage.save_memory("preference", "喜欢咖啡", "auto", "low",
                                  subject="用户", predicate="饮品偏好", value="咖啡")
        new = storage.save_memory("preference", "改喝茶", "auto", "low",
                                  subject="用户", predicate="饮品偏好", value="茶")
        storage.supersede_memory(old, new)
        active = storage.list_memories(active_only=True)
        self.assertEqual([new], [item.id for item in active])
        retired = next(item for item in storage.list_memories(active_only=False) if item.id == old)
        self.assertEqual("superseded", retired.status)
        self.assertIsNotNone(retired.valid_to)
        operations = [(task.memory_id, task.operation) for task in storage.pending_index_tasks()]
        self.assertIn((old, "remove"), operations)
        history = storage.connection.execute(
            "SELECT memory_id, replacement_id, action FROM memory_history").fetchall()
        self.assertEqual([(old, new, "updated")], [tuple(row) for row in history])
        storage.close()

    def test_fact_lookup_is_scoped_by_user(self):
        storage = self.make_storage()
        storage.save_memory("preference", "喜欢咖啡", "auto", "low", user_id="user_001",
                            subject="用户", predicate="饮品偏好", value="咖啡")
        storage.save_memory("preference", "喜欢茶", "auto", "low", user_id="user_002",
                            subject="用户", predicate="饮品偏好", value="茶")
        mine = storage.active_facts("user_001", "用户", "饮品偏好")
        self.assertEqual(["咖啡"], [item.value for item in mine])
        self.assertEqual(2, len(storage.list_memories()))
        storage.close()

    def test_candidate_user_id_survives_the_save(self):
        storage = self.make_storage()
        candidate = MemoryCandidate("preference", "喜欢茶", Sensitivity.LOW, "auto",
                                    user_id="user_002")
        storage.save_memory(candidate)
        self.assertEqual("user_002", storage.list_memories()[0].user_id)
        storage.close()

    def test_saving_the_same_fact_twice_returns_the_existing_row(self):
        storage = self.make_storage()
        first = storage.save_memory("preference", "喜欢无糖拿铁", "auto", "low")
        again = storage.save_memory("preference", "  喜欢无糖拿铁  ", "auto", "low")
        self.assertEqual(first, again)
        self.assertEqual(1, len(storage.list_memories()))
        storage.close()
    def test_deduplication_is_scoped_by_category_and_reactivates_after_delete(self):
        storage = self.make_storage()
        first = storage.save_memory("preference", "喜欢无糖拿铁", "auto", "low")
        hobby = storage.save_memory("hobby", "喜欢无糖拿铁", "auto", "low")
        self.assertNotEqual(first, hobby)
        storage.delete_memory(first)
        revived = storage.save_memory("preference", "喜欢无糖拿铁", "auto", "low")
        self.assertNotIn(revived, (first, hobby))
        self.assertEqual([hobby, revived], [item.id for item in storage.list_memories()])
        storage.close()

    def test_event_crud_supports_ui(self):
        storage = self.make_storage()
        event_id = storage.add_event("纪念日晚餐", "2026-07-16T18:00:00+08:00", True)
        self.assertEqual("纪念日晚餐", storage.list_events()[0].title)
        storage.update_event(event_id, "纪念日约会", "2026-07-16T19:00:00+08:00", False)
        self.assertEqual("纪念日约会", storage.list_events()[0].title)
        storage.delete_event(event_id)
        self.assertEqual([], storage.list_events())
        storage.close()

    def test_messages_persist_across_reopen_and_return_chronologically(self):
        storage = self.make_storage()
        first = storage.add_message("user", "你好", session_id="morning", created_at="2026-07-15T09:00:00+08:00")
        second = storage.add_message("assistant", "我在", session_id="morning", incomplete=True, created_at="2026-07-15T09:01:00+08:00")
        storage.close()
        reopened = self.make_storage()
        rows = reopened.recent_messages(limit=2)
        self.assertEqual([first, second], [row.id for row in rows])
        self.assertEqual(["你好", "我在"], [row.content for row in rows])
        self.assertTrue(rows[-1].incomplete)
        with self.assertRaises(FrozenInstanceError):
            rows[0].content = "changed"
        reopened.close()

    def test_memory_candidate_can_be_saved(self):
        storage = self.make_storage()
        candidate = MemoryCandidate("preference", "喜欢散步", Sensitivity.LOW, "auto")
        memory_id = storage.save_memory(candidate)
        saved = storage.list_memories()[0]
        self.assertEqual(memory_id, saved.id)
        self.assertEqual("low", saved.sensitivity)
        storage.close()

    def test_memory_conflict_replaces_active_value_and_records_history(self):
        storage = self.make_storage()
        first = storage.save_memory("favorite_food", "喜欢草莓", "user_explicit", "low")
        second = storage.replace_memory(first, "最近更喜欢芒果", "user_explicit", "low")
        active = storage.list_memories(active_only=True)
        self.assertEqual([second], [item.id for item in active])
        all_rows = storage.list_memories(active_only=False)
        self.assertEqual([False, True], [item.active for item in all_rows])
        history = storage.connection.execute(
            "SELECT memory_id, replacement_id, action FROM memory_history"
        ).fetchone()
        self.assertEqual((first, second, "replaced"), tuple(history))
        storage.close()

    def test_delete_and_clear_memories_commit_immediately(self):
        storage = self.make_storage()
        first = storage.save_memory("preference", "喜欢蓝色", "user_explicit", "low")
        storage.save_memory("preference", "喜欢音乐", "user_explicit", "low")
        storage.delete_memory(first)
        other_connection = sqlite3.connect(str(self.paths.database_file))
        self.assertEqual(1, other_connection.execute("SELECT COUNT(*) FROM memories WHERE active = 1").fetchone()[0])
        other_connection.close()
        storage.clear_memories()
        self.assertEqual([], storage.list_memories())
        storage.close()

    def test_due_events_returns_only_active_events_due_by_now(self):
        storage = self.make_storage()
        due = storage.add_event("吃药", "2026-07-15T10:00:00+08:00", important=True)
        storage.add_event("复诊", "2026-07-16T10:00:00+08:00")
        rows = storage.due_events("2026-07-15T12:00:00+08:00")
        self.assertEqual([due], [row.id for row in rows])
        self.assertTrue(rows[0].important)
        storage.close()

    def test_prune_refuses_to_delete_unsummarized_raw_messages(self):
        storage = self.make_storage()
        storage.add_message("user", "旧消息", created_at="2026-05-01T00:00:00+08:00")
        storage.add_message("user", "新消息", created_at="2026-07-01T00:00:00+08:00")
        with self.assertRaisesRegex(RuntimeError, "summarized"):
            storage.prune_raw_messages(now="2026-07-15T00:00:00+08:00", retention_days=30)
        self.assertEqual(2, len(storage.recent_messages(limit=20)))
        storage.close()


if __name__ == "__main__":
    unittest.main()
