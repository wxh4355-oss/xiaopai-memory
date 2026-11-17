import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from puppy_agent.config import AppSettings
from puppy_agent.retrieval import (MemoryRetriever, lexical_similarity, recency_score)
from puppy_agent.storage import Storage


NOW = datetime(2026, 11, 5, tzinfo=timezone.utc).astimezone()


class FakeIndex:
    available = True

    def __init__(self, scores=(), error=None):
        self.scores = list(scores)
        self.error = error
        self.searches = []
        self.synced = 0

    def search(self, query, user_id="user_001", limit=20):
        self.searches.append((query, user_id, limit))
        if self.error:
            raise self.error
        return list(self.scores)

    def sync_pending(self, limit=50):
        self.synced += 1
        return 0

    def ensure_ready(self):
        return 0


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.storage = Storage(Path(self.temp.name) / "r.db")
        self.settings = AppSettings()

    def tearDown(self):
        self.storage.close()
        self.temp.cleanup()

    def remember(self, content, value, category="preference", user="user_001", created=None):
        return self.storage.save_memory(
            category, content, "auto", "low", user_id=user, subject="用户",
            predicate="饮品偏好" if category == "preference" else "其他",
            value=value, importance=0.5, created_at=created)

    def retriever(self, index=None):
        return MemoryRetriever(self.storage, index, self.settings, now=lambda: NOW)

    def test_lexical_similarity_prefers_overlapping_wording(self):
        self.assertGreater(
            lexical_similarity("我很喜欢喝咖啡", "喜欢喝无糖拿铁"),
            lexical_similarity("我很喜欢喝咖啡", "生日是三月"),
        )

    def test_recency_decays_with_age(self):
        self.assertAlmostEqual(1.0, recency_score(NOW.isoformat(), NOW), places=2)
        old = (NOW - timedelta(days=240)).isoformat()
        self.assertAlmostEqual(0.25, recency_score(old, NOW), places=2)

    def test_without_an_index_only_relevant_memories_are_recalled(self):
        coffee = self.remember("喜欢喝无糖拿铁", "无糖拿铁")
        self.remember("生日是三月十四日", "三月十四日", category="important_date")
        rows = self.retriever().retrieve("我很喜欢喝咖啡")
        self.assertEqual([coffee], [row.id for row in rows])

    def test_budget_limits_how_many_lines_are_injected(self):
        for index in range(6):
            self.remember("喜欢喝第{}号饮料".format(index), "饮料{}".format(index),
                          created=(NOW - timedelta(days=index)).isoformat())
        retriever = self.retriever()
        scored = retriever.candidates("饮料")
        self.assertGreater(len(scored), 3)
        selected = retriever.select(scored, budget_chars=30)
        self.assertLess(len(selected), len(scored))
        self.assertGreaterEqual(len(selected), 1)

    def test_vector_backend_wins_when_it_returns_hits(self):
        coffee = self.remember("喜欢喝无糖拿铁", "无糖拿铁")
        self.remember("生日是三月十四日", "三月十四日", category="important_date")
        index = FakeIndex([(coffee, 0.93)])
        retriever = self.retriever(index)
        rows = retriever.retrieve("帮我选杯咖啡")
        self.assertEqual([coffee], [row.id for row in rows])
        self.assertEqual("chroma", retriever.last_backend)

    def test_vector_hits_are_rechecked_against_sqlite(self):
        coffee = self.remember("喜欢喝无糖拿铁", "无糖拿铁")
        self.storage.delete_memory(coffee)
        index = FakeIndex([(coffee, 0.99)])
        retriever = self.retriever(index)
        self.assertEqual([], retriever.retrieve("帮我选杯咖啡"))

    def test_a_stale_index_pointing_at_nothing_falls_back_to_lexical(self):
        coffee = self.remember("喜欢喝无糖拿铁", "无糖拿铁")
        index = FakeIndex([])
        retriever = self.retriever(index)
        self.assertEqual([coffee], [row.id for row in retriever.retrieve("咖啡")])
        self.assertEqual("lexical", retriever.last_backend)

    def test_index_errors_degrade_to_lexical_recall(self):
        coffee = self.remember("喜欢喝无糖拿铁", "无糖拿铁")
        index = FakeIndex(error=RuntimeError("chroma exploded"))
        retriever = self.retriever(index)
        self.assertEqual([coffee], [row.id for row in retriever.retrieve("咖啡")])
        self.assertEqual("lexical", retriever.last_backend)

    def test_memories_of_another_user_are_never_recalled(self):
        self.remember("喜欢喝咖啡", "咖啡", user="user_002")
        self.assertEqual([], self.retriever().retrieve("咖啡"))

    def test_no_match_still_returns_high_importance_grounding(self):
        memory_id = self.storage.save_memory(
            "important_date", "纪念日是七月十六日", "giver", "low",
            subject="用户", predicate="纪念日", value="七月十六日", importance=1.0)
        retriever = self.retriever()
        rows = retriever.retrieve("今天天气怎么样")
        self.assertEqual([memory_id], [row.id for row in rows])
        self.assertEqual("lexical", retriever.last_backend)


if __name__ == "__main__":
    unittest.main()
