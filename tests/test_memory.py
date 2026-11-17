"""All personal-looking inputs below are fabricated classification fixtures."""
import tempfile
import unittest
from pathlib import Path

from puppy_agent import facts
from puppy_agent.config import AppSettings
from puppy_agent.memory import MemoryService, classify_candidate
from puppy_agent.models import Sensitivity
from puppy_agent.storage import Storage


class FakeKimi:
    def __init__(self, result=None, error=None):
        self.result = result or {"memories": []}
        self.error = error

    def complete_json(self, messages):
        if self.error:
            raise self.error
        return self.result


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.storage = Storage(Path(self.temp.name) / "test.db")

    def tearDown(self):
        self.storage.close()
        self.temp.cleanup()

    def test_address_candidate_requires_confirmation(self):
        candidate = classify_candidate("address", "我住在虚构测试路 0 号", "auto")
        self.assertEqual(Sensitivity.SENSITIVE, candidate.sensitivity)

    def test_model_candidates_are_validated_and_limited(self):
        entries = [
            {"category": "preference", "content": "喜欢草莓", "sensitivity": "low"},
            {"category": "unknown", "content": "x", "sensitivity": "low"},
            {"category": "preference", "content": "", "sensitivity": "low"},
            {"category": "preference", "content": "x" * 301, "sensitivity": "low"},
        ] + [{"category": "hobby", "content": str(i), "sensitivity": "low"} for i in range(8)]
        service = MemoryService(self.storage, FakeKimi({"memories": entries}), AppSettings())
        candidates = service.candidates_for_turn("我喜欢草莓")
        self.assertEqual(5, len(candidates))
        self.assertTrue(all(c.category in {"preference", "hobby"} for c in candidates))

    def test_sensitive_model_label_cannot_be_downgraded(self):
        service = MemoryService(self.storage, FakeKimi({"memories": [
            {"category": "health", "content": "最近失眠", "sensitivity": "low"}
        ]}), AppSettings())
        self.assertEqual(Sensitivity.SENSITIVE, service.candidates_for_turn("最近失眠")[0].sensitivity)

    def test_explicit_remember_command_survives_model_failure_and_keeps_sensitivity(self):
        service = MemoryService(self.storage, FakeKimi(error=RuntimeError("down")), AppSettings())
        candidates = service.candidates_for_turn("记住我住在虚构测试路 0 号")
        self.assertEqual(1, len(candidates))
        self.assertEqual("address", candidates[0].category)
        self.assertEqual(Sensitivity.SENSITIVE, candidates[0].sensitivity)

    def test_explicit_sensitive_commands_are_conservative_when_provider_fails(self):
        service = MemoryService(self.storage, FakeKimi(error=RuntimeError("down")), AppSettings())
        cases = [
            ("记住我和家里人吵架了", "family_conflict"),
            ("记住我和男朋友正在冷战", "relationship_conflict"),
            ("记住我有轻生的念头", "crisis"),
            ("记住我的银行卡账号是 TEST-ACCOUNT", "account"),
            ("记住我的邮箱是 demo@example.invalid", "contact"),
            ("记住我最近被诊断为高血压", "health"),
        ]
        for text, category in cases:
            with self.subTest(text=text):
                candidate = service.candidates_for_turn(text)[0]
                self.assertEqual(category, candidate.category)
                self.assertEqual(Sensitivity.SENSITIVE, candidate.sensitivity)

    def test_extraction_failure_is_recorded_instead_of_looking_like_no_facts(self):
        service = MemoryService(self.storage, FakeKimi(error=RuntimeError("invalid temperature")), AppSettings())
        self.assertEqual([], service.candidates_for_turn("今天天气不错"))
        self.assertEqual(1, service.failure_count)
        self.assertEqual(1, service.extraction_count)
        self.assertIn("invalid temperature", service.last_error)

    def test_successful_extraction_clears_the_recorded_failure(self):
        service = MemoryService(self.storage, FakeKimi({"memories": []}), AppSettings())
        service.candidates_for_turn("今天天气不错")
        self.assertIsNone(service.last_error)
        self.assertEqual(0, service.failure_count)

    def test_accepting_the_same_fact_twice_keeps_one_row(self):
        service = MemoryService(self.storage, FakeKimi({"memories": []}), AppSettings())
        candidate = classify_candidate("preference", "喜欢无糖拿铁", "auto")
        self.assertEqual(service.accept(candidate), service.accept(candidate))
        self.assertEqual(1, len(self.storage.list_memories()))

    def test_summarization_failure_retains_expired_rows(self):
        self.storage.add_message("user", "旧消息", created_at="2020-01-01T00:00:00+00:00")
        service = MemoryService(self.storage, FakeKimi(error=RuntimeError("down")), AppSettings(retention_days=30))
        self.assertEqual(0, service.summarize_expired_messages(now="2026-07-15T00:00:00+00:00"))
        self.assertEqual(["旧消息"], [m.content for m in self.storage.recent_messages(10)])

    def test_summarize_before_prune_saves_covered_range(self):
        first = self.storage.add_message("user", "甲", created_at="2020-01-01T00:00:00+00:00")
        last = self.storage.add_message("assistant", "乙", created_at="2020-01-01T00:01:00+00:00")
        service = MemoryService(self.storage, FakeKimi({"summary": "用户说甲，助手答乙"}), AppSettings(retention_days=30))
        self.assertEqual(2, service.summarize_expired_messages(now="2026-07-15T00:00:00+00:00"))
        row = self.storage.connection.execute("SELECT * FROM summaries").fetchone()
        self.assertEqual((first, last), (row["first_message_id"], row["last_message_id"]))
        self.assertEqual([], self.storage.recent_messages(10))

    def test_pruning_deletes_only_exact_expired_rows_selected_for_summary(self):
        first = self.storage.add_message("user", "旧甲", created_at="2020-01-01T00:00:00+00:00")
        middle = self.storage.add_message("user", "新消息", created_at="2026-07-14T00:00:00+00:00")
        last = self.storage.add_message("assistant", "旧乙", created_at="2020-01-01T00:01:00+00:00")
        service = MemoryService(self.storage, FakeKimi({"summary": "旧对话"}), AppSettings(retention_days=30))
        self.assertEqual(2, service.summarize_expired_messages(now="2026-07-15T00:00:00+00:00"))
        self.assertEqual([middle], [m.id for m in self.storage.recent_messages(10)])
        rows = self.storage.connection.execute(
            "SELECT first_message_id, last_message_id FROM summaries ORDER BY id"
        ).fetchall()
        self.assertEqual([(first, first), (last, last)], [tuple(row) for row in rows])


class MemoryDecisionTests(unittest.TestCase):
    """The accept() path: structured fact -> ADD / UPDATE / NOOP."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.storage = Storage(Path(self.temp.name) / "decisions.db")
        self.service = MemoryService(self.storage, FakeKimi({"memories": []}), AppSettings())

    def tearDown(self):
        self.storage.close()
        self.temp.cleanup()

    def candidate(self, value, predicate="饮品偏好", category="preference", turn=""):
        return classify_candidate(category, "喜欢" + value, "auto",
                                  predicate=predicate, value=value, turn_text=turn)

    def test_new_fact_is_added_with_its_source_message(self):
        memory_id = self.service.accept(self.candidate("无糖拿铁"), turn_text="我喜欢无糖拿铁")
        row = self.storage.list_memories()[0]
        self.assertEqual(memory_id, row.id)
        self.assertEqual("无糖拿铁", row.value)
        self.assertEqual("饮品偏好", row.predicate)
        self.assertEqual("active", row.status)
        self.assertEqual(facts.ADD, self.service.last_decision.action)

    def test_repeating_a_fact_is_a_noop(self):
        first = self.service.accept(self.candidate("无糖拿铁"))
        second = self.service.accept(self.candidate("无糖拿铁"))
        self.assertEqual(first, second)
        self.assertEqual(1, len(self.storage.list_memories()))
        self.assertEqual(facts.NOOP, self.service.last_decision.action)

    def test_explicit_correction_supersedes_and_keeps_history(self):
        old = self.service.accept(self.candidate("咖啡"), turn_text="我喜欢咖啡")
        new = self.service.accept(self.candidate("茶"), turn_text="最近不喝咖啡了，改喝茶")
        self.assertNotEqual(old, new)
        active = self.storage.list_memories(active_only=True)
        self.assertEqual([new], [row.id for row in active])
        retired = next(row for row in self.storage.list_memories(active_only=False)
                       if row.id == old)
        self.assertEqual("superseded", retired.status)
        self.assertIsNotNone(retired.valid_to)
        history = self.storage.connection.execute(
            "SELECT memory_id, replacement_id, action FROM memory_history").fetchall()
        self.assertEqual((old, new, "updated"), tuple(history[0]))
        self.assertEqual(facts.UPDATE, self.service.last_decision.action)

    def test_correction_with_a_renamed_property_still_updates_the_category(self):
        old = self.service.accept(self.candidate("咖啡"), turn_text="我喜欢咖啡")
        new = self.service.accept(
            self.candidate("茶", predicate="饮品习惯"), turn_text="最近不喝咖啡了，改喝茶")
        self.assertEqual([new], [row.id for row in self.storage.list_memories()])
        retired = next(row for row in self.storage.list_memories(active_only=False)
                       if row.id == old)
        self.assertEqual("superseded", retired.status)

    def test_plain_statement_with_a_new_value_coexists(self):
        first = self.service.accept(self.candidate("咖啡"), turn_text="我喜欢咖啡")
        second = self.service.accept(self.candidate("茶"), turn_text="我也喜欢茶")
        self.assertEqual({first, second},
                         {row.id for row in self.storage.list_memories()})

    def test_change_description_candidates_are_dropped(self):
        service = MemoryService(self.storage, FakeKimi({"memories": [
            {"category": "preference", "predicate": "饮品偏好", "value": "茶",
             "content": "改喝茶", "importance": 0.6, "sensitivity": "low"},
            {"category": "preference", "predicate": "饮品习惯变化", "value": "从咖啡转为茶",
             "content": "从咖啡转为茶", "importance": 0.6, "sensitivity": "low"},
        ]}), AppSettings())
        candidates = service.candidates_for_turn("最近不喝咖啡了，改喝茶")
        self.assertEqual(["饮品偏好"], [c.predicate for c in candidates])

    def test_known_predicates_are_offered_to_the_extractor(self):
        self.service.accept(self.candidate("无糖拿铁"))
        prompt = self.service._extraction_system_prompt()
        self.assertIn("饮品偏好", prompt)

    def test_extraction_attaches_the_source_message_id(self):
        service = MemoryService(self.storage, FakeKimi({"memories": [
            {"category": "preference", "predicate": "饮品偏好", "value": "无糖拿铁",
             "content": "喜欢喝无糖拿铁", "importance": 0.7, "sensitivity": "low"},
        ]}), AppSettings())
        candidates = service.candidates_for_turn("我喜欢喝无糖拿铁", message_id=42)
        self.assertEqual(42, candidates[0].source_message_id)
        service.accept(candidates[0])
        self.assertEqual(42, self.storage.list_memories()[0].source_message_id)


if __name__ == "__main__":
    unittest.main()
