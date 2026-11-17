import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from puppy_agent.config import AppSettings
from puppy_agent.conversation import ConversationService
from puppy_agent.memory import MemoryService, classify_candidate
from puppy_agent.retrieval import MemoryRetriever
from puppy_agent.storage import Storage


class ContextualMemoryTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "memory.db"
        self.moment = "2026-11-01T19:00:00+08:00"
        self.clock = lambda: datetime.fromisoformat(self.moment)
        self.settings = AppSettings()
        self.model = Mock()

    def remember(self, service, value, scene="evening_return", valid_to=None, text=""):
        return service.accept(classify_candidate(
            "preference", "开车听：" + value, "test", predicate="驾车音频偏好",
            value=value, scene=scene, valid_to=valid_to, turn_text=text))

    def test_time_and_scene_gates_survive_reopen_and_only_report_real_selection(self):
        with Storage(self.path, now=self.clock) as storage:
            memory = MemoryService(storage, self.model, self.settings)
            evening = self.remember(memory, "轻音乐")
            daytime = self.remember(memory, "播客", "daytime")
            exception = self.remember(memory, "保持安静", valid_to="2026-11-02T00:00:00+08:00")
        with Storage(self.path, now=self.clock) as storage:
            retriever = MemoryRetriever(storage, settings=self.settings, now=self.clock, policy="v1")
            memory = MemoryService(storage, self.model, self.settings)
            service = ConversationService(storage, self.model, self.settings, memory,
                                          retriever=retriever, now=self.clock)
            self.addCleanup(service.close)
            events = []
            service.set_recall_publisher(events.append)
            query = "今天下班回家的路上听点什么？"
            service._context(query)
            self.assertEqual([exception], events[-1]["selected"])
            self.assertEqual({daytime: "scene_mismatch", evening: "exception_preferred",
                              exception: "selected"},
                             {entry["id"]: entry["reason"] for entry in events[-1]["decisions"]})
            self.moment = "2026-11-02T19:00:00+08:00"
            service._context(query)
            self.assertEqual([evening], events[-1]["selected"])
            updated = self.remember(memory, "不推荐音乐", text="以后晚上开车都别推荐音乐了。")
            service._context(query)
            self.assertEqual([updated], events[-1]["selected"])
            service._context("白天开车听点什么？")
            self.assertEqual([daytime], events[-1]["selected"])
            service._context("明天会下雨吗？")
            self.assertEqual([], events[-1]["selected"])
            self.assertEqual([], events[-1]["candidates"])
            self.assertEqual(evening, storage.memory_history(updated)[0]["memory_id"])
            self.moment = "2026-11-02T18:59:59+08:00"
            service._context(query)
            self.assertEqual([evening], events[-1]["selected"])

    def test_feedback_is_scoped_to_a_selected_version_and_needs_confirmation(self):
        with Storage(self.path, now=self.clock) as storage:
            memory = MemoryService(storage, self.model, self.settings)
            evening = self.remember(memory, "轻音乐")
            trace = {"selected": [evening], "scene": "evening_return", "recall_id": "turn-1"}
            feedback = memory.record_feedback(trace, evening)
            self.assertEqual("not_applicable", feedback["action"])
            self.assertTrue(storage.memory_by_id(evening).active)
            with self.assertRaises(ValueError):
                memory.record_feedback(trace, evening + 1)
            self.moment = "2026-11-02T19:00:00+08:00"
            replacement = classify_candidate("preference", "保持安静", "test",
                                             predicate="驾车音频偏好", scene="evening_return")
            confirmation = memory.record_feedback(trace, evening, replacement)
            self.assertEqual("confirmed_correction", confirmation["action"])
            self.assertFalse(storage.memory_by_id(evening).active)
            self.assertTrue(storage.memory_by_id(confirmation["replacement_id"]).active)

    def test_edit_preserves_scene_and_expiration(self):
        with Storage(self.path, now=self.clock) as storage:
            memory = MemoryService(storage, self.model, self.settings)
            expires = "2026-11-02T00:00:00+08:00"
            previous = self.remember(memory, "保持安静", valid_to=expires)
            edited = storage.memory_by_id(storage.replace_memory(previous, "不要放歌", "edit", "low"))
            self.assertEqual(("evening_return", expires, True), (edited.scene, edited.valid_to, edited.temporary))

    def test_live_extraction_recognizes_temporary_intent_without_a_health_fact(self):
        with Storage(self.path, now=self.clock) as storage:
            memory = MemoryService(storage, self.model, self.settings)
            self.model.complete_json.return_value = {"memories": [{
                "category": "preference", "predicate": "驾车音频偏好",
                "value": "不要播放音乐", "content": "今晚返程不要播放音乐", "sensitivity": "low",
                "scene": "evening_return", "temporary": True,
            }]}
            user_text = "今天头疼，回去别放音乐。"
            message = storage.add_message("user", user_text, session_id="first")
            candidates = memory.candidates_for_turn(user_text, message_id=message)
            self.assertEqual(1, len(candidates))
            fact = storage.memory_by_id(memory.accept(candidates[0]))
            self.assertEqual("evening_return", fact.scene)
            self.assertEqual("2026-11-02T00:00:00+08:00", fact.valid_to)
            self.assertEqual(user_text, storage.message_by_id(fact.source_message_id).content)

    def test_disabled_contextual_recall_does_not_inject_expired_memories(self):
        self.settings.retrieval_enabled = False
        with Storage(self.path, now=self.clock) as storage:
            memory = MemoryService(storage, self.model, self.settings)
            self.remember(memory, "保持安静", valid_to="2026-11-02T00:00:00+08:00")
            self.moment = "2026-11-02T19:00:00+08:00"
            retriever = MemoryRetriever(storage, settings=self.settings, now=self.clock, policy="v1")
            service = ConversationService(storage, self.model, self.settings, memory,
                                          retriever=retriever, now=self.clock)
            self.addCleanup(service.close)
            self.assertNotIn("保持安静", service._context("回家路上听点什么？")[0]["content"])

    def test_confirmed_scene_correction_shadows_default_only_in_that_scene(self):
        with Storage(self.path, now=self.clock) as storage:
            memory = MemoryService(storage, self.model, self.settings)
            default = self.remember(memory, "轻音乐", "default")
            replacement = classify_candidate("preference", "保持安静", "test")
            feedback = memory.record_feedback({
                "selected": [default], "scene": "evening_return", "recall_id": "turn-1",
            }, default, replacement)
            retriever = MemoryRetriever(storage, settings=self.settings, now=self.clock, policy="v1")
            service = ConversationService(storage, self.model, self.settings, memory,
                                          retriever=retriever, now=self.clock)
            self.addCleanup(service.close)
            events = []
            service.set_recall_publisher(events.append)
            service._context("晚上开车听点什么？")
            self.assertEqual([feedback["replacement_id"]], events[-1]["selected"])
            self.assertIn({"id": default, "reason": "scene_override"}, events[-1]["decisions"])
            service._context("白天开车听点什么？")
            self.assertEqual([default], events[-1]["selected"])

    def test_explicit_remember_keeps_scope_and_expiry_with_or_without_extraction(self):
        with Storage(self.path, now=self.clock) as storage:
            memory = MemoryService(storage, self.model, self.settings)
            text = "记住今晚回去别放音乐"
            self.model.complete_json.side_effect = RuntimeError("offline")
            offline, = memory.candidates_for_turn(text)
            self.assertEqual(("evening_return", "2026-11-02T00:00:00+08:00"),
                             (offline.scene, offline.valid_to))
            self.model.complete_json.side_effect = None
            self.model.complete_json.return_value = {"memories": [{
                "category": "preference", "predicate": "驾车音频偏好",
                "value": "保持安静", "content": "今晚返程保持安静", "sensitivity": "low",
                "scene": "evening_return", "temporary": True,
            }]}
            online, = memory.candidates_for_turn(text)
            self.assertEqual((offline.predicate, offline.scene, offline.valid_to),
                             (online.predicate, online.scene, online.valid_to))

    def test_supersede_retry_keeps_single_history_link_and_index_removal(self):
        with Storage(self.path, now=self.clock) as storage:
            memory = MemoryService(storage, self.model, self.settings)
            old = self.remember(memory, "轻音乐")
            new = self.remember(memory, "保持安静")
            storage.supersede_memory(old, new)
            storage.supersede_memory(old, new)
            self.assertEqual(1, len(storage.memory_history(old)))
            self.assertEqual(1, sum(task.memory_id == old and task.operation == "remove"
                                    for task in storage.pending_index_tasks()))


