import unittest

from puppy_agent import facts
from puppy_agent.models import MemoryCandidate, Sensitivity


class Row:
    """Minimal stand-in for an active MemoryRow."""

    def __init__(self, memory_id, value):
        self.id = memory_id
        self.value = value
        self.content = value


def candidate(value, predicate="饮品偏好", category="preference"):
    return MemoryCandidate(category, "喜欢" + value, Sensitivity.LOW, "auto",
                           predicate=predicate, value=value)


class CorrectionIntentTests(unittest.TestCase):
    def test_correction_markers_are_detected(self):
        for text in ("最近不喝咖啡了，改喝茶",
                     "我现在住杭州，之前住上海",
                     "其实我换成无糖的了"):
            with self.subTest(text=text):
                self.assertTrue(facts.correction_intent(text))

    def test_plain_statement_is_not_a_correction(self):
        self.assertFalse(facts.correction_intent("我喜欢喝无糖拿铁"))
        self.assertFalse(facts.correction_intent("今天有点累"))


class DecisionTests(unittest.TestCase):
    def test_new_fact_is_added(self):
        decision = facts.decide(candidate("无糖拿铁"), [], "我喜欢喝无糖拿铁")
        self.assertEqual(facts.ADD, decision.action)
        self.assertTrue(decision.writes)

    def test_same_value_is_a_noop_even_in_a_correction(self):
        decision = facts.decide(candidate("无糖拿铁"), [Row(4, "无糖拿铁")], "我改喝无糖拿铁了")
        self.assertEqual(facts.NOOP, decision.action)
        self.assertEqual(4, decision.memory_id)

    def test_different_value_without_correction_coexists(self):
        """喜欢咖啡 and 喜欢茶 may both be true."""
        decision = facts.decide(candidate("茶"), [Row(4, "咖啡")], "我也喜欢喝茶")
        self.assertEqual(facts.ADD, decision.action)
        self.assertEqual((), decision.supersede)

    def test_explicit_correction_supersedes_every_active_value(self):
        rows = [Row(4, "咖啡"), Row(7, "果汁")]
        decision = facts.decide(candidate("茶"), rows, "最近不喝咖啡了，改喝茶")
        self.assertEqual(facts.UPDATE, decision.action)
        self.assertEqual(7, decision.memory_id)
        self.assertEqual((7, 4), decision.supersede)

    def test_near_duplicate_wording_is_a_noop(self):
        decision = facts.decide(candidate("无糖拿铁"), [Row(4, "无糖拿铁")],
                                "我喜欢无糖拿铁", similarity=0.97)
        self.assertEqual(facts.NOOP, decision.action)

    def test_empty_value_is_rejected(self):
        decision = facts.decide(candidate(""), [], "没什么")
        self.assertEqual(facts.NOOP, decision.action)


if __name__ == "__main__":
    unittest.main()
