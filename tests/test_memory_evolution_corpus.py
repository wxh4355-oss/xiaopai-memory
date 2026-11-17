from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import unittest

from scripts.memory_evolution_corpus import scale_dataset


DATASET = Path(__file__).with_name("memory_evolution_cases.json")
FAMILIES = {"correction", "expiry", "scene", "unrelated"}


class MemoryEvolutionCorpusTests(unittest.TestCase):
    def setUp(self):
        self.source = json.loads(DATASET.read_text(encoding="utf-8"))

    def test_scale_one_keeps_original_cases_and_demo(self):
        self.assertEqual(self.source, scale_dataset(self.source, 1))

    def test_scale_rejects_invalid_multipliers(self):
        for multiplier in (0, -1, 1.5, True):
            with self.subTest(multiplier=multiplier):
                with self.assertRaises(ValueError):
                    scale_dataset(self.source, multiplier)

    def test_scale_100_preserves_split_and_family_sizes_and_original_trajectories(self):
        expanded = scale_dataset(self.source, 100)
        self.assertEqual(100, expanded.get("multiplier"))
        cases = expanded["cases"]
        original = json.loads(DATASET.read_text(encoding="utf-8"))
        self.assertEqual(original, self.source)
        self.assertEqual(original["cases"], cases[:32])
        self.assertEqual(original["demo"], expanded["demo"])
        self.assertEqual(3200, len({case["id"] for case in cases}))
        self.assertEqual(3200, len(cases))
        self.assertEqual(
            {(split, family): count for split, count in (("debug", 200), ("holdout", 600))
             for family in FAMILIES},
            Counter((case["split"], case["id"].split("-")[0]) for case in cases),
        )
        self.assertEqual(3600, sum(
            step["expected"] is not None
            for case in cases if case["split"] == "holdout" for step in case["steps"]
        ))

    def test_scale_100_is_reproducible_and_separates_topics_and_users(self):
        first = scale_dataset(self.source, 100)
        self.assertEqual(first, scale_dataset(self.source, 100))
        splits = {split: {"users": set(), "topics": set()}
                  for split in ("debug", "holdout")}
        for case in first["cases"]:
            split = splits[case["split"]]
            split["users"].add(case["user_id"])
            for step in case["steps"]:
                split["users"].add(step.get("user_id", case["user_id"]))
                split["topics"].update(write["predicate"] for write in step["writes"])
        self.assertFalse(splits["debug"]["users"] & splits["holdout"]["users"])
        self.assertFalse(splits["debug"]["topics"] & splits["holdout"]["topics"])
        self.assertGreaterEqual(len(splits["debug"]["topics"]), 8)
        self.assertGreaterEqual(len(splits["holdout"]["topics"]), 24)

    def test_scale_100_varies_text_preference_values_and_timing(self):
        cases = scale_dataset(self.source, 100)["cases"][32:]
        signatures, vocabulary, elapsed = set(), set(), set()
        for case in cases:
            steps = case["steps"]
            signatures.add(tuple(step["text"] for step in steps))
            vocabulary.update(write["value"] for step in steps for write in step["writes"])
            times = [datetime.fromisoformat(step["at"]) for step in steps]
            self.assertEqual(sorted(times), times, case["id"])
            elapsed.add(times[-1] - times[0])
            self.assertEqual(len(steps), len({step["session_id"] for step in steps}))
            if case["id"].startswith("expiry-"):
                expiry = datetime.fromisoformat(steps[1]["writes"][0]["valid_to"])
                self.assertLess(times[2], expiry)
                self.assertLessEqual(expiry, times[3])
        self.assertEqual(len(cases), len(signatures))
        self.assertGreaterEqual(len(vocabulary), 120)
        self.assertGreaterEqual(len(elapsed), 40)
