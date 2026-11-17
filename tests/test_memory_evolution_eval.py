import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.eval_memory_evolution import load_dataset, run_evaluation


class MemoryEvolutionEvaluationTests(unittest.TestCase):
    def test_rejects_leaked_users_and_duplicate_step_ids(self):
        for defect in ("user", "step", "split", "demo_id"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as folder:
                dataset = load_dataset()
                held = next(case for case in dataset["cases"] if case["split"] == "holdout")
                if defect == "user":
                    debug = next(case for case in dataset["cases"] if case["split"] == "debug")
                    held["steps"][-1]["user_id"] = debug["user_id"]
                elif defect == "step":
                    held["steps"][-1]["id"] = held["steps"][0]["id"]
                elif defect == "split":
                    held["split"] = "other"
                else:
                    held["id"] = dataset["demo"]["id"]
                path = Path(folder) / "cases.json"
                path.write_text(json.dumps(dataset), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_dataset(path)

    def test_custom_ids_do_not_become_database_paths(self):
        dataset = load_dataset()
        dataset["cases"] = dataset["cases"][:2]
        dataset["cases"][0]["id"] = "../crossed"
        dataset["cases"][1]["id"] = "Unicode案例"
        with tempfile.TemporaryDirectory() as folder:
            report = run_evaluation(dataset, commit="test", replay_dir=folder, jobs=2)
            self.assertFalse(report["errors"])
            self.assertEqual(5, len(list((Path(folder) / report["run_id"]).glob("*.db"))))
            self.assertFalse(list(Path(folder).glob("*.db")))
            self.assertEqual(["../crossed", "Unicode案例"], [case["id"] for case in report["cases"]])

    def test_cli_expands_and_exports_the_input_and_scaled_report(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "eval_memory_evolution.py"
        with tempfile.TemporaryDirectory() as folder:
            dataset_path = Path(folder) / "cases.json"
            report_path = Path(folder) / "report.json"
            result = subprocess.run(
                [sys.executable, str(script), "--scale", "2", "--jobs", "2",
                 "--dataset-output", str(dataset_path), "--output", str(report_path)],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            self.assertEqual(0, result.returncode, result.stderr)
            dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(64, len(dataset["cases"]))
            self.assertEqual({"case_count": 64, "multiplier": 2,
                              "split_counts": {"debug": 16, "holdout": 48}},
                             report["dataset_profile"])
            self.assertEqual(72, report["metrics"]["v1"]["samples"])
            self.assertEqual(2, report["jobs"])
            self.assertEqual([], report["cases"])
            self.assertGreater(report["case_pages"]["page_size"], 0)

    def test_grouped_dataset_has_disjoint_users_and_full_demo(self):
        dataset = load_dataset()
        self.assertGreaterEqual(len(dataset["cases"]), 30)
        self.assertLessEqual(len(dataset["cases"]), 50)
        groups = {
            split: {case["user_id"] for case in dataset["cases"] if case["split"] == split}
            for split in ("debug", "holdout")
        }
        self.assertTrue(all(groups.values()))
        self.assertFalse(groups["debug"] & groups["holdout"])
        self.assertEqual(len(dataset["demo"]["steps"]), 6)

    def test_replay_reports_actual_injection_and_uses_holdout_denominators(self):
        dataset = load_dataset()
        report = run_evaluation(dataset, commit="test")
        self.assertFalse(report["errors"])
        self.assertEqual(report["commit"], "test")
        for policy in ("v0", "v1"):
            metrics = report["metrics"][policy]
            self.assertIsNone(metrics["answer_adherence"])
            self.assertEqual(metrics["cross_user_injected"]["num"], 0)
            self.assertGreater(metrics["cross_user_challenge"]["den"], 0)
            self.assertEqual(metrics["samples"], sum(
                step["expected"] is not None
                for case in dataset["cases"] if case["split"] == "holdout"
                for step in case["steps"]
            ))
        demo = report["demo"]
        self.assertEqual(len(demo), 6)
        self.assertIn("轻音乐", demo[1]["expected"])
        self.assertTrue(demo[1]["outcome"]["recalled"])
        selected = demo[1]["trace"]["selected"]
        self.assertTrue(any(row["id"] in selected and row["source_message"]
                            for row in demo[1]["memories"]))
        feedback_turn = demo[3]["followups"][0]
        self.assertEqual([row["action"] for row in feedback_turn["feedback"]], ["not_applicable"])
        self.assertEqual([row["action"] for row in demo[4]["feedback"]], ["confirmed_correction"])
        observed, confirmed = feedback_turn["feedback"][0], demo[4]["feedback"][0]
        self.assertEqual(observed["recall_id"], confirmed["recall_id"])
        self.assertEqual(observed["memory_id"], confirmed["memory_id"])
        self.assertEqual(len(feedback_turn["memories"]), len(demo[3]["memories"]))
        self.assertTrue(all(turn["outcome"]["recalled"] for turn in demo[4]["followups"]))
        self.assertEqual(demo[-1]["trace"]["selected"], [])

    def test_explicit_switch_retires_the_old_value_under_both_policies(self):
        dataset = load_dataset()
        dataset["cases"] = [case for case in dataset["cases"]
                            if case["id"] in ("correction-01", "correction-02")]
        report = run_evaluation(dataset, commit="test", split="debug")
        self.assertFalse(report["errors"])
        for case in report["cases"]:
            for policy in ("v0", "v1"):
                probe = case[policy][-1]
                with self.subTest(case=case["id"], policy=policy):
                    self.assertTrue(probe["outcome"]["recalled"])
                    self.assertEqual(probe["outcome"]["stale"], [])
                    self.assertTrue(any(row["status"] == "superseded" for row in probe["memories"]))
