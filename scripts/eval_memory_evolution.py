"""Replay synthetic Chinese conversations against two memory policies."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys
from tempfile import TemporaryDirectory
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT))

from puppy_agent.config import AppSettings
from puppy_agent.conversation import ConversationService
from puppy_agent.memory import MemoryService, classify_candidate
from puppy_agent.memory_context import timestamp
from puppy_agent.retrieval import MemoryRetriever
from puppy_agent.storage import Storage
from puppy_runtime.memory_evolution_report import write_report
from scripts.memory_evolution_corpus import scale_dataset

DATASET = ROOT / "tests" / "memory_evolution_cases.json"
OUTPUT = ROOT / ".runtime" / "memory-evolution" / "report.json"
POLICIES = ("v0", "v1")
LIVE_IDS = ("correction-03", "expiry-04", "scene-05", "unrelated-06")
OFFLINE_REPLY = "离线回放：未调用模型生成回答。"


def load_dataset(path=DATASET):
    dataset = json.loads(Path(path).read_text(encoding="utf-8"))
    _validate_dataset(dataset)
    return dataset


def _validate_dataset(dataset):
    cases = dataset["cases"]
    demo = dataset["demo"]
    case_ids = set()
    groups = {"debug": set(), "holdout": set()}
    for case in [demo, *cases]:
        case_id = case["id"]
        if not isinstance(case_id, str) or not case_id.strip() or case_id in case_ids:
            raise ValueError("轨迹 ID 缺失或重复")
        case_ids.add(case_id)
        turns = [turn for step in case["steps"] for turn in [step, *step.get("followups", [])]]
        step_ids = [turn["id"] for turn in turns]
        if len(set(step_ids)) != len(step_ids):
            raise ValueError(f"重复的轨迹步骤 ID：{case_id}")
        stamps = [datetime.fromisoformat(turn["at"]) for turn in turns]
        if stamps != sorted(stamps):
            raise ValueError(f"轨迹时间顺序错误：{case_id}")
        if case is not demo:
            split = case["split"]
            if split not in groups:
                raise ValueError(f"未知的轨迹分组：{split}")
            groups[split].add(case["user_id"])
            groups[split].update(turn.get("user_id", case["user_id"]) for turn in turns)
    if groups["debug"] & groups["holdout"]:
        raise ValueError("调试集与留出集用户重叠")


def _matches(rows, expected):
    return all(any(value == row["value"] or value in row["content"] for row in rows)
               for value in expected)


def _outcome(step, rows, user, live_reply=None):
    expected = step["expected"]
    stale = step.get("stale", [])
    return {
        "recalled": _matches(rows, expected) if expected else None,
        "forbidden": [value for value in step["forbidden"] if _matches(rows, [value])],
        "stale": [value for value in stale if _matches(rows, [value])],
        "irrelevant": any(row["category"] == "preference" and not row["valid_to"]
                          for row in rows) if step.get("unrelated") else None,
        "cross_user": [row["id"] for row in rows if row["user_id"] != user],
        "answer_ok": (all(value in live_reply for value in expected)
                      and all(value not in live_reply for value in step["forbidden"])
                      if live_reply is not None and expected else None),
        "answer_stale": (any(value in live_reply for value in stale)
                         if live_reply is not None and stale else None),
    }


def _snapshot(storage, at):
    memories = []
    for row in storage.list_memories(active_only=False):
        data = asdict(row)
        source = storage.message_by_id(row.source_message_id) if row.source_message_id else None
        data["source_message"] = asdict(source) if source else None
        data["history"] = storage.memory_history(row.id)
        if row.active and row.valid_to and timestamp(row.valid_to) <= timestamp(at):
            data["status"] = "expired"
        memories.append(data)
    return memories


def _turn(storage, step, user, policy, prior, live):
    at = step["at"]
    now = lambda: datetime.fromisoformat(at)
    settings = AppSettings(user_id=user, user_name="虚构用户", giver_name="虚构用户")
    session = step["session_id"]
    if live is not None:
        raise ValueError("This public export supports offline synthetic replay only")
    client = None
    memory = MemoryService(storage, client, settings)
    retriever = MemoryRetriever(storage, settings=settings, now=now, policy=policy)
    service = ConversationService(storage, client, settings, session_id=session,
                                  retriever=retriever, memory_service=memory, now=now)
    events = []
    service.set_recall_publisher(events.append)
    try:
        message_id = storage.add_message("user", step["text"], session_id=session)

        def candidate(spec):
            return classify_candidate("preference", source="offline-replay", user_id=user,
                                      source_message_id=message_id, turn_text=step["text"], **spec)

        feedback = []
        if step.get("feedback"):
            spec = step["feedback"]
            previous = prior[spec["from_step"]]
            selected = previous["trace"]["selected"]
            target = next(row for row in previous["memories"]
                          if row["id"] in selected and row["value"] == spec["target"])
            replacement = candidate(spec["replacement"]) if "replacement" in spec else None
            feedback.append(memory.record_feedback(previous["trace"], target["id"],
                                                   replacement=replacement))
        if not live:
            for spec in step["writes"]:
                memory.accept(candidate(spec), turn_text=step["text"])
        context = service._context(step["text"], scene=step["scene"])
        trace = events[-1]
        if trace["phase"] != "complete":
            raise RuntimeError("检索追踪未完成")
        reply = "".join(client.stream_chat(context)) if client else OFFLINE_REPLY
        if client and not reply.strip():
            raise RuntimeError("真实模型返回空回答")
        if client:
            storage.add_message("assistant", reply, session_id=session)
        usage = client.last_usage if client else None
        if client:
            for extracted in memory.candidates_for_turn(step["text"], message_id=message_id):
                memory.accept(extracted, turn_text=step["text"])
            if memory.last_error:
                raise RuntimeError("真实模型提取失败")
        storage.connection.execute(
            "CREATE TABLE IF NOT EXISTS replay_recalls "
            "(session_id TEXT, step_id TEXT, created_at TEXT, payload TEXT)")
        storage.connection.executemany(
            "INSERT INTO replay_recalls VALUES (?, ?, ?, ?)",
            [(session, step["id"], at, json.dumps(event, ensure_ascii=False)) for event in events])
        storage.connection.commit()
        memories = _snapshot(storage, at)
        rows = [row for row in memories if row["id"] in trace["selected"]]
        return {
            "id": step["id"], "label": step["label"], "at": at, "scene": step["scene"],
            "user_id": user, "session_id": session, "user_text": step["text"],
            "reply": reply, "trace": trace, "memories": memories, "feedback": feedback,
            "expected": step["expected"], "forbidden": step["forbidden"],
            "stale_expected": step.get("stale", []), "usage": usage,
            "outcome": _outcome(step, rows, user, reply if client else None),
        }
    finally:
        service.close()


def _replay(case, policy, path, live=None):
    results = []
    prior = {}
    for step in case["steps"]:
        # Recreate Storage, MemoryService, retriever and conversation after each
        # independent session. Only SQLite carries state across these steps.
        with Storage(path, now=lambda: datetime.fromisoformat(step["at"])) as storage:
            result = _turn(storage, step, step.get("user_id", case["user_id"]), policy, prior, live)
            prior[step["id"]] = result
        followups = []
        for turn in step.get("followups", []):
            with Storage(path, now=lambda: datetime.fromisoformat(turn["at"])) as storage:
                followups.append(_turn(storage, turn, turn.get("user_id", case["user_id"]), policy, prior, live))
        result["followups"] = followups
        results.append(result)
    return results


def _rate(values, *, unmeasured=False):
    flags = [bool(value) for value in values if value is not None]
    if unmeasured and not flags:
        return None
    return {"num": sum(flags), "den": len(flags)}


def _metrics(rows, live_rows):
    probes = [row for row in rows if row["expected"] is not None]
    timings = sorted(row["trace"]["elapsed_ms"] for row in probes)
    challenge = [row for row in probes
                 if any(item["user_id"] != row["user_id"] for item in row["memories"])]
    tokens = [row["usage"]["prompt_tokens"] for row in live_rows
              if row["usage"] and isinstance(row["usage"].get("prompt_tokens"), int)]
    return {
        "samples": len(probes),
        "preference_recalled": _rate(row["outcome"]["recalled"] for row in probes),
        "stale_injected": _rate(row["outcome"]["stale"] for row in probes if row["stale_expected"]),
        "irrelevant_injected": _rate(row["outcome"]["irrelevant"] for row in probes),
        "cross_user_injected": _rate(row["outcome"]["cross_user"] for row in probes),
        "cross_user_challenge": _rate(row["outcome"]["cross_user"] for row in challenge),
        "answer_adherence": _rate((row["outcome"]["answer_ok"] for row in live_rows), unmeasured=True),
        "answer_stale": _rate((row["outcome"]["answer_stale"] for row in live_rows), unmeasured=True),
        "p95_ms": timings[math.ceil(len(timings) * .95) - 1] if timings else None,
        "mean_injected_chars": mean(row["trace"]["injected_chars"] for row in probes) if probes else None,
        "mean_prompt_tokens": mean(tokens) if tokens else None,
        "prompt_token_samples": len(tokens),
    }


def _incorrect(outcome):
    return (outcome["recalled"] is False or bool(outcome["forbidden"]) or bool(outcome["stale"])
            or bool(outcome["irrelevant"]) or bool(outcome["cross_user"]))


def run_evaluation(dataset, *, commit, replay_dir=None, live=None, split=None,
                   jobs=1, on_progress=None):
    _validate_dataset(dataset)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    cases, errors, regressions, live_cases = [], [], [], []
    demo = []
    debug_example = None
    base = Path(replay_dir) / run_id if replay_dir else None
    if base:
        base.mkdir(parents=True)
    directory = nullcontext(base) if base else TemporaryDirectory(prefix="puppy-memory-")

    def replay(case, storage_key, policy, folder, case_errors, live_config=None):
        suffix = "-live" if live_config else ""
        try:
            return _replay(case, policy, Path(folder) / f"{storage_key}-{policy}{suffix}.db", live_config)
        except Exception as exc:
            case_errors.append({"case_id": case["id"], "policy": policy,
                                "mode": "live" if live_config else "offline",
                                "error": type(exc).__name__ if live_config else f"{type(exc).__name__}: {exc}"})
            return []

    source_cases = [case for case in dataset["cases"] if not split or case["split"] == split]
    with directory as folder:
        def compare(entry):
            number, case = entry
            storage_key = f"case-{number:05d}"
            case_errors = []
            identity = {key: case[key] for key in ("id", "split", "title")}
            compared = {**identity, **{policy: replay(case, storage_key, policy, folder, case_errors)
                                       for policy in POLICIES}}
            live_result = None
            if live and case["id"] in LIVE_IDS:
                live_result = {**identity, **{policy: replay(case, storage_key, policy, folder, case_errors, live)
                                             for policy in POLICIES}}
            return compared, live_result, case_errors

        with (ThreadPoolExecutor(max_workers=jobs) if jobs > 1 else nullcontext()) as pool:
            indexed_cases = enumerate(source_cases, start=1)
            results = pool.map(compare, indexed_cases) if pool else map(compare, indexed_cases)
            for compared, live_result, case_errors in results:
                cases.append(compared)
                errors.extend(case_errors)
                if live_result is not None:
                    live_cases.append(live_result)
                for before, after in zip(compared["v0"], compared["v1"]):
                    location = {"case_id": compared["id"], "step_id": before["id"]}
                    if compared["split"] == "holdout":
                        if not _incorrect(before["outcome"]) and _incorrect(after["outcome"]):
                            regressions.append(location)
                    elif (debug_example is None and before["outcome"]["irrelevant"]
                          and after["outcome"]["irrelevant"] is False):
                        debug_example = {
                            **location,
                            "cause": "V0 在无相关证据时仍采用长期偏好。",
                            "change": "V1 过滤缺乏相关证据的记录，该轮注入集合为空。",
                        }
                if on_progress:
                    on_progress(len(cases), len(source_cases))
        demo = replay(dataset["demo"], "demo", "v1", folder, errors)

    # Only complete paired trajectories contribute to either policy's denominator.
    paired = [case for case in cases if case["split"] == "holdout" and case["v0"] and case["v1"]]
    live_paired = [case for case in live_cases if case["v0"] and case["v1"]]
    metrics = {policy: _metrics([row for case in paired for row in case[policy]],
                                [row for case in live_paired for row in case[policy]])
               for policy in POLICIES}
    settings = AppSettings()
    return {
        "schema_version": 1, "run_id": run_id, "commit": commit, "synthetic": True,
        "dataset_version": dataset["version"], "mode": "offline+live-sample" if live else "offline",
        "dataset_profile": {"case_count": len(dataset["cases"]),
                            "split_counts": dict(Counter(case["split"] for case in dataset["cases"])),
                            "multiplier": dataset.get("multiplier", 1)},
        "jobs": jobs,
        "model": live["model"] if live else None,
        "provider": live["provider"] if live else None, "policies": list(POLICIES),
        "budget": {"memory_chars": settings.memory_budget_chars, "memory_items": settings.memory_max_items,
                   "context_chars": settings.context_char_budget},
        "cases": cases, "live_cases": live_cases, "demo": demo, "metrics": metrics,
        "errors": errors, "regressions": regressions, "debug_example": debug_example,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--replay-dir", type=Path, help="保留按运行编号隔离的 SQLite 文件")
    parser.add_argument("--split", choices=("debug", "holdout"))
    parser.add_argument("--scale", type=int, default=1, choices=range(1, 101), metavar="1–100",
                        help="用合成变体扩充原始轨迹，默认 1")
    parser.add_argument("--dataset-output", type=Path, help="保存本次使用的完整轨迹")
    parser.add_argument("--jobs", type=int, default=1, choices=range(1, 9), metavar="1–8",
                        help="离线并发轨迹数，默认 1")
    args = parser.parse_args()
    if args.output.resolve() == args.cases.resolve():
        parser.error("评测报告必须与输入轨迹分开")
    if args.dataset_output and args.dataset_output.resolve() in (args.cases.resolve(), args.output.resolve()):
        parser.error("轨迹输出必须与输入和评测报告分开")
    live = None
    # A downloaded source archive is also runnable without Git or repository history.
    try:
        if not (ROOT / ".git").exists():
            raise FileNotFoundError("no export repository")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                         stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True,
                                            stderr=subprocess.DEVNULL).strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit, dirty = "source-export", None
    dataset = scale_dataset(load_dataset(args.cases), args.scale)
    if args.dataset_output:
        args.dataset_output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".memory-corpus-", dir=args.dataset_output.parent) as folder:
            pending = Path(folder) / "cases.json"
            with pending.open("w", encoding="utf-8") as output:
                json.dump(dataset, output, ensure_ascii=False, indent=2)
                output.write("\n")
            pending.replace(args.dataset_output)

    def progress(done, total):
        if total > 50 and (done % 200 == 0 or done == total):
            print(f"已回放 {done}/{total} 条轨迹", file=sys.stderr, flush=True)

    report = run_evaluation(dataset, commit=commit, replay_dir=args.replay_dir,
                            live=live, split=args.split, jobs=args.jobs, on_progress=progress)
    report["working_tree_dirty"] = dirty
    write_report(report, args.output, paged=len(report["cases"]) > 50)
    print(f"{report['run_id']}  {args.output}")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    return int(bool(report["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
