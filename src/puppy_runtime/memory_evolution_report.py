"""Write and read local synthetic memory evaluation reports."""

import json
import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory


REPORT_MAX_BYTES = 8 * 1024 * 1024
PAGE_SIZE = 24
_CASE_MAX_BYTES = 2 * 1024 * 1024
_RUN_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
_DEFAULT_METRIC = 1
_RECALLED_METRIC = 2
_STALE_METRIC = 4
_IRRELEVANT_METRIC = 8
_CHALLENGE_METRIC = 16
_ALL_METRICS = 31
PAGE_METRICS = {
    None: _DEFAULT_METRIC, "samples": _DEFAULT_METRIC, "p95_ms": _DEFAULT_METRIC,
    "mean_injected_chars": _DEFAULT_METRIC, "cross_user_injected": _DEFAULT_METRIC,
    "preference_recalled": _RECALLED_METRIC, "stale_injected": _STALE_METRIC,
    "irrelevant_injected": _IRRELEVANT_METRIC, "cross_user_challenge": _CHALLENGE_METRIC,
}


def valid_run_id(value):
    return isinstance(value, str) and _RUN_ID.fullmatch(value) is not None


def _encode(value, max_bytes):
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n"
    if len(raw) > max_bytes:
        raise ValueError("memory evaluation report exceeds size limit")
    return raw


def _read_json(path):
    with path.open("rb") as stream:
        raw = stream.read(REPORT_MAX_BYTES + 1)
    if len(raw) > REPORT_MAX_BYTES:
        raise ValueError("memory evaluation report exceeds size limit")
    return json.loads(raw.decode("utf-8"))


def _validate_report(report):
    if not (
        isinstance(report, dict)
        and type(report.get("schema_version")) is int and report["schema_version"] == 1
        and report.get("synthetic") is True
        and isinstance(report.get("run_id"), str) and report["run_id"].strip()
        and isinstance(report.get("demo"), list)
        and isinstance(report.get("cases"), list)
        and isinstance(report.get("metrics"), dict)
    ):
        raise ValueError("invalid synthetic report")
    if "case_pages" in report:
        pages = report["case_pages"]
        if (pages != {"page_size": PAGE_SIZE} or type(pages["page_size"]) is not int
                or report["cases"] or not valid_run_id(report["run_id"])):
            raise ValueError("invalid paged report")


def read_report(path):
    report = _read_json(path)
    _validate_report(report)
    # The event stream serializes strict UTF-8 and the browser expects finite JSON numbers.
    json.dumps(report, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return report


def _bundle_path(path, run_id):
    if not valid_run_id(run_id):
        raise ValueError("invalid paged run ID")
    return path.parent.resolve() / f"{path.name}.{run_id}.cases"


def _bundle_file(bundle, name):
    path = bundle / name
    if path.resolve(strict=True) != path or not path.is_file():
        raise ValueError("invalid paged report path")
    return path


def _pairs(case):
    if not (isinstance(case, dict) and isinstance(case.get("id"), str)
            and isinstance(case.get("title", ""), str) and case.get("split") in ("debug", "holdout")):
        raise ValueError("invalid case in memory evaluation archive")
    policies = []
    for key in ("v0", "v1"):
        steps = case.get(key)
        if not isinstance(steps, list) or not all(
            isinstance(step, dict) and isinstance(step.get("id"), str) and 0 < len(step["id"]) <= 256
            for step in steps
        ):
            raise ValueError("invalid step in memory evaluation archive")
        policies.append({step["id"]: step for step in steps})
    v0, v1 = policies
    for step_id in dict.fromkeys([*v0, *v1]):
        yield step_id, v0.get(step_id), v1.get(step_id)


def _metric_mask(v0, v1):
    # Match the comparison view's membership rules, including unpaired probes.
    latest = v1 if v1 is not None else v0
    steps = [step for step in (v0, v1) if step is not None]
    outcomes = [step.get("outcome") or {} for step in steps]
    mask = _DEFAULT_METRIC if latest.get("expected") is not None else 0
    if any(outcome.get("recalled") is not None for outcome in outcomes):
        mask |= _RECALLED_METRIC
    if any(step.get("stale_expected") for step in steps):
        mask |= _STALE_METRIC
    if any(outcome.get("irrelevant") is not None for outcome in outcomes):
        mask |= _IRRELEVANT_METRIC
    if any(step.get("expected") is not None and any(
        item.get("user_id") != step.get("user_id") for item in step.get("memories", [])
    ) for step in steps):
        mask |= _CHALLENGE_METRIC
    return mask


def _step_failed(step):
    if step is None or step.get("error"):
        return True
    outcome = step.get("outcome") or {}
    return (outcome.get("recalled") is False or outcome.get("answer_ok") is False or any(
        outcome.get(key) for key in ("forbidden", "stale", "irrelevant", "cross_user", "answer_stale")
    ))


def _write_bundle(bundle, cases, run_id):
    failures, passes = [], []
    header = {"schema_version": 1, "run_id": run_id}
    bundle.mkdir()
    with (bundle / "cases.jsonl").open("wb") as stream:
        stream.write(_encode(header, _CASE_MAX_BYTES))
        for case in cases:
            offset = stream.tell()
            for step_id, v0, v1 in _pairs(case):
                if case["split"] == "holdout":
                    mask = _metric_mask(v0, v1)
                    if mask:
                        entries = failures if _step_failed(v0) or _step_failed(v1) else passes
                        entries.append((offset, step_id, mask))
            stream.write(_encode(case, _CASE_MAX_BYTES))
        archive_bytes = stream.tell()
    index = {**header, "archive_bytes": archive_bytes, "entries": failures + passes}
    (bundle / "index.json").write_bytes(_encode(index, REPORT_MAX_BYTES))


def write_report(report, path, *, paged=False):
    """Publish one summary after its run-specific artifacts have been written."""
    path = Path(path)
    _validate_report(report)
    summary = {**report, "cases": [], "case_pages": {"page_size": PAGE_SIZE}} if paged else report
    raw = _encode(summary, REPORT_MAX_BYTES)
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".memory-evaluation-", dir=path.parent) as directory:
        staging = Path(directory)
        pending = staging / "report.json"
        pending.write_bytes(raw)
        if paged:
            run_id = report["run_id"]
            bundle = _bundle_path(path, run_id)
            artifacts = staging / "cases"
            _write_bundle(artifacts, report["cases"], run_id)
            # Keep a published bundle immutable if two writes use the same run ID.
            artifacts.rename(bundle)
        try:
            pending.replace(path)
        except OSError:
            if paged:
                bundle.rename(artifacts)
            raise


def _archive_line(stream):
    raw = stream.readline(_CASE_MAX_BYTES + 1)
    if len(raw) > _CASE_MAX_BYTES or not raw.endswith(b"\n"):
        raise ValueError("invalid memory evaluation archive line")
    return json.loads(raw.decode("utf-8"))


def _read_page_positions(index, run_id, metric, start, archive_bytes, header_bytes):
    if not (isinstance(index, dict) and type(index.get("schema_version")) is int
            and index["schema_version"] == 1 and index.get("run_id") == run_id
            and type(index.get("archive_bytes")) is int and index["archive_bytes"] == archive_bytes
            and isinstance(index.get("entries"), list)):
        raise ValueError("invalid memory evaluation index")
    seen = set()
    total, positions = 0, []
    for item in index["entries"]:
        if not isinstance(item, list) or len(item) != 3:
            raise ValueError("invalid memory evaluation index entry")
        offset, step_id, mask = item
        if not (type(offset) is int and header_bytes <= offset < archive_bytes
                and isinstance(step_id, str) and 0 < len(step_id) <= 256
                and type(mask) is int and 0 < mask <= _ALL_METRICS and (offset, step_id) not in seen):
            raise ValueError("invalid memory evaluation index position")
        seen.add((offset, step_id))
        if mask & PAGE_METRICS[metric]:
            if start <= total < start + PAGE_SIZE:
                positions.append((offset, step_id, mask))
            total += 1
    if start and start >= total:
        raise ValueError("memory evaluation page out of range")
    return total, positions


def _sample_step(step):
    # Nested followups remain in the archive. The comparison renders only this step.
    return {key: value for key, value in step.items() if key != "followups"} if step is not None else None


def read_page(path, run_id, metric, page):
    bundle = _bundle_path(path, run_id)
    index = _read_json(_bundle_file(bundle, "index.json"))
    with _bundle_file(bundle, "cases.jsonl").open("rb") as stream:
        header = _encode({"schema_version": 1, "run_id": run_id}, _CASE_MAX_BYTES)
        if stream.read(len(header)) != header:
            raise ValueError("memory evaluation archive run mismatch")
        total, positions = _read_page_positions(index, run_id, metric, page * PAGE_SIZE,
                                               os.fstat(stream.fileno()).st_size, stream.tell())
        rows, cached = [], {}
        for offset, step_id, mask in positions:
            if offset not in cached:
                stream.seek(offset - 1)
                if stream.read(1) != b"\n":
                    raise ValueError("invalid memory evaluation archive offset")
                case = _archive_line(stream)
                steps = {key: (v0, v1) for key, v0, v1 in _pairs(case)}
                if case["split"] != "holdout":
                    raise ValueError("index points to a case outside the holdout set")
                cached[offset] = case, steps
            case, steps = cached[offset]
            if step_id not in steps or _metric_mask(*steps[step_id]) != mask:
                raise ValueError("index points to an invalid case sample")
            v0, v1 = steps[step_id]
            rows.append({"case_id": case["id"], "title": case.get("title", case["id"]),
                         "id": step_id, "v0": _sample_step(v0), "v1": _sample_step(v1)})
    _encode(rows, REPORT_MAX_BYTES)
    return total, rows
