"""Report local recall stages without sharing queries or remembered text."""

from uuid import uuid4


class RecallTrace:
    CANDIDATE_LIMIT = 160

    def __init__(self, publish):
        self._publish = publish
        self.recall_id = uuid4().hex if publish is not None else None
        self.candidates = None
        self.candidate_count = None
        self.selected = ()
        self.strategy = None
        self.fell_back = False
        self.scene = None
        self.policy = None
        self.decisions = {}
        self.elapsed_ms = None
        self.injected_chars = None

    def __enter__(self):
        self._emit("searching")
        return self if self.recall_id is not None else None

    def __exit__(self, error_type, _error, _traceback):
        if error_type is not None:
            self.selected = ()
        self._emit("failed" if error_type is not None or self.fell_back else "complete")

    def record_candidates(self, ids, strategy):
        candidates = self._ids(ids)
        self.candidate_count = len(candidates)
        self.candidates = candidates[: self.CANDIDATE_LIMIT]
        self.strategy = strategy

    def record_evaluated(self, scene, reasons):
        self.scene = scene
        self.decisions = reasons

    @staticmethod
    def _ids(values):
        return [value for value in values if type(value) is int and value > 0]

    def _emit(self, phase):
        if self._publish is None:
            return
        event = {
            "kind": "memory.recall",
            "phase": phase,
            "recall_id": self.recall_id,
            "selected": self._ids(self.selected),
        }
        if self.candidates is not None:
            event["candidates"] = self.candidates
            event["candidate_count"] = self.candidate_count
        if self.strategy is not None:
            event["strategy"] = self.strategy
        if self.policy is not None and phase != "searching":
            selected = set(event["selected"])
            decisions = dict(self.decisions)
            for memory_id in self.candidates or ():
                decisions[memory_id] = "selected" if memory_id in selected else "budget"
            for memory_id in event["selected"]:
                decisions[memory_id] = "selected"
            # Preserve selected and candidate evidence before historical skips.
            order = list(dict.fromkeys([*event["selected"], *(self.candidates or ()), *decisions]))
            event.update(
                policy=self.policy, scene=self.scene,
                decisions=[{"id": memory_id, "reason": decisions[memory_id]}
                           for memory_id in order[:self.CANDIDATE_LIMIT]],
                elapsed_ms=self.elapsed_ms, injected_chars=self.injected_chars,
            )
        try:
            self._publish(event)
        except Exception:
            # Presentation observers must never interrupt retrieval or generation.
            pass
