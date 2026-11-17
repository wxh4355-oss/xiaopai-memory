"""Relevant-memory retrieval and budgeted context selection.

Two backends share one interface:

* the Chroma index, when a DashScope key makes embeddings possible;
* a purely local bigram scorer, which keeps the app useful offline and is also
  the automatic fallback when the index is empty, stale or unavailable.

Whatever the backend returns is re-checked against SQLite so a superseded or
deleted row can never reach the model.
"""
import math
import re
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Tuple

from .persona import memory_line
from .memory_context import driving_audio, scene_for, validity_reason


CJK = re.compile(r"[^0-9a-z\u4e00-\u9fff]+")
WEIGHTS = {"similarity": 0.7, "importance": 0.2, "recency": 0.1}
RECENCY_HALF_LIFE_DAYS = 120.0
MIN_LEXICAL_EVIDENCE = 0.12
MIN_VECTOR_EVIDENCE = 0.45


def _bigrams(text: str) -> set:
    cleaned = CJK.sub("", str(text or "").lower())
    if not cleaned:
        return set()
    if len(cleaned) < 2:
        return {cleaned}
    return {cleaned[index:index + 2] for index in range(len(cleaned) - 1)}


def lexical_similarity(query: str, text: str) -> float:
    """Cosine over character bigrams: no tokenizer, no model, no network."""
    left, right = _bigrams(query), _bigrams(text)
    if not left or not right:
        return 0.0
    shared = len(left & right)
    if not shared:
        return 0.0
    score = shared / math.sqrt(len(left) * len(right))
    return min(1.0, score * 1.2)


def recency_score(created_at: Optional[str], now: Optional[datetime] = None) -> float:
    if not created_at:
        return 0.5
    try:
        moment = datetime.fromisoformat(str(created_at))
    except ValueError:
        return 0.5
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    current = now or datetime.now(timezone.utc).astimezone()
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (current - moment).total_seconds() / 86400.0)
    return 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)


def fuse(similarity: float, importance: float, recency: float) -> float:
    return (WEIGHTS["similarity"] * max(0.0, min(1.0, similarity))
            + WEIGHTS["importance"] * max(0.0, min(1.0, float(importance)))
            + WEIGHTS["recency"] * max(0.0, min(1.0, recency)))


class MemoryRetriever:
    def __init__(self, storage, index=None, settings=None, now=None, *, policy="v0"):
        if policy not in ("v0", "v1"):
            raise ValueError("unsupported retrieval policy")
        self.storage = storage
        self.index = index
        self.settings = settings
        self._now = now or (lambda: datetime.now(timezone.utc).astimezone())
        self.policy = policy
        self.last_backend = "none"

    def _setting(self, name, default):
        return getattr(self.settings, name, default) if self.settings is not None else default

    @property
    def user_id(self) -> str:
        return str(self._setting("user_id", "user_001") or "user_001")

    def _active_by_id(self) -> dict:
        user_id = self.user_id
        return {row.id: row for row in self.storage.list_memories(active_only=True)
                if str(row.user_id or "user_001") == user_id}

    def candidates(self, query: str, limit: Optional[int] = None, *, on_ranked=None,
                   on_evaluated=None, scene=None) -> List[Tuple[object, float]]:
        """Return scored (row, score) pairs, newest evidence already filtered."""
        limit = int(limit if limit is not None else self._setting("retrieval_limit", 20))
        if self.policy == "v1":
            return self._contextual_candidates(query, limit, scene, on_ranked, on_evaluated)
        active = self._active_by_id()
        if not active:
            self.last_backend = "none"
            if on_ranked is not None:
                on_ranked((), "none")
            return []
        current = self._now()
        scored = {}
        vector_scores = {}
        if self.index is not None and self.index.available:
            try:
                vector_scores = dict(self.index.search(query, user_id=self.user_id, limit=limit))
            except Exception:
                # A broken index degrades to lexical recall; it never breaks a reply.
                vector_scores = {}
        if vector_scores:
            backend = "chroma"
            for memory_id, similarity in vector_scores.items():
                row = active.get(memory_id)  # SQLite stays the source of truth
                if row is None:
                    continue
                scored[row.id] = fuse(similarity, row.importance, recency_score(row.created_at, current))
        else:
            backend = "lexical"
            for row in active.values():
                similarity = lexical_similarity(query, row.content)
                if similarity <= 0:
                    continue
                scored[row.id] = fuse(similarity, row.importance, recency_score(row.created_at, current))
            if not scored:
                # Nothing matches the wording: fall back to importance and recency
                # so a brand-new session still has some grounding.
                for row in active.values():
                    scored[row.id] = fuse(0.0, row.importance, recency_score(row.created_at, current))
        ordered = sorted(scored.items(), key=lambda item: (-item[1], item[0]))
        ranked = [(active[memory_id], score) for memory_id, score in ordered[:max(1, limit)]]
        self.last_backend = backend
        if on_ranked is not None:
            on_ranked((row.id for row, _ in ranked), backend)
        return ranked

    def _contextual_candidates(self, query, limit, scene, on_ranked, on_evaluated):
        current = self._now()
        scene = scene_for(query, current, scene)
        eligible, reasons = [], {}
        for row in self.storage.list_memories(active_only=False):
            if row.user_id != self.user_id:
                continue
            reason = validity_reason(row, current)
            if reason is None and row.scene not in ("default", scene):
                reason = "scene_mismatch"
            if reason is None:
                eligible.append(row)
            else:
                reasons[row.id] = reason

        vectors = {}
        if eligible and self.index is not None and self.index.available:
            try:
                vectors = dict(self.index.search(query, user_id=self.user_id, limit=limit))
            except Exception:
                pass
        driving_query = driving_audio(query)
        scored = []
        for row in eligible:
            lexical = max(lexical_similarity(query, text)
                          for text in (row.content, row.predicate, row.value))
            if driving_query and row.predicate == "驾车音频偏好":
                lexical = max(0.6, lexical)
            vector = vectors.get(row.id, 0.0)
            if lexical < MIN_LEXICAL_EVIDENCE and vector < MIN_VECTOR_EVIDENCE:
                reasons[row.id] = "insufficient_evidence"
                continue
            similarity = max(lexical, vector if vector >= MIN_VECTOR_EVIDENCE else 0.0)
            scored.append((row, fuse(similarity, row.importance, recency_score(row.created_at, current))))

        scored.sort(key=lambda item: (not item[0].temporary, item[0].scene != scene, -item[1], -item[0].id))
        specific_keys = {
            (row.subject, row.category, row.predicate)
            for row, _ in scored if scene != "default" and row.scene == scene and not row.temporary
        }
        filtered, exception_scopes = [], {}
        for row, score in scored:
            key = (row.subject, row.category, row.predicate)
            previous = exception_scopes.get(key)
            if row.predicate and previous is not None and previous in ("default", row.scene, scene):
                reasons[row.id] = "exception_preferred"
                continue
            if row.predicate and row.scene == "default" and not row.temporary and key in specific_keys:
                reasons[row.id] = "scene_override"
                continue
            if row.predicate and row.temporary:
                exception_scopes[key] = row.scene
            filtered.append((row, score))

        limit = max(1, limit)
        ranked = filtered[:limit]
        for row, _ in filtered[limit:]:
            reasons[row.id] = "ranked_out"
        for row, _ in ranked:
            reasons[row.id] = "candidate"
        self.last_backend = ("chroma" if vectors else "lexical") if eligible else "none"
        if on_ranked is not None:
            on_ranked((row.id for row, _ in ranked), self.last_backend)
        if on_evaluated is not None:
            on_evaluated(scene, reasons)
        return ranked

    def select(self, scored: Iterable[Tuple[object, float]], budget_chars: int) -> List:
        """Greedy budget fill that always keeps the best line when possible.

        The character budget alone used to admit every recalled candidate
        because memory lines are short; capping the item count is what actually
        keeps unrelated chatter out of the prompt.
        """
        budget = max(0, int(budget_chars))
        max_items = max(1, int(self._setting("memory_max_items", 6)))
        chosen, used = [], 0
        for row, _score in scored:
            if len(chosen) >= max_items:
                break
            line = memory_line(row)
            if not line:
                continue
            cost = len(line) + 1
            if used + cost > budget:
                if self.policy == "v0" and not chosen and budget > 0:
                    chosen.append(row)
                    used += cost
                continue
            chosen.append(row)
            used += cost
        return chosen

    def retrieve(self, query: str, budget_chars: Optional[int] = None, *, on_ranked=None,
                 on_evaluated=None, scene=None) -> List:
        budget = int(budget_chars if budget_chars is not None
                     else self._setting("memory_budget_chars", 1200))
        if self.policy == "v1":
            scored = self.candidates(query, scene=scene, on_ranked=on_ranked, on_evaluated=on_evaluated)
        else:
            scored = self.candidates(query) if on_ranked is None else self.candidates(query, on_ranked=on_ranked)
        return self.select(scored, budget)

    def sync_index(self, limit: int = 20) -> int:
        """Catch the index up before a search so fresh facts are visible."""
        if self.index is None or not self.index.available:
            return 0
        try:
            return self.index.sync_pending(limit)
        except Exception:
            return 0

    def ensure_index(self) -> int:
        """Background entry point: rebuild after a model change, then drain."""
        if self.index is None or not self.index.available:
            return 0
        try:
            return self.index.ensure_ready()
        except Exception:
            return 0
