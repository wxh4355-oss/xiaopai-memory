from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import threading
import time
from typing import Callable, Optional

from .config import AppSettings
from .memory import MemoryService
from .persistence import SerializedPersistence
from .persona import build_system_prompt, memory_line, memory_text
from .memory_context import scene_for
from .recall_trace import RecallTrace
from .retrieval import MemoryRetriever


class ConversationService:
    def __init__(self, storage, kimi, settings: AppSettings,
                 memory_service: Optional[MemoryService] = None,
                 session_id: str = "default", system_prompt: Optional[str] = None,
                 clock=time.monotonic, retriever=None, now=None):
        self.storage, self.kimi, self.settings = storage, kimi, settings
        self.memory_service = memory_service or MemoryService(storage, kimi, settings)
        self.retriever = retriever or MemoryRetriever(storage, settings=settings)
        self.session_id, self.system_prompt = session_id, system_prompt
        self._clock = clock
        self._now = now or datetime.now
        self._profile_recall = now is not None
        self._pending, self._pending_lock, self._next_sequence = [], threading.RLock(), 0
        self._warning_callback = lambda message: None
        self._latency_callback = lambda kind, milliseconds: None
        self._publish_recall = None
        self.persistence = SerializedPersistence(
            storage, on_warning=lambda message: self._warning_callback(message),
            on_latency=lambda kind, ms: self._latency_callback(kind, ms))
        self._memory_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="puppy-memory")
        self._close_lock = threading.Lock()
        self._closed = False

    def set_background_callbacks(self, on_warning=None, on_latency=None):
        if on_warning is not None: self._warning_callback = on_warning
        if on_latency is not None: self._latency_callback = on_latency

    def set_recall_publisher(self, publish):
        self._publish_recall = publish

    @staticmethod
    def _notify(callback, *args):
        try:
            callback(*args)
        except Exception:
            pass

    def close(self, wait=True):
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        closer = getattr(self.kimi, "close", None)
        if closer is not None:
            closer()
        # Shutdown must not leave Tk waiting on a stalled provider or disk worker.
        self.persistence.close(wait=wait, timeout=2.0 if wait else None)
        self._memory_executor.shutdown(wait=False, cancel_futures=True)

    def _recent_summaries(self, remaining):
        """Newest summaries first, contiguous, until the budget is spent.

        Only the newest summary used to be read, so every earlier compression
        was stored in SQLite and then never used again.
        """
        limit = max(1, int(getattr(self.settings, "context_summary_limit", 3)))
        rows = self.storage.connection.execute(
            "SELECT content FROM summaries WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (self.session_id, limit)).fetchall()
        selected = []
        for row in rows:
            content = "Earlier summary: " + (row["content"] or "").strip()
            if len(content) > remaining:
                break
            selected.append(content)
            remaining -= len(content)
        return list(reversed(selected)), remaining

    def _fallback_memories(self, budget):
        user_id = str(getattr(self.settings, "user_id", "user_001") or "user_001")
        limit = max(1, int(getattr(self.settings, "memory_max_items", 6)))
        selected = []
        for row in self.storage.active_facts(user_id):
            line = memory_line(row)
            cost = len(line) + 1
            if line and cost <= budget:
                selected.append(row)
                budget -= cost
                if len(selected) == limit:
                    break
        return selected

    def _memories_for(self, query: str, trace=None, scene=None):
        """Only the memories that matter for this turn, within their own budget."""
        retriever = self.retriever
        contextual = isinstance(retriever, MemoryRetriever) and retriever.policy == "v1"
        budget = max(0, int(getattr(self.settings, "memory_budget_chars", 1200)))
        if not getattr(self.settings, "retrieval_enabled", True) or retriever is None:
            if contextual:
                return []
            return self._fallback_memories(budget)
        try:
            retriever.sync_index(limit=20)
            if contextual:
                selected = retriever.retrieve(
                    query, budget_chars=budget, scene=scene,
                    on_ranked=trace.record_candidates if trace else None,
                    on_evaluated=trace.record_evaluated if trace else None)
            elif trace is not None and type(retriever) is MemoryRetriever:
                selected = retriever.retrieve(query, budget_chars=budget, on_ranked=trace.record_candidates)
            else:
                # Injected retrievers keep their existing contract, with no candidate claim.
                selected = retriever.retrieve(query, budget_chars=budget)
        except Exception:
            # Retrieval must never stop a reply; fall back to the safest set.
            if trace is not None:
                trace.fell_back = True
                trace.strategy = "none" if contextual else "active-fallback"
            if contextual:
                return []
            return self._fallback_memories(budget)
        if selected:
            try:
                self.storage.touch_memory_usage([row.id for row in selected])
            except Exception:
                pass
        return selected

    def _context(self, query: str = "", *, scene=None):
        with RecallTrace(self._publish_recall) as trace:
            if trace is not None:
                policy = getattr(self.retriever, "policy", None)
                if policy == "v1" or self._profile_recall:
                    trace.policy = policy
                    trace.scene = scene_for(query, self._now(), scene)
            return self._build_context(query, trace, scene)

    def _build_context(self, query, trace, scene=None):
        started = time.perf_counter()
        memories = self._memories_for(query, trace, scene)
        if trace is not None:
            trace.elapsed_ms = (time.perf_counter() - started) * 1000
        memory_offsets = [] if trace is not None else None
        prompt = self.system_prompt or build_system_prompt(
            self.settings, memories, self._now(), memory_offsets=memory_offsets
        )
        budget = min(4000, max(0, int(getattr(self.settings, "context_char_budget", 4000))))
        prompt = prompt[:budget]
        context, remaining = [{"role": "system", "content": prompt}], budget - len(prompt)
        summaries, remaining = self._recent_summaries(remaining)
        for content in summaries:
            context.append({"role": "system", "content": content})
        limit = min(12, max(1, int(getattr(self.settings, "context_message_limit", 12))))
        rows = self.storage.recent_messages(limit=limit, session_id=self.session_id)
        merged = [(row.id * 2, 0, row.role, row.content) for row in rows
                  if row.session_id == self.session_id and row.content.strip()]
        with self._pending_lock:
            pending = list(self._pending)
        committed_messages = {(item[2], item[3]) for item in merged}
        for sequence, role, content, incomplete, after_message_id in pending:
            if (role, content) not in committed_messages:
                merged.append((after_message_id * 2 + 1, sequence, role, content))
        merged.sort(key=lambda item: (item[0], item[1]))
        if merged and len(merged[-1][3]) <= budget and len(merged[-1][3]) > remaining:
            prompt = prompt[:budget - len(merged[-1][3])]
            context = ([{"role": "system", "content": prompt}] if prompt else [])
            remaining = budget - len(prompt)
        selected = []
        for index, (_, _, role, content) in enumerate(reversed(merged[-limit:])):
            if remaining <= 0: break
            kept = content[:remaining] if index == 0 else content
            if index and len(kept) > remaining: continue
            if kept.strip():
                selected.append({"role": role, "content": kept}); remaining -= len(kept)
        context.extend(reversed(selected))
        if trace is not None:
            # The turn budget and latest-message reservation may truncate the prompt.
            trace.selected = [memory_id for memory_id, start in memory_offsets if start < len(prompt)]
            if trace.policy is not None:
                by_id = {row.id: row for row in memories}
                trace.injected_chars = sum(
                    len(memory_line(by_id[memory_id])) - len(memory_text(by_id[memory_id]))
                    + min(len(memory_text(by_id[memory_id])), len(prompt) - start)
                    for memory_id, start in memory_offsets if start < len(prompt)
                )
        return context

    def send(self, text: str, on_delta: Callable[[str], None],
             on_stream_finished: Callable[[str], None],
             on_memory_candidate: Optional[Callable[[object], None]] = None) -> str:
        if on_memory_candidate is None:
            on_memory_candidate, on_stream_finished = on_stream_finished, lambda reply: None
        if not isinstance(text, str) or not text.strip(): raise ValueError("message must not be blank")
        started = self._clock()
        user_message_id = self.storage.add_message("user", text, session_id=self.session_id)
        elapsed = max(0.0, self._clock() - started) * 1000
        if elapsed > 100:
            self._notify(self._latency_callback, "persistence", elapsed)
        messages, chunks = self._context(text), []
        try:
            for delta in self.kimi.stream_chat(messages):
                chunks.append(delta); on_delta(delta)
        except Exception:
            partial = "".join(chunks)
            if partial.strip(): self._queue_assistant(partial, user_message_id, incomplete=True)
            raise
        reply = "".join(chunks)
        on_stream_finished(reply)
        self._queue_assistant(reply, user_message_id)
        self._memory_executor.submit(self._extract_memory, text, user_message_id,
                                     on_memory_candidate)
        return reply

    def _queue_assistant(self, content, after_message_id, incomplete=False):
        with self._pending_lock:
            self._next_sequence += 1; sequence = self._next_sequence
            self._pending.append((sequence, "assistant", content, incomplete, after_message_id))
        future = self.persistence.submit("assistant", content, self.session_id, incomplete)
        def completed(done):
            if done.exception() is None:
                with self._pending_lock:
                    self._pending = [item for item in self._pending if item[0] != sequence]
        future.add_done_callback(completed)
        return future

    def _extract_memory(self, text, message_id, callback):
        try:
            for candidate in self.memory_service.candidates_for_turn(text, message_id=message_id):
                callback(candidate)
            if self.retriever is not None:
                self.retriever.ensure_index()
        except Exception:
            pass
