"""Single-worker FIFO persistence for conversation replies."""
from concurrent.futures import Future
import queue
from threading import Thread
import time


class SerializedPersistence:
    def __init__(self, storage, clock=time.monotonic, on_latency=None, on_warning=None):
        self.storage = storage
        self.clock = clock
        self.on_latency = on_latency or (lambda kind, milliseconds: None)
        self.on_warning = on_warning or (lambda message: None)
        self._queue = queue.Queue()
        self._closed = False
        self._worker = Thread(target=self._run, daemon=True, name="puppy-persistence")
        self._worker.start()

    @staticmethod
    def _notify(callback, *args):
        try:
            callback(*args)
        except Exception:
            pass

    def submit(self, role, content, session_id, incomplete=False):
        if self._closed:
            raise RuntimeError("persistence is closed")
        future = Future()
        self._queue.put((role, content, session_id, incomplete, future))
        return future

    def close(self, wait=True, timeout=None):
        if not self._closed:
            self._closed = True
            self._queue.put(None)
        if wait:
            self._worker.join(timeout)

    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            role, content, session_id, incomplete, future = item
            error = None
            for _ in range(2):
                started = self.clock()
                try:
                    result = self.storage.add_message(role, content, session_id=session_id,
                                                      incomplete=incomplete)
                    error = None
                except Exception as exc:
                    error = exc
                elapsed = max(0.0, self.clock() - started) * 1000
                if elapsed > 100:
                    self._notify(self.on_latency, "persistence", elapsed)
                if error is None:
                    future.set_result(result)
                    break
            if error is not None:
                self._notify(
                    self.on_warning,
                    "Puppy could not save this reply locally; displayed text is unchanged.",
                )
                future.set_exception(error)
