"""One inference in flight, no backlog, and generation-scoped results. No ROS."""

from dataclasses import dataclass
from threading import Condition
from threading import Thread
import time


@dataclass(frozen=True)
class InferenceResult:
    generation: int
    context: object
    value: object
    error: Exception | None
    submitted_ns: int
    finished_ns: int


class InferenceWorker:
    """The worker owns compute state. The caller alone accepts and publishes results."""

    def __init__(self, compute, reset, *, clock=time.monotonic_ns):
        self._compute, self._reset, self._clock = compute, reset, clock
        self._condition = Condition()
        self._generation = 0
        self._job = self._result = None
        self._busy = False
        self._closed = False
        self._thread = Thread(target=self._run, name='policy-inference', daemon=True)
        self._thread.start()

    @property
    def ready(self):
        with self._condition:
            return not self._closed and not self._busy and self._result is None

    def invalidate(self):
        with self._condition:
            self._generation += 1
            self._result = None
            # Running native code cannot be cancelled. Its result will be discarded.

    def submit(self, payload, context):
        """Transfer an immutable/private input snapshot. Reject instead of queuing."""
        with self._condition:
            if self._closed or self._busy or self._result is not None:
                return False
            self._busy = True
            self._job = (self._generation, payload, context, self._clock())
            self._condition.notify()
            return True

    def poll(self):
        with self._condition:
            result, self._result = self._result, None
            return result

    def close(self, timeout=1.0):
        with self._condition:
            self._closed = True
            self._generation += 1
            self._result = None
            self._condition.notify()
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _run(self):
        previous_generation = None
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._job is not None)
                if self._closed:
                    return
                generation, payload, context, started = self._job
                self._job = None
            value = error = None
            try:
                if generation != previous_generation:
                    self._reset()
                    previous_generation = generation
                value = self._compute(payload)
            except Exception as exc:
                error = exc
            finished = self._clock()
            with self._condition:
                self._busy = False
                if not self._closed and generation == self._generation:
                    self._result = InferenceResult(
                        generation, context, value, error, started, finished
                    )
