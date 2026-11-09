"""Bounded disk worker. Progress describes only successfully synchronized records."""

import os
from queue import Empty
from queue import Full
from queue import Queue
from threading import Event
from threading import Lock
from threading import Thread
import time


class CaptureWriter:
    def __init__(self, path, capacity=1024, interval=0.05, sync=os.fsync):
        if capacity <= 0 or interval <= 0:
            raise ValueError('Invalid writer bounds')
        self.path, self.interval, self.sync = path, interval, sync
        self.queue = Queue(maxsize=capacity)
        self.stop = Event()
        self.lock = Lock()
        self._progress = None
        self._error = None
        self.thread = Thread(target=self._run, name='joint-capture-disk', daemon=True)
        self.thread.start()

    def submit(self, line, progress):
        if self.error or self.stop.is_set():
            raise RuntimeError('Capture writer unavailable')
        try:
            self.queue.put_nowait((line, dict(progress)))
        except Full as error:
            raise RuntimeError('Capture writer queue overflow') from error

    @property
    def error(self):
        with self.lock:
            return self._error

    def progress(self):
        with self.lock:
            return dict(self._progress) if self._progress else None

    def close(self):
        self.stop.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError('Capture writer shutdown timeout')
        if self.error:
            raise RuntimeError(self.error)

    def _run(self):
        try:
            with self.path.open('x') as stream:
                deadline = time.monotonic() + self.interval
                pending = None
                while not self.stop.is_set() or not self.queue.empty() or pending is not None:
                    try:
                        line, pending = self.queue.get(
                            timeout=max(0, min(0.01, deadline - time.monotonic()))
                        )
                        stream.write(line)
                    except Empty:
                        pass
                    now = time.monotonic()
                    if now >= deadline or (self.stop.is_set() and self.queue.empty()):
                        if pending is not None:
                            stream.flush()
                            started = time.monotonic()
                            self.sync(stream.fileno())
                            completed = time.monotonic()
                            with self.lock:
                                self._progress = dict(
                                    pending,
                                    committed_at=completed,
                                    sync_duration_ms=1000 * (completed - started),
                                )
                            pending = None
                        deadline = time.monotonic() + self.interval
        except Exception as error:
            with self.lock:
                self._error = str(error)


def durable_ready(progress, now, max_age):
    if not progress or progress['targets'] < 2 or progress['feedback'] < 2:
        return False
    return all(
        value is not None and 0 <= now - value <= max_age
        for value in (progress['last_target'], progress['last_feedback'], progress['committed_at'])
    )
