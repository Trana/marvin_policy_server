"""Bounded IPC reader for a management-capable policy node.

Only the ROS executor thread may change policy state. The reader copies packets
into a small queue and triggers an executor wake-up. Control acknowledgements
use a separate condition so a service can confirm native Stop without waiting
for its own executor to process the acknowledgement.
The queue holds at most queue_limit work packets plus one replaceable STATUS.
"""

from collections import deque
from dataclasses import replace
import json
import socket
from threading import Condition, Lock, Thread
import time

from .native_protocol import Kind, Packet, receive, send, validate_socket


class NativeChannel:
    def __init__(self, connection, wake, *, queue_limit=4):
        validate_socket(connection)
        if type(queue_limit) is not int or queue_limit < 1:
            raise ValueError('Native queue limit must be positive')
        self._connection = connection
        self._wake = wake
        self._limit = queue_limit
        self._condition = Condition()
        self._send_lock = Lock()
        self._control_lock = Lock()
        self._packets = deque()
        self._waiting_sequence = None
        self._ack = None
        self._sequence = 0
        self._closed = False
        self._error = None
        self._thread = Thread(target=self._read, name='native-policy-ipc', daemon=True)
        self._thread.start()

    def _check(self):
        if self._error is not None:
            raise RuntimeError(f'Native policy transport failed: {self._error}')
        if self._closed:
            raise RuntimeError('Native policy transport is closed')

    def _fail(self, error):
        with self._condition:
            if self._closed:
                return
            if self._error is None:
                self._error = error
            self._packets.clear()
            self._condition.notify_all()
        # The native owner must observe EOF even if Python's executor is stalled.
        try:
            self._connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self._wake()
        except Exception:
            pass  # The executor may already be shutting down.

    def send(self, packet):
        with self._send_lock:
            with self._condition:
                self._check()
            try:
                send(self._connection, packet)
            except (OSError, ValueError) as exc:
                self._fail(exc)
                raise RuntimeError(f'Native policy send failed: {exc}') from exc

    def control(self, packet, *, timeout=.25):
        """Confirm an operation with a matching ACK, or fail the entire channel."""
        if packet.kind not in {Kind.PREPARE, Kind.CONFIGURE, Kind.ACTIVATE, Kind.DEACTIVATE,
                               Kind.INVALIDATE, Kind.SHUTDOWN}:
            raise ValueError('Packet is not a native control operation')
        if not 0 < timeout <= 5.:
            raise ValueError('Control timeout must be in (0,5] seconds')
        with self._control_lock:
            with self._condition:
                self._check()
                self._sequence += 1
                self._waiting_sequence = self._sequence
                self._ack = None
            self.send(replace(packet, sequence=self._sequence))
            deadline = time.monotonic() + timeout
            with self._condition:
                while self._ack is None:
                    self._check()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        # Condition uses an RLock, so _fail can notify waiters here.
                        self._fail(TimeoutError('Native operation acknowledgement expired'))
                        self._check()
                    self._condition.wait(remaining)
                self._check()
                ack = self._ack
                self._ack = None
                self._waiting_sequence = None
                return ack

    def drain(self):
        """Called by the executor, never by the reader thread."""
        with self._condition:
            self._check()
            packets = tuple(self._packets)
            self._packets.clear()
            return packets

    def _read(self):
        try:
            while True:
                packet = receive(self._connection)
                with self._condition:
                    if self._closed:
                        return
                    if packet.kind == Kind.ACK:
                        if packet.sequence == self._waiting_sequence and self._ack is None:
                            self._ack = packet
                            self._condition.notify_all()
                        continue
                    # STATUS is a replaceable snapshot, not ordered inference work.
                    # Append the newest at its receive position so COMMIT/SNAPSHOT
                    # ordering remains intact. ACKs use the separate path above.
                    if packet.kind == Kind.STATUS:
                        self._packets = deque(
                            item for item in self._packets if item.kind != Kind.STATUS)
                    work_count = sum(item.kind != Kind.STATUS for item in self._packets)
                    if packet.kind != Kind.STATUS and work_count >= self._limit:
                        # Format only on failure. Source age includes transport time,
                        # not just time spent in this Python queue.
                        now = time.monotonic_ns()
                        def describe(item):
                            return {
                                'kind': item.kind.name, 'sequence': item.sequence,
                                'generation': item.generation, 'flags': int(item.flags),
                                'source_age_ms': (now - item.requested_ns) / 1e6
                                if 0 < item.requested_ns <= now else None,
                            }
                        detail = {'limit': self._limit,
                                  'queued': [describe(item) for item in self._packets],
                                  'incoming': describe(packet)}
                        raise RuntimeError('Native input queue exceeded its bounded capacity: ' +
                                           json.dumps(detail, separators=(',', ':')))
                    self._packets.append(packet)
                self._wake()
        except Exception as exc:
            self._fail(exc)

    def close(self):
        with self._condition:
            self._closed = True
            self._packets.clear()
            self._condition.notify_all()
        try:
            self._connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._thread.join(timeout=1.)
        self._connection.close()
        return not self._thread.is_alive()
