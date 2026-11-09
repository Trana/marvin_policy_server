"""Constant-space local timing counters; detailed events are opt-in."""

from bisect import bisect_left
import math
import time

AGE_BOUNDS_MS = (
    0.0,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    7.5,
    10.0,
    12.5,
    15.0,
    20.0,
    25.0,
    30.0,
    40.0,
    50.0,
    75.0,
    100.0,
    150.0,
    250.0,
    500.0,
    1000.0,
)


class Timing:
    def __init__(self, mode='metrics', capture_seconds=30.0, *, clock=time.monotonic_ns):
        if mode not in {'off', 'metrics', 'capture'}:
            raise ValueError('tracing_mode must be off, metrics or capture')
        if not math.isfinite(capture_seconds) or not 0 < capture_seconds <= 300:
            raise ValueError('trace_capture_seconds must be in (0, 300]')
        self.mode = mode
        self.clock = clock
        self.started = clock()
        self.deadline = self.started + int(capture_seconds * 1e9)
        self.previous_tick = None
        self.previous_callback_end = None
        self.stage_started = None
        self.counters = {}
        self.durations = {}
        self.histograms = {}

    def observe_ms(self, name, value):
        """Fixed-bin distributions; percentile values are bucket upper bounds."""
        if self.mode == 'off':
            return
        if not math.isfinite(value) or value < 0:
            self.count(name + '_invalid')
            return
        if name not in self.histograms:
            self.histograms[name] = [0, 0.0, 0.0, [0] * (len(AGE_BOUNDS_MS) + 1)]
        entry = self.histograms[name]
        entry[0] += 1
        entry[1] += value
        entry[2] = max(entry[2], value)
        entry[3][bisect_left(AGE_BOUNDS_MS, value)] += 1

    def _distribution_snapshot(self):
        result = {}
        for name, (count, total, maximum, buckets) in self.histograms.items():
            item = {
                'count': count,
                'mean': total / count,
                'max': maximum,
                'bounds_ms': list(AGE_BOUNDS_MS),
                'bucket_counts': list(buckets),
            }
            for percentile in (50, 95, 99):
                target, cumulative = math.ceil(count * percentile / 100), 0
                for index, amount in enumerate(buckets):
                    cumulative += amount
                    if cumulative >= target:
                        item[f'p{percentile}_upper_ms'] = (
                            AGE_BOUNDS_MS[index] if index < len(AGE_BOUNDS_MS) else None
                        )
                        break
            result[name] = item
        return result

    def capture_enabled(self):
        if self.mode == 'capture' and self.clock() >= self.deadline:
            self.mode = 'metrics'
        return self.mode == 'capture'

    def count(self, name):
        self.counters[name] = self.counters.get(name, 0) + 1

    def duration(self, name, ns):
        count, total, maximum = self.durations.get(name, (0, 0, 0))
        self.durations[name] = (count + 1, total + ns, max(maximum, ns))

    def tick(self, expected_ns):
        now = self.clock()
        if self.previous_callback_end is not None:
            self.duration('between_callbacks', now - self.previous_callback_end)
        if self.previous_tick is not None:
            gap = now - self.previous_tick
            self.duration('tick_interval', gap)
            if gap > expected_ns * 1.5:
                self.count('late_tick_intervals')
        self.previous_tick = now
        self.stage_started = now
        self.count('ticks')
        return now

    def stage(self, name):
        """Record one completed, non-overlapping callback section."""
        if self.mode == 'off' or self.stage_started is None:
            return
        now = self.clock()
        self.duration(name, now - self.stage_started)
        self.stage_started = now

    def checkpoint(self, name, started):
        """Measure a nested section without changing callback stage boundaries."""
        if self.mode == 'off':
            return started
        now = self.clock()
        self.duration(name, now - started)
        return now

    def finish_callback(self, started, expected_ns):
        now = self.clock()
        elapsed = now - started
        self.duration('callback', elapsed)
        if self.stage_started is not None:
            self.duration('callback_tail', now - self.stage_started)
        if elapsed > expected_ns:
            self.count('callback_over_budget')
        self.previous_callback_end = now
        self.stage_started = None

    def snapshot(self):
        self.capture_enabled()
        now = self.clock()
        elapsed = (now - self.started) / 1e9
        result = {
            'schema_version': 'runtime-timing.v1',
            'source': 'policy',
            'mode': self.mode,
            'window_s': elapsed,
            'counts': self.counters,
            'rates_hz': {k: v / elapsed for k, v in self.counters.items()} if elapsed else {},
            'durations_ms': {
                k: {'count': n, 'mean': total / n / 1e6, 'max': maximum / 1e6}
                for k, (n, total, maximum) in self.durations.items()
            },
            'distributions_ms': self._distribution_snapshot(),
        }
        self.started = now
        self.counters = {}
        self.durations = {}
        self.histograms = {}
        return result
