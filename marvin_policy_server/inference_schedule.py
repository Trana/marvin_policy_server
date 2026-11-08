"""Single-threaded ROS-clock deadlines; no catch-up inference bursts."""
from collections import deque


class InferenceSchedule:
    """Maintain cyclic deadlines independently of publication tick count."""

    def __init__(self, period_ns):
        periods = (period_ns,) if type(period_ns) is int else period_ns
        if (not isinstance(periods, (list, tuple)) or not periods or len(periods) > 1024 or
                any(type(p) is not int or p <= 0 for p in periods)):
            raise ValueError('Inference periods must be positive integer nanoseconds')
        self.periods_ns = tuple(periods)
        self.period_ns = self.periods_ns[0]
        self.cycle_ns = sum(self.periods_ns)
        self.reset()

    def reset(self):
        """Forget timing across activation, invalid inputs or a clock rewind."""
        self.deadline_ns = None
        self.last_clock_ns = None
        self.last_observation_ns = None
        self.phase = 0
        self.skipped_slots = 0
        self.observations = deque(maxlen=101)

    def due(self, now_ns):
        if self.last_clock_ns is not None and now_ns < self.last_clock_ns:
            self.reset()
        self.last_clock_ns = now_ns
        if self.deadline_ns is None:
            self.deadline_ns = now_ns
        return now_ns >= self.deadline_ns

    def observation_dt(self, now_ns):
        if self.last_observation_ns is None:
            return self.period_ns / 1e9
        return max(0, now_ns - self.last_observation_ns) / 1e9

    def complete(self, observation_ns, finished_ns):
        """Advance once on admission, skipping expired slots without bursts."""
        if finished_ns < observation_ns:
            self.reset()
            return
        self.last_observation_ns = observation_ns
        self.last_clock_ns = finished_ns
        self.observations.append(observation_ns)
        deadline = self.deadline_ns if self.deadline_ns is not None else observation_ns
        cycles = max(0, (finished_ns - deadline) // self.cycle_ns)
        deadline += cycles * self.cycle_ns
        advances = cycles * len(self.periods_ns)
        while deadline <= finished_ns:
            deadline += self.periods_ns[self.phase]
            self.phase = (self.phase + 1) % len(self.periods_ns)
            advances += 1
        self.skipped_slots += max(0, advances - 1)
        self.deadline_ns = deadline

    def observed(self):
        stamps = self.observations
        rate = ((len(stamps)-1)*1e9/(stamps[-1]-stamps[0])
                if len(stamps) > 1 and stamps[-1] > stamps[0] else None)
        return dict(sample_count=len(stamps), mean_hz=rate,
                    last_observation_ns=self.last_observation_ns, skipped_slots=self.skipped_slots)
