"""Single-threaded ROS-clock deadlines; no catch-up inference bursts."""


class InferenceSchedule:
    """Maintain phase-aligned deadlines independently of publication tick count."""

    def __init__(self, period_ns: int = 20_000_000):
        """Initialize a 50 Hz schedule by default; timestamps must share one clock."""
        if not isinstance(period_ns, int) or period_ns <= 0:
            raise ValueError('Inference period must be positive integer nanoseconds')
        self.period_ns = period_ns
        self.reset()

    def reset(self) -> None:
        """Forget timing across activation, invalid inputs or a clock rewind."""
        self.deadline_ns = None
        self.last_clock_ns = None
        self.last_observation_ns = None

    def due(self, now_ns: int) -> bool:
        """Report whether to infer, resetting automatically on backward time."""
        if self.last_clock_ns is not None and now_ns < self.last_clock_ns:
            self.reset()
        self.last_clock_ns = now_ns
        if self.deadline_ns is None:
            self.deadline_ns = now_ns
        return now_ns >= self.deadline_ns

    def observation_dt(self, now_ns: int) -> float:
        """Return observation elapsed seconds, not the publication tick interval."""
        if self.last_observation_ns is None:
            return self.period_ns / 1e9
        return max(0, now_ns - self.last_observation_ns) / 1e9

    def complete(self, observation_ns: int, finished_ns: int) -> None:
        """Advance past callback completion, skipping missed slots without bursts."""
        if finished_ns < observation_ns:
            self.reset()
            return
        self.last_observation_ns = observation_ns
        self.last_clock_ns = finished_ns
        deadline = self.deadline_ns if self.deadline_ns is not None else observation_ns
        elapsed_periods = (finished_ns - deadline) // self.period_ns
        self.deadline_ns = deadline + (elapsed_periods + 1) * self.period_ns
