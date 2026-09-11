"""Optional joint-position conditioning for runtime diagnostics."""

import math

import numpy as np


class ExponentialJointPositionFilter:
    """Apply a continuous-time first-order low-pass at inference updates."""

    def __init__(self, tau_seconds: float):
        """Configure a positive filter time constant."""
        self.tau_seconds = float(tau_seconds)
        if not np.isfinite(self.tau_seconds) or self.tau_seconds <= 0.0:
            raise ValueError('tau_seconds must be finite and greater than zero')
        self._value: np.ndarray | None = None
        self._last_time_seconds: float | None = None

    def reset(self) -> None:
        """Discard filter state at a policy activation boundary."""
        self._value = None
        self._last_time_seconds = None

    def update(
        self, value: np.ndarray, time_seconds: float
    ) -> np.ndarray:
        """Filter one finite position vector using elapsed wall/sim time."""
        current = np.asarray(value, dtype=np.float64).reshape(-1)
        now = float(time_seconds)
        if not np.isfinite(current).all() or not np.isfinite(now):
            raise ValueError('joint-position filter input must be finite')
        if self._value is None or self._value.shape != current.shape:
            self._value = current.copy()
            self._last_time_seconds = now
            return self._value.copy()
        elapsed = max(0.0, now - float(self._last_time_seconds))
        alpha = -math.expm1(-elapsed / self.tau_seconds)
        self._value += alpha * (current - self._value)
        self._last_time_seconds = now
        return self._value.copy()


__all__ = ['ExponentialJointPositionFilter']
