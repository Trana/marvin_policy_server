"""
Wall-time pacing for isolated simulation diagnostics, not robot control.

Call after each simulation update with its simulation time and time.monotonic().
Sleep for the returned duration before the next update. Never change physics dt.
The caller must stop the trial on ValueError and preserve the failed capture.
"""

import math


class SimulationPacer:
    """Fixed clock mapping with bounded lag; no silent re-anchoring or time skips."""

    def __init__(self, simulation_time, wall_time, *, max_lag_s, max_step_s):
        values = (simulation_time, wall_time, max_lag_s, max_step_s)
        if not all(math.isfinite(value) for value in values):
            raise ValueError('Pacing values must be finite')
        if max_lag_s <= 0 or max_step_s <= 0:
            raise ValueError('Pacing bounds must be positive')
        self._sim_origin = self._last_sim = simulation_time
        self._wall_origin = self._last_wall = wall_time
        self._max_lag = max_lag_s
        self._max_step = max_step_s
        self._failed = False

    def wait_seconds(self, simulation_time, wall_time):
        """
        Return required wait, rejecting clock resets, large steps and overruns.

        Equal simulation stamps are allowed during app updates without physics.
        A failed pacer stays failed: create a new capture to establish new origins.
        This does not prove accurate sleep, source timestamps or real-time speed.
        Validate the resulting recorded clocks independently.
        """
        if self._failed:
            raise ValueError('Pacing failed; a new trial is required')
        try:
            if not all(math.isfinite(value) for value in (simulation_time, wall_time)):
                raise ValueError('Clock values must be finite')
            if simulation_time < self._last_sim or wall_time < self._last_wall:
                raise ValueError('Clock moved backwards')
            if simulation_time - self._last_sim > self._max_step + 1e-12:
                raise ValueError('Simulation step exceeded pacing bound')
            ahead = (simulation_time - self._sim_origin) - (wall_time - self._wall_origin)
            if ahead < -self._max_lag - 1e-12:
                raise ValueError('Simulation fell behind wall time')
        except ValueError:
            self._failed = True
            raise
        self._last_sim = simulation_time
        self._last_wall = wall_time
        return max(0.0, ahead)
