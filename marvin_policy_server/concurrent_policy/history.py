"""Reset-safe fixed estimator history used by the ROS deployment runtime."""

from __future__ import annotations

import numpy as np


class EstimatorFeatureHistory:
    """Maintain one fixed-length feature sequence for policy inference."""

    def __init__(self, history_length: int, feature_count: int):
        """Allocate an initially empty history."""
        self.history_length = int(history_length)
        self.feature_count = int(feature_count)
        self.values = np.zeros(
            (self.history_length, self.feature_count), dtype=np.float32
        )
        self.count = 0

    @property
    def ready(self) -> bool:
        """Return whether all slots represent distinct received frames."""
        return self.count >= self.history_length

    def reset(self) -> None:
        """Discard history at a new activation boundary."""
        self.values.fill(0.0)
        self.count = 0

    def push(self, features: np.ndarray) -> np.ndarray:
        """Append one frame, repeating the first across empty slots."""
        frame = np.asarray(features, dtype=np.float32).reshape(-1)
        if (
            frame.shape != (self.feature_count,)
            or not np.isfinite(frame).all()
        ):
            raise ValueError(
                f'Expected {self.feature_count} finite concurrent-estimator '
                f'features, got {frame.shape}'
            )
        if self.count == 0:
            self.values[:] = frame
        else:
            self.values[:-1] = self.values[1:].copy()
            self.values[-1] = frame
        self.count = min(self.count + 1, self.history_length)
        return self.values


__all__ = ['EstimatorFeatureHistory']
