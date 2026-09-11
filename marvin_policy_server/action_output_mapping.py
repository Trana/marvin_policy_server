"""Provide policy-action to joint-position mappings."""

from __future__ import annotations

import numpy as np


class SmoothBoundedActionMapper:
    """Map raw actions smoothly into asymmetric joint-position limits."""

    def __init__(
        self,
        default_position: np.ndarray,
        minimum_position: np.ndarray,
        maximum_position: np.ndarray,
        *,
        local_action_scale: float,
    ) -> None:
        """Initialize the mapper from joint defaults and soft limits."""
        self._default = np.asarray(
            default_position, dtype=np.float64
        ).reshape(-1)
        self._minimum = np.asarray(
            minimum_position, dtype=np.float64
        ).reshape(-1)
        self._maximum = np.asarray(
            maximum_position, dtype=np.float64
        ).reshape(-1)
        self._local_action_scale = float(local_action_scale)

        if not (
            self._default.shape == self._minimum.shape == self._maximum.shape
        ):
            raise ValueError(
                'Default, minimum, and maximum positions must have '
                'matching shapes'
            )
        if self._default.size == 0:
            raise ValueError('At least one joint is required')
        if (
            not np.isfinite(self._local_action_scale)
            or self._local_action_scale <= 0.0
        ):
            raise ValueError(
                'local_action_scale must be finite and greater than zero'
            )
        if not (
            np.isfinite(self._default).all()
            and np.isfinite(self._minimum).all()
            and np.isfinite(self._maximum).all()
        ):
            raise ValueError(
                'Smooth bounded mapping requires finite positions and limits'
            )

        self._lower_range = self._default - self._minimum
        self._upper_range = self._maximum - self._default
        invalid_lower = np.any(self._lower_range <= 0.0)
        invalid_upper = np.any(self._upper_range <= 0.0)
        if invalid_lower or invalid_upper:
            raise ValueError(
                'Every default position must lie strictly inside its limits'
            )

    def map_action(self, raw_action: np.ndarray) -> np.ndarray:
        """Return targets with the linear mapping's slope at zero."""
        action = np.asarray(raw_action, dtype=np.float64)
        if action.shape != self._default.shape:
            raise ValueError(
                f'Action shape {action.shape} does not match joint shape '
                f'{self._default.shape}'
            )
        if not np.isfinite(action).all():
            raise ValueError('Raw actions must be finite')

        positive = action >= 0.0
        active_range = np.where(positive, self._upper_range, self._lower_range)
        offset = active_range * np.tanh(
            action * self._local_action_scale / active_range
        )
        return self._default + offset


__all__ = ['SmoothBoundedActionMapper']
