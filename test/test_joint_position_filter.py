"""Test optional runtime joint-position filtering."""

from marvin_policy_server.joint_position_filter import (
    ExponentialJointPositionFilter,
)

import numpy as np

import pytest


def test_filter_initializes_without_a_startup_transient() -> None:
    """Use the first measured position directly."""
    position_filter = ExponentialJointPositionFilter(0.1)

    np.testing.assert_allclose(
        position_filter.update(np.array([1.0, -2.0]), 4.0),
        [1.0, -2.0],
    )


def test_filter_uses_elapsed_time_and_resets() -> None:
    """Match the continuous first-order response and reset semantics."""
    position_filter = ExponentialJointPositionFilter(0.1)
    position_filter.update(np.zeros(2), 1.0)

    result = position_filter.update(np.ones(2), 1.1)
    np.testing.assert_allclose(result, 1.0 - np.exp(-1.0))

    position_filter.reset()
    np.testing.assert_allclose(
        position_filter.update(np.array([3.0, 4.0]), 9.0),
        [3.0, 4.0],
    )


@pytest.mark.parametrize('tau', [0.0, -0.1, float('nan')])
def test_filter_rejects_invalid_time_constant(tau: float) -> None:
    """Require a positive finite time constant."""
    with pytest.raises(ValueError):
        ExponentialJointPositionFilter(tau)
