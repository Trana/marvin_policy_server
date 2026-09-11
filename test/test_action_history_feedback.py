"""Test actor action-history feedback selection."""

from marvin_policy_server.action_history_feedback import (
    action_history_feedback,
)

import numpy as np

import pytest


def test_raw_mode_preserves_policy_action() -> None:
    """Feed the raw actor output back without sharing mutable storage."""
    raw = np.array([2.0, -3.0])
    result = action_history_feedback(
        'raw', raw, np.array([0.2, -0.4]), np.zeros(2), 0.25
    )

    np.testing.assert_allclose(result, raw)
    assert result is not raw


def test_zero_mode_removes_actor_action_history() -> None:
    """Replace the history frame with zeros for the ablation."""
    result = action_history_feedback(
        'zero',
        np.array([2.0, -3.0]),
        np.array([0.2, -0.4]),
        np.zeros(2),
        0.25,
    )

    np.testing.assert_allclose(result, np.zeros(2))


def test_applied_mode_inverts_target_scaling() -> None:
    """Represent the final safety-limited policy target as an action."""
    result = action_history_feedback(
        'applied',
        np.array([9.0, -9.0]),
        np.array([0.6, -0.7]),
        np.array([0.1, -0.2]),
        0.25,
    )

    np.testing.assert_allclose(result, [2.0, -2.0])


@pytest.mark.parametrize('mode', ['bad', '', 'RAW'])
def test_invalid_mode_is_rejected(mode: str) -> None:
    """Reject unknown action-history representations."""
    with pytest.raises(ValueError):
        action_history_feedback(
            mode, np.zeros(2), np.zeros(2), np.zeros(2), 0.25
        )
