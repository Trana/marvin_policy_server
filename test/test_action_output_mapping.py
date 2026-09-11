"""Test the smooth bounded policy-action output mapping."""

from marvin_policy_server.action_output_mapping import (
    SmoothBoundedActionMapper,
)

import numpy as np

import pytest


def test_smooth_mapping_preserves_default_and_local_linear_gain() -> None:
    """Preserve the default pose and trained local action gain."""
    mapper = SmoothBoundedActionMapper(
        np.array([0.1, -0.2]),
        np.array([-0.4, -1.0]),
        np.array([0.3, 0.7]),
        local_action_scale=0.25,
    )

    np.testing.assert_allclose(mapper.map_action(np.zeros(2)), [0.1, -0.2])
    epsilon = 1.0e-6
    positive_slope = (
        mapper.map_action(np.full(2, epsilon)) - np.array([0.1, -0.2])
    ) / epsilon
    negative_slope = (
        mapper.map_action(np.full(2, -epsilon)) - np.array([0.1, -0.2])
    ) / -epsilon
    np.testing.assert_allclose(positive_slope, 0.25, rtol=1.0e-6, atol=1.0e-8)
    np.testing.assert_allclose(negative_slope, 0.25, rtol=1.0e-6, atol=1.0e-8)


def test_smooth_mapping_approaches_asymmetric_limits() -> None:
    """Approach each joint's asymmetric soft limits without crossing."""
    default = np.array([0.1, -0.2])
    minimum = np.array([-0.4, -1.0])
    maximum = np.array([0.3, 0.7])
    mapper = SmoothBoundedActionMapper(
        default,
        minimum,
        maximum,
        local_action_scale=0.25,
    )

    positive = mapper.map_action(np.full(2, 1.0e6))
    negative = mapper.map_action(np.full(2, -1.0e6))

    assert np.all(positive <= maximum)
    assert np.all(positive > default)
    assert np.all(negative >= minimum)
    assert np.all(negative < default)
    np.testing.assert_allclose(positive, maximum)
    np.testing.assert_allclose(negative, minimum)


@pytest.mark.parametrize(
    ('default', 'minimum', 'maximum'),
    [
        ([0.0], [0.0], [1.0]),
        ([0.0], [-1.0], [0.0]),
        ([0.0], [-np.inf], [1.0]),
    ],
)
def test_smooth_mapping_rejects_invalid_limits(
    default, minimum, maximum
) -> None:
    """Reject incomplete limits or defaults on an endpoint."""
    with pytest.raises(ValueError):
        SmoothBoundedActionMapper(
            np.asarray(default),
            np.asarray(minimum),
            np.asarray(maximum),
            local_action_scale=0.25,
        )
