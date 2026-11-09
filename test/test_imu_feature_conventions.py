"""Known-pose checks independent of physical sensor mounting and ROS."""

import math

from marvin_policy_server.sensor_snapshot import SensorSnapshot
from marvin_policy_server.velocity_estimator_features import VelocityEstimatorFeatureBuilder
import numpy as np
import pytest


@pytest.mark.parametrize(
    'axis,angle,expected',
    [
        (1, 0.0, [0.0, 0.0, -1.0]),
        (1, math.pi / 6, [0.5, 0.0, -math.sqrt(3) / 2]),
        (1, -math.pi / 6, [-0.5, 0.0, -math.sqrt(3) / 2]),
        (0, math.pi / 6, [0.0, -0.5, -math.sqrt(3) / 2]),
    ],
)
def test_gravity_free_acceleration_conversion_and_axis_signs(axis, angle, expected):
    names = [f'joint_{i}' for i in range(12)]
    q = [0.0, 0.0, 0.0, math.cos(angle / 2)]
    q[axis] = math.sin(angle / 2)
    snapshot = SensorSnapshot.from_values(
        names, 1, 1, [0.0] * 24 + q + [0.1, 0.2, 0.3] + [0.0] * 3
    )
    native = VelocityEstimatorFeatureBuilder(names, np.zeros(12))
    converted = VelocityEstimatorFeatureBuilder(
        names, np.zeros(12), add_gravity_to_linear_acceleration=True
    )
    raw = native.build(snapshot.joint, snapshot.imu, np.zeros(12))
    corrected = converted.build(snapshot.joint, snapshot.imu, np.zeros(12))
    np.testing.assert_allclose(raw[:3], 0.0)
    np.testing.assert_allclose(raw[6:9], expected, atol=1e-12)
    np.testing.assert_allclose(corrected[:3], -9.81 * np.array(expected), atol=1e-12)
    np.testing.assert_array_equal(corrected[3:], raw[3:])
    np.testing.assert_allclose(corrected[3:6], [0.1, 0.2, 0.3])


def test_mg_divided_by_100_is_scaled_before_gravity_and_does_not_mutate_input():
    names = [f'joint_{i}' for i in range(12)]
    snapshot = SensorSnapshot.from_values(
        names, 1, 1, [0.0] * 24 + [0.0, 0.0, 0.0, 1.0] + [0.0] * 3 + [1.0, 2.0, -0.4]
    )
    builder = VelocityEstimatorFeatureBuilder(
        names, np.zeros(12), acceleration_scale=0.980665, add_gravity_to_linear_acceleration=True
    )
    feature = builder.build(snapshot.joint, snapshot.imu, np.zeros(12))
    np.testing.assert_allclose(feature[:3], [0.980665, 1.96133, 9.81 - 0.4 * 0.980665])
    assert snapshot.imu.linear_acceleration.z == -0.4


@pytest.mark.parametrize('scale', [0.0, -1.0, float('nan'), float('inf')])
def test_invalid_acceleration_scale_rejected(scale):
    with pytest.raises(ValueError, match='acceleration_scale'):
        VelocityEstimatorFeatureBuilder(
            [str(i) for i in range(12)], np.zeros(12), acceleration_scale=scale
        )
