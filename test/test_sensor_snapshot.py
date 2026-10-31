"""Replay identical sensor values through ROS and ROS-independent builders."""

from dataclasses import FrozenInstanceError, fields
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from marvin_policy_server.inference_inputs import runner_inputs
from marvin_policy_server.observation import ObservationBuilder, ObservationConfig, ObservationState
from marvin_policy_server.sensor_snapshot import SensorSnapshot
from marvin_policy_server.velocity_estimator_features import VelocityEstimatorFeatureBuilder


NAMES = tuple(f'joint_{i}' for i in range(12))


def values():
    return list(np.linspace(-.3, .7, 12)) + list(np.linspace(-1., 1., 12)) + [
        .1, -.2, .05, .97, .2, -.3, .4, .5, -.6, 9.5]


def state():
    return ObservationState(
        lin_vel_b=np.zeros(3), action_history=np.zeros((2, 12)),
        default_pos=np.linspace(-.2, .2, 12),
        **{name: np.full((2, width), np.nan) for name, width in (
            ('base_lin_vel_history', 3), ('base_ang_vel_history', 3),
            ('projected_gravity_history', 3), ('velocity_commands_history', 3),
            ('joint_pos_history', 12), ('joint_vel_history', 12))})


def test_snapshot_copies_and_validates_values():
    data = values()
    snapshot = SensorSnapshot.from_values(NAMES, 1234567890, 2000000001, data)
    data[0] = 999
    assert snapshot.joint.position[0] == -.3
    assert snapshot.joint.header.stamp.sec == 1
    assert snapshot.joint.header.stamp.nanosec == 234567890
    assert snapshot.imu.header.stamp.nanosec == 1
    assert snapshot.imu.orientation.w == .97
    with pytest.raises(FrozenInstanceError):
        snapshot.imu.orientation.w = 0


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
@pytest.mark.parametrize('index', [0, 12, 24, 28, 31])
def test_nonfinite_input_is_rejected(bad, index):
    data = values()
    data[index] = bad
    with pytest.raises(ValueError, match='non-finite'):
        SensorSnapshot.from_values(NAMES, 1, 1, data)


@pytest.mark.parametrize('names,js,imu,data', [
    (NAMES, -1, 1, values()), (NAMES, 1, -1, values()),
    (NAMES, 1.5, 1, values()), (NAMES, True, 1, values()),
    (NAMES, 1, 1, values()[:-1]), (NAMES, 1, 1, values() + [0]),
    ((NAMES[0],) * 12, 1, 1, values()),
    (('',) + NAMES[1:], 1, 1, values()), ((), 1, 1, values()),
])
def test_invalid_snapshot_is_rejected(names, js, imu, data):
    with pytest.raises(ValueError):
        SensorSnapshot.from_values(names, js, imu, data)


@pytest.mark.parametrize('locked', [False, True])
@pytest.mark.parametrize('add_gravity', [False, True])
def test_ros_and_snapshot_observations_features_and_histories_match(locked, add_gravity):
    sensor_msgs = pytest.importorskip('sensor_msgs.msg')
    geometry_msgs = pytest.importorskip('geometry_msgs.msg')
    joint, imu, command = sensor_msgs.JointState(), sensor_msgs.Imu(), geometry_msgs.Twist()
    joint.name = list(reversed(NAMES))  # Transport order must not affect policy order.
    command.linear.x, command.linear.y, command.angular.z = .4, -.1, .2
    original, isolated = state(), state()
    builder = ObservationBuilder(NAMES, ObservationConfig(2, 2, 2, 2, 2, 2, 2))
    features = VelocityEstimatorFeatureBuilder(
        NAMES, original.default_pos, add_gravity_to_linear_acceleration=add_gravity)
    overrides = {} if not locked else {
        'command_override': np.array([.2, .1, -.3]),
        'base_lin_vel_override': np.array([.1, -.2, .3]),
        'imu_override': {'base_lin_vel': np.zeros(3), 'base_ang_vel': np.ones(3),
                         'projected_gravity': np.array([0., 0., -1.])},
        'joint_position_relative_override': np.linspace(-.1, .1, 12),
        'joint_velocity_override': np.linspace(-.5, .5, 12),
    }
    for frame in range(20):
        data = values()
        data[0] += frame * .003
        data[12] -= frame * .01
        joint.position = list(reversed(data[:12]))
        joint.velocity = list(reversed(data[12:24]))
        for name, value in zip(('x', 'y', 'z', 'w'), data[24:28]):
            setattr(imu.orientation, name, value)
        for name, value in zip(('x', 'y', 'z'), data[28:31]):
            setattr(imu.angular_velocity, name, value)
        for name, value in zip(('x', 'y', 'z'), data[31:34]):
            setattr(imu.linear_acceleration, name, value)
        joint.header.stamp.sec = imu.header.stamp.sec = 1
        joint.header.stamp.nanosec = imu.header.stamp.nanosec = frame * 20_000_000
        snapshot = SensorSnapshot.from_messages(NAMES, joint, imu)
        expected = builder.build(joint, imu, command, .02, original, **overrides)
        actual = builder.build(snapshot.joint, snapshot.imu, command, .02, isolated, **overrides)
        np.testing.assert_array_equal(actual, expected)
        previous_action = np.linspace(-1., 1., 12) * frame
        filtered = overrides.get('joint_position_relative_override')
        np.testing.assert_array_equal(
            features.build(snapshot.joint, snapshot.imu, previous_action, filtered),
            features.build(joint, imu, previous_action, filtered))
        for field in fields(original):
            np.testing.assert_array_equal(getattr(original, field.name), getattr(isolated, field.name))
        original.action_history[-1] = isolated.action_history[-1] = previous_action


def test_message_missing_joint_is_not_silently_zeroed():
    snapshot = SensorSnapshot.from_values(NAMES, 1, 1, values())
    with pytest.raises(ValueError, match='missing'):
        SensorSnapshot.from_messages(NAMES + ('absent',), snapshot.joint, snapshot.imu)


def test_runner_request_preserves_overrides_and_feature_inputs():
    from types import SimpleNamespace
    observation, action = np.arange(108), np.arange(12)
    overrides = dict(estimated_velocity_override=np.ones(3),
                     estimated_foot_positions_override=np.ones(12),
                     estimated_foot_height_override=np.ones(4),
                     estimated_contact_override=np.zeros(4))
    calls = []
    def build(*args, **kwargs):
        calls.append((args, kwargs))
        return np.arange(45)
    args, kwargs = runner_inputs(
        observation, feature_builder=SimpleNamespace(build=build),
        joint_state='joint', imu='imu', previous_action=action,
        filtered_joint_position='filtered', **overrides)
    assert args[0] is observation
    np.testing.assert_array_equal(args[1], np.arange(45))
    assert kwargs.keys() == {'infer', *overrides}
    assert kwargs['infer'] is True
    for name, value in overrides.items():
        assert kwargs[name] is value
    assert calls[0][0][:2] == ('joint', 'imu')
    assert calls[0][0][2] is action
    assert calls[0][1] == {'joint_position_relative_override': 'filtered'}
    args, kwargs = runner_inputs(observation)
    assert args == (observation,)
    assert kwargs == {'infer': True}


def test_computation_imports_without_ros():
    source = '''
import sys
class BlockRos:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'rclpy', 'sensor_msgs', 'geometry_msgs'}:
            raise RuntimeError('Computation imported ROS: ' + fullname)
sys.meta_path.insert(0, BlockRos())
from marvin_policy_server.observation import ObservationBuilder
from marvin_policy_server.velocity_estimator_features import VelocityEstimatorFeatureBuilder
from marvin_policy_server.sensor_snapshot import SensorSnapshot
from marvin_policy_server.inference_inputs import runner_inputs
'''
    result = subprocess.run([sys.executable, '-c', source],
                            cwd=Path(__file__).parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
