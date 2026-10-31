"""Deterministic same-artifact replay. No sensor or command ROS endpoints."""

import argparse
from copy import deepcopy
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    if os.environ.get('ROS_DOMAIN_ID') != '184' or os.environ.get('ROS_LOCALHOST_ONLY') != '1':
        raise SystemExit('Replay requires isolated localhost domain 184')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--env', required=True, type=Path)
    parser.add_argument('--mapping-binary', required=True, type=Path)
    parser.add_argument('--capture', type=Path)
    args = parser.parse_args()
    import numpy as np
    import rclpy
    import torch
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import JointState, Imu
    from marvin_policy_server.marvin_policy_server import MarvinPolicyServer
    from marvin_policy_server.concurrent_policy.runner import ConcurrentPolicyRunner
    from marvin_policy_server.inference_inputs import runner_inputs
    from marvin_policy_server.native_protocol import Packet, Kind
    from marvin_policy_server.sensor_snapshot import SensorSnapshot
    from marvin_policy_server.action_output_mapping import SmoothBoundedActionMapper

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    options = {'policy_path': str(args.model.resolve()), 'env_path': str(args.env.resolve()),
               'compact_estimator': 'true', 'async_inference': 'false',
               'lean_qos_events': 'true', 'tracing_mode': '"off"'}
    ros_args = ['--ros-args']
    for key, value in options.items(): ros_args += ['-p', f'{key}:={value}']
    rclpy.init(args=ros_args)
    node = None
    try:
        node = MarvinPolicyServer(native_io=True)
        assert node._joint_publisher is node._joint_publisher_traced is None
        assert node._joint_state_sub is node._imu_sub is node._timer is None
        assert node._concurrent_policy_manifest is not None
        runners = [ConcurrentPolicyRunner(node.policy, node._concurrent_policy_manifest, decimation=4)
                   for _ in range(2)]
        states = [deepcopy(node._obs_state) for _ in range(2)]
        previous = [np.zeros(12), np.zeros(12)]
        rng = np.random.default_rng(20261011)
        digest = hashlib.sha256()
        mapping_lines, expected_mapping = [], []
        count = 256
        capture = json.loads(args.capture.read_text()) if args.capture else None
        if capture:
            assert capture['schema'] == 'native-input-replay.v1'
            assert capture['joint_names'] == list(node.joint_names)
            assert len(capture['frames']) == count
        for frame in range(count):
            if frame == 160:
                for runner in runners: runner.reset()
                states = [deepcopy(node._obs_state) for _ in range(2)]
                previous = [np.zeros(12), np.zeros(12)]
            positions = node.default_pos + rng.normal(0., .08, 12)
            velocities = rng.normal(0., .3, 12)
            quat = np.r_[rng.normal(0., .04, 3), 1.]
            quat /= np.linalg.norm(quat)
            gyro, acc = rng.normal(0., .15, 3), rng.normal(0., .2, 3) + [0., 0., 9.81]
            data = tuple(np.r_[positions, velocities, quat, gyro, acc])
            stamp = 1_000_000_000 + frame * 20_000_000
            imu_stamp = stamp
            if capture:
                record = capture['frames'][frame]
                data = tuple(record['values'])
                positions, velocities = np.asarray(data[:12]), np.asarray(data[12:24])
                quat, gyro, acc = data[24:28], data[28:31], data[31:34]
                stamp, imu_stamp = record['joint_ns'], record['imu_ns']
            packet = Packet(Kind.SNAPSHOT, sequence=frame + 1, generation=1,
                            ros_ns=stamp, joint_ns=stamp, imu_ns=imu_stamp, values=data)
            encoded = packet.encode()
            digest.update(encoded)
            decoded = Packet.decode(encoded)
            snapshot = SensorSnapshot.from_values(node.joint_names, decoded.joint_ns,
                                                  decoded.imu_ns, decoded.values)
            # The ROS branch has a different transport ordering.
            order = rng.permutation(12)
            joint = JointState(name=[node.joint_names[i] for i in order],
                               position=positions[order].tolist(), velocity=velocities[order].tolist())
            imu = Imu()
            joint.header.stamp.sec, joint.header.stamp.nanosec = divmod(stamp, 1_000_000_000)
            imu.header.stamp.sec, imu.header.stamp.nanosec = divmod(imu_stamp, 1_000_000_000)
            for names, values, obj in [('xyzw', quat, imu.orientation),
                                       ('xyz', gyro, imu.angular_velocity),
                                       ('xyz', acc, imu.linear_acceleration)]:
                for name, value in zip(names, values): setattr(obj, name, float(value))
            command = Twist()
            command.linear.x, command.linear.y, command.angular.z = .2, -.03, .1
            locked = 80 <= frame < 160
            obs_kwargs = {'base_lin_vel_override': np.zeros(3)}
            if locked: obs_kwargs['command_override'] = np.zeros(3)
            model_overrides = {'estimated_velocity_override': np.array([.1, 0., 0.])} if frame >= 120 else {}
            observations, features, actions = [], [], []
            for branch, (js, inertial) in enumerate(((joint, imu), (snapshot.joint, snapshot.imu))):
                obs = node._obs_builder.build(js, inertial, command, .02, states[branch], **obs_kwargs)
                inputs, kwargs = runner_inputs(obs, feature_builder=node._concurrent_policy_feature_builder,
                                               joint_state=js, imu=inertial,
                                               previous_action=previous[branch], **model_overrides)
                action = np.clip(runners[branch].step(*inputs, **kwargs), -10., 10.)
                observations.append(obs)
                features.append(inputs[1])
                actions.append(action.copy())
                previous[branch] = action.copy()
                history = states[branch].action_history
                if len(history):
                    history[:-1] = history[1:]
                    history[-1] = action
            for pair in (observations, features, actions,
                         [runner.history.values for runner in runners],
                         [runner.latest_estimated_state for runner in runners]):
                np.testing.assert_array_equal(pair[0], pair[1])
            for field in fields(states[0]):
                np.testing.assert_array_equal(getattr(states[0], field.name), getattr(states[1], field.name))
            for smooth in (False, True):
                factor = (frame % 32) / 31.
                values = [int(smooth), node._action_scale, factor]
                for row in zip(node.default_pos, node._safety_min_pos, node._safety_max_pos, actions[1]):
                    values.extend(row)
                mapping_lines.append(' '.join(map(str, values)))
                mapped = (SmoothBoundedActionMapper(node.default_pos, node._safety_min_pos,
                          node._safety_max_pos, local_action_scale=node._action_scale).map_action(actions[0] * factor)
                          if smooth else node.default_pos + actions[0] * factor * node._action_scale)
                expected_mapping.append(np.clip(mapped, node._safety_min_pos, node._safety_max_pos))
        result = subprocess.run([str(args.mapping_binary.resolve()), '--map'],
                                input='\n'.join(mapping_lines) + '\n', text=True, capture_output=True, check=True)
        mapped = np.asarray([[float(v) for v in line.split()] for line in result.stdout.splitlines()])
        np.testing.assert_allclose(mapped, expected_mapping, rtol=1e-13, atol=1e-13)
        print('NATIVE_ARTIFACT_REPLAY ' + json.dumps({
            'frames': count, 'transport_sha256': digest.hexdigest(),
            'input_source': 'recorded Isaac' if capture else 'synthetic',
            'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
            'exact_observations_features_histories_actions': True,
            'cpp_mapping_cases': len(mapped),
            'cpp_mapping_max_abs_error': float(np.max(np.abs(mapped - expected_mapping))),
            'includes_command_override_estimator_override_and_reset': True}))
    finally:
        if node is not None:
            node._close_velocity_estimator_shadow()
            node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()


if __name__ == '__main__':
    main()
