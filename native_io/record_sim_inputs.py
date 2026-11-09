"""Record passive Isaac sensor input for deterministic artifact replay."""

import argparse
import json
import os
from pathlib import Path
import time


def main():
    if os.environ.get('ROS_DOMAIN_ID') != '10' or os.environ.get('ROS_LOCALHOST_ONLY') != '1':
        raise SystemExit('Recording requires localhost-only simulation domain 10')
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from marvin_policy_server.env_config_loader import EnvConfigLoader
    from marvin_policy_server.sensor_snapshot import SensorSnapshot
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Imu
    from sensor_msgs.msg import JointState

    names = EnvConfigLoader(args.env).get_joint_names()
    rclpy.init(args=[])
    node = rclpy.create_node('passive_sim_input_recorder')
    latest, frames = {}, []
    node.create_subscription(
        JointState,
        '/isaac_joint_states',
        lambda m: latest.update(joint=m),
        qos_profile_sensor_data,
    )
    node.create_subscription(
        Imu, '/isaac_imu', lambda m: latest.update(imu=m), qos_profile_sensor_data
    )
    previous = None
    try:
        deadline = time.monotonic() + 45.0
        while len(frames) < 256:
            assert time.monotonic() < deadline, 'Isaac sensor recording timed out'
            rclpy.spin_once(node, timeout_sec=0.02)
            if len(latest) != 2:
                continue
            sample = SensorSnapshot.from_messages(names, latest['joint'], latest['imu'])
            js, im = sample.joint, sample.imu
            joint_ns = js.header.stamp.sec * 1_000_000_000 + js.header.stamp.nanosec
            imu_ns = im.header.stamp.sec * 1_000_000_000 + im.header.stamp.nanosec
            if previous is not None and joint_ns - previous < 20_000_000:
                continue
            previous = joint_ns
            q, w, a = im.orientation, im.angular_velocity, im.linear_acceleration
            frames.append({
                'joint_ns': joint_ns,
                'imu_ns': imu_ns,
                'values': list(js.position + js.velocity) + [
                    q.x,
                    q.y,
                    q.z,
                    q.w,
                    w.x,
                    w.y,
                    w.z,
                    a.x,
                    a.y,
                    a.z,
                ],
            })
        with args.output.open('x') as output:
            json.dump(
                {
                    'schema': 'native-input-replay.v1',
                    'source': 'Isaac ROS sensors',
                    'joint_names': names,
                    'frames': frames,
                },
                output,
            )
        print(f'Recorded {len(frames)} passive Isaac frames to {args.output}')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
