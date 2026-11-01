"""Isolated constructor/service check. No native process or command publisher."""

import argparse
import json
import os
from pathlib import Path


def main():
    if os.environ.get('ROS_DOMAIN_ID') != '184' or os.environ.get('ROS_LOCALHOST_ONLY') != '1':
        raise SystemExit('Ownership check requires private domain 184 and localhost-only discovery')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--env', required=True, type=Path)
    args = parser.parse_args()
    import rclpy
    import torch
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from std_srvs.srv import Trigger
    from marvin_policy_server.marvin_policy_server import MarvinPolicyServer
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    options = {
        'policy_path': str(args.model.resolve()), 'env_path': str(args.env.resolve()),
        'compact_estimator': 'true', 'async_inference': 'false',
        'lean_qos_events': 'true', 'tracing_mode': '"off"',
        'publish_trace_events': 'false',
    }
    ros_args = ['--ros-args']
    for name, value in options.items():
        ros_args += ['-p', f'{name}:={value}']
    rclpy.init(args=ros_args)
    node = client_node = executor = None
    try:
        node = MarvinPolicyServer(native_io=True)
        assert node._timer is None
        assert node._joint_state_sub is node._imu_sub is None
        assert node._joint_publisher is node._joint_publisher_traced is None
        assert node._async_worker is None
        assert not node._activation_mgr.is_active()
        client_node = Node('native_ownership_check')
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        executor.add_node(client_node)
        client = client_node.create_client(Trigger, '/get_policy_observations')
        assert client.wait_for_service(timeout_sec=5.), 'Management service not discovered'
        future = client.call_async(Trigger.Request())
        executor.spin_until_future_complete(future, timeout_sec=5.)
        assert future.done() and future.result().success, 'Management request did not complete'
        payload = json.loads(future.result().message)
        assert payload['observations'], 'Observation API returned no fields'
        print('NATIVE_OWNERSHIP_AND_MANAGEMENT passed')
    finally:
        if executor is not None:
            executor.shutdown()
        if client_node is not None:
            client_node.destroy_node()
        if node is not None:
            node._close_velocity_estimator_shadow()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
