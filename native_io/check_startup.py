"""USB-C-safe installed-package lifecycle check on private ROS domain 184.

No sensor source, motor API, CAN interface or physical ROS domain is used.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    if os.environ.get('ROS_DOMAIN_ID') != '184' or os.environ.get('ROS_LOCALHOST_ONLY') != '1':
        raise SystemExit('Startup check requires localhost-only domain 184')
    if os.environ.get('MARVIN_POLICY_NATIVE_PROFILE', 'test') != 'test':
        raise SystemExit('Startup check requires the test profile')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--env', required=True, type=Path)
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--log-dir', required=True, type=Path)
    args = parser.parse_args()
    args.log_dir.mkdir(exist_ok=False)
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from std_srvs.srv import SetBool, Trigger
    from marvin_trace_msgs.msg import TracedFloat64MultiArray
    from std_msgs.msg import Float64MultiArray
    rclpy.init(args=[])
    observer = rclpy.create_node('private_policy_startup_observer')
    commands = []
    observer.create_subscription(TracedFloat64MultiArray, '/policy_probe/commands_traced',
                                 commands.append, qos_profile_sensor_data)
    observer.create_subscription(Float64MultiArray, '/policy_probe/commands',
                                 commands.append, qos_profile_sensor_data)
    options = {
        'policy_path': str(args.model.resolve()), 'env_path': str(args.env.resolve()),
        'native_io_executable': str(args.binary.resolve()), 'compact_estimator': 'true',
        'lean_qos_events': 'true', 'use_sim_time': 'false', 'tracing_mode': '"off"',
        'operator_input_required': 'true', 'joint_states_topic': '/policy_probe/joints',
        'imu_topic': '/policy_probe/imu', 'command_topic': '/policy_probe/commands',
        'traced_command_topic': '/policy_probe/commands_traced',
        'trace_event_topic': '/policy_probe/events',
    }
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
              for name, path in [('policy.pt', args.model), ('env.yaml', args.env),
                                 ('metadata.json', args.model.parent / 'metadata.json')]}
    results = []
    try:
        for cycle, backend in enumerate(('native', 'native', 'python')):
            # Fresh service names prevent stale discovery from a prior cycle.
            ns = f'/policy_probe/startup_{cycle}'
            module = 'native_server' if backend == 'native' else 'marvin_policy_server'
            command = [sys.executable, '-m', 'marvin_policy_server.' + module,
                       '--ros-args', '-r', f'__ns:={ns}',
                       '-r', '/operator/motion_permitted:=/policy_probe/permission',
                       '-r', '/trace/metrics:=/policy_probe/metrics',
                '-r', 'execution/policy/commands:=/policy_probe/execution/policy/commands']
            for name, value in dict(options, async_inference=str(backend == 'python').lower()).items():
                command += ['-p', f'{name}:={value}']
            active = observer.create_client(SetBool, ns + '/set_active')
            status = observer.create_client(Trigger, ns + '/get_policy_observations')
            child = None
            with (args.log_dir / f'{cycle}-{backend}.log').open('x') as log:
                process = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
                try:
                    def call(client, request):
                        deadline = time.monotonic() + 30.
                        while not client.wait_for_service(timeout_sec=.2):
                            assert process.poll() is None, f'Policy startup failed. Read {log.name}'
                            assert time.monotonic() < deadline, 'Management service unavailable'
                        future = client.call_async(request)
                        rclpy.spin_until_future_complete(observer, future, timeout_sec=5.)
                        assert future.done() and future.result() is not None, 'Management request timed out'
                        return future.result()

                    result = call(status, Trigger.Request())
                    assert result.success, result.message
                    payload = json.loads(result.message)
                    for name, digest in hashes.items():
                        assert payload['loaded_artifacts'][name]['sha256'] == digest, 'Incorrect loaded model/env'
                    if backend == 'native':
                        execution = payload['execution']
                        assert execution['io_backend'] == 'native' and execution['profile'] == 'test'
                        assert not execution['native_active'] and not execution['sensors_valid']
                        children = Path(f'/proc/{process.pid}/task/{process.pid}/children').read_text().split()
                        assert len(children) == 1, f'Unexpected native children: {children}'
                        child = int(children[0])
                        rejected = call(active, SetBool.Request(data=True))
                        assert not rejected.success and 'sensor' in rejected.message.lower(), rejected.message
                    stopped = call(active, SetBool.Request(data=False))
                    assert stopped.success, stopped.message
                    deadline = time.monotonic() + 1.
                    while time.monotonic() < deadline:
                        rclpy.spin_once(observer, timeout_sec=.02)
                        assert process.poll() is None, 'Policy exited unexpectedly'
                    assert not commands, 'Inactive policy published a command'
                    process.send_signal(signal.SIGINT)
                    assert process.wait(timeout=10.) == 0, 'Policy shutdown did not finish cleanly'
                    if child is not None:
                        assert not Path(f'/proc/{child}').exists(), 'Native helper survived graceful shutdown'
                    results.append({'cycle': cycle, 'backend': backend, 'model_hashes': hashes,
                                    'commands': len(commands), 'clean_shutdown': True})
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGINT)
                        try:
                            process.wait(timeout=10.)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait(timeout=5.)
                    observer.destroy_client(active)
                    observer.destroy_client(status)
        print('USB_STARTUP_CHECK ' + json.dumps(results))
    finally:
        observer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
