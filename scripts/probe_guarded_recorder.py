"""Synthetic 200 Hz capture in isolated domain 231. No control services or CAN."""
import argparse
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=12)
    parser.add_argument('--stop-signal', choices=['INT', 'TERM'])
    args = parser.parse_args()
    if os.environ.get('ROS_DOMAIN_ID') != '231' or os.environ.get('ROS_LOCALHOST_ONLY') != '1':
        parser.error('Requires localhost-only domain 231')
    if not 5 <= args.seconds <= 60:
        parser.error('Probe duration must be 5 to 60 seconds')
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState
    from runtime_interfaces.msg import JointTestTargetSample
    rclpy.init()
    node = rclpy.create_node('synthetic_capture_probe')
    qos = QoSProfile(depth=64, reliability=ReliabilityPolicy.BEST_EFFORT)
    targets = node.create_publisher(JointTestTargetSample, '/capture_probe/joint_test/target_samples',
                                   QoSProfile(depth=64, reliability=ReliabilityPolicy.RELIABLE))
    feedback = node.create_publisher(JointState, '/capture_probe/feedback', qos)
    joints = ['test_joint_'+str(i) for i in range(12)]
    child = subprocess.Popen([sys.executable, str(Path(__file__).with_name('record_joint_test_guarded.py')),
        '--controller', '/capture_probe', '--feedback', '/capture_probe/feedback',
        '--instance', 'synthetic-probe', '--activation', '1', '--joints', *joints,
        '--seconds', str(args.seconds), '--output', str(args.output)])
    sequence = 0
    publish_after = None
    def tick():
        nonlocal sequence, publish_after
        if publish_after is None:
            if (targets.get_subscription_count() and feedback.get_subscription_count() and
                    (args.output/'health.json').exists()):
                publish_after = time.monotonic()+1
            return
        if time.monotonic() < publish_after:
            return
        sequence += 1
        stamp = node.get_clock().now().to_msg()
        target = JointTestTargetSample(controller_instance_id='synthetic-probe', activation_generation=1,
            update_sequence=sequence, sampled_at_monotonic_s=time.monotonic(),
            stamp_sec=stamp.sec, stamp_nanosec=stamp.nanosec,
            joint_names=joints, positions=[.1]*12)
        targets.publish(target)
        state = JointState(name=joints, position=[.1]*12, velocity=[0.]*12, effort=[float('nan')]*12)
        state.header.stamp = stamp
        feedback.publish(state)
    node.create_timer(.005, tick)
    seen = set()
    ready = unhealthy = 0
    unhealthy_details = []
    max_sync = max_queue = 0
    deadline = time.monotonic()+args.seconds+8
    stop_sent = None
    try:
        while child.poll() is None and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.01)
            if args.stop_signal and stop_sent is None and ready >= 30:
                stop_sent = time.monotonic()
                child.send_signal(getattr(signal, 'SIG'+args.stop_signal))
            try:
                health = json.loads((args.output/'health.json').read_text())
                key = health['sampled_at_monotonic_s']
                if key not in seen:
                    seen.add(key)
                    if health['targets'] >= 2:
                        ready += bool(health['ready'])
                        unhealthy += not health['ready']
                        if not health['ready']:
                            durable = health['durable']
                            unhealthy_details.append(dict(running=health['running'], targets=health['targets'],
                                committed=durable is not None,
                                durable_target_age_ms=(1000*(key-durable['last_target'])
                                    if durable and durable['last_target'] is not None else None)))
                    max_queue = max(max_queue, health['writer_queue'])
                    max_sync = max(max_sync, (health['durable'] or {}).get('sync_duration_ms', 0))
            except (FileNotFoundError, json.JSONDecodeError):
                pass
        if child.poll() is None:
            raise RuntimeError('Synthetic capture timeout')
        final = json.loads((args.output/'health.json').read_text())
        print(json.dumps(dict(exit_code=child.returncode, targets=final['targets'], feedback=final['feedback'],
            missing=final['missing'], fault=final['fault'], healthy_checks=ready,
            unhealthy_checks=unhealthy, unhealthy_details=unhealthy_details,
            max_sync_ms=max_sync, max_writer_queue=max_queue)))
        if child.returncode or final['fault'] or final['missing'] or not ready:
            raise RuntimeError('Synthetic recorder validation failed')
        assert final['finalized'] and not final['running'] and final['writer_queue'] == 0
        assert final['durable']['targets'] == final['targets']
        assert final['durable']['feedback'] == final['feedback']
        if args.stop_signal:
            assert stop_sent is not None and time.monotonic()-stop_sent < 8
            print(json.dumps(dict(stop_signal=args.stop_signal,
                shutdown_ms=1000*(time.monotonic()-stop_sent), finalized=True)))
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=6)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
