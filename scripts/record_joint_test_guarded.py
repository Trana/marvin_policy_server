#!/usr/bin/env python3
"""Persist local joint-test samples and expose recording health. Never sends commands."""

import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import signal
import tempfile
import time
import uuid

from joint_capture_writer import CaptureWriter
from joint_capture_writer import durable_ready
from joint_test_capture_guard import CaptureGuard

# Storage allowance only. SD-card sync stalls must not block live callbacks.
# 1024 records at two 200 Hz streams cover 2.56 s; overflow still fails closed.
# Live source/receipt freshness stays at 150 ms, independent of this allowance.
MAX_DISK_LAG_S = 2.0


class LoopTiming:
    """Bounded diagnostics, without logging or disk writes in callbacks."""

    def __init__(self):
        self.maximum_ms = {}
        self.slow = deque(maxlen=32)

    def observe(self, phase, started, finished):
        elapsed = (finished - started) * 1000
        self.maximum_ms[phase] = max(self.maximum_ms.get(phase, 0.0), elapsed)
        if elapsed >= 25:
            self.slow.append(
                {'phase': phase, 'started_monotonic_s': started, 'elapsed_ms': elapsed}
            )

    def snapshot(self):
        return {'maximum_ms': dict(self.maximum_ms), 'slow': list(self.slow)}


class LiveHealthFile:
    """Keep live health off the capture disk, then replace its link at shutdown."""

    def __init__(self, output, ram_root='/dev/shm'):
        self.output = Path(output) / 'health.json'
        self.ram = tempfile.TemporaryDirectory(prefix='joint-capture-health-', dir=ram_root)
        self.live = Path(self.ram.name) / 'health.json'
        try:
            self.output.symlink_to(self.live)
        except BaseException:
            self.ram.cleanup()
            raise

    def write(self, value):
        destination = self.output if value.get('finalized') else self.live
        temporary = destination.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, allow_nan=False))
        # Replaces the link itself on finalization, never follows it for writes.
        temporary.replace(destination)

    def close(self):
        self.ram.cleanup()


def optional_feedback_fields(sample):
    """Keep missing effort/velocity as null, never alter required positions."""
    return {
        key: (
            [v if math.isfinite(v) else None for v in value]
            if key in ('velocity', 'effort')
            else value
        )
        for key, value in sample.items()
    }


def sample_fields(kind, msg):
    """Copy the fixed recording schema without recursive ROS type inspection."""
    if kind == 'target':
        return {
            'controller_instance_id': msg.controller_instance_id,
            'activation_generation': msg.activation_generation,
            'update_sequence': msg.update_sequence,
            'sampled_at_monotonic_s': msg.sampled_at_monotonic_s,
            'stamp_sec': msg.stamp_sec,
            'stamp_nanosec': msg.stamp_nanosec,
            'joint_names': list(msg.joint_names),
            'positions': list(msg.positions),
        }
    if kind != 'feedback':
        raise ValueError('Unknown sample kind')
    return optional_feedback_fields({
        'header': {
            'stamp': {'sec': msg.header.stamp.sec, 'nanosec': msg.header.stamp.nanosec},
            'frame_id': msg.header.frame_id,
        },
        'name': list(msg.name),
        'position': list(msg.position),
        'velocity': list(msg.velocity),
        'effort': list(msg.effort),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller', required=True)
    parser.add_argument('--feedback', required=True)
    parser.add_argument('--instance', required=True)
    parser.add_argument('--activation', required=True, type=int)
    parser.add_argument('--joints', required=True, nargs='+')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seconds', type=float, default=60.0)
    args = parser.parse_args()
    if os.environ.get('ROS_LOCALHOST_ONLY') != '1' or os.environ.get('ROS_DOMAIN_ID') not in (
        '0',
        '231',
    ):
        parser.error('Requires explicit local robot domain 0 or isolated simulation domain 231')
    if (
        not 0 < args.seconds <= 600
        or not args.controller.startswith('/')
        or not args.feedback.startswith('/')
    ):
        parser.error('Invalid endpoint or capture duration')
    guard = CaptureGuard(args.instance, args.activation, args.joints)
    args.output.mkdir(parents=True, exist_ok=False)
    import rclpy
    from rclpy.qos import DurabilityPolicy
    from rclpy.qos import QoSProfile
    from rclpy.qos import ReliabilityPolicy
    from rclpy.signals import SignalHandlerOptions
    from runtime_interfaces.msg import JointTestTargetSample
    from sensor_msgs.msg import JointState

    capture_id = str(uuid.uuid4())
    # Signals request loop exit only. Do not tear down ROS or interrupt a disk
    # progress lock inside a signal handler.
    stopping = False

    def request_stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    node = rclpy.create_node('guarded_joint_test_recorder')
    start = time.monotonic()
    metadata = {
        'schema': 'guarded-joint-capture.v1',
        'capture_id': capture_id,
        'instance': args.instance,
        'activation': args.activation,
        'joints': args.joints,
        'controller': args.controller,
        'feedback': args.feedback,
        'domain': os.environ['ROS_DOMAIN_ID'],
        'clock': (
            'monotonic timestamps share the local host kernel; ROS stamps retain source clock'
        ),
    }
    metadata['optional_feedback_nonfinite'] = 'velocity/effort nonfinite values are null, not zero'
    metadata['receive_depth'] = 64
    metadata['target_reliability'] = 'reliable'
    metadata['feedback_reliability'] = 'best_effort'
    metadata['writer_capacity'] = 1024
    metadata['max_live_age_s'] = guard.max_age
    metadata['max_disk_lag_s'] = MAX_DISK_LAG_S
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    health_file = LiveHealthFile(args.output)
    writer = CaptureWriter(args.output / 'samples.jsonl')
    timing = LoopTiming()
    qos = QoSProfile(
        depth=64, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE
    )

    def health(running, finalized=False):
        now = time.monotonic()
        progress = writer.progress()
        live_ready = guard.live_ready(now)
        disk_ready = durable_ready(progress, now, MAX_DISK_LAG_S)
        value = {
            'schema': 'joint-capture-health.v1',
            'capture_id': capture_id,
            'instance': args.instance,
            'activation': args.activation,
            'sampled_at_monotonic_s': now,
            'ready': running and not writer.error and live_ready and disk_ready,
            'live_ready': live_ready,
            'disk_ready': disk_ready,
            'max_live_age_s': guard.max_age,
            'max_disk_lag_s': MAX_DISK_LAG_S,
            'running': running,
            'finalized': finalized,
            'fault': guard.fault or writer.error,
            'targets': guard.target_count,
            'feedback': guard.feedback_count,
            'missing': guard.missing,
            'target_positions': guard.positions,
            'last_target_source': guard.last_source,
            'feedback_positions': guard.feedback_positions,
            'durable': progress,
            'writer_queue': writer.queue.qsize(),
            'loop_timing': timing.snapshot(),
        }
        started = time.monotonic()
        health_file.write(value)
        timing.observe('health_write', started, time.monotonic())

    try:

        def record(kind, msg):
            now = time.monotonic()
            sample = sample_fields(kind, msg)
            line = (
                json.dumps(
                    {
                        'kind': kind,
                        'received_monotonic_s': now,
                        'received_unix_ns': time.time_ns(),
                        'sample': sample,
                    },
                    allow_nan=False,
                )
                + '\n'
            )
            if kind == 'target':
                guard.target(msg, now)
            else:
                guard.feedback(msg.name, msg.position, now)
            writer.submit(
                line,
                {
                    'last_target': guard.last_target,
                    'last_feedback': guard.last_feedback,
                    'targets': guard.target_count,
                    'feedback': guard.feedback_count,
                },
            )
            timing.observe('record_' + kind, now, time.monotonic())

        node.create_subscription(
            JointTestTargetSample,
            args.controller + '/joint_test/target_samples',
            lambda msg: record('target', msg),
            QoSProfile(
                depth=64,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            ),
        )
        node.create_subscription(
            JointState, args.feedback, lambda msg: record('feedback', msg), qos
        )
        next_health = start
        while not stopping and time.monotonic() - start < args.seconds:
            spin_started = time.monotonic()
            rclpy.spin_once(node, timeout_sec=0.005)
            timing.observe('spin_once', spin_started, time.monotonic())
            if time.monotonic() >= next_health:
                health(True)
                next_health = time.monotonic() + 0.05
            if guard.fault or writer.error:
                raise RuntimeError(guard.fault or writer.error)
    except KeyboardInterrupt:
        pass
    except BaseException:
        guard.reject('RECORDER_FAILED')
        raise
    finally:
        try:
            try:
                health(False)
            finally:
                writer.close()
            health(False, finalized=True)
        finally:
            try:
                health_file.close()
            finally:
                try:
                    node.destroy_node()
                finally:
                    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
