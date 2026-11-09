#!/usr/bin/env python3
"""Bounded read-only snapshot check. Ages here are at inspection, NOT inference."""

import argparse
import json
import math
import time

from control_msgs.msg import DynamicJointState
from marvin_policy_server.encoder_snapshot import decode_snapshot
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topic', default='/dynamic_joint_states')
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument(
        '--same-host',
        action='store_true',
        required=True,
        help='Confirm driver and inspector share Linux host/boot',
    )
    parser.add_argument('--joints', nargs='+', required=True)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 60:
        parser.error('seconds must be in (0, 60]')
    rclpy.init()
    node = rclpy.create_node('encoder_snapshot_inspector')
    result = {
        'messages': 0,
        'valid_messages': 0,
        'invalid_messages': 0,
        'max_receipt_age_at_inspection_ms': 0.0,
        'joints': args.joints,
        'topic': args.topic,
    }

    def receive(message):
        result['messages'] += 1
        try:
            _, receipts = decode_snapshot(message, args.joints, JointState)
            now = time.monotonic()
            if any(
                not math.isfinite(stamp) or stamp <= 0 or stamp > now
                for stamp in receipts.values()
            ):
                raise ValueError('Missing/invalid monotonic receipt')
            oldest = max((now - stamp) * 1000 for stamp in receipts.values())
            result['max_receipt_age_at_inspection_ms'] = max(
                oldest, result['max_receipt_age_at_inspection_ms']
            )
            result['valid_messages'] += 1
        except ValueError as exc:
            result['invalid_messages'] += 1
            result['last_error'] = str(exc)

    subscription = node.create_subscription(
        DynamicJointState, args.topic, receive, qos_profile_sensor_data
    )
    try:
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=max(0.0, min(0.1, deadline - time.monotonic())))
    finally:
        node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()
        print(json.dumps(result, sort_keys=True))
    return 0 if result['valid_messages'] and not result['invalid_messages'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
