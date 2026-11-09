#!/usr/bin/env python3
"""Compare stamped targets and joint feedback in a completed SQLite bag, read-only."""

import argparse
from bisect import bisect_right
import json
import math
from pathlib import Path
import sqlite3


def summarize(commands, feedback, names, start_ns, end_ns, max_command_age_ns):
    """
    Rows: (ROS stamp, values) and (ROS stamp, names, positions).

    Use the latest preceding target, without interpolation or fitted time shifts.
    Require ordered unique stamps to avoid silently merging clock-reset epochs.
    """
    if not names or len(set(names)) != len(names):
        raise ValueError('Command joint names must be nonempty and unique')
    if not 0 < start_ns < end_ns or max_command_age_ns <= 0:
        raise ValueError('Require a positive time window and maximum command age')
    for rows in (commands, feedback):
        stamps = [row[0] for row in rows]
        if any(stamp <= 0 for stamp in stamps) or any(b <= a for a, b in zip(stamps, stamps[1:])):
            raise ValueError(
                'Nonpositive, duplicate or regressing ROS stamps; select a clean clock epoch'
            )
    for _, values in commands:
        if len(values) != len(names) or not all(math.isfinite(v) for v in values):
            raise ValueError('Invalid command shape or nonfinite command value')
    stamps = [row[0] for row in commands]
    errors = {name: [] for name in names}
    ages = []
    counts = {
        'in_window': 0,
        'matched': 0,
        'no_preceding_command': 0,
        'stale_command': 0,
        'invalid_feedback': 0,
    }
    for stamp, observed_names, positions in feedback:
        if not start_ns <= stamp < end_ns:
            continue
        counts['in_window'] += 1
        if (
            len(observed_names) != len(positions)
            or len(set(observed_names)) != len(observed_names)
            or not set(names).issubset(observed_names)
            or not all(math.isfinite(v) for v in positions)
        ):
            counts['invalid_feedback'] += 1
            continue
        index = bisect_right(stamps, stamp) - 1
        if index < 0:
            counts['no_preceding_command'] += 1
            continue
        age = stamp - stamps[index]
        if age > max_command_age_ns:
            counts['stale_command'] += 1
            continue
        measured = dict(zip(observed_names, positions))
        for name, target in zip(names, commands[index][1]):
            errors[name].append(target - measured[name])
        counts['matched'] += 1
        ages.append(age / 1e6)
    joints = {}
    for name, values in errors.items():
        joints[name] = {
            'samples': len(values),
            'mean_target_minus_measured_rad': sum(values) / len(values) if values else None,
            'rmse_rad': math.sqrt(sum(v * v for v in values) / len(values)) if values else None,
            'max_abs_error_rad': max(map(abs, values)) if values else None,
        }
    return {
        'schema': 'joint-tracking.v1',
        'window': {'start_ros_ns': start_ns, 'end_ros_ns': end_ns},
        'input_counts': {'commands': len(commands), 'feedback': len(feedback)},
        'counts': counts,
        'joints': joints,
        'max_command_age_ms': max_command_age_ns / 1e6,
        'mean_matched_command_age_ms': sum(ages) / len(ages) if ages else None,
        'limits': (
            'Sample-weighted ROS target-minus-feedback error, not CAN latency or motor torque. '
            'Header stamps must share one clock epoch. Feedback stamps may describe publication, '
            'not encoder sampling. No interpolation, lag fitting or ground-contact inference. '
            'Low error alone does not prove good locomotion or complete capture.'
        ),
    }


def read_bag(bag, command_topic, feedback_topic, source_id=None, selection=None):
    if source_id is not None and not 0 <= source_id <= 255:
        raise ValueError('Trace source ID must be in [0, 255]')
    from marvin_trace_msgs.msg import TracedFloat64MultiArray
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import JointState

    files = sorted(bag.glob('*.db3'))
    if not files:
        raise ValueError('No uncompressed SQLite bag files found')
    recorded = []
    types = {
        command_topic: ('marvin_trace_msgs/msg/TracedFloat64MultiArray', TracedFloat64MultiArray),
        feedback_topic: ('sensor_msgs/msg/JointState', JointState),
    }
    for path in files:
        with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
            recorded.extend(
                db.execute(
                    'SELECT messages.timestamp, topics.name, topics.type, messages.data '
                    'FROM messages JOIN topics ON topics.id=messages.topic_id '
                    'WHERE topics.name IN (?, ?)',
                    (command_topic, feedback_topic),
                )
            )
    commands, feedback = [], []
    source_counts = {}
    # Receive order exposes header clock regressions. Never sort by header stamp.
    for _, topic, message_type, data in sorted(recorded, key=lambda row: row[0]):
        expected, cls = types[topic]
        if message_type != expected:
            raise ValueError(f'Unexpected message type on {topic}: {message_type}')
        msg = deserialize_message(data, cls)
        if topic == command_topic:
            # Producer contract: source[8], instance[16], counter[40].
            source = int(msg.trace_id) >> 56
            source_counts[str(source)] = source_counts.get(str(source), 0) + 1
            if source_id is not None and source != source_id:
                continue
        stamp = msg.stamp if topic == command_topic else msg.header.stamp
        ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        if topic == command_topic:
            commands.append((ns, list(msg.data)))
        else:
            feedback.append((ns, list(msg.name), list(msg.position)))
    if selection is not None:
        selection.update(
            source_id=source_id,
            recorded_by_source=source_counts,
            selected=len(commands),
            excluded=sum(source_counts.values()) - len(commands),
        )
    return commands, feedback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', type=Path)
    parser.add_argument(
        '--joint-names',
        required=True,
        nargs='+',
        help='Exact command-array order from the captured runtime configuration',
    )
    parser.add_argument('--start-ros-ns', required=True, type=int)
    parser.add_argument('--end-ros-ns', required=True, type=int)
    parser.add_argument('--max-command-age-ms', type=float, default=50)
    parser.add_argument('--command-topic', default='/marvin_joint_controller/commands_traced')
    parser.add_argument('--feedback-topic', default='/joint_states')
    parser.add_argument(
        '--source-id',
        type=int,
        choices=range(256),
        metavar='0..255',
        help='Select one trace producer before clock validation; sim policy is normally 1',
    )
    args = parser.parse_args()
    if not math.isfinite(args.max_command_age_ms) or args.max_command_age_ms <= 0:
        parser.error('Maximum command age must be finite and positive')
    selection = {}
    commands, feedback = read_bag(
        args.bag, args.command_topic, args.feedback_topic, args.source_id, selection
    )
    result = summarize(
        commands,
        feedback,
        args.joint_names,
        args.start_ros_ns,
        args.end_ros_ns,
        int(args.max_command_age_ms * 1e6),
    )
    result['topics'] = {'command': args.command_topic, 'feedback': args.feedback_topic}
    result['command_joint_names'] = args.joint_names
    result['trace_selection'] = selection
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
