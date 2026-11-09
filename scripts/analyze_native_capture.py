#!/usr/bin/env python3
"""Join native diagnostics to recorded traced commands. Read-only, offline."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sqlite3
import statistics


def distribution(values):
    """Linear-interpolated sample percentiles; never infer missing samples."""
    ordered = sorted(values)

    def percentile(q):
        if not ordered:
            return None
        index = (len(ordered) - 1) * q
        lower = math.floor(index)
        upper = math.ceil(index)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    return {
        'count': len(values),
        'mean': statistics.mean(values) if values else None,
        'max': max(values) if values else None,
        'p50': percentile(0.5),
        'p95': percentile(0.95),
        'p99': percentile(0.99),
    }


def command_limit_occupancy(
    rows, commands, limits, margin_rad=0.01, start_ros_ns=None, end_ros_ns=None
):
    """Count unique, value-verified first publications, not time at a limit."""
    metadata = [row for row in rows if row.get('event') == 'metadata']
    if len(metadata) != 1:
        raise ValueError('Limit analysis requires exactly one metadata record')
    names = metadata[0].get('joint_names', [])
    if len(names) != 12 or len(set(names)) != 12:
        raise ValueError('Limit analysis requires twelve unique joint names')
    if not math.isfinite(margin_rad) or margin_rad < 0:
        raise ValueError('Limit margin must be finite and nonnegative')
    for bound in (start_ros_ns, end_ros_ns):
        if bound is not None and (type(bound) is not int or bound < 0):
            raise ValueError('Window bounds must be nonnegative integer ROS timestamps')
    if start_ros_ns is not None and end_ros_ns is not None and start_ros_ns >= end_ros_ns:
        raise ValueError('Window start must precede end')
    bounds = []
    for name in names:
        item = limits[name]
        lower, upper = item['min_position'], item['max_position']
        if (
            not item.get('enabled', False)
            or not all(map(math.isfinite, (lower, upper)))
            or lower >= upper
        ):
            raise ValueError(f'Invalid or disabled limits for {name}')
        bounds.append((lower, upper))
    commits = [row for row in rows if row.get('event') == 'publication_commit']
    counts = Counter(row['trace_id'] for row in commits)
    samples = []
    outside_window = 0
    for row in commits:
        matches = commands.get(row['trace_id'], [])
        if counts[row['trace_id']] != 1 or len(matches) != 1:
            continue
        stamp = matches[0][0]
        if (start_ros_ns is not None and stamp < start_ros_ns) or (
            end_ros_ns is not None and stamp >= end_ros_ns
        ):
            outside_window += 1
            continue
        expected, actual = row.get('values', [])[12:], matches[0][1]
        if (
            len(expected) == 12
            and len(actual) == 12
            and all(
                math.isfinite(a) and math.isfinite(b) and abs(a - b) <= 1e-9
                for a, b in zip(expected, actual)
            )
        ):
            samples.append(actual)
    joints = {}
    for index, (name, (lower, upper)) in enumerate(zip(names, bounds)):
        values = [sample[index] for sample in samples]
        near_lower = sum(abs(value - lower) <= margin_rad for value in values)
        near_upper = sum(abs(value - upper) <= margin_rad for value in values)
        outside = sum(value < lower - 1e-9 or value > upper + 1e-9 for value in values)
        joints[name] = {
            'samples': len(values),
            'near_lower': near_lower,
            'near_upper': near_upper,
            'outside_limits': outside,
            'min_rad': min(values) if values else None,
            'max_rad': max(values) if values else None,
        }
    return {
        'margin_rad': margin_rad,
        'accepted_commits': len(samples),
        'excluded_commits': len(commits) - len(samples),
        'joints': joints,
        'outside_window': outside_window,
        'window': {'start_ros_ns': start_ros_ns, 'end_ros_ns': end_ros_ns, 'end_exclusive': True},
        'limits': (
            'Counts of verified first publications, not time occupancy, clipping events, '
            'physical joint positions or motor tracking. Ramp and override effects may be present.'
        ),
    }


def summarize(rows, commands):
    commits = [row for row in rows if row.get('event') == 'publication_commit']
    counts = Counter(row['trace_id'] for row in commits)
    matched, mismatched, ambiguous = 0, 0, 0
    ages = {
        key: []
        for key in (
            'request_to_publication_ms',
            'joint_age_at_publication_ms',
            'imu_age_at_publication_ms',
        )
    }
    invalid_times = 0
    for row in commits:
        matches = commands.get(row['trace_id'], [])
        if counts[row['trace_id']] != 1 or len(matches) > 1:
            ambiguous += 1
            continue
        if not matches:
            continue
        stamp, values = matches[0]
        matched += 1
        expected = row['values'][12:]
        if (
            len(expected) != 12
            or len(values) != 12
            or any(
                not math.isfinite(a) or not math.isfinite(b) or abs(a - b) > 1e-9
                for a, b in zip(expected, values)
            )
        ):
            mismatched += 1
        for metric, field in zip(ages, ('ros_ns', 'joint_ns', 'imu_ns')):
            if row[field] <= 0 or stamp < row[field]:
                invalid_times += 1
            else:
                ages[metric].append((stamp - row[field]) / 1e6)
    summary = rows[-1] if rows and rows[-1].get('event') == 'summary' else None
    recorded_events = sum(row.get('event') not in {'metadata', 'summary'} for row in rows)
    declared = summary.get('records') if summary else None
    dropped = summary.get('dropped') if summary else None

    def valid_count(value):
        return type(value) is int and value >= 0

    counts_consistent = valid_count(declared) and declared == recorded_events
    lossless_export = bool(summary and counts_consistent and valid_count(dropped) and dropped == 0)
    return {
        'commits': len(commits),
        'matched': matched,
        'ambiguous': ambiguous,
        'unmatched': len(commits) - matched - ambiguous,
        'output_mismatches': mismatched,
        'invalid_time_pairs': invalid_times,
        'timing': {name: distribution(values) for name, values in ages.items()},
        'complete_export': summary is not None,
        'export_quality': {
            'recorded_events': recorded_events,
            'declared_records': declared,
            'dropped_records': dropped,
            'counts_consistent': counts_consistent,
            'lossless_export': lossless_export,
        },
        'limits': (
            'ROS-clock stamp differences only. Not CAN receipt, motor response, or ground-truth'
            ' sensor latency. Complete export means a closing summary exists. Lossless export'
            ' checks diagnostic counts only, not bag completeness or trial validity.'
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('diagnostics', type=Path)
    parser.add_argument('bag', type=Path)
    parser.add_argument('--topic', default='/marvin_joint_controller/commands_traced')
    parser.add_argument(
        '--joint-limits',
        type=Path,
        help='Canonical YAML joint limits for optional command analysis',
    )
    parser.add_argument('--limit-margin-rad', type=float, default=0.01)
    parser.add_argument(
        '--limit-start-ros-ns',
        type=int,
        help='Inclusive publication timestamp for limit analysis only',
    )
    parser.add_argument(
        '--limit-end-ros-ns',
        type=int,
        help='Exclusive publication timestamp for limit analysis only',
    )
    args = parser.parse_args()
    if not args.joint_limits and (
        args.limit_start_ros_ns is not None or args.limit_end_ros_ns is not None
    ):
        parser.error('Limit time windows require --joint-limits')
    from marvin_trace_msgs.msg import TracedFloat64MultiArray
    from rclpy.serialization import deserialize_message

    commands = {}
    files = sorted(args.bag.glob('*.db3'))
    if not files:
        parser.error('No uncompressed SQLite bag files found')
    for path in files:
        with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
            for (data,) in db.execute(
                'SELECT messages.data FROM messages JOIN topics '
                'ON topics.id=messages.topic_id WHERE topics.name=?',
                (args.topic,),
            ):
                msg = deserialize_message(data, TracedFloat64MultiArray)
                commands.setdefault(str(msg.trace_id), []).append(
                    (msg.stamp.sec * 1_000_000_000 + msg.stamp.nanosec, list(msg.data))
                )
    rows = [json.loads(line) for line in args.diagnostics.read_text().splitlines()]
    report = summarize(rows, commands)
    if args.joint_limits:
        import yaml

        limits = yaml.safe_load(args.joint_limits.read_text())['limits']
        report['command_limit_occupancy'] = command_limit_occupancy(
            rows,
            commands,
            limits,
            args.limit_margin_rad,
            args.limit_start_ros_ns,
            args.limit_end_ros_ns,
        )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
