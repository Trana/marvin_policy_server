#!/usr/bin/env python3
"""Compare sparse cached velocity estimates with completed, same-host odometry bags."""
import argparse
from bisect import bisect_left, bisect_right
import hashlib
import json
import math
from pathlib import Path
import sqlite3


def finite_vector(values):
    return (isinstance(values, list) and len(values) >= 3
            and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v) for v in values[:3]))


def compare(status, odometry, start_ns, end_ns, radius_ns=100_000_000,
            max_cache_age_s=0.1):
    """Odometry rows: (bag receive ns, header ns, xyz velocity).

    Receive clocks must share one host. Cached completion time is approximate,
    not observation acquisition time. No interpolation or fitted time shift.
    """
    if not 0 < start_ns < end_ns or radius_ns <= 0:
        raise ValueError('Require a positive window and comparison radius')
    if not math.isfinite(max_cache_age_s) or max_cache_age_s <= 0:
        raise ValueError('Cache age limit must be finite and positive')
    for axis in (0, 1):
        stamps = [row[axis] for row in odometry]
        if any(t <= 0 for t in stamps) or any(b <= a for a, b in zip(stamps, stamps[1:])):
            raise ValueError('Odometry clocks must be positive and strictly increasing')
    if any(not finite_vector(row[2]) or len(row[2]) != 3 for row in odometry):
        raise ValueError('Invalid odometry velocity')
    times = [row[0] for row in odometry]
    samples = []
    counts = dict(status_rows=len(status), outside_window=0, failed=0,
                  unavailable=0, stale=0, boundary=0, no_nearby_odometry=0)
    for row in status:
        if not row.get('success'):
            counts['failed'] += 1
            continue
        received, requested = row['received_unix_ns'], row['request_unix_ns']
        if requested > received:
            raise ValueError('Status response precedes its request')
        if not start_ns <= requested <= received < end_ns:
            counts['outside_window'] += 1
            continue
        state = json.loads(row['message'])
        snapshot = state.get('concurrent_policy', {})
        values, age = snapshot.get('estimated_state'), snapshot.get('snapshot_age_seconds')
        if (not snapshot.get('history_ready') or not finite_vector(values)
                or not isinstance(age, (int, float)) or isinstance(age, bool)
                or not math.isfinite(age) or age < 0):
            counts['unavailable'] += 1
            continue
        if age > max_cache_age_s:
            counts['stale'] += 1
            continue
        approximate_ns = received - int(age * 1e9)
        # Do not mix a response from this phase with a cached result from another.
        if approximate_ns - radius_ns < start_ns or approximate_ns + radius_ns >= end_ns:
            counts['boundary'] += 1
            continue
        left = bisect_left(times, approximate_ns - radius_ns)
        right = bisect_right(times, approximate_ns + radius_ns)
        nearby = odometry[left:right]
        if not nearby:
            counts['no_nearby_odometry'] += 1
            continue
        closest = min(nearby, key=lambda r: abs(r[0] - approximate_ns))
        samples.append(dict(
            status_received_unix_ns=received, approximate_result_unix_ns=approximate_ns,
            cache_age_seconds=age, request_duration_ms=(received-requested)/1e6,
            odometry_receive_unix_ns=closest[0], odometry_header_ns=closest[1],
            estimate_xyz=values[:3], nearby_odometry_xyz=closest[2],
            estimate_minus_odometry_xyz=[a-b for a, b in zip(values[:3], closest[2])],
            odometry_min_xyz=[min(r[2][i] for r in nearby) for i in range(3)],
            odometry_max_xyz=[max(r[2][i] for r in nearby) for i in range(3)],
        ))
    return dict(
        schema='sparse-estimator-comparison.v1', counts=counts, matched=len(samples),
        window=dict(start_unix_ns=start_ns, end_unix_ns=end_ns),
        radius_ms=radius_ns/1e6, max_cache_age_seconds=max_cache_age_s, samples=samples,
        mean_difference_xyz=([sum(s['estimate_minus_odometry_xyz'][i] for s in samples)
                              /len(samples) for i in range(3)] if samples else None),
        limitations='Sparse exploratory samples, not full-rate estimator accuracy. '
        'Receive clocks must share one host. Result age is not observation acquisition age. '
        'Frame labels do not verify the publisher implementation. Nearby ranges are not '
        'confidence intervals. Signed errors can cancel. No physical motion inference.',
    )


def read_odometry(paths, topic, child_frame):
    from rclpy.serialization import deserialize_message
    from nav_msgs.msg import Odometry
    rows = []
    for path in paths:
        with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
            for received, kind, data in db.execute(
                'SELECT m.timestamp,t.type,m.data FROM messages m JOIN topics t '
                'ON m.topic_id=t.id WHERE t.name=?', (topic,)):
                if kind != 'nav_msgs/msg/Odometry':
                    raise ValueError('Unexpected odometry message type')
                msg = deserialize_message(data, Odometry)
                if msg.child_frame_id != child_frame:
                    raise ValueError('Unexpected odometry child frame')
                stamp = msg.header.stamp.sec*1_000_000_000 + msg.header.stamp.nanosec
                velocity = msg.twist.twist.linear
                rows.append((received, stamp, [velocity.x, velocity.y, velocity.z]))
    return sorted(rows, key=lambda row: row[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--start-unix-ns', required=True, type=int)
    parser.add_argument('--end-unix-ns', required=True, type=int)
    parser.add_argument('--same-host-receive-clock', required=True, action='store_true',
                        help='Confirm bag recorder and status capture used the same host clock')
    parser.add_argument('--topic', default='/isaac_odometry')
    parser.add_argument('--child-frame', default='trunk')
    args = parser.parse_args()
    status_path = args.capture / 'status.jsonl'
    bags = sorted((args.capture / 'topics').glob('*.db3'))
    if not bags:
        parser.error('No completed SQLite bag found')
    status = [json.loads(line) for line in status_path.read_text().splitlines()]
    result = compare(status, read_odometry(bags, args.topic, args.child_frame),
                     args.start_unix_ns, args.end_unix_ns)
    result['inputs'] = []
    for path in [status_path, *bags]:
        with path.open('rb') as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                digest.update(chunk)
            result['inputs'].append(dict(path=str(path.resolve()),
                                         sha256=digest.hexdigest()))
    result['topic'] = args.topic
    result['expected_child_frame'] = args.child_frame
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
