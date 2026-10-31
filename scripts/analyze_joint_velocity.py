#!/usr/bin/env python3
"""Read-only position/velocity consistency check for completed ROS SQLite bags."""
import argparse
import json
import math
from pathlib import Path
import sqlite3


def analyze(rows, window_s=0.1, max_gap_s=0.05, min_speed=0.1, min_windows=20):
    """Rows contain (header ns, names, positions, velocities).

    Compare displacement with trapezoidal velocity integration over nonoverlapping
    windows. This is an observation consistency check, not a motor response model.
    """
    if (not all(math.isfinite(v) and v > 0 for v in (window_s, max_gap_s, min_speed))
            or min_windows < 1):
        raise ValueError('Require positive finite analysis limits')
    names = None
    previous = anchor = None
    integrated = {}
    errors = {}
    gaps = 0
    for stamp, row_names, positions, velocities in rows:
        if (stamp <= 0 or (previous and stamp <= previous[0])
                or not row_names or len(set(row_names)) != len(row_names)
                or len(positions) != len(row_names) or len(velocities) != len(row_names)
                or not all(math.isfinite(v) for v in [*positions, *velocities])):
            raise ValueError('Invalid joint state or non-increasing header clock')
        if names is None:
            names = list(row_names)
            errors = {name: [] for name in names}
        if set(row_names) != set(names):
            raise ValueError('Joint set changed inside capture')
        pos, vel = dict(zip(row_names, positions)), dict(zip(row_names, velocities))
        current = (stamp, pos, vel)
        dt = (stamp - previous[0])*1e-9 if previous else None
        if previous is None or dt > max_gap_s:
            gaps += int(previous is not None)
            anchor = current
            integrated = dict.fromkeys(names, 0.)
        else:
            for name in names:
                integrated[name] += 0.5*(previous[2][name]+vel[name])*dt
            elapsed = (stamp-anchor[0])*1e-9
            if elapsed >= window_s:
                for name in names:
                    derivative = (pos[name]-anchor[1][name])/elapsed
                    reported = integrated[name]/elapsed
                    if max(abs(derivative), abs(reported)) >= min_speed:
                        errors[name].append((derivative, reported))
                anchor = current
                integrated = dict.fromkeys(names, 0.)
        previous = current
    joints = {}
    for name, pairs in errors.items():
        count = len(pairs)
        enough = count >= min_windows
        joints[name] = dict(
            moving_windows=count, sufficient_motion_windows=enough,
            derivative_minus_velocity_rmse=(math.sqrt(sum((d-v)**2 for d, v in pairs)/count)
                                             if enough else None),
            derivative_plus_velocity_rmse=(math.sqrt(sum((d+v)**2 for d, v in pairs)/count)
                                            if enough else None),
            same_sign_fraction=(sum(d*v > 0 for d, v in pairs)/count if enough else None),
        )
    return dict(schema='joint-velocity-consistency.v1', window_s=window_s,
                max_gap_s=max_gap_s, min_speed_rad_s=min_speed, min_windows=min_windows,
                clock_gap_resets=gaps, joints=joints,
                limitations='Insufficient movement is inconclusive, not a pass. '
                'Compare minus and plus residuals; lower plus residual suggests a sign reversal. '
                'Sensor filtering, sample age and clock errors also affect residuals. '
                'Window averages can hide oscillation. This does not validate physical joint axes.')


def read_bag(paths, topic):
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import JointState
    recorded = []
    for path in paths:
        with sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True) as db:
            recorded.extend(db.execute(
                'SELECT m.timestamp,t.type,m.data FROM messages m JOIN topics t '
                'ON t.id=m.topic_id WHERE t.name=?', (topic,)))
    rows = []
    for _, kind, data in sorted(recorded, key=lambda row: row[0]):
        if kind != 'sensor_msgs/msg/JointState':
            raise ValueError('Unexpected joint state message type')
        msg = deserialize_message(data, JointState)
        rows.append((msg.header.stamp.sec*1_000_000_000+msg.header.stamp.nanosec,
                     list(msg.name), list(msg.position), list(msg.velocity)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', type=Path)
    parser.add_argument('--topic', default='/joint_states')
    parser.add_argument('--start-ros-ns', required=True, type=int)
    parser.add_argument('--end-ros-ns', required=True, type=int)
    args = parser.parse_args()
    if not 0 < args.start_ros_ns < args.end_ros_ns:
        parser.error('Require positive increasing phase boundaries')
    paths = sorted(args.bag.glob('*.db3'))
    if not paths:
        parser.error('No completed SQLite bag found')
    rows = read_bag(paths, args.topic)
    # Validate the whole recording first, so filtering cannot hide clock resets.
    analyze(rows)
    result = analyze([r for r in rows if args.start_ros_ns <= r[0] < args.end_ros_ns])
    result.update(topic=args.topic, start_ros_ns=args.start_ros_ns, end_ros_ns=args.end_ros_ns)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
