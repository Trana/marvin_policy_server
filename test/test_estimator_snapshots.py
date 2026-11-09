import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'snapshots', Path(__file__).parents[1] / 'scripts/analyze_estimator_snapshots.py'
)
snapshots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snapshots)


def status(received=500_000_000, age=0.01, values=None):
    return {
        'success': True,
        'request_unix_ns': received - 1000,
        'received_unix_ns': received,
        'message': json.dumps({
            'concurrent_policy': {
                'history_ready': True,
                'snapshot_age_seconds': age,
                'estimated_state': [1.0, 2.0, 3.0] if values is None else values,
            }
        }),
    }


def test_pairing_and_local_range():
    result = snapshots.compare(
        [status()],
        [
            (480_000_000, 100, [0.5, 1.0, 2.0]),
            (490_000_000, 200, [0.8, 2.0, 3.0]),
            (510_000_000, 300, [1.2, 3.0, 4.0]),
        ],
        1,
        1_000_000_000,
    )
    assert result['matched'] == 1
    assert result['mean_difference_xyz'] == pytest.approx([0.2, 0, 0])
    sample = result['samples'][0]
    assert sample['approximate_result_unix_ns'] == 490_000_000
    assert sample['odometry_min_xyz'] == [0.5, 1, 2]
    assert sample['odometry_max_xyz'] == [1.2, 3, 4]


@pytest.mark.parametrize(
    'age,counter',
    [(0.2, 'stale'), (-1, 'unavailable'), (float('nan'), 'unavailable'), (None, 'unavailable')],
)
def test_invalid_and_stale_cache(age, counter):
    result = snapshots.compare([status(age=age)], [], 1, 1_000_000_000)
    assert result['counts'][counter] == 1
    assert result['mean_difference_xyz'] is None


def test_boundary_and_missing_are_not_zero_error():
    result = snapshots.compare([status(received=50_000_000), status()], [], 1, 1_000_000_000)
    assert result['counts']['boundary'] == 1
    assert result['counts']['no_nearby_odometry'] == 1
    assert result['mean_difference_xyz'] is None


@pytest.mark.parametrize(
    'rows',
    [
        [(200, 10, [0, 0, 0]), (100, 20, [0, 0, 0])],
        [(100, 20, [0, 0, 0]), (200, 10, [0, 0, 0])],
        [(100, 10, [float('inf'), 0, 0])],
    ],
)
def test_bad_odometry_rejected(rows):
    with pytest.raises(ValueError):
        snapshots.compare([], rows, 1, 1_000_000_000)


def test_request_crosses_phase_and_invalid_vector():
    row = status()
    row['request_unix_ns'] = 10
    result = snapshots.compare([row, status(values=[1, 2])], [], 100, 1_000_000_000)
    assert result['counts']['outside_window'] == 1
    assert result['counts']['unavailable'] == 1


def test_sqlite_readonly_reader(tmp_path):
    serialization = pytest.importorskip('rclpy.serialization')
    msgs = pytest.importorskip('nav_msgs.msg')
    import sqlite3

    msg = msgs.Odometry()
    msg.child_frame_id = 'trunk'
    msg.header.stamp.sec = 1
    msg.twist.twist.linear.x = 0.25
    path = tmp_path / 'bag.db3'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE topics (id INTEGER, name TEXT, type TEXT)')
        db.execute('CREATE TABLE messages (topic_id INTEGER, timestamp INTEGER, data BLOB)')
        db.execute("INSERT INTO topics VALUES (1, '/odom', 'nav_msgs/msg/Odometry')")
        db.execute(
            'INSERT INTO messages VALUES (1, 123, ?)', (serialization.serialize_message(msg),)
        )
    before = path.read_bytes()
    assert snapshots.read_odometry([path], '/odom', 'trunk') == [
        (123, 1_000_000_000, [0.25, 0.0, 0.0])
    ]
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match='child frame'):
        snapshots.read_odometry([path], '/odom', 'wrong')
