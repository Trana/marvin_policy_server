import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    'tracking', Path(__file__).parents[1] / 'scripts/analyze_joint_tracking.py')
tracking = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracking)


def test_named_feedback_and_latest_preceding_command():
    result = tracking.summarize(
        [(10, [1., 2.]), (20, [3., 4.])],
        [(15, ['b', 'a'], [1.5, .5]), (20, ['a', 'b'], [2., 3.])],
        ['a', 'b'], 1, 30, 10)
    assert result['counts']['matched'] == 2
    for joint in result['joints'].values():
        assert joint['mean_target_minus_measured_rad'] == .75
        assert joint['rmse_rad'] == pytest.approx((.625)**.5)


def test_window_staleness_and_missing_are_not_zero_error():
    result = tracking.summarize(
        [(10, [1.])], [(5, ['a'], [0]), (15, ['a'], [0]),
                      (25, ['a'], [0]), (30, ['a'], [0])], ['a'], 5, 30, 10)
    assert result['counts'] == dict(in_window=3, matched=1, no_preceding_command=1,
                                   stale_command=1, invalid_feedback=0)
    empty = tracking.summarize([], [(5, ['a'], [0])], ['a'], 1, 10, 10)
    assert empty['joints']['a']['rmse_rad'] is None


@pytest.mark.parametrize('row', [(15, ['b'], [0]), (15, ['a'], []),
                                     (15, ['a', 'a'], [0, 0]), (15, ['a'], [float('nan')])])
def test_invalid_feedback_is_counted(row):
    result = tracking.summarize([(10, [1])], [row], ['a'], 1, 30, 10)
    assert result['counts']['invalid_feedback'] == 1
    assert result['counts']['matched'] == 0


@pytest.mark.parametrize('stamps', [(10, 10), (20, 10), (0, 10)])
def test_clock_errors_rejected(stamps):
    with pytest.raises(ValueError):
        tracking.summarize([(t, [0]) for t in stamps], [], ['a'], 1, 30, 10)
    with pytest.raises(ValueError):
        tracking.summarize([], [(t, ['a'], [0]) for t in stamps], ['a'], 1, 30, 10)


def test_invalid_commands_rejected():
    for values in ([], [float('inf')], [0, 1]):
        with pytest.raises(ValueError):
            tracking.summarize([(10, values)], [], ['a'], 1, 30, 10)


def test_no_future_target_leaks_into_error():
    result = tracking.summarize([(20, [100])], [(10, ['a'], [0])], ['a'], 1, 30, 10)
    assert result['counts']['no_preceding_command'] == 1
    assert result['joints']['a']['samples'] == 0


def test_real_ros_serialization_and_split_sqlite_reader(tmp_path):
    serialization = pytest.importorskip('rclpy.serialization')
    trace = pytest.importorskip('marvin_trace_msgs.msg')
    sensors = pytest.importorskip('sensor_msgs.msg')
    import sqlite3

    command = trace.TracedFloat64MultiArray()
    command.stamp.sec = 1
    command.trace_id = (1 << 56) | 123
    command.data = [1., 2.]
    feedback = sensors.JointState()
    feedback.header.stamp.sec = 1
    feedback.header.stamp.nanosec = 10_000_000
    feedback.name = ['b', 'a']
    feedback.position = [1.5, .5]
    # Filenames do not imply temporal order. The reader uses bag receive time.
    for filename, topic, kind, msg, received in (
        ('z.db3', '/commands', 'marvin_trace_msgs/msg/TracedFloat64MultiArray', command, 100),
        ('a.db3', '/feedback', 'sensor_msgs/msg/JointState', feedback, 200),
    ):
        with sqlite3.connect(tmp_path / filename) as db:
            db.execute('CREATE TABLE topics (id INTEGER, name TEXT, type TEXT)')
            db.execute('CREATE TABLE messages (topic_id INTEGER, timestamp INTEGER, data BLOB)')
            db.execute('INSERT INTO topics VALUES (1, ?, ?)', (topic, kind))
            db.execute('INSERT INTO messages VALUES (1, ?, ?)',
                       (received, serialization.serialize_message(msg)))
    commands, observed = tracking.read_bag(tmp_path, '/commands', '/feedback')
    result = tracking.summarize(commands, observed, ['a', 'b'],
                                1_000_000_000, 2_000_000_000, 50_000_000)
    assert result['counts']['matched'] == 1
    assert result['mean_matched_command_age_ms'] == 10
    assert result['joints']['a']['rmse_rad'] == .5
    assert result['joints']['b']['rmse_rad'] == .5
    # An IK command uses wall time on the same topic. Never repair this by sorting stamps.
    command.trace_id = (2 << 56) | 456
    command.stamp.sec = 1_700_000_000
    with sqlite3.connect(tmp_path / 'z.db3') as db:
        db.execute('INSERT INTO messages VALUES (1, ?, ?)',
                   (90, serialization.serialize_message(command)))
    mixed, observed = tracking.read_bag(tmp_path, '/commands', '/feedback')
    with pytest.raises(ValueError, match='regressing'):
        tracking.summarize(mixed, observed, ['a', 'b'], 1, 2_000_000_000, 50_000_000)
    selection = {}
    selected, observed = tracking.read_bag(tmp_path, '/commands', '/feedback', 1, selection)
    assert selected == commands
    assert selection == dict(source_id=1, recorded_by_source={'2': 1, '1': 1},
                             selected=1, excluded=1)
    absent, _ = tracking.read_bag(tmp_path, '/commands', '/feedback', 3)
    assert absent == []


def test_invalid_source_id_rejected(tmp_path):
    for source in (-1, 256):
        with pytest.raises(ValueError, match='source ID'):
            tracking.read_bag(tmp_path, '/commands', '/feedback', source)
