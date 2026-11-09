import importlib.util
import json
from pathlib import Path
import sqlite3

import pytest

spec = importlib.util.spec_from_file_location(
    'quality', Path(__file__).parents[1] / 'scripts/reference_capture_quality.py'
)
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


def test_split_bag_gaps_and_missing_data(tmp_path):
    (tmp_path / 'topics').mkdir()
    (tmp_path / 'metadata.json').write_text(json.dumps({'topics': ['/imu', '/missing']}))
    for index, stamps in enumerate(([0, 20_000_000], [100_000_000, 100_000_000])):
        with sqlite3.connect(tmp_path / 'topics' / f'bag_{index}.db3') as db:
            db.execute('CREATE TABLE topics (id INTEGER, name TEXT)')
            db.execute('CREATE TABLE messages (topic_id INTEGER, timestamp INTEGER)')
            db.execute("INSERT INTO topics VALUES (1, '/imu')")
            db.executemany('INSERT INTO messages VALUES (1, ?)', [(s,) for s in stamps])
    (tmp_path / 'status.jsonl').write_text('{"success":true}\n{"error":"timeout"}\nbroken\n')
    result = quality.report(tmp_path)
    assert result['topics']['/imu']['count'] == 4
    assert result['topics']['/imu']['max_gap_ms'] == 80
    assert result['topics']['/imu']['duplicate_timestamps'] == 1
    assert result['empty_topics'] == ['/missing']
    assert result['topics']['/missing']['mean_recorded_hz'] is None
    assert result['status']['successful'] == 1
    assert result['status']['unsuccessful'] == 1
    assert result['status']['malformed'] == 1
    json.dumps(result, allow_nan=False)


def test_no_bag_is_not_success(tmp_path):
    (tmp_path / 'metadata.json').write_text('{"topics":[]}')
    with pytest.raises(ValueError, match='No SQLite'):
        quality.report(tmp_path)


def test_single_sample_has_no_rate():
    assert quality.summarize_timestamps([42])['mean_recorded_hz'] is None


def test_typed_snapshot_is_received_evidence_not_a_failed_policy_query(tmp_path):
    (tmp_path / 'topics').mkdir()
    (tmp_path / 'metadata.json').write_text('{"topics":[]}')
    with sqlite3.connect(tmp_path / 'topics/bag.db3') as db:
        db.execute('CREATE TABLE topics (id INTEGER, name TEXT)')
        db.execute('CREATE TABLE messages (topic_id INTEGER, timestamp INTEGER)')
    (tmp_path / 'status.jsonl').write_text(
        '{"state":{"phase":"execution_hold","reason_code":"LEASE_EXPIRED"}}\n'
        '{"error":"status timeout"}\n'
    )
    status = quality.report(tmp_path)['status']
    assert status['typed_snapshots'] == 1
    assert status['successful'] == 0  # Legacy policy success flags only.
    assert status['unsuccessful'] == 1
