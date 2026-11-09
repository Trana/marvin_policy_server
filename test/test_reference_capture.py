"""Validate capture admission without importing ROS or starting a recorder."""

import hashlib
import importlib.util
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    'capture_reference_run', Path(__file__).parents[1] / 'scripts/capture_reference_run.py'
)
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def test_default_sim_endpoints_include_namespaced_commands_and_clocks():
    topics, service = capture.capture_endpoints('sim')
    assert service == '/sim/get_policy_observations'
    assert {
        '/sim/marvin_joint_controller/commands_traced',
        '/isaac_joint_commands',
        '/isaac_joint_states',
        '/isaac_imu',
        '/isaac_odometry',
        '/clock',
    } <= set(topics)
    assert '/marvin_joint_controller/commands_traced' not in topics
    assert '/sim/joint_states' not in topics
    assert len(topics) == len(set(topics))


def test_robot_endpoints_preserve_real_namespace():
    topics, service = capture.capture_endpoints('robot')
    assert service == '/get_policy_observations'
    assert {'/marvin_joint_controller/commands_traced', '/joint_states', '/imu'} <= set(topics)
    assert not any(topic.startswith(('/sim/', '/isaac_', '/clock')) for topic in topics)
    with pytest.raises(ValueError):
        capture.capture_endpoints('other')


def test_joint_test_endpoints_and_domain_are_isolated(monkeypatch):
    topics, service = capture.capture_endpoints('sim_joint_test')
    assert service == '/joint_test/joint_test_controller/joint_test/state'
    assert set(topics) == {
        '/joint_test/isaac_joint_commands',
        '/joint_test/isaac_joint_states',
        '/joint_test/joint_test_controller/joint_test/target_samples',
        '/joint_test/joint_test_controller/status',
        '/clock',
        '/rosout',
    }
    monkeypatch.setenv('ROS_DOMAIN_ID', '231')
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    args = ['--target', 'sim_joint_test', '--output', '/unused']
    with pytest.raises(SystemExit):
        capture.parse_args(args)
    args += ['--artifact', '/supplied/profile.yaml']
    assert capture.parse_args(args).target == 'sim_joint_test'
    monkeypatch.setenv('ROS_DOMAIN_ID', '10')
    with pytest.raises(SystemExit):
        capture.parse_args(args)


def test_typed_status_keeps_evidence_but_omits_movement_challenge():
    state = SimpleNamespace(
        phase='execution_hold',
        reason_code='LEASE_EXPIRED',
        applied_sequence=7,
        permit_challenge='short-lived',
        motion_available=False,
    )
    state.get_fields_and_field_types = lambda: {
        key: 'unused'
        for key in (
            'phase',
            'reason_code',
            'applied_sequence',
            'permit_challenge',
            'motion_available',
        )
    }
    row = capture.status_payload('sim_joint_test', SimpleNamespace(state=state))
    assert row == {
        'state': {
            'phase': 'execution_hold',
            'reason_code': 'LEASE_EXPIRED',
            'applied_sequence': 7,
            'motion_available': False,
        }
    }
    assert capture.status_payload('robot_joint_test', SimpleNamespace(state=state)) == row
    assert capture.status_payload('robot', SimpleNamespace(success=True, message='cached')) == {
        'success': True,
        'message': 'cached',
    }


def test_robot_joint_recording_requires_explicit_provenance(monkeypatch):
    monkeypatch.setenv('ROS_DOMAIN_ID', '0')
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    args = ['--target', 'robot_joint_test', '--output', '/unused']
    with pytest.raises(SystemExit):
        capture.parse_args(args)
    args += ['--artifact', '/supplied/profile.yaml']
    with pytest.raises(SystemExit):
        capture.parse_args(args)
    args += [
        '--joint-test-controller',
        '/joint_test_controller',
        '--joint-test-command-topic',
        '/joint_test_controller/applied_targets',
    ]
    assert capture.parse_args(args).target == 'robot_joint_test'
    topics, service = capture.capture_endpoints(
        'robot_joint_test', '/joint_test_controller', '/joint_test_controller/applied_targets'
    )
    assert service == '/joint_test_controller/joint_test/state'
    assert '/joint_states' in topics and '/dynamic_joint_states' in topics
    assert '/joint_test_controller/joint_test/target_samples' in topics
    assert not any(topic.startswith(('/sim/', '/isaac_', '/clock')) for topic in topics)
    assert '/marvin_joint_controller/commands' not in topics
    monkeypatch.setenv('ROS_DOMAIN_ID', '231')
    with pytest.raises(SystemExit):
        capture.parse_args(args)


@pytest.mark.parametrize(
    'topic',
    [
        '/marvin_joint_controller/commands',
        '/marvin_joint_controller/commands_traced',
        '/joint_states',
        '/dynamic_joint_states',
        '/joint_test_controller/status',
        'relative',
        '/bad name',
        '/',
    ],
)
def test_robot_joint_recording_rejects_ambiguous_sources(topic):
    with pytest.raises(ValueError):
        capture.capture_endpoints('robot_joint_test', '/joint_test_controller', topic)


@pytest.mark.parametrize('seconds', ['nan', 'inf', '0', '-1', '301'])
def test_invalid_duration(monkeypatch, seconds):
    monkeypatch.setenv('ROS_DOMAIN_ID', '0')
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    with pytest.raises(SystemExit):
        capture.parse_args(['--target', 'robot', '--output', '/unused', '--seconds', seconds])


@pytest.mark.parametrize('target,domain', [('robot', '0'), ('sim', '10')])
def test_target_domain(monkeypatch, target, domain):
    monkeypatch.setenv('ROS_DOMAIN_ID', domain)
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    args = ['--target', target, '--output', '/unused']
    assert capture.parse_args(args).seconds == 120
    monkeypatch.setenv('ROS_DOMAIN_ID', '184')
    with pytest.raises(SystemExit):
        capture.parse_args(args)


def test_requires_localhost(monkeypatch):
    monkeypatch.setenv('ROS_DOMAIN_ID', '0')
    monkeypatch.delenv('ROS_LOCALHOST_ONLY', raising=False)
    with pytest.raises(SystemExit):
        capture.parse_args(['--target', 'robot', '--output', '/unused'])


def test_artifact_identity(tmp_path):
    path = tmp_path / 'policy.pt'
    path.write_bytes(b'reference bytes')
    result = capture.file_identity(path)
    assert result['sha256'] == hashlib.sha256(b'reference bytes').hexdigest()
    assert result['size'] == 15
    assert 'verify against runtime' in result['evidence']
    with pytest.raises(FileNotFoundError):
        capture.file_identity(tmp_path / 'missing')


def test_events_append_without_ros(tmp_path):
    (tmp_path / 'metadata.json').write_text(json.dumps({'hostname': socket.gethostname()}))
    first = capture.mark_event(tmp_path, 'grounded')
    second = capture.mark_event(tmp_path, 'lifted')
    rows = [json.loads(line) for line in (tmp_path / 'events.jsonl').read_text().splitlines()]
    assert rows == [first, second]
    assert second['monotonic_ns'] >= first['monotonic_ns']
    assert first['source'] == 'operator'
    with pytest.raises(ValueError):
        capture.mark_event(tmp_path, '\n')


def test_event_rejects_missing_capture_or_different_host(tmp_path):
    with pytest.raises(FileNotFoundError):
        capture.mark_event(tmp_path, 'grounded')
    (tmp_path / 'metadata.json').write_text('{"hostname":"not-this-host"}')
    with pytest.raises(ValueError):
        capture.mark_event(tmp_path, 'grounded')
    assert not (tmp_path / 'events.jsonl').exists()


def test_finalized_capture_rejects_event_mutation(tmp_path):
    (tmp_path / 'complete.json').write_text('{}')
    with pytest.raises(ValueError, match='finalized'):
        capture.mark_event(tmp_path, 'late note')


def test_topics_only_and_session_config(monkeypatch):
    monkeypatch.setenv('ROS_DOMAIN_ID', '0')
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    args = capture.parse_args([
        '--target',
        'robot',
        '--output',
        '/unused',
        '--topics-only',
        '--session-config',
        '/supplied/session.json',
    ])
    assert args.topics_only and args.session_config == Path('/supplied/session.json')


def test_progress_does_not_open_database(tmp_path, monkeypatch):
    (tmp_path / 'topics').mkdir()
    (tmp_path / 'topics/data.db3').write_bytes(b'not a real database')
    (tmp_path / 'recorder.log').write_text("Subscribed to topic '/joint_states'\n")
    original = Path.open

    def guarded_open(self, *args, **kwargs):
        assert self.suffix != '.db3', 'Live database must never be opened'
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', guarded_open)
    progress = capture.capture_progress(tmp_path, 'recording')
    assert progress['bag_bytes'] == 19
    assert progress['subscribed_topics'] == ['/joint_states']
    assert json.loads((tmp_path / 'progress.json').read_text()) == progress


def test_clean_stop_admission(tmp_path):
    spec = importlib.util.spec_from_file_location(
        'capture_control', Path(__file__).parents[1] / 'scripts/capture_control.py'
    )
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)
    capture.capture_progress(tmp_path, 'recording')
    assert control.control(tmp_path, stop=True)['stop_requested']
    assert (tmp_path / 'stop.request').exists()
    capture.capture_progress(tmp_path, 'failed')
    with pytest.raises(ValueError, match='not recording'):
        control.control(tmp_path, stop=True)
    capture.capture_progress(tmp_path, 'recording')
    data = json.loads((tmp_path / 'progress.json').read_text())
    data['observed_unix_ns'] = 1
    (tmp_path / 'progress.json').write_text(json.dumps(data))
    with pytest.raises(ValueError, match='stale'):
        control.control(tmp_path, stop=True)
