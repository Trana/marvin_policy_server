"""Fake service responses only. No ROS nodes, discovery or hardware access."""

from importlib import import_module
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

pytest.importorskip('rclpy')
rclpy = import_module('rclpy')

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location(
    'chirp_client', ROOT / 'scripts/run_robot_joint_chirp.py'
)
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


def test_terminal_failure_preserves_tracking_evidence():
    result = client.terminal_failure(
        'NOT_READY',
        NS(
            phase='execution_hold',
            reason_code='TRACKING_FAULT',
            tracking_fault_latched=True,
            feedback_fault_latched=False,
        ),
    )
    assert 'TRACKING_FAULT' in result
    assert 'tracking_fault=True' in result
    assert 'feedback_fault=False' in result


@pytest.mark.parametrize(
    'case',
    [
        'stop_only',
        'humble_stop',
        'inactive',
        'false_switch_ack',
        'prepared_eof',
        'prepared_changed',
        'prepared_inactive',
        'prepared_cleanup_timeout',
    ],
)
def test_stop_and_handover_fences(monkeypatch, tmp_path, case):
    calls = []
    state = ['inactive' if case == 'inactive' or case.startswith('prepared_') else 'active']
    state_reads = [0]

    def respond(endpoint, request):
        calls.append((endpoint, request))
        if endpoint.endswith('list_hardware_components'):
            if case == 'humble_stop':
                return NS(component=[NS(name='robot', state=NS(label=state[0]))])
            return NS(
                component=[
                    NS(
                        name='robot',
                        plugin_name='odrive_ros2_control_plugin/ODriveHardwareInterface',
                        state=NS(label=state[0]),
                    )
                ]
            )
        if endpoint.endswith('get_parameters'):
            return NS(
                values=[
                    NS(
                        type=4,
                        string_value=(
                            '<robot><ros2_control name="robot"><hardware>'
                            '<plugin>odrive_ros2_control_plugin/ODriveHardwareInterface</plugin>'
                            '</hardware></ros2_control></robot>'
                        ),
                    )
                ]
            )
        if endpoint.endswith('list_controllers'):
            return NS(
                controller=[
                    NS(
                        name='marvin_joint_controller',
                        state='active',
                        claimed_interfaces=['joint/position'],
                    ),
                    NS(name='joint_test_controller', state='inactive', claimed_interfaces=[]),
                ]
            )
        if endpoint.endswith('switch_controller'):
            return NS(ok=True)  # Humble acknowledgement without actual handover.
        if endpoint.endswith('joint_test/state'):
            state_reads[0] += 1
            return NS(
                state=NS(
                    controller_instance_id='instance',
                    activation_generation=(state_reads[0] if case == 'prepared_changed' else 0),
                )
            )
        if endpoint.endswith('set_hardware_component_state'):
            assert request.target_state.label == 'inactive'
            state[0] = 'inactive'
            return NS(ok=True)
        raise AssertionError('Unexpected call: ' + endpoint)

    class FakeClient:
        def __init__(self, endpoint):
            self.endpoint = endpoint

        def wait_for_service(self, **_):
            return True

        def call_async(self, request):
            result = respond(self.endpoint, request)
            return NS(done=lambda: True, result=lambda: result)

    node = NS(
        create_client=lambda _kind, endpoint: FakeClient(endpoint), destroy_node=lambda: None
    )
    monkeypatch.setattr(rclpy, 'init', lambda **_: None)
    monkeypatch.setattr(rclpy, 'create_node', lambda _: node)
    monkeypatch.setattr(rclpy, 'spin_until_future_complete', lambda *_, **__: None)
    monkeypatch.setattr(rclpy, 'try_shutdown', lambda: None)
    monkeypatch.setattr(client.signal, 'signal', lambda *args: None)
    monkeypatch.setenv('ROS_DOMAIN_ID', '0')
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    profile = ROOT.parent / 'marvin/bringup/config/robot_joint_test_chirp.yaml'
    argv = [
        'chirp',
        '--operator-ready',
        '--profile',
        str(profile),
        '--output',
        str(tmp_path / 'capture'),
    ]
    if case in ('stop_only', 'humble_stop'):
        argv.append('--stop-only')
    if case.startswith('prepared_'):
        argv.append('--wait-for-operator')
        monkeypatch.setattr(
            sys,
            'stdin',
            io.StringIO('' if case in ('prepared_eof', 'prepared_cleanup_timeout') else '\n'),
        )
        monkeypatch.setattr(client.select, 'select', lambda *args: ([sys.stdin], [], []))
        monkeypatch.setattr(rclpy, 'spin_once', lambda *args, **kwargs: None)

        def spawn_recorder(args):
            assert args[args.index('--activation') + 1] == (
                '2' if case == 'prepared_changed' else '1'
            )
            output = tmp_path / 'capture'
            output.mkdir()
            (output / 'health.json').write_text(
                json.dumps({'running': True, 'fault': None, 'feedback': 2})
            )

            def wait(timeout):
                if case == 'prepared_cleanup_timeout' and timeout == 8:
                    raise client.subprocess.TimeoutExpired('recorder', timeout)
                return 0

            return NS(
                poll=lambda: None, send_signal=lambda _: None, wait=wait, terminate=lambda: None
            )

        monkeypatch.setattr(client.subprocess, 'Popen', spawn_recorder)
    monkeypatch.setattr(sys, 'argv', argv)
    if case in ('stop_only', 'humble_stop'):
        client.main()
        assert not any(path.endswith('switch_controller') for path, _ in calls)
    else:
        with pytest.raises(
            RuntimeError,
            match=(
                'Requires active|Exclusive handover|Operator input|Controller changed|Recorder'
                ' shutdown'
            ),
        ):
            client.main()
    stops = [request for path, request in calls if path.endswith('set_hardware_component_state')]
    assert len(stops) == (0 if case == 'inactive' else 1)
    assert not any('/joint_test/control' in path for path, _ in calls)
    if case.startswith('prepared_'):
        assert not any(path.endswith('switch_controller') for path, _ in calls)
        timing = json.loads((tmp_path / 'capture/service-timings.json').read_text())
        assert timing['records'] and all(r['completed'] for r in timing['records'])
        assert all(r['elapsed_ms'] >= 0 for r in timing['records'])
        assert 'permit_challenge' not in json.dumps(timing)


@pytest.mark.parametrize('case', ['clean', 'failed', 'timeout_term', 'timeout_kill'])
def test_recorder_shutdown_is_bounded(case):
    actions = []

    class Child:
        returncode = None

        def poll(self):
            return None

        def send_signal(self, value):
            actions.append(('signal', value))

        def terminate(self):
            actions.append(('terminate',))

        def kill(self):
            actions.append(('kill',))

        def wait(self, timeout):
            actions.append(('wait', timeout))
            if (
                case.startswith('timeout')
                and timeout == 8
                or case == 'timeout_kill'
                and ('kill',) not in actions
            ):
                raise client.subprocess.TimeoutExpired('recorder', timeout)
            return 1 if case == 'failed' else 0

    if case == 'clean':
        client.stop_recorder(Child())
    else:
        with pytest.raises(RuntimeError, match='finalization is unconfirmed'):
            client.stop_recorder(Child())
    assert actions[0] == ('signal', client.signal.SIGINT)
    assert (('terminate',) in actions) == case.startswith('timeout')
    assert (('kill',) in actions) == (case == 'timeout_kill')


def test_recorder_stop_does_not_repeat_signal_during_cleanup():
    actions = []
    child = NS(
        poll=lambda: None, send_signal=lambda _: actions.append('signal'), wait=lambda **_: 0
    )
    client.stop_recorder(child, signal_sent=True)
    assert actions == []


@pytest.mark.parametrize(
    'joint', ['FR_thigh_joint', 'FL_thigh_joint', 'RR_thigh_joint', 'RL_thigh_joint']
)
def test_half_degree_thigh_profile(joint):
    profile = client.yaml.safe_load(
        (ROOT.parent / 'marvin/bringup/config/robot_joint_test_thigh_chirp.yaml').read_text()
    )['/joint_test_controller']['ros__parameters']
    assert client.approved_amplitude(profile, joint) == pytest.approx(0.008726646259971648)
    original = client.yaml.safe_load(
        (ROOT.parent / 'marvin/bringup/config/robot_joint_test_chirp.yaml').read_text()
    )['/joint_test_controller']['ros__parameters']
    for key in ('lower_rad', 'upper_rad', 'speed_rad_s', 'acceleration_rad_s2'):
        assert profile['joint_test_profile.' + key] == original['joint_test_profile.' + key]
    assert profile['max_handover_error'] == original['max_handover_error']
    with pytest.raises(ValueError):
        client.approved_amplitude(profile, 'FR_calf_joint')
    profile['joint_test_profile.max_amplitude_rad'][profile['joints'].index(joint)] = 0.02
    with pytest.raises(ValueError):
        client.approved_amplitude(profile, joint)
