"""Management-side ownership checks without ROS nodes or command publishers."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from marvin_policy_server.execution_outcome import ExecutionOutcome
from marvin_policy_server.marvin_policy_server import MarvinPolicyServer
from marvin_policy_server.native_protocol import Kind, Packet
from marvin_policy_server.native_server import NativePolicyServer
from marvin_policy_server.native_profile import MARVIN_JOINTS, MARVIN_TOPICS


def server_without_node():
    server = object.__new__(NativePolicyServer)
    server._native_generation = 3
    server._pending_result = Packet(Kind.RESULT, sequence=7, generation=3, trace_id=42)
    server._history_requested = True
    server._timing = SimpleNamespace(count=Mock())
    server._policy_runner = SimpleNamespace(reset=Mock())
    server._inference_schedule = SimpleNamespace(reset=Mock())
    server._velocity_estimator = SimpleNamespace(reset=Mock())
    server._latest_estimated_velocity = np.ones(3)
    server.joint_names = [f'joint_{i}' for i in range(12)]
    server.action = np.ones(12)
    return server


@pytest.mark.parametrize('direct_output', [False, True])
def test_result_records_effective_observation_command_not_raw_input(direct_output):
    server = server_without_node()
    server._frame = Packet(Kind.SNAPSHOT, sequence=7, values=(0.,) * 58)
    server._native_channel = SimpleNamespace(send=Mock())
    server._apply_safety_position_limits = lambda values: values
    server._nan_obs_warned = direct_output
    server._joint_command_output_locked = False
    server._native_diagnostics = SimpleNamespace(record=Mock())
    server._obs_state = SimpleNamespace(velocity_commands_history=np.array([[.2, -.1, .3]]))
    server._cmd_vel = SimpleNamespace(linear=SimpleNamespace(x=99., y=99.), angular=SimpleNamespace(z=99.))
    server._publish_joint_command([0.] * 12)
    captured = server._native_diagnostics.record.call_args.kwargs['effective_velocity_command']
    if direct_output:
        assert captured is None
    else:
        np.testing.assert_array_equal(captured, [.2, -.1, .3])


def test_fault_logs_last_request_timing_once_without_changing_stop():
    server = server_without_node()
    server._logger = Mock()
    server._native_fault = None
    server._native_request_timing = {'sequence': 7, 'generation': 3, 'callback_ms': 12.0}
    server._activation_mgr = SimpleNamespace(state=SimpleNamespace(active=True))
    server._fail_native('Inference deadline exceeded', notify=False)
    server._fail_native('Inference deadline exceeded', notify=False)
    assert not server._activation_mgr.state.active
    assert server._pending_result is None
    messages = [call.args[0] for call in server._logger.error.call_args_list]
    assert len(messages) == 2
    assert messages[1].startswith('NATIVE_PYTHON_FAULT ')
    assert '"sequence":7' in messages[1]


def test_history_commits_once_after_confirmed_publication(monkeypatch):
    advance = Mock()
    monkeypatch.setattr(MarvinPolicyServer, '_advance_action_history', advance)
    server = server_without_node()
    server._native_diagnostics = SimpleNamespace(record=Mock())
    commit = replace(server._pending_result, kind=Kind.COMMIT, values=(.1,) * 12 + (.2,) * 12)
    server._commit(commit)
    server._commit(commit)
    advance.assert_called_once()
    np.testing.assert_array_equal(server._latest_policy_joint_command, [.1] * 12)
    np.testing.assert_array_equal(server._last_published_joint_command, [.2] * 12)
    assert server._pending_result is None
    server._native_diagnostics.record.assert_called_once_with('publication_commit', commit)


@pytest.mark.parametrize('change', [{'sequence': 6}, {'generation': 2}])
def test_old_commits_cannot_advance_history(monkeypatch, change):
    advance = Mock()
    monkeypatch.setattr(MarvinPolicyServer, '_advance_action_history', advance)
    server = server_without_node()
    server._native_diagnostics = SimpleNamespace(record=Mock())
    commit = replace(server._pending_result, kind=Kind.COMMIT, values=(.1,) * 24, **change)
    server._commit(commit)
    advance.assert_not_called()
    assert server._pending_result is not None
    server._native_diagnostics.record.assert_not_called()


@pytest.mark.parametrize('change', [{'values': (.1,) * 23}, {'trace_id': 43}])
def test_malformed_commit_is_rejected(change):
    server = server_without_node()
    commit = replace(server._pending_result, kind=Kind.COMMIT, values=(.1,) * 24)
    with pytest.raises(ValueError, match='Invalid native publication commit'):
        server._commit(replace(commit, **change))


def test_generation_change_discards_pending_result_and_resets_inference():
    server = server_without_node()
    server._observe_generation(4)
    assert server._pending_result is None
    assert not server._history_requested
    assert server._latest_estimated_velocity is None
    np.testing.assert_array_equal(server.action, np.zeros(12))
    for component in (server._policy_runner, server._inference_schedule, server._velocity_estimator):
        component.reset.assert_called_once()
    server._observe_generation(3)
    assert server._native_generation == 4
    server._policy_runner.reset.assert_called_once()


@pytest.mark.parametrize('domain,localhost', [('0', '1'), ('184', '0'), ('', '1')])
def test_physical_or_network_domain_rejected_before_ros_init(monkeypatch, domain, localhost):
    monkeypatch.delenv('MARVIN_POLICY_NATIVE_PROFILE', raising=False)
    monkeypatch.setenv('ROS_DOMAIN_ID', domain)
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', localhost)
    with pytest.raises(ValueError, match='Native I/O'):
        NativePolicyServer()


@pytest.mark.parametrize('parameter,value,accepted', [
    ('inference_max_age_ms', 100.0, True), ('inference_max_age_ms', 1000.0, False),
    ('publish_period_ms', 10, False), ('joy_topic', '/other/joy', False),
    ('legacy', True, False), ('traced', False, False), ('limits', False, False),
    ('finite', False, False),
])
def test_physical_profile_checks_without_creating_ros_node(monkeypatch, parameter, value, accepted):
    monkeypatch.setenv('ROS_DOMAIN_ID', '0')
    server = object.__new__(NativePolicyServer)
    server._native_profile = 'marvin-v1'
    server.joint_names = MARVIN_JOINTS
    server._operator_permission = SimpleNamespace(required=True)
    server._native_topics = lambda: MARVIN_TOPICS
    server.get_namespace = lambda: '/'
    server.resolve_topic_name = lambda name: name
    params = dict(use_sim_time=False, inference_max_age_ms=100., publish_period_ms=5, joy_topic='/joy')
    params[parameter] = value
    server.get_parameter = lambda key: SimpleNamespace(value=params[key])
    server._safety_limits_enabled = value if parameter == 'limits' else True
    server._safety_limits_loaded = True
    server._safety_min_pos = np.full(12, -1.)
    server._safety_max_pos = np.full(12, np.inf if parameter == 'finite' else 1.)
    server._publish_legacy_commands = value if parameter == 'legacy' else False
    server._publish_traced_commands = value if parameter == 'traced' else True
    if accepted:
        server._validate_native_profile()
    else:
        with pytest.raises(ValueError):
            server._validate_native_profile()


def test_execution_release_requires_fresh_current_generation():
    import time
    from marvin_policy_server.native_protocol import Flags
    server = server_without_node()
    server._native_status_flags = Flags.OUTPUT_RELEASED
    server._native_status_generation = 3
    server._native_status_at = time.monotonic()
    assert server._execution_state()['output_released']
    server._native_status_generation = 2
    assert not server._execution_state()['output_released']
    server._native_status_generation = 3
    server._native_status_at -= 1
    assert not server._execution_state()['output_released']


def test_python_execution_release_waits_for_ramp_completion():
    from marvin_policy_server.activation_manager import ActivationManager, ActivationConfig
    server = object.__new__(MarvinPolicyServer)
    server._activation_mgr = ActivationManager(ActivationConfig())
    state = server._activation_mgr.state
    assert server._execution_state()['output_released']
    state.active = True
    assert not server._execution_state()['output_released']
    state.active = False
    state.deactivation_start_time = 10.
    assert not server._execution_state()['output_released']
    state.deactivation_start_time = None
    assert server._execution_state()['output_released']


@pytest.mark.parametrize('exit_code', [0, 1, -15])
def test_confirmed_child_exit_proves_release_even_with_stale_status(exit_code):
    server = server_without_node()
    server._native_child = SimpleNamespace(poll=lambda: exit_code)
    assert server._execution_state() == {'active': False, 'output_released': True}


def test_transport_fault_does_not_prove_release_while_child_lives():
    server = server_without_node()
    server._native_child = SimpleNamespace(poll=lambda: None)
    server._native_fault = 'socket closed'
    server._native_status_flags = 0
    server._native_status_generation = 3
    server._native_status_at = 0.
    assert not server._execution_state()['output_released']


def test_stop_confirms_reaped_child_without_using_failed_transport():
    server = server_without_node()
    server._native_child = SimpleNamespace(poll=lambda: 1)
    server._native_channel = SimpleNamespace(control=Mock())
    server._fail_native = Mock()
    response = server._set_active_cb(SimpleNamespace(data=False), SimpleNamespace())
    assert response.success
    server._native_channel.control.assert_not_called()
    server._fail_native.assert_called_once()


def test_preparation_failure_cannot_activate_native_output(monkeypatch):
    server = server_without_node()
    server._activation_mgr = SimpleNamespace(state=SimpleNamespace(active=False))
    server._native_channel = SimpleNamespace(control=Mock())
    def reject(_self, request, response):
        response.success, response.message = False, 'A fresh preparation is required'
        return response
    monkeypatch.setattr(MarvinPolicyServer, '_set_active_cb', reject)
    response = server._set_active_cb(SimpleNamespace(data=True), SimpleNamespace())
    assert not response.success
    server._native_channel.control.assert_not_called()


def test_native_activation_occurs_only_after_parent_validation(monkeypatch):
    from marvin_policy_server.native_protocol import Flags
    server = server_without_node()
    server._activation_mgr = SimpleNamespace(state=SimpleNamespace(active=False, activation_start_time=None))
    order = []
    def validate(_self, active):
        response = ExecutionOutcome()
        order.append('validated')
        response.success = True
        return response
    def native(packet):
        order.append(packet.kind)
        return Packet(Kind.ACK, generation=4, flags=Flags.SUCCESS, ros_ns=10_000_000_000)
    monkeypatch.setattr(MarvinPolicyServer, 'request_activation', validate)
    server._native_channel = SimpleNamespace(control=native)
    response = server._set_active_cb(SimpleNamespace(data=True), SimpleNamespace())
    assert response.success
    assert order == ['validated', Kind.ACTIVATE]
    assert server._activation_mgr.state.activation_start_time == 10.


def test_failed_native_activation_invalidates_local_active_state(monkeypatch):
    server = server_without_node()
    server._logger = Mock()
    server._native_fault = None
    server._activation_mgr = SimpleNamespace(state=SimpleNamespace(active=False))
    def validate(_self, active):
        response = ExecutionOutcome()
        server._activation_mgr.state.active = True
        response.success = True
        return response
    monkeypatch.setattr(MarvinPolicyServer, 'request_activation', validate)
    server._native_channel = SimpleNamespace(control=lambda packet: Packet(Kind.ACK), send=Mock())
    response = server._set_active_cb(SimpleNamespace(data=True), SimpleNamespace())
    assert not response.success
    assert not server._activation_mgr.state.active
    assert server._native_channel.send.call_args.args[0].kind == Kind.FAULT
