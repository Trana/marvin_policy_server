"""ROS callbacks only translate results from backend lifecycle operations."""

from types import SimpleNamespace
from unittest.mock import Mock

from marvin_policy_server.execution_outcome import ExecutionOutcome
from marvin_policy_server.marvin_policy_server import MarvinPolicyServer
from marvin_policy_server.native_server import NativePolicyServer
import pytest


@pytest.mark.parametrize('backend', [MarvinPolicyServer, NativePolicyServer])
@pytest.mark.parametrize('success', [False, True])
def test_callbacks_preserve_backend_outcome(backend, success):
    server = object.__new__(backend)
    result = ExecutionOutcome(success, 'backend result')
    server.prepare_execution = Mock(return_value=result)
    server.request_activation = Mock(return_value=result)
    response = SimpleNamespace()
    assert (
        server._prepare_execution(
            SimpleNamespace(generation='generation-1', initial_state_json='{}'), response
        )
        is response
    )
    server.prepare_execution.assert_called_once_with('generation-1', '{}')
    assert (response.success, response.message) == (success, 'backend result')
    assert server._set_active_cb(SimpleNamespace(data=False), response) is response
    server.request_activation.assert_called_once_with(False)
    assert (response.success, response.message) == (success, 'backend result')


def test_reload_pending_blocks_prepare_activation_and_debugging():
    server = object.__new__(MarvinPolicyServer)
    server._reload_adapter = SimpleNamespace(manager=SimpleNamespace(blocked=True))
    assert not server.prepare_execution('generation', '{}').success
    assert not server.request_activation(True).success
    assert server._apply_policy_input_lock('command', False)[0] is False


@pytest.mark.parametrize('body', ['[]', 'null', '"invalid"'])
def test_reload_service_rejects_non_object_json(body):
    import json

    from marvin_policy_server.policy_reload import PolicyReload
    from marvin_policy_server.reload_adapter import ReloadAdapter

    adapter = object.__new__(ReloadAdapter)
    adapter.manager = object.__new__(PolicyReload)
    response = adapter.handle(SimpleNamespace(request_json=body), SimpleNamespace())
    assert json.loads(response.response_json)['state'] == 'failed'
    assert 'object' in json.loads(response.response_json)['message']


def test_inactive_install_rotates_session_and_clears_history_without_commands():
    from pathlib import Path

    from marvin_policy_server.policy_control import PolicyControl
    from marvin_policy_server.reload_adapter import ReloadAdapter
    import numpy as np

    node = SimpleNamespace(
        _native_io=True,
        _execution_gateway=True,
        _execution_state=lambda: {'output_released': True},
        _activation_mgr=SimpleNamespace(is_active=lambda: False),
        joint_names=[f'joint{i}' for i in range(12)],
        _loaded_artifacts={'policy.pt': {'sha256': 'old'}},
        _policy_control=PolicyControl(lambda: {}, lambda *args: None),
        _inference_schedule=SimpleNamespace(reset=Mock()),
        _joint_position_filter=SimpleNamespace(reset=Mock()),
        _observation_status_payload=lambda: {},
        _apply_policy_input_lock=lambda *args: None,
    )
    old_session = node._policy_control.session_id
    adapter = object.__new__(ReloadAdapter)
    adapter.node = node
    adapter.loaded_context = None
    adapter.observation_template = SimpleNamespace(action_history=np.zeros((2, 12)))
    context = SimpleNamespace(
        policy=object(),
        runner=object(),
        manifest=None,
        estimator=None,
        artifacts={'policy.pt': {'sha256': 'new'}},
        path=Path('/policies/bundle/policy.pt'),
    )
    result = adapter.install(context)
    assert node._policy_control.session_id != old_session
    assert result['old_session_id'] == old_session and result['new_artifacts'] == context.artifacts
    assert node._policy_runner is context.runner and node.policy is context.policy
    assert not node._joint_command_output_locked and node._locked_joint_command is None
    assert node._pending_result is None and not node._history_requested
    assert node._execution_preparation.generation is None
    assert not np.shares_memory(
        node._obs_state.action_history, adapter.observation_template.action_history
    )
