"""ROS callbacks only translate results from backend lifecycle operations."""
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

from marvin_policy_server.execution_outcome import ExecutionOutcome
from marvin_policy_server.marvin_policy_server import MarvinPolicyServer
from marvin_policy_server.native_server import NativePolicyServer


@pytest.mark.parametrize('backend', [MarvinPolicyServer, NativePolicyServer])
@pytest.mark.parametrize('success', [False, True])
def test_callbacks_preserve_backend_outcome(backend, success):
    server = object.__new__(backend)
    result = ExecutionOutcome(success, 'backend result')
    server.prepare_execution = Mock(return_value=result)
    server.request_activation = Mock(return_value=result)
    response = SimpleNamespace()
    assert server._prepare_execution(
        SimpleNamespace(generation='generation-1', initial_state_json='{}'), response
    ) is response
    server.prepare_execution.assert_called_once_with('generation-1', '{}')
    assert (response.success, response.message) == (success, 'backend result')
    assert server._set_active_cb(SimpleNamespace(data=False), response) is response
    server.request_activation.assert_called_once_with(False)
    assert (response.success, response.message) == (success, 'backend result')
