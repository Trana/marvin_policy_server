import time
from types import SimpleNamespace as NS

from marvin_policy_server.async_policy import RunnerTask
from marvin_policy_server.inference_worker import InferenceResult
import numpy as np
import pytest


def test_runner_task_copies_action_and_state():
    runner = NS(
        current_action=np.ones(2),
        latest_estimated_state=np.arange(11.0),
        latest_foot_positions=np.arange(12.0),
        latest_result_monotonic=123.0,
        history=NS(count=1, values=np.ones((2, 3))),
    )
    runner.step = lambda *a, **kw: runner.current_action
    result, state = RunnerTask(runner)(((np.zeros(3),), {'infer': True}))
    runner.current_action[:] = 9
    runner.latest_estimated_state[:] = 9
    runner.latest_foot_positions[:] = 9
    runner.history.values[:] = 9
    np.testing.assert_array_equal(result, np.ones(2))
    np.testing.assert_array_equal(state['latest_estimated_state'], np.arange(11.0))
    np.testing.assert_array_equal(state['latest_foot_positions'], np.arange(12.0))
    assert state['latest_result_monotonic'] == 123.0
    np.testing.assert_array_equal(state['history'].values, np.ones((2, 3)))


def server():
    pytest.importorskip('rclpy')
    from marvin_policy_server.inference_schedule import InferenceSchedule
    from marvin_policy_server.marvin_policy_server import MarvinPolicyServer

    node = object.__new__(MarvinPolicyServer)
    node._async_pending_ns = None
    node._async_fault = None
    node._async_has_action = False
    node._inference_max_age_ns = 100_000_000
    node._timing = None
    node._policy_runner = NS()
    node.action = np.zeros(2)
    node._inference_schedule = InferenceSchedule(20_000_000)
    node._publish_trace_event = lambda *a, **kw: None
    node._logger = NS(error=lambda message: None)
    node._async_worker = NS(invalidate=lambda: None, poll=lambda: None)
    return node


def test_node_accepts_result_and_copies_estimator_state():
    node = server()
    node._async_worker.poll = lambda: InferenceResult(
        0,
        (1_000_000_000, 12),
        (np.ones(2), {'latest_estimated_state': np.arange(11.0)}),
        None,
        10,
        10_000_010,
    )
    assert node._poll_async_inference(1_020_000_000) == 12
    assert node._async_has_action
    np.testing.assert_array_equal(node._latest_estimated_velocity, [0, 1, 2])
    assert node._inference_schedule.deadline_ns is None  # Submission owns scheduling.


@pytest.mark.parametrize(
    'observation,now,duration,error',
    [
        (1_000_000_000, 1_010_000_000, 110_000_000, None),
        (1_000_000_000, 1_110_000_000, 10_000_000, None),
        (1_000_000_000, 900_000_000, 10_000_000, None),
        (1_000_000_000, 1_010_000_000, 10_000_000, ValueError('bad model')),
    ],
)
def test_expired_rewound_or_failed_result_never_becomes_an_action(
    observation, now, duration, error
):
    node = server()
    node.action = np.zeros(2)
    node._async_worker.poll = lambda: InferenceResult(
        0, (observation, 1), (np.ones(2), {}), error, 0, duration
    )
    assert node._poll_async_inference(now) is None
    assert node._async_fault
    assert not node._async_has_action
    np.testing.assert_array_equal(node.action, [0, 0])


def test_timeout_without_result_requires_explicit_reactivation():
    node = server()
    node._async_pending_ns = time.monotonic_ns() - 200_000_000
    node._poll_async_inference(1_000_000_000)
    assert node._async_fault == 'Inference deadline exceeded'
    node._invalidate_async_inference()
    assert node._async_fault  # Stale inputs/override changes must not clear a fault.


@pytest.mark.parametrize('action', [np.array([np.nan, 1.0]), np.ones(3)])
def test_invalid_action_does_not_replace_last_safe_action(action):
    node = server()
    node._async_worker.poll = lambda: InferenceResult(0, (0, 1), (action, {}), None, 0, 10)
    assert node._poll_async_inference(10) is None
    np.testing.assert_array_equal(node.action, [0.0, 0.0])
    assert node._async_fault
