"""Unit tests for the concurrent policy deployment adapter."""

import json
from pathlib import Path

from marvin_policy_server.concurrent_policy.history import (
    EstimatorFeatureHistory,
)
from marvin_policy_server.concurrent_policy.manifest import (
    load_concurrent_policy_manifest,
)
from marvin_policy_server.concurrent_policy.runner import (
    ConcurrentPolicyRunner,
)

import numpy as np

import torch


class _Policy:
    def __call__(self, observation, history):
        action = observation[:, :2] + history[:, -1, :2]
        estimated_state = torch.arange(11, dtype=torch.float32).unsqueeze(0)
        return action, estimated_state


class _Estimator:
    def actor_estimates(self, history):
        batch = history.shape[0]
        velocity = torch.tensor([[0.4, -0.2, 0.1]], dtype=torch.float32).repeat(batch, 1)
        height = torch.zeros((batch, 4), dtype=torch.float32)
        contact = torch.full((batch, 4), 0.5, dtype=torch.float32)
        foot_positions = torch.zeros((batch, 4, 3), dtype=torch.float32)
        return velocity, height, contact, foot_positions


class _Actor:
    def __call__(self, observation):
        return observation[:, :2]


class _SplitPolicy(_Policy):
    estimator = _Estimator()
    actor = _Actor()


class _AuxActor:
    def __call__(self, observation):
        return torch.stack(
            (
                observation[:, 4],
                observation[:, 16] + observation[:, 20],
            ),
            dim=-1,
        )


class _AuxSplitPolicy(_Policy):
    estimator = _Estimator()
    actor = _AuxActor()


def _manifest(directory: Path):
    policy_path = directory / 'combined_policy.pt'
    policy_path.write_bytes(b'test')
    (directory / 'metadata.json').write_text(
        json.dumps(
            {
                'schema': 'marvin.concurrent_policy_export.v1',
                'artifacts': {'combined_policy': 'combined_policy.pt'},
                'dimensions': {
                    'base_policy_observation': 4,
                    'actor_observation': 24,
                    'raw_estimator_feature': 3,
                    'estimator_history': 2,
                    'estimated_state': 11,
                    'joint_action': 2,
                },
            }
        ),
        encoding='utf-8',
    )
    return policy_path, load_concurrent_policy_manifest(policy_path)


def test_history_repeats_first_frame_and_resets():
    """The first received frame fills warmup history and reset clears it."""
    history = EstimatorFeatureHistory(3, 2)
    values = history.push(np.array([1.0, 2.0]))
    np.testing.assert_allclose(values, [[1.0, 2.0]] * 3)
    assert not history.ready

    history.push(np.array([3.0, 4.0]))
    history.push(np.array([5.0, 6.0]))
    assert history.ready
    history.reset()
    assert history.count == 0
    assert not history.ready
    np.testing.assert_allclose(history.values, 0.0)


def test_manifest_and_runner_preserve_decimation(tmp_path: Path):
    """The runner infers only on decimated ticks and holds its action."""
    _, manifest = _manifest(tmp_path)
    assert manifest is not None
    runner = ConcurrentPolicyRunner(_Policy(), manifest, decimation=2)

    first = runner.step(
        np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.5, 1.5, 2.5])
    )
    held = runner.step(None)
    second = runner.step(
        np.array([2.0, 3.0, 4.0, 5.0]), np.array([1.0, 2.0, 3.0])
    )

    np.testing.assert_allclose(first, [1.5, 3.5])
    np.testing.assert_allclose(held, first)
    np.testing.assert_allclose(second, [3.0, 5.0])
    np.testing.assert_allclose(runner.latest_estimated_state, np.arange(11))
    assert runner.history.count == 2


def test_runner_can_replace_only_estimated_velocity(tmp_path: Path):
    """Diagnostic override keeps the raw estimate visible but changes actor input."""
    _, manifest = _manifest(tmp_path)
    assert manifest is not None
    runner = ConcurrentPolicyRunner(_SplitPolicy(), manifest, decimation=1)

    action = runner.step(
        np.array([9.0, 8.0, 7.0, 6.0]),
        np.array([0.5, 1.5, 2.5]),
        estimated_velocity_override=np.array([0.0, 0.0, 0.0]),
    )

    np.testing.assert_allclose(action, [0.0, 0.0])
    np.testing.assert_allclose(runner.latest_estimated_state[:3], [0.4, -0.2, 0.1])
    np.testing.assert_allclose(runner.latest_effective_velocity, [0.0, 0.0, 0.0])


def test_runner_can_replace_estimator_auxiliary_actor_inputs(tmp_path: Path):
    """Override FK, height, and contact while preserving raw estimates."""
    _, manifest = _manifest(tmp_path)
    assert manifest is not None
    runner = ConcurrentPolicyRunner(
        _AuxSplitPolicy(), manifest, decimation=1
    )

    action = runner.step(
        np.array([9.0, 8.0, 7.0, 6.0]),
        np.array([0.5, 1.5, 2.5]),
        estimated_foot_positions_override=np.full(12, 0.25),
        estimated_foot_height_override=np.full(4, 0.3),
        estimated_contact_override=np.full(4, 0.4),
    )

    np.testing.assert_allclose(action, [0.25, 0.7])
    np.testing.assert_allclose(
        runner.latest_estimated_state,
        [0.4, -0.2, 0.1, 0.0, 0.0, 0.0, 0.0,
         0.5, 0.5, 0.5, 0.5],
    )
    np.testing.assert_allclose(runner.latest_foot_positions, np.zeros(12))
    np.testing.assert_allclose(runner.current_foot_positions(), np.zeros(12))
