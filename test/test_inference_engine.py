"""Real CPU graph conversion and runner-contract comparisons; no ROS or hardware."""

import builtins
from types import SimpleNamespace
from unittest.mock import patch

from marvin_policy_server.concurrent_policy.runner import ConcurrentPolicyRunner
from marvin_policy_server.inference_engine import engine_identity, prepare_policy
from marvin_policy_server.policy_runner import PolicyRunner, PolicyRunnerConfig
import numpy as np
import pytest
import torch


class Estimator(torch.nn.Module):
    @torch.jit.export
    def actor_estimates(self, history):
        velocity = history[:, -1, :3]
        height = history[:, -1, :1].repeat(1, 4)
        contact = height.sigmoid()
        feet = history[:, -1, :3].unsqueeze(1).repeat(1, 4, 1)
        return velocity, height, contact, feet


class Combined(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.estimator = Estimator()
        self.actor = torch.nn.Linear(24, 2)

    def forward(self, observation, history):
        action, state, _ = self.inference_snapshot(observation, history)
        return action, state

    @torch.jit.export
    def inference_snapshot(self, observation, history):
        velocity, height, contact, feet = self.estimator.actor_estimates(history)
        observation = observation.clone()
        observation[:, :3] = velocity
        action = self.actor(torch.cat((observation, feet.flatten(start_dim=-2), height, contact), -1))
        return action, torch.cat((velocity, height, contact), -1), feet


def manifest():
    return SimpleNamespace(base_observation_count=4, actor_observation_count=24,
                           action_count=2, estimated_state_count=11,
                           history_length=2, raw_feature_count=3)


@pytest.mark.parametrize('engine', ['torchscript', 'onnxruntime'])
def test_combined_preserves_live_diagnostics_overrides_hold_and_reset(engine):
    source = torch.jit.script(Combined().eval())
    spec = manifest()
    model = prepare_policy(source, manifest=spec, observation_count=4, action_count=2, engine=engine)
    expected = ConcurrentPolicyRunner(source, spec, decimation=1)
    actual = ConcurrentPolicyRunner(model, spec, decimation=1)
    rng = np.random.default_rng(59)
    for overrides in ({}, {'estimated_velocity_override': np.ones(3)},
                      {'estimated_foot_positions_override': np.ones(12)},
                      {'estimated_foot_height_override': np.ones(4)},
                      {'estimated_contact_override': np.ones(4)},
                      dict(estimated_velocity_override=np.ones(3), estimated_contact_override=np.zeros(4))):
        observation, features = rng.normal(size=4), rng.normal(size=3)
        np.testing.assert_allclose(actual.step(observation, features, **overrides),
                                   expected.step(observation, features, **overrides), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(actual.latest_estimated_state, expected.latest_estimated_state, atol=1e-5)
        np.testing.assert_allclose(actual.latest_foot_positions, expected.latest_foot_positions, atol=1e-5)
        np.testing.assert_allclose(actual.latest_effective_velocity, expected.latest_effective_velocity, atol=1e-5)
        held = actual.current_action.copy()
        actual.step(None, infer=False)
        np.testing.assert_array_equal(actual.current_action, held)
        expected.step(None, infer=False)
    actual.reset()
    assert not actual.history.ready and actual.latest_estimated_state is None
    assert actual.latest_foot_positions is None
    np.testing.assert_array_equal(actual.current_action, np.zeros(2))
    assert engine_identity(model)['engine'] == engine


def test_explicit_torchscript_never_imports_onnx_and_invalid_name_fails():
    source = torch.nn.Linear(4, 2).eval()
    real_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in ('onnx', 'onnxruntime'):
            raise ImportError('dependency unavailable')
        return real_import(name, *args, **kwargs)
    with patch('builtins.__import__', side_effect=guarded):
        assert prepare_policy(source, manifest=None, observation_count=4, action_count=2,
                              engine='torchscript') is source
        with pytest.raises(RuntimeError, match='dependencies are missing'):
            prepare_policy(source, manifest=None, observation_count=4, action_count=2)
    with pytest.raises(ValueError, match='inference_engine'):
        prepare_policy(source, manifest=None, observation_count=4, action_count=2, engine='automatic')


def test_legacy_actor_default_onnx_and_input_output_refusals():
    source = torch.nn.Linear(4, 2).eval()
    model = prepare_policy(source, manifest=None, observation_count=4, action_count=2)
    runner = PolicyRunner(model, 2, PolicyRunnerConfig())
    values = np.array([1., -2., 3., -4.])
    np.testing.assert_allclose(runner.step(values), source(torch.tensor(values).float()).detach().numpy(), atol=1e-5)
    for invalid in (torch.zeros(2, 4), torch.full((1, 4), float('nan')), torch.zeros(1, 4).double()):
        with pytest.raises(ValueError, match='ONNX input'):
            model(invalid)
    with patch.object(model.session, 'run', return_value=[np.full((1, 2), np.nan, dtype=np.float32)]):
        with pytest.raises(ValueError, match='ONNX output'):
            model(torch.zeros(1, 4))


@pytest.mark.parametrize('snapshot', [False, True])
def test_combined_sources_without_split_operations(snapshot):
    source = torch.jit.script(Combined().eval())
    class ForwardOnly(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = source
        def forward(self, observation, history):
            return self.model(observation, history)
    class SnapshotOnly(ForwardOnly):
        @torch.jit.export
        def inference_snapshot(self, observation, history):
            return self.model.inference_snapshot(observation, history)
    selected = torch.jit.script(SnapshotOnly() if snapshot else ForwardOnly())
    model = prepare_policy(selected, manifest=manifest(), observation_count=4, action_count=2)
    assert hasattr(model, 'inference_snapshot') == snapshot
    assert not hasattr(model, 'actor') and not hasattr(model, 'estimator')
    observation, history = torch.randn(1, 4), torch.randn(1, 2, 3)
    for actual, expected in zip(model(observation, history), selected(observation, history)):
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_external_estimator_preserves_normalization_history_and_reset():
    from pathlib import Path
    from marvin_policy_server.velocity_estimator import VelocityEstimator
    path = Path(__file__).parents[1] / 'policy' / 'velocity_estimator.pt'
    original = VelocityEstimator(path, engine='torchscript')
    converted = VelocityEstimator(path, engine='onnxruntime')
    rng = np.random.default_rng(59)
    for _ in range(original.config.history_length + 3):
        frame = rng.normal(size=original.raw_feature_count)
        np.testing.assert_allclose(converted.predict(frame), original.predict(frame), rtol=1e-4, atol=1e-5)
        assert converted.history_ready == original.history_ready
    converted.reset()
    assert converted.history_count == 0 and not converted.history_ready
    assert converted.loaded_artifact == original.loaded_artifact


def test_conversion_parity_failure_is_fatal():
    from marvin_policy_server.inference_engine import _Graph
    with patch.object(_Graph, '__call__', return_value=torch.full((1, 2), 999.)):
        with pytest.raises(AssertionError):
            prepare_policy(torch.nn.Linear(4, 2).eval(), manifest=None, observation_count=4, action_count=2)
