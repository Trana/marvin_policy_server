"""Cache equivalence and invalidation without ROS or robot hardware."""

import numpy as np
import pytest

from marvin_policy_server.command_clip_cache import CommandClipCache


def test_identical_values_reuse_clipping_but_return_independent_lists(monkeypatch):
    cache = CommandClipCache()
    command, lower, upper = np.array([2., -.5]), np.array([-1., -1.]), np.ones(2)
    values, indices = cache.apply(command, lower, upper)
    assert values == [1., -.5]
    assert indices == (0,)
    values[0] = 999.
    monkeypatch.setattr(np, 'clip', lambda *args: pytest.fail('unexpected recalculation'))
    assert cache.apply(command.copy(), lower.copy(), upper.copy()) == ([1., -.5], (0,))


@pytest.mark.parametrize('changed', ['command', 'lower', 'upper'])
def test_in_place_changes_invalidate(changed):
    cache = CommandClipCache()
    arrays = dict(command=np.array([2., -2.]), lower=np.array([-1., -1.]), upper=np.ones(2))
    cache.apply(**arrays)
    arrays[changed][:] *= .5
    values, indices = cache.apply(**arrays)
    expected = np.clip(arrays['command'], arrays['lower'], arrays['upper'])
    np.testing.assert_array_equal(values, expected)
    assert indices == tuple(np.flatnonzero(np.abs(expected-arrays['command']) > 1e-9))


def test_repeated_and_changing_commands_match_reference():
    rng = np.random.default_rng(19)
    cache = CommandClipCache()
    lower, upper = -np.ones(12), np.ones(12)
    commands = [rng.normal(size=12) for _ in range(50)]
    commands += [np.array([np.nan, np.inf, -np.inf] + [0.]*9)]
    with np.errstate(invalid='ignore'):
        for command in commands:
            expected = np.clip(command, lower, upper)
            for _ in range(4):
                values, indices = cache.apply(command, lower, upper)
                np.testing.assert_array_equal(values, expected)
                assert indices == tuple(np.flatnonzero(np.abs(expected-command) > 1e-9))


def test_node_preserves_warnings_enable_flags_and_shape_fallback():
    pytest.importorskip('rclpy')
    from types import SimpleNamespace as NS
    from marvin_policy_server.marvin_policy_server import MarvinPolicyServer

    node = object.__new__(MarvinPolicyServer)
    node._safety_limits_enabled = node._safety_limits_loaded = True
    node._safety_min_pos, node._safety_max_pos = -np.ones(2), np.ones(2)
    node.joint_names = ['hip', 'knee']
    node._last_safety_clip_warn_time = 0.
    now = [1_000_000_000]
    warnings = []
    node.get_clock = lambda: NS(now=lambda: NS(nanoseconds=now[0]))
    node._logger = NS(warn=warnings.append)
    assert node._apply_safety_position_limits([2., 0.]) == [1., 0.]
    assert len(warnings) == 1
    now[0] += 500_000_000
    assert node._apply_safety_position_limits([2., 0.]) == [1., 0.]
    assert len(warnings) == 1
    now[0] += 500_000_000
    node._apply_safety_position_limits([2., 0.])
    assert len(warnings) == 2  # Cache hits still emit rate-limited warnings.
    for flag in ('_safety_limits_enabled', '_safety_limits_loaded'):
        setattr(node, flag, False)
        assert node._apply_safety_position_limits([2., 0.]) == [2., 0.]
        setattr(node, flag, True)
    node._safety_max_pos[0] = .25
    assert node._apply_safety_position_limits([2., 0.]) == [.25, 0.]
    assert node._apply_safety_position_limits([2.]) == [2.]
    assert 'shape mismatch' in warnings[-1]
