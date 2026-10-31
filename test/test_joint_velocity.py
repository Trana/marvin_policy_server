import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'joint_velocity', Path(__file__).parents[1] / 'scripts/analyze_joint_velocity.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def rows(sign=1, speed=1):
    return [(1_000_000_000+i*20_000_000, ['a'], [speed*i*.02], [speed*sign])
            for i in range(151)]


@pytest.mark.parametrize('sign', [1, -1])
@pytest.mark.parametrize('speed', [1, -2])
def test_sign_detection(sign, speed):
    joint = module.analyze(rows(sign, speed))['joints']['a']
    assert joint['sufficient_motion_windows']
    assert joint['same_sign_fraction'] == (1 if sign == 1 else 0)
    expected = 'minus' if sign == 1 else 'plus'
    assert joint[f'derivative_{expected}_velocity_rmse'] < 1e-12


def test_no_motion_is_inconclusive():
    joint = module.analyze(rows(speed=0))['joints']['a']
    assert not joint['sufficient_motion_windows']
    assert joint['derivative_minus_velocity_rmse'] is None


def test_short_capture_and_gaps():
    sample = rows()
    result = module.analyze(sample[:30]+sample[80:])
    assert result['clock_gap_resets'] == 1
    assert not result['joints']['a']['sufficient_motion_windows']


def test_invalid_state_and_clock_rejected():
    for sample in [[rows()[0], rows()[0]], list(reversed(rows())),
                   [(1, ['a'], [0], [])], [(1, ['a'], [0], [float('nan')])]]:
        with pytest.raises(ValueError):
            module.analyze(sample)


def test_reordered_names_remain_named():
    sample = []
    for i, (stamp, _, pos, vel) in enumerate(rows()):
        names, positions, velocities = ['a', 'b'], [pos[0], -pos[0]], [vel[0], -vel[0]]
        if i % 2:
            names.reverse(); positions.reverse(); velocities.reverse()
        sample.append((stamp, names, positions, velocities))
    result = module.analyze(sample)
    for joint in result['joints'].values():
        assert joint['same_sign_fraction'] == 1
