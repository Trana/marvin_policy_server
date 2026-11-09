import json

from marvin_policy_server.execution_preparation import PolicyPreparation
import pytest


def payload():
    return json.dumps({
        'joint_names': list(range(12)),
        'position_unit': 'rad',
        'model': 'marvin',
        'sampled_at': 10.0,
        'joint_age': 0.0,
        'imu_age': 0.0,
        'positions': [0.0] * 12,
        'last_targets': [0.2] * 12,
    })


def test_prepare_cannot_start_active_policy_and_first_command_preserves_entry_target():
    now = [10.0]
    p = PolicyPreparation(range(12), clock=lambda: now[0])
    with pytest.raises(ValueError):
        p.prepare('epoch', payload(), released=False)
    p.prepare('epoch', payload(), released=True)
    assert p.command([1.0] * 12) is None
    p.start()
    assert p.command([1.0] * 12) == (0.2,) * 12
    now[0] = 12.0
    assert p.command([1.0] * 12) == (1.0,) * 12
    assert p.sequence == 2


def test_expired_prepare_cannot_activate_and_bad_state_is_rejected():
    now = [10.0]
    p = PolicyPreparation(range(12), clock=lambda: now[0])
    p.prepare('epoch', payload(), released=True)
    now[0] = 16.0
    with pytest.raises(ValueError):
        p.start()
    with pytest.raises(ValueError):
        p.prepare('new', payload(), released=True)
