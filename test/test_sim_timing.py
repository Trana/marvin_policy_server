import json

from marvin_policy_server.sim_timing import SimActionRelease
from marvin_policy_server.sim_timing import simulation_gate
import numpy as np
import pytest

PROFILE = {'revision': 'marvin-timing-v1', 'physics_dt': 0.005, 'inference_delay_ms': [15, 15]}


def gate():
    return SimActionRelease(PROFILE, np.zeros(2))


def test_early_completion_holds_until_observation_deadline():
    g = gate()
    g.submit([1, 2], [3, 4], 0, 5_000_000, 7)
    assert g.take(10_000_000) is None
    np.testing.assert_array_equal(g.last_target, [0, 0])
    assert g.take(15_000_000)[2] == 7
    assert g.take(20_000_000) is None
    assert g.last_release['release_ns'] == 15_000_000


def test_late_completion_does_not_add_another_delay():
    g = gate()
    g.submit([1, 2], [3, 4], 0, 25_000_000, 7)
    assert g.take(20_000_000) is None
    assert g.take(25_000_000) is not None


def test_clear_reset_and_clock_rewind_drop_pending():
    g = gate()
    g.observe(20, True)
    g.submit([1, 2], [3, 4], 0, 0, 7)
    assert g.observe(10, True)
    assert g.take(30_000_000) is None
    g.submit([1, 2], [3, 4], 0, 0, 7)
    g.clear()
    assert g.pending is None
    g.reset([8, 9])
    np.testing.assert_array_equal(g.last_target, [8, 9])


def test_identity_and_scene_session():
    g = gate()
    assert not g.readiness(0)[0]
    payload = {
        'profile': PROFILE,
        'session_id': 'a',
        'sim_time_ns': 0,
        'imu_acceleration_convention': 'specific_force',
    }
    assert g.identity(json.dumps(payload))
    assert g.readiness(50_000_000)[0]
    assert not g.readiness(400_000_000)[0]
    g.submit([1, 2], [3, 4], 0, 0, 7)
    payload['session_id'] = 'b'
    assert g.identity(json.dumps(payload))
    assert g.pending is None
    payload['profile'] = {}
    g.identity(json.dumps(payload))
    assert not g.readiness(0)[0]


def test_hardware_and_untimed_are_unchanged():
    assert simulation_gate(PROFILE, [0], use_sim_time=False, native_io=True) is None
    assert simulation_gate(None, [0], use_sim_time=True, native_io=True) is None
    with pytest.raises(ValueError, match='Python'):
        simulation_gate(PROFILE, [0], use_sim_time=True, native_io=True)


@pytest.mark.parametrize('convention', [None, 'kinematic_acceleration'])
def test_missing_or_incompatible_imu_contract_prevents_inference(convention):
    g = gate()
    g.submit([1, 2], [3, 4], 0, 0, 7)
    g.identity(
        json.dumps({
            'profile': PROFILE,
            'session_id': 'a',
            'sim_time_ns': 0,
            'imu_acceleration_convention': convention,
        })
    )
    assert not g.readiness(0)[0]
    assert g.pending is None
    assert 'specific-force' in g.error


def test_cadence_warning_is_independent_of_delay_and_rate_limited():
    g = gate()
    for tick in range(101):
        now = tick * 40_000_000
        g.submit([0, 0], [0, 0], now, now, tick)
        g.take(now + 15_000_000)
    assert '25.0 Hz' in g.cadence_warning(now)
    assert g.cadence_warning(now + 1) is None
    g.reset([0, 0])
    assert g.cadence_warning(now + 10_000_000_000) is None
