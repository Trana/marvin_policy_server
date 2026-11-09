from marvin_policy_server.inference_schedule import InferenceSchedule
from marvin_policy_server.policy_cadence import PolicyCadence
import pytest


def config(**changes):
    return dict(sim={'dt': 0.005}, decimation=4, **changes)


@pytest.mark.parametrize(
    'ticks,expected', [([4, 5], [0, 20, 45, 65, 90, 110]), ([6], [0, 30, 60, 90, 120, 150])]
)
def test_saved_policy_selects_schedule(ticks, expected):
    cadence = PolicyCadence.resolve(config(policy_interval_ticks=ticks))
    cadence.validate_backend(5_000_000)
    schedule = InferenceSchedule(cadence.periods_ns)
    actual = []
    for ms in range(151):
        if schedule.due(ms * 1_000_000):
            actual.append(ms)
            schedule.complete(ms * 1_000_000, ms * 1_000_000)
    assert actual[:6] == expected


@pytest.mark.parametrize('ticks', [[], [True], [0], [-1], [4.5], None, '4', [4] * 1025])
def test_invalid_pattern_never_falls_back(ticks):
    with pytest.raises(ValueError):
        PolicyCadence.resolve(config(policy_interval_ticks=ticks))


@pytest.mark.parametrize('dt', [None, True, 0, -1, float('nan'), float('inf'), 0.0000000001])
def test_invalid_physics_dt(dt):
    with pytest.raises(ValueError):
        PolicyCadence.resolve({'sim': {'dt': dt}, 'decimation': 4})


def test_missing_timing_and_backend_capabilities():
    with pytest.raises(ValueError):
        PolicyCadence.resolve({'sim': {'dt': 0.005}})
    cadence = PolicyCadence.resolve(config(policy_interval_ticks=[4, 5]))
    for options in ({'native': True}, {'delay_max_ns': 25_000_000}):
        with pytest.raises(ValueError):
            cadence.validate_backend(5_000_000, **options)
    with pytest.raises(ValueError):
        cadence.validate_backend(10_000_000)
    PolicyCadence.resolve(config()).validate_backend(5_000_000, native=True)


def test_export_agreement_and_digest():
    env = config(policy_interval_ticks=[4, 5])
    training = {
        'policy_rate_hz': 44.44444444444444,
        'policy_cadence': {
            'schema': 'marvin.policy-cadence.v1',
            'physics_dt_s': 0.005,
            'pattern_ticks': [4, 5],
            'mean_policy_hz': 44.44444444444444,
        },
    }
    a = PolicyCadence.resolve(env, training)
    assert (
        a.projection() == PolicyCadence.resolve(dict(env, unrelated='new artifact')).projection()
    )
    training['policy_cadence']['pattern_ticks'] = [4]
    with pytest.raises(ValueError, match='Export cadence'):
        PolicyCadence.resolve(env, training)
    with pytest.raises(ValueError, match='Export policy rate'):
        PolicyCadence.resolve(env, {'policy_rate_hz': 50})


def test_cyclic_jump_reset_and_no_burst():
    schedule = InferenceSchedule((20, 25))
    assert schedule.due(0)
    schedule.complete(0, 65)
    assert schedule.deadline_ns == 90
    assert schedule.skipped_slots == 3
    assert not schedule.due(65)
    assert schedule.due(1_000_000_000)
    schedule.complete(1_000_000_000, 1_000_000_000)
    assert schedule.deadline_ns > 1_000_000_000
    assert not schedule.due(1_000_000_000)
    assert schedule.due(0)
    schedule.complete(0, 0)
    assert schedule.deadline_ns == 20
    assert schedule.observed()['sample_count'] == 1
