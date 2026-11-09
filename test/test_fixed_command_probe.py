import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    'fixed_command_probe', Path(__file__).parents[1] / 'scripts/fixed_command_probe.py'
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def environment():
    env = {'collections': __import__('collections'), 'DELAY_STEPS': 6}
    exec(probe.HELPERS, env)
    clock = SimpleNamespace(value=0)
    env['_target_probe_time'] = SimpleNamespace(
        monotonic_ns=lambda: clock.value, time_ns=lambda: 10**18 + clock.value
    )
    db = SimpleNamespace(per_instance_state=SimpleNamespace())
    calls = []
    dc = SimpleNamespace(set_dof_position_target=lambda h, q: calls.append((h, q)) or 73)
    return env, clock, db, dc, calls


def test_records_six_callback_delay_and_preserves_setter_result():
    env, clock, db, dc, calls = environment()
    for index in range(20):
        clock.value = index * 5_000_000
        env['_target_probe_accept'](db, ['FL_calf_joint'], [index])
        target = index if index < 6 else index - 6
        assert env['_target_probe_set'](db, dc, 42, target, 'FL_calf_joint') == 73
    rows = db.per_instance_state._target_probe_rows
    assert len(calls) == len(rows) == 20
    assert all(r['shadow_matches'] for r in rows)
    assert all(
        r['setter_return_monotonic_ns'] - r['accepted_monotonic_ns'] == 30_000_000
        for r in rows[6:]
    )
    assert [r['input_sequence'] for r in rows[:7]] == [1, 2, 3, 4, 5, 6, 1]


def test_logging_failure_does_not_change_setter_and_memory_stays_bounded():
    env, clock, db, dc, calls = environment()
    env['_target_probe_json'] = SimpleNamespace(
        dumps=lambda _: (_ for _ in ()).throw(RuntimeError())
    )
    for index in range(205):
        env['_target_probe_accept'](db, ['FL_calf_joint'], [0])
        assert env['_target_probe_set'](db, dc, 42, 0, 'FL_calf_joint') == 73
    assert len(calls) == 205
    assert len(db.per_instance_state._target_probe_rows) == 5
    assert len(db.per_instance_state._target_probe_queue) == 6


def test_setter_failure_propagates_without_success_record():
    env, clock, db, dc, calls = environment()
    env['_target_probe_accept'](db, ['FL_calf_joint'], [0])
    dc.set_dof_position_target = lambda *_: (_ for _ in ()).throw(RuntimeError('setter'))
    with pytest.raises(RuntimeError, match='setter'):
        env['_target_probe_set'](db, dc, 42, 0, 'FL_calf_joint')
    assert not db.per_instance_state._target_probe_rows


def test_mismatch_is_recorded_not_corrected():
    env, clock, db, dc, calls = environment()
    env['_target_probe_accept'](db, ['FL_calf_joint'], [1])
    env['_target_probe_set'](db, dc, 42, 2, 'FL_calf_joint')
    assert calls == [(42, 2)]
    assert db.per_instance_state._target_probe_rows[0]['shadow_matches'] is False


def test_unknown_script_rejected():
    with pytest.raises(ValueError):
        probe.instrument('def compute(db): return True')


def test_reset_discards_previous_activation_metadata():
    env, clock, db, dc, calls = environment()
    for _ in range(10):
        env['_target_probe_accept'](db, ['FL_calf_joint'], [1])
    env['_target_probe_reset'](db)
    env['_target_probe_accept'](db, ['FL_calf_joint'], [2])
    env['_target_probe_set'](db, dc, 42, 2, 'FL_calf_joint')
    row = db.per_instance_state._target_probe_rows[0]
    assert row['callback_sequence'] == row['input_sequence'] == 1
    assert row['shadow_matches']


def test_numpy_input_comparison_is_json_serializable(capsys):
    import json

    import numpy as np

    env, clock, db, dc, calls = environment()
    for _ in range(100):
        env['_target_probe_accept'](db, ['FL_calf_joint'], [np.float32(0.5)])
        env['_target_probe_set'](db, dc, 42, 0.5, 'FL_calf_joint')
    output = capsys.readouterr().out.strip()
    assert output.startswith('SIM_TARGET_PROBE ')
    rows = json.loads(output.split(' ', 1)[1])
    assert len(rows) == 100 and all(r['shadow_matches'] is True for r in rows)
    assert len(calls) == 100


def test_joint_selection_records_only_selected_joint():
    env, clock, db, dc, calls = environment()
    exec(probe.HELPERS.replace("name == 'FL_calf_joint'", "name == 'FL_thigh_joint'"), env)
    env['_target_probe_accept'](db, ['FL_thigh_joint', 'FL_calf_joint'], [0.2, 0.5])
    for name, target in [('FL_thigh_joint', 0.2), ('FL_calf_joint', 0.5)]:
        assert env['_target_probe_set'](db, dc, 42, target, name) == 73
    assert len(calls) == 2
    assert [r['joint'] for r in db.per_instance_state._target_probe_rows] == ['FL_thigh_joint']


@pytest.mark.parametrize('joint', ['trunk', 'FL_hip_backlash_joint', "FL_calf_joint'; bad"])
def test_invalid_joint_selection(joint):
    with pytest.raises(ValueError, match='12 motor joints'):
        probe.instrument('', joint)
