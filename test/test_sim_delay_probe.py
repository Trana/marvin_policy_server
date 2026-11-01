import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    'probe', Path(__file__).parents[1] / 'scripts/prepare_sim_delay_probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


SOURCE = '''import collections
def setup(db):
    db.internal_state.buffer = collections.deque()
def compute(db):
    current_val = db.inputs.positionCommand
    if current_val:
        db.internal_state.buffer.append(list(current_val))
        delay_steps = 6
        print(len(db.internal_state.buffer))
        if len(db.internal_state.buffer) > delay_steps:
            db.outputs.delayed_position_command = db.internal_state.buffer.popleft()
        else:
            db.outputs.delayed_position_command = list(current_val)
    return True
'''


def test_wrapper_keeps_targets_and_measures_buffer_age(capsys):
    env = {}
    exec(probe.instrument(SOURCE), env)
    clock = SimpleNamespace(monotonic=lambda: clock.value, value=0.)
    env['_probe_time'] = clock
    db = SimpleNamespace(internal_state=SimpleNamespace(), inputs=SimpleNamespace(), outputs=SimpleNamespace())
    env['setup'](db)
    actual = []
    for index in range(30):
        clock.value = index * .1
        db.inputs.positionCommand = [index]
        db.inputs.probeSimulationTime = index * .005
        assert env['compute'](db)
        actual.append(db.outputs.delayed_position_command[0])
    assert actual == list(range(6)) + list(range(24))
    line = capsys.readouterr().out.strip()
    data = json.loads(line.removeprefix('SIM_DELAY_PROBE '))
    assert data['callback_hz'] == pytest.approx(10)
    assert data['buffer_age_mean_ms'] == pytest.approx(600)
    assert data['output_mismatches'] == 0
    assert data['passthrough_count'] == 6
    assert data['buffer_sim_age_mean_ms'] == pytest.approx(30)
    assert data['callback_sim_hz'] == pytest.approx(200)
    assert data['real_time_factor'] == pytest.approx(.05)
    assert data['window_start_monotonic_s'] == 0.
    assert data['window_end_monotonic_s'] == pytest.approx(2.9)
    assert data['window_start_sim_s'] == 0.
    assert data['window_end_sim_s'] == pytest.approx(.145)
    assert data['oldest_buffered_input_sim_s'] == pytest.approx(0.)


def test_unknown_source_refused():
    with pytest.raises(ValueError):
        probe.instrument(SOURCE.replace('delay_steps = 6', 'delay_steps = 4'))


def test_missing_clock_is_not_reported_as_zero_delay(capsys):
    env = {}
    exec(probe.instrument(SOURCE), env)
    db = SimpleNamespace(internal_state=SimpleNamespace(),
                         inputs=SimpleNamespace(positionCommand=[1.]), outputs=SimpleNamespace())
    env['setup'](db)
    for _ in range(30):
        env['compute'](db)
    data = json.loads(capsys.readouterr().out.strip().removeprefix('SIM_DELAY_PROBE '))
    assert data['buffer_sim_age_mean_ms'] is None
    assert data['invalid_sim_age_samples'] == 24
    assert data['output_mismatches'] == 0
    assert data['window_start_sim_s'] is None
    assert data['window_end_sim_s'] is None
    assert data['oldest_buffered_input_sim_s'] is None
