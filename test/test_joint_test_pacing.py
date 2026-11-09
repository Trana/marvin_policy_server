"""Deterministic pacing tests; no Isaac runtime, sleeping or robot connection."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'joint_test_pacing', Path(__file__).parents[1] / 'scripts/joint_test_pacing.py'
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
SimulationPacer = module.SimulationPacer


def pacer():
    return SimulationPacer(10.0, 100.0, max_lag_s=0.05, max_step_s=0.02)


def test_waits_for_simulation_without_accumulating_computation_drift():
    clock = pacer()
    wall = 100.0
    for tick in range(1, 1001):
        wall += 0.001  # simulated cost of app.update()
        wait = clock.wait_seconds(10.0 + tick * 0.005, wall)
        assert wait == pytest.approx(0.004, abs=1e-10)
        wall += wait
    assert wall == pytest.approx(105.0)


def test_small_lag_keeps_original_mapping():
    clock = pacer()
    assert clock.wait_seconds(10.005, 100.009) == 0
    assert clock.wait_seconds(10.010, 100.010) == pytest.approx(0)
    assert clock.wait_seconds(10.015, 100.011) == pytest.approx(0.004)


def test_no_physics_update_is_allowed_but_stall_is_bounded():
    clock = pacer()
    assert clock.wait_seconds(10.0, 100.01) == 0
    with pytest.raises(ValueError, match='fell behind'):
        clock.wait_seconds(10.0, 100.06)


@pytest.mark.parametrize(
    'sim,wall',
    [
        (9.99, 100.0),
        (10.0, 99.99),
        (10.03, 100.03),
        (10.005, 100.1),
        (float('nan'), 100.0),
        (10.0, float('inf')),
    ],
)
def test_invalid_clock_or_overrun_latches_failure(sim, wall):
    clock = pacer()
    with pytest.raises(ValueError):
        clock.wait_seconds(sim, wall)
    with pytest.raises(ValueError, match='new trial'):
        clock.wait_seconds(10.005, 100.005)


@pytest.mark.parametrize('lag,step', [(0, 0.02), (0.05, -1), (float('nan'), 0.02)])
def test_invalid_configuration(lag, step):
    with pytest.raises(ValueError):
        SimulationPacer(0.0, 0.0, max_lag_s=lag, max_step_s=step)
