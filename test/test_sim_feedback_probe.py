"""Probe must preserve original computation and use read-only getters."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

SCRIPTS = Path(__file__).parents[1] / 'scripts'


def load_wrapper(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        'feedback_probe', SCRIPTS / 'prepare_sim_feedback_probe.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FEEDBACK_WRAPPER


def test_readonly_probe_keeps_compute_result_and_rate(monkeypatch, capsys):
    calls = []

    class Articulation:
        dof_names = ['joint']

        def __init__(self, path, reset_xform_properties):
            assert path == '/World/marvin_isaacsim_ros'
            assert reset_xform_properties is False

        def initialize(self):
            calls.append('initialize')

        def get_joint_positions(self):
            return np.array([0.2])

        def get_joint_velocities(self):
            return np.array([0.3])

    monkeypatch.setitem(
        sys.modules, 'isaacsim.core.prims', SimpleNamespace(SingleArticulation=Articulation)
    )
    scope = {'compute': lambda db: calls.append('compute') or 73, '_probe_json': json}
    exec(load_wrapper(monkeypatch), scope)
    db = SimpleNamespace(inputs=SimpleNamespace(probeSimulationTime=1.0))
    assert scope['compute'](db) == 73
    db.inputs.probeSimulationTime = 1.01
    assert scope['compute'](db) == 73
    assert calls == ['compute', 'initialize', 'compute']
    output = capsys.readouterr().out
    assert output.count('SIM_FEEDBACK_PROBE ') == 1
    assert json.loads(output.split(' ', 1)[1])['velocity'] == [0.3]


def test_probe_read_failure_does_not_replace_original_output(monkeypatch, capsys):
    scope = {'compute': lambda db: 42, '_probe_json': json}
    exec(load_wrapper(monkeypatch), scope)
    scope['_feedback_articulation'] = SimpleNamespace(dof_names=['joint'])
    assert scope['compute'](SimpleNamespace(inputs=SimpleNamespace(probeSimulationTime=1.0))) == 42
    assert 'SIM_FEEDBACK_PROBE_ERROR' in capsys.readouterr().out
