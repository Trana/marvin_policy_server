"""Offline PACE data validation; no robot, ROS, or motion access."""
import importlib.util
import hashlib
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('pace_export', Path(__file__).parents[1] / 'scripts/export_pace_capture.py')
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def fixture():
    commands = [(100, [1., 2.]), (200, [3., 4.]), (300, [5., 6.])]
    feedback = [(100, ['b', 'a'], [20., 10.]), (200, ['b', 'a'], [40., 30.]),
                (300, ['b', 'a'], [60., 50.])]
    return commands, feedback


def test_hold_interpolation_and_joint_order():
    result = export.resample(*fixture(), ['a', 'b'], 100, 300, 50, 100)
    assert result['des_dof_pos'] == [[1., 2.], [1., 2.], [3., 4.], [3., 4.]]
    assert result['dof_pos'] == [[10., 20.], [20., 30.], [30., 40.], [40., 50.]]
    assert result['time'] == [0., 50e-9, 100e-9, 150e-9]


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'regression', 'nan', 'shape', 'mapping'])
def test_reject_bad_stream(fault):
    commands, feedback = fixture()
    if fault == 'missing':
        commands = []
    elif fault == 'duplicate':
        commands[1] = commands[0]
    elif fault == 'regression':
        commands.reverse()
    elif fault == 'nan':
        commands[0][1][0] = float('nan')
    elif fault == 'shape':
        commands[0][1].pop()
    else:
        feedback[0][1][0] = 'unknown'
    with pytest.raises(ValueError):
        export.resample(commands, feedback, ['a', 'b'], 100, 300, 50, 100)


@pytest.mark.parametrize('start,end,dt,gap', [(90, 300, 50, 100), (100, 301, 50, 100),
    (100, 300, 50, 99), (100, 300, 0, 100), (100, 300, 500, 100)])
def test_reject_bad_window(start, end, dt, gap):
    with pytest.raises(ValueError):
        export.resample(*fixture(), ['a', 'b'], start, end, dt, gap)


def make_receipt(root):
    (root / 'data').write_bytes(b'recorded')
    receipt = {'schema': 'reference-complete.v1', 'files': {
        'data': {'sha256': hashlib.sha256(b'recorded').hexdigest(), 'size': 8}}}
    (root / 'complete.json').write_text(json.dumps(receipt))
    return receipt


def test_capture_integrity(tmp_path):
    receipt = make_receipt(tmp_path)
    assert export.verify_capture(tmp_path) == receipt
    (tmp_path / 'data').write_bytes(b'modified')
    with pytest.raises(ValueError, match='checksum'):
        export.verify_capture(tmp_path)


@pytest.mark.parametrize('fault', ['extra', 'missing', 'escape'])
def test_capture_inventory(tmp_path, fault):
    receipt = make_receipt(tmp_path)
    if fault == 'extra':
        (tmp_path / 'extra.db3').write_bytes(b'')
    elif fault == 'missing':
        (tmp_path / 'data').unlink()
    else:
        receipt['files']['../outside'] = receipt['files'].pop('data')
        (tmp_path / 'complete.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        export.verify_capture(tmp_path)
