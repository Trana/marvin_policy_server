"""Run with PyTorch installed; no ROS dependency."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
torch = pytest.importorskip('torch')
SCRIPT = Path(__file__).parents[1] / 'scripts/pace_json_to_tensor.py'


def test_tensor_conversion_and_no_overwrite(tmp_path):
    source, output = tmp_path / 'data.json', tmp_path / 'data.pt'
    source.write_text(json.dumps(dict(time=[0., .02, .04],
                                     des_dof_pos=[[.1], [.2], [.3]], dof_pos=[[0.], [.1], [.2]])))
    command = [sys.executable, str(SCRIPT), str(source), str(output)]
    subprocess.run(command, check=True, capture_output=True)
    data = torch.load(output, weights_only=True)
    assert set(data) == {'time', 'des_dof_pos', 'dof_pos'}
    assert data['dof_pos'].shape == (3, 1)
    before = output.read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert output.read_bytes() == before


@pytest.mark.parametrize('times', [[0., .02, .07], [0., .02, .02], [0., float('nan'), .04]])
def test_invalid_grid(tmp_path, times):
    source, output = tmp_path / 'data.json', tmp_path / 'data.pt'
    source.write_text(json.dumps(dict(time=times, des_dof_pos=[[0.]]*3, dof_pos=[[0.]]*3)))
    assert subprocess.run([sys.executable, str(SCRIPT), str(source), str(output)], capture_output=True).returncode != 0
    assert not output.exists()
