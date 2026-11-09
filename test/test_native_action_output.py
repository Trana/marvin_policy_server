"""Compare C++ high-rate mappings against the existing Python mapping."""

from pathlib import Path
import shutil
import subprocess

from marvin_policy_server.action_output_mapping import SmoothBoundedActionMapper
import numpy as np
import pytest


def test_native_mapping_matches_python(tmp_path):
    compiler = shutil.which('c++')
    if compiler is None:
        pytest.skip('C++ compiler unavailable')
    source = Path(__file__).parents[1] / 'native_io' / 'action_output_test.cpp'
    binary = tmp_path / 'action_output_test'
    subprocess.run(
        [
            compiler,
            '-std=c++17',
            '-Wall',
            '-Wextra',
            '-Wpedantic',
            '-Werror',
            '-O2',
            str(source),
            '-o',
            str(binary),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True)
    random = np.random.default_rng(42)
    lines, expected = [], []
    for smooth in (False, True):
        for factor in (0.0, 0.001, 0.25, 0.5, 0.999, 1.0):
            for _ in range(20):
                default = random.uniform(-1.0, 1.0, 12)
                lower = default - random.uniform(0.01, 2.0, 12)
                upper = default + random.uniform(0.01, 2.0, 12)
                action = random.uniform(-10.0, 10.0, 12)
                scale = float(random.uniform(0.01, 1.0))
                values = [int(smooth), scale, factor]
                for joint in zip(default, lower, upper, action):
                    values.extend(joint)
                lines.append(' '.join(str(value) for value in values))
                mapped = (
                    SmoothBoundedActionMapper(
                        default, lower, upper, local_action_scale=scale
                    ).map_action(action * factor)
                    if smooth
                    else default + action * factor * scale
                )
                expected.append(np.clip(mapped, lower, upper))
    result = subprocess.run(
        [str(binary), '--map'],
        input='\n'.join(lines) + '\n',
        text=True,
        capture_output=True,
        check=True,
    )
    actual = np.asarray([[float(v) for v in line.split()] for line in result.stdout.splitlines()])
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
