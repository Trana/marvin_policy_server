"""The monitoring helper must reject physical or network-visible ROS domains."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('domain,localhost', [('0', '1'), ('10', '1'), ('184', '0'), ('', '')])
@pytest.mark.parametrize('mode', ['server', 'fresh', 'discovery'])
def test_probe_rejects_non_private_environment(domain, localhost, mode):
    script = Path(__file__).resolve().parents[1] / 'native_io/monitoring_probe.py'
    result = subprocess.run(
        [sys.executable, str(script), mode],
        env=dict(os.environ, ROS_DOMAIN_ID=domain, ROS_LOCALHOST_ONLY=localhost),
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode != 0
    assert 'requires localhost-only domain 184' in result.stderr


@pytest.mark.parametrize(
    'extra',
    [
        ['--backend', 'python'],
        ['--backend', 'native', '--check-parent-death'],
        ['--backend', 'native', '--monitor', 'fresh', '--monitor-during-stop'],
    ],
)
def test_inactive_check_rejects_activation_test_options(extra):
    script = Path(__file__).resolve().parents[1] / 'native_io/check_performance.py'
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            '--inactive-only',
            '--binary',
            '/unused',
            '--model',
            '/unused',
            '--env',
            '/unused',
            '--log',
            '/unused',
            *extra,
        ],
        env=dict(os.environ, ROS_DOMAIN_ID='184', ROS_LOCALHOST_ONLY='1'),
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode != 0
    assert '--inactive-only requires native backend' in result.stderr
