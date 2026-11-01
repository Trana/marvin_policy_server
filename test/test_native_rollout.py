"""Preparation does not need root or service control."""

import pytest

from marvin_policy_server.native_rollout import render


def test_drop_in_retains_existing_runner_and_explicit_safety_profile():
    text = render('/home/mikael/Development/native-policy-rollout.test/install/setup.bash',
                  '/home/mikael/Development/policy-selection.test/scripts/run-motion-runtime')
    assert 'MARVIN_POLICY_IO_BACKEND=native' in text
    assert 'MARVIN_POLICY_NATIVE_PROFILE=marvin-v1' in text
    assert 'ROBOT_RUNTIME_LOCALHOST_ONLY=1 ROBOT_RUNTIME_OPERATOR_INPUT_REQUIRED=true ROS_DOMAIN_ID=0' in text
    assert '\nExecStart=\nExecStart=/usr/bin/env ' in text
    assert 'policy-selection.test/scripts/run-motion-runtime' in text
    assert '[Install]' not in text and 'ExecStartPost' not in text


@pytest.mark.parametrize('path', ['/tmp/test', '/home/mikael/Development/../bad',
                                 '/home/mikael/Development/test name',
                                 '/home/mikael/Development/%n',
                                 '/home/mikael/Development/x\nExecStart=bad',
                                 '/home/mikael/Development/$EVIL', 'relative'])
def test_unsafe_systemd_paths_rejected(path):
    with pytest.raises(ValueError):
        render(path, '/home/mikael/Development/valid/runner')
