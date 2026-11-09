"""Identical offline admission matrix for the Python and C++ deployment guards."""

from pathlib import Path
import subprocess

from marvin_policy_server.native_profile import MARVIN_JOINTS
from marvin_policy_server.native_profile import MARVIN_TOPICS
from marvin_policy_server.native_profile import validate_environment
from marvin_policy_server.native_profile import validate_profile
import pytest


@pytest.fixture(scope='module')
def checker(tmp_path_factory):
    binary = tmp_path_factory.mktemp('native-profile') / 'profile_check'
    source = Path(__file__).resolve().parents[1] / 'native_io/profile_check.cpp'
    subprocess.run(
        ['c++', '-std=c++17', '-Wall', '-Wextra', '-Werror', str(source), '-o', str(binary)],
        check=True,
    )
    return binary


CASES = [({}, True)]
CASES += [
    ({key: value}, False)
    for key, value in (
        ('domain', '184'),
        ('domain', '10'),
        ('domain', ''),
        ('localhost', '0'),
        ('localhost', ''),
        ('profile', 'test'),
        ('profile', 'marvin'),
        ('namespace', '/sim'),
        ('use_sim_time', True),
        ('required', False),
        ('names', tuple(reversed(MARVIN_JOINTS))),
    )
]
CASES += [
    ({'topics': dict(MARVIN_TOPICS, **{key: '/unexpected/topic'})}, False) for key in MARVIN_TOPICS
]
CASES += [
    (
        {
            'profile': 'test',
            'domain': '184',
            'topics': {key: '/policy_probe/' + key for key in MARVIN_TOPICS},
        },
        True,
    ),
    ({'profile': 'test', 'domain': '10', 'namespace': '/sim', 'use_sim_time': True}, True),
    ({'profile': 'test', 'domain': '10', 'namespace': '/sim'}, False),
]


@pytest.mark.parametrize('changes,accepted', CASES)
def test_python_cpp_admission_agree_without_ros(checker, changes, accepted):
    fields = {
        'domain': '0',
        'localhost': '1',
        'profile': 'marvin-v1',
        'namespace': '/',
        'use_sim_time': False,
        'required': True,
        'names': MARVIN_JOINTS,
        'topics': MARVIN_TOPICS,
    }
    fields.update(changes)
    try:
        validate_environment(fields['domain'], fields['localhost'], fields['profile'])
        validate_profile(**{key: value for key, value in fields.items() if key != 'localhost'})
        python_accepted = True
    except ValueError:
        python_accepted = False
    args = [fields[key] for key in ('domain', 'localhost', 'profile', 'namespace')]
    args += [str(fields[key]).lower() for key in ('use_sim_time', 'required')]
    args += [fields['topics'][key] for key in MARVIN_TOPICS] + list(fields['names'])
    result = subprocess.run([str(checker), *args], capture_output=True, text=True)
    assert python_accepted == accepted
    assert (result.returncode == 0) == accepted, result.stderr
