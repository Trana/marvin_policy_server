"""Execution experiments remain opt-in and can be scoped to a service."""

import importlib.util
from pathlib import Path

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch.utilities import perform_substitutions
import pytest


@pytest.mark.parametrize(
    'name,variable,default,override',
    [
        ('io_backend', 'MARVIN_POLICY_IO_BACKEND', 'python', 'native'),
        ('async_inference', 'MARVIN_POLICY_ASYNC_INFERENCE', 'false', 'true'),
        ('compact_estimator', 'MARVIN_POLICY_COMPACT_ESTIMATOR', 'false', 'true'),
        ('lean_qos_events', 'MARVIN_POLICY_LEAN_QOS_EVENTS', 'false', 'true'),
        ('inference_max_age_ms', 'MARVIN_POLICY_INFERENCE_MAX_AGE_MS', '100.0', '80.0'),
    ],
)
def test_execution_launch_environment(name, variable, default, override):
    path = Path(__file__).parents[1] / 'launch' / 'marvin_policy_server.launch.py'
    spec = importlib.util.spec_from_file_location('policy_launch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = {
        a.name: a
        for a in module.generate_launch_description().entities
        if isinstance(a, DeclareLaunchArgument)
    }
    context = LaunchContext()
    context.environment.pop(variable, None)
    assert perform_substitutions(context, args[name].default_value) == default
    context.environment[variable] = override
    assert perform_substitutions(context, args[name].default_value) == override


@pytest.mark.parametrize(
    'backend,requested,executable,effective',
    [
        ('python', 'true', 'marvin_policy_server', 'true'),
        ('python', 'false', 'marvin_policy_server', 'false'),
        ('native', 'true', 'marvin_policy_server_native', 'false'),
        ('native', 'false', 'marvin_policy_server_native', 'false'),
    ],
)
def test_native_backend_owns_scheduling(backend, requested, executable, effective):
    path = Path(__file__).parents[1] / 'launch' / 'marvin_policy_server.launch.py'
    spec = importlib.util.spec_from_file_location('policy_launch_selection', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    context = LaunchContext()
    context.launch_configurations.update(io_backend=backend, async_inference=requested)
    assert module.PolicyExecutable().perform(context) == executable
    assert module.PolicyAsyncSetting().perform(context) == effective
