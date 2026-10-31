"""Explicit native deployment contracts. These checks do not authorize motion."""

MARVIN_JOINTS = tuple(f'{leg}_{axis}_joint'
                      for axis in ('hip', 'thigh', 'calf')
                      for leg in ('FL', 'RL', 'FR', 'RR'))
MARVIN_TOPICS = {
    'joint_states_topic': '/joint_states',
    'dynamic_joint_states_topic': '',
    'imu_topic': '/imu',
    'permission_topic': '/operator/motion_permitted',
    'command_topic': '/marvin_joint_controller/commands',
    'traced_command_topic': '/marvin_joint_controller/commands_traced',
    'trace_event_topic': '/trace/events',
    'metrics_topic': '/trace/metrics',
    'candidate_topic': '/execution/policy/commands',
}


def validate_environment(domain, localhost, profile):
    if localhost != '1':
        raise ValueError('Native I/O requires localhost-only discovery')
    if profile == 'test' and domain in {'10', '184'}:
        return
    if profile == 'marvin-v1' and domain == '0':
        return
    raise ValueError('Native I/O is not enabled for physical robot domains without the explicit marvin-v1 profile')


def validate_profile(domain, profile, namespace, use_sim_time, required, names, topics):
    """Validate resolved names, before the native helper creates publishers."""
    if profile == 'marvin-v1':
        if domain != '0' or namespace != '/' or use_sim_time or not required:
            raise ValueError('marvin-v1 requires domain 0, root namespace, wall time and operator permission')
        if tuple(names) != MARVIN_JOINTS:
            raise ValueError('marvin-v1 joint order must match the physical command controller')
        if topics != MARVIN_TOPICS:
            raise ValueError('marvin-v1 requires the reviewed physical topic names without remapping')
    elif profile == 'test' and domain == '10':
        if namespace != '/sim' or not use_sim_time:
            raise ValueError('Native simulation requires use_sim_time=true and namespace=/sim')
    elif profile == 'test' and domain == '184':
        if any(value and not value.startswith('/policy_probe/') for value in topics.values()):
            raise ValueError('Isolated domain requires /policy_probe/ topics')
    else:
        raise ValueError('Unknown native deployment profile')
