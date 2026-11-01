"""Prepare an opt-in systemd drop-in. Never install it or control a service."""

import argparse
from pathlib import Path
import re

DROP_IN_NAME = 'zzzzz-native-policy.conf'


def deployment_path(value):
    path = Path(value)
    if (not re.fullmatch(r'/home/mikael/Development/[A-Za-z0-9_./-]+', str(path))
            or '..' in path.parts or '%' in str(path)):
        raise ValueError('Use an absolute deployment path under /home/mikael/Development without shell or systemd substitutions')
    return path


def render(setup, runner):
    setup, runner = deployment_path(setup), deployment_path(runner)
    return f'''# Prepared only. Installation requires supervised physical acceptance.
# Remove only this drop-in to restore the preceding Python deployment.
[Service]
Environment=MARVIN_POLICY_IO_BACKEND=native
Environment=MARVIN_POLICY_NATIVE_PROFILE=marvin-v1
Environment=MARVIN_POLICY_ASYNC_INFERENCE=false
Environment=MARVIN_POLICY_COMPACT_ESTIMATOR=true
Environment=MARVIN_POLICY_LEAN_QOS_EVENTS=true
Environment=MARVIN_POLICY_INFERENCE_MAX_AGE_MS=100.0
ExecStart=
ExecStart=/usr/bin/env ROBOT_CONTROL_WORKSPACE_SETUP={setup} ROBOT_RUNTIME_LOCALHOST_ONLY=1 ROBOT_RUNTIME_OPERATOR_INPUT_REQUIRED=true ROS_DOMAIN_ID=0 /bin/bash {runner}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setup', required=True, type=Path)
    parser.add_argument('--runner', required=True, type=Path)
    parser.add_argument('--output-directory', required=True, type=Path)
    args = parser.parse_args()
    text = render(args.setup, args.runner)
    if not args.setup.is_file() or not args.runner.is_file():
        parser.error('Both setup and the existing runtime runner must exist')
    if args.setup.name != 'setup.bash' or args.setup.parent.name != 'install':
        parser.error('Use the staged colcon workspace install/setup.bash')
    prefix = args.setup.parent / 'marvin_policy_server'
    helper = prefix / 'lib/marvin_policy_server/marvin_policy_io'
    if not helper.is_file():
        parser.error('The staged native helper is missing')
    output_dir = deployment_path(args.output_directory)
    # Do not permit this preparation tool to write into an active configuration directory.
    if output_dir.resolve() != output_dir or not output_dir.is_relative_to(args.setup.parent.parent):
        parser.error('Output must be a real directory inside the staged workspace')
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / DROP_IN_NAME
    with output.open('x') as stream:
        stream.write(text)
    print(f'Prepared {output}. No service configuration changed. No service started.')


if __name__ == '__main__':
    main()
