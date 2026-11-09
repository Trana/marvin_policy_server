#!/usr/bin/env python3
"""Bounded, read-only ROS bag and cached status capture. Never activates motion."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time

TARGET_DOMAINS = {'robot': '0', 'sim': '10', 'sim_joint_test': '231', 'robot_joint_test': '0'}
JOINT_TEST_TARGETS = {'sim_joint_test', 'robot_joint_test'}


def capture_progress(output, state, reason=None):
    """Inspect logs and file sizes only. Never open a live SQLite database."""
    log = output / 'recorder.log'
    text = log.read_text(errors='replace') if log.exists() else ''
    progress = {
        'schema': 'capture-progress.v1',
        'state': state,
        'reason': reason,
        'observed_unix_ns': time.time_ns(),
        'subscribed_topics': sorted(set(re.findall(r"Subscribed to topic '([^']+)'", text))),
        'bag_bytes': sum(p.stat().st_size for p in (output / 'topics').glob('*') if p.is_file()),
        'limits': 'Subscriptions and file size do not prove message arrival or usable data.',
    }
    temporary = output / 'progress.json.tmp'
    temporary.write_text(json.dumps(progress, indent=2))
    temporary.replace(output / 'progress.json')
    return progress


def capture_endpoints(target, controller=None, command_topic=None):
    """Return default launch endpoints with separate target-specific streams."""
    common = [
        '/joy',
        '/operator/motion_permitted',
        '/operator/watchdog_ready',
        '/trace/metrics',
        '/rosout',
    ]
    if target == 'robot':
        return (
            common
            + [
                '/joint_states',
                '/dynamic_joint_states',
                '/imu',
                '/grav',
                '/marvin_joint_controller/commands_traced',
                '/marvin_joint_controller/commands',
            ],
            '/get_policy_observations',
        )
    if target == 'sim':
        # Record the actual policy feedback input, not an assumed broadcaster path.
        return (
            common
            + [
                '/clock',
                '/sim/marvin_joint_controller/commands_traced',
                '/sim/marvin_joint_controller/commands',
                '/isaac_joint_commands',
                '/isaac_joint_states',
                '/isaac_imu',
                '/isaac_odometry',
            ],
            '/sim/get_policy_observations',
        )
    if target == 'sim_joint_test':
        return (
            [
                '/joint_test/isaac_joint_commands',
                '/joint_test/isaac_joint_states',
                '/joint_test/joint_test_controller/joint_test/target_samples',
                '/joint_test/joint_test_controller/status',
                '/clock',
                '/rosout',
            ],
            '/joint_test/joint_test_controller/joint_test/state',
        )
    if target == 'robot_joint_test':
        validate_robot_test_endpoints(controller, command_topic)
        return (
            [
                command_topic,
                controller + '/joint_test/target_samples',
                '/joint_states',
                '/dynamic_joint_states',
                controller + '/status',
                '/rosout',
            ],
            controller + '/joint_test/state',
        )
    raise ValueError(f'Unknown capture target: {target}')


def validate_robot_test_endpoints(controller, command_topic):
    pattern = r'/[A-Za-z_][A-Za-z0-9_]*(?:/[A-Za-z_][A-Za-z0-9_]*)*'
    if not all(
        isinstance(value, str) and re.fullmatch(pattern, value)
        for value in (controller, command_topic)
    ):
        raise ValueError(
            'Robot joint tests require explicit absolute controller and command-topic names'
        )
    if command_topic in {
        '/joint_states',
        '/dynamic_joint_states',
        '/rosout',
        controller + '/status',
        '/marvin_joint_controller/commands',
        '/marvin_joint_controller/commands_traced',
    }:
        raise ValueError(
            'The command topic must expose test-controller targets, not feedback or legacy IK'
            ' input'
        )


def status_payload(target, response):
    if target in JOINT_TEST_TARGETS:
        # Scalar typed snapshot. Do not retain short-lived movement challenges.
        state = response.state
        return {
            'state': {
                name: getattr(state, name)
                for name in state.get_fields_and_field_types()
                if name != 'permit_challenge'
            }
        }
    return {'success': response.success, 'message': response.message}


def file_identity(path):
    """Record supplied disk bytes, not a claim about the loaded runtime model."""
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'Artifact changed during hashing: {path}')
    return {
        'path': str(path),
        'sha256': digest.hexdigest(),
        'size': after.st_size,
        'evidence': 'supplied disk artifact; verify against runtime loaded identity',
    }


def mark_event(output, label):
    """Append an operator observation on the capture host; never control motion."""
    if not label.strip() or len(label) > 500 or '\n' in label:
        raise ValueError('Event must contain 1–500 characters on one line')
    if (output / 'complete.json').exists():
        raise ValueError('Capture is finalized; keep later observations in a separate file')
    metadata = json.loads((output / 'metadata.json').read_text())
    if metadata.get('hostname') != socket.gethostname():
        raise ValueError('Record events on the original capture host')
    row = {
        'label': label,
        'unix_ns': time.time_ns(),
        'monotonic_ns': time.monotonic_ns(),
        'source': 'operator',
        'meaning': 'entry time, not measured physical event time',
    }
    with (output / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')
    return row


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', choices=tuple(TARGET_DOMAINS), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=120)
    parser.add_argument(
        '--topics-only',
        action='store_true',
        help='Do not query status services; subscribe and record only',
    )
    parser.add_argument(
        '--session-config',
        type=Path,
        help='JSON test context copied into metadata; not applied to hardware',
    )
    parser.add_argument(
        '--artifact',
        type=Path,
        action='append',
        default=[],
        help='Policy/config file to hash before recording; repeatable',
    )
    parser.add_argument(
        '--joint-test-controller', help='Explicit robot test controller namespace; no default'
    )
    parser.add_argument(
        '--joint-test-command-topic', help='Explicit robot test target telemetry topic; no default'
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or not 1 <= args.seconds <= 300:
        parser.error('seconds must be between 1 and 300')
    domain = TARGET_DOMAINS[args.target]
    if os.environ.get('ROS_DOMAIN_ID') != domain or os.environ.get('ROS_LOCALHOST_ONLY') != '1':
        parser.error(f'{args.target} requires ROS_DOMAIN_ID={domain} and ROS_LOCALHOST_ONLY=1')
    if args.target in JOINT_TEST_TARGETS and not args.artifact:
        parser.error('Joint tests require --artifact for the supplied test configuration')
    if args.target == 'robot_joint_test':
        try:
            validate_robot_test_endpoints(
                args.joint_test_controller, args.joint_test_command_topic
            )
        except ValueError as error:
            parser.error(str(error))
    elif args.joint_test_controller or args.joint_test_command_topic:
        parser.error('Explicit joint-test endpoints apply only to robot_joint_test')
    return args


def main(argv=None):
    args = parse_args(argv)
    # Fail before starting ROS or a bag if supplied provenance is unreadable.
    artifacts = [file_identity(path) for path in args.artifact]
    session = json.loads(args.session_config.read_text()) if args.session_config else None
    if session is not None and not isinstance(session, dict):
        raise ValueError('Session configuration must be a JSON object')
    import rclpy
    from std_srvs.srv import Trigger

    service_type = Trigger
    status_type = 'std_srvs/srv/Trigger'
    status_period = 1.0
    if args.target in JOINT_TEST_TARGETS:
        from runtime_interfaces.srv import GetJointTestState

        service_type = GetJointTestState
        status_type = 'runtime_interfaces/srv/GetJointTestState'
        status_period = 0.2

    topics, status_service = capture_endpoints(
        args.target, args.joint_test_controller, args.joint_test_command_topic
    )
    # Refuse to overwrite an earlier trial, including partial recordings.
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {
        'schema': 'reference-capture.v2',
        'target': args.target,
        'hostname': socket.gethostname(),
        'started_monotonic_ns': time.monotonic_ns(),
        'artifacts': artifacts,
        'session': session,
        'topics_only': args.topics_only,
        'started_unix_ns': time.time_ns(),
        'seconds': args.seconds,
        'topics': topics,
        'status_service': status_service,
        'status_period_s': status_period,
        'status_type': status_type,
        'ros_domain_id': TARGET_DOMAINS[args.target],
        'command_evidence': (
            'operator-selected test-controller telemetry; verify publisher and applied-write'
            ' semantics'
            if args.target == 'robot_joint_test'
            else 'target-specific topic capture'
        ),
        'clocks': {
            'bag': 'recorder receipt time; not sensor acquisition time',
            'source_headers': 'preserved in ROS messages; may use simulation time',
            'status_monotonic': 'capture-host monotonic time; not comparable across hosts/boots',
        },
        'limits': (
            'Status snapshots are not atomic inference frames or ground truth. '
            'Missing topics are not zero measurements. Recording adds load.'
        ),
    }
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    # Preserve configuration bytes with the recording, not only host-local paths.
    for index, artifact in enumerate(artifacts):
        destination = args.output / 'artifacts' / f'{index:03d}-{Path(artifact["path"]).name}'
        destination.parent.mkdir(exist_ok=True)
        with Path(artifact['path']).open('rb') as source, destination.open('xb') as dest:
            import shutil

            shutil.copyfileobj(source, dest)
        if file_identity(destination)['sha256'] != artifact['sha256']:
            raise ValueError('Artifact changed while copying')
    # Mixed publishers can offer different QoS on the command topic. A reliable
    # subscription would silently omit the native best-effort publisher.
    qos = args.output / 'recording_qos.yaml'
    qos.write_text(
        ''.join(
            f'{topic}:\n  history: keep_last\n  depth: 200\n'
            '  reliability: best_effort\n  durability: volatile\n'
            for topic in topics
        )
    )
    rclpy.init(args=[])
    node = rclpy.create_node('reference_run_status_capture')
    client = None if args.topics_only else node.create_client(service_type, status_service)
    bag = None
    duration_completed = False
    stop_reason = 'duration'
    try:
        with (
            (args.output / 'recorder.log').open('x') as log,
            (args.output / 'status.jsonl').open('x') as status,
        ):
            bag = subprocess.Popen(
                [
                    'ros2',
                    'bag',
                    'record',
                    '--storage',
                    'sqlite3',
                    '--qos-profile-overrides-path',
                    str(qos),
                    '-o',
                    str(args.output / 'topics'),
                    *topics,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = time.monotonic() + args.seconds
            next_request = 0.0
            next_progress = 0.0
            pending = None
            requested_ns = None
            requested_at = None
            while time.monotonic() < deadline:
                if bag.poll() is not None:
                    raise RuntimeError(
                        f'Bag recorder exited: {bag.returncode}; inspect recorder.log'
                    )
                if (args.output / 'stop.request').exists():
                    stop_reason = 'operator_request'
                    break
                rclpy.spin_once(node, timeout_sec=0.05)
                now = time.monotonic()
                if now >= next_progress:
                    capture_progress(args.output, 'recording')
                    next_progress = now + 1.0
                row = None
                if pending is not None and pending.done():
                    response = pending.result()
                    row = dict(
                        request_unix_ns=requested_ns,
                        received_unix_ns=time.time_ns(),
                        request_monotonic_ns=int(requested_at * 1_000_000_000),
                        received_monotonic_ns=time.monotonic_ns(),
                        latency_ms=(now - requested_at) * 1000,
                        **status_payload(args.target, response),
                    )
                    pending = None
                elif pending is not None and now - requested_at > 2:
                    client.remove_pending_request(pending)
                    pending.cancel()
                    pending = None
                    row = {'request_unix_ns': requested_ns, 'error': 'status timeout'}
                if row is not None:
                    status.write(json.dumps(row) + '\n')
                    status.flush()
                if client is not None and pending is None and now >= next_request:
                    next_request = now + status_period
                    if client.service_is_ready():
                        requested_ns, requested_at = time.time_ns(), now
                        pending = client.call_async(service_type.Request())
                    else:
                        status.write(
                            json.dumps({
                                'received_unix_ns': time.time_ns(),
                                'error': 'status service unavailable',
                            })
                            + '\n'
                        )
                        status.flush()
            duration_completed = True
    except KeyboardInterrupt:
        pass
    finally:
        if bag is not None and bag.poll() is None:
            os.killpg(bag.pid, signal.SIGINT)
            try:
                bag.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(bag.pid, signal.SIGTERM)
                bag.wait(timeout=5)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        capture_progress(
            args.output,
            (
                'finalized'
                if duration_completed and bag is not None and bag.returncode == 0
                else 'failed'
            ),
            stop_reason if duration_completed else 'interrupted_or_recorder_error',
        )
    if duration_completed and bag is not None and bag.returncode == 0:
        # Only finalized, bounded captures receive a transfer/export receipt.
        files = {
            str(path.relative_to(args.output)): file_identity(path)
            for path in sorted(args.output.rglob('*'))
            if path.is_file()
        }
        (args.output / 'complete.json').write_text(
            json.dumps(
                {'schema': 'reference-complete.v1', 'stop_reason': stop_reason, 'files': files},
                indent=2,
            )
        )
    else:
        raise RuntimeError(
            f'Capture incomplete: {args.output}; no completion receipt. Motion state was not'
            ' changed.'
        )
    print(
        f'Capture finalized: {args.output}. Check topic coverage before fitting. Motion state was'
        ' not changed.'
    )


if __name__ == '__main__':
    main()
