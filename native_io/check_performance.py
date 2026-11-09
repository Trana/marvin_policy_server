"""Bounded private-domain performance comparison with an external observer."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def cpu_tree(pid):
    """Read CPU ticks for this owned process and its descendants only."""
    result = {}
    pending = [pid]
    while pending:
        current = pending.pop()
        try:
            text = Path(f'/proc/{current}/stat').read_text()
            fields = text[text.rindex(')') + 2:].split()
            result[current] = int(fields[11]) + int(fields[12])
            pending.extend(
                int(value)
                for value in Path(f'/proc/{current}/task/{current}/children').read_text().split()
            )
        except FileNotFoundError:
            pass
    return result


def main():
    if os.environ.get('ROS_DOMAIN_ID') != '184' or os.environ.get('ROS_LOCALHOST_ONLY') != '1':
        raise SystemExit('Performance check requires localhost-only domain 184')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gateway',
        action='store_true',
        help='Measure supervisor candidate output and command validation',
    )
    parser.add_argument('--backend', choices=['python', 'native'], required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--env', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=20.0)
    parser.add_argument(
        '--sensor-profile',
        choices=['stress', 'robot'],
        default='stress',
        help='Robot uses 200 Hz joint states, 50 Hz IMU/permission/joy, 10 Hz watchdog',
    )
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--check-parent-death', action='store_true')
    parser.add_argument('--diagnostics', type=Path, help='New native diagnostic output file')
    parser.add_argument('--bag', type=Path, help='New private-domain SQLite bag directory')
    parser.add_argument(
        '--managed-recorder-root', type=Path, help='Private-domain managed recorder output'
    )
    parser.add_argument(
        '--managed-recorder-worker', type=Path, help='Reviewed private-domain capture adapter'
    )
    parser.add_argument(
        '--monitor', choices=['none', 'persistent', 'fresh', 'discovery'], default='none'
    )
    parser.add_argument(
        '--measure-status',
        action='store_true',
        help='Time status callbacks in the private test process',
    )
    parser.add_argument(
        '--inactive-only',
        action='store_true',
        help='Observe startup without any activation request',
    )
    parser.add_argument(
        '--monitor-during-stop',
        action='store_true',
        help='Deliberately overlap fresh client discovery with deactivation',
    )
    args = parser.parse_args()
    if bool(args.managed_recorder_root) != bool(args.managed_recorder_worker):
        parser.error('Both managed recorder arguments are required')
    if args.bag and args.managed_recorder_root:
        parser.error('Choose one recorder, not both')
    if args.inactive_only and (
        args.backend != 'native' or args.check_parent_death or args.monitor_during_stop
    ):
        parser.error(
            '--inactive-only requires native backend without parent-death or stop-overlap testing'
        )
    if args.monitor_during_stop and args.monitor not in {'fresh', 'discovery'}:
        parser.error('--monitor-during-stop requires fresh or discovery monitoring')
    if not 10 <= args.seconds <= 60:
        parser.error('--seconds must be between 10 and 60')
    if args.check_parent_death and args.backend != 'native':
        parser.error('--check-parent-death requires the native backend')
    if args.diagnostics and (args.backend != 'native' or args.check_parent_death):
        parser.error('--diagnostics requires native backend and orderly shutdown')
    # Match run-motion-runtime before importing numerical libraries or spawning
    # children. Torch's runtime thread setter alone did not reproduce this setup.
    thread_environment = {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    os.environ.update(thread_environment)
    os.environ['RECORDER_PROBE_SENSOR_PROFILE'] = args.sensor_profile
    import uuid

    from main.command_gateway import CommandGateway
    from marvin_policy_server.env_config_loader import EnvConfigLoader
    from marvin_trace_msgs.msg import TracedFloat64MultiArray
    import numpy as np
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from runtime_interfaces.msg import ExecutorCommand
    from runtime_interfaces.srv import PrepareExecution
    from std_msgs.msg import String
    from std_srvs.srv import SetBool
    from std_srvs.srv import Trigger

    config = EnvConfigLoader(args.env)
    options = {
        'execution_gateway': str(args.gateway).lower(),
        'policy_path': str(args.model.resolve()),
        'env_path': str(args.env.resolve()),
        'native_io_executable': str(args.binary.resolve()),
        'async_inference': str(args.backend == 'python').lower(),
        'compact_estimator': 'true',
        'lean_qos_events': 'true',
        'tracing_mode': 'metrics',
        'publish_trace_events': 'false',
        'operator_input_required': 'true',
        'joint_states_topic': '/policy_probe/joints',
        'same_host_dynamic_joint_states_topic': '/policy_probe/joints',
        'imu_topic': '/policy_probe/imu',
        'command_topic': '/policy_probe/commands',
        'traced_command_topic': '/policy_probe/commands_traced',
        'trace_event_topic': '/policy_probe/events',
        'publish_traced_commands': 'true',
        'publish_legacy_commands': 'false',
    }
    module = 'native_server' if args.backend == 'native' else 'marvin_policy_server'
    if args.diagnostics:
        options['native_diagnostic_path'] = str(args.diagnostics.resolve())
    command = [
        sys.executable,
        '-m',
        'marvin_policy_server.' + module,
        '--ros-args',
        '-r',
        '/operator/motion_permitted:=/policy_probe/permission',
        '-r',
        '/trace/metrics:=/policy_probe/metrics',
        '-r',
        'execution/policy/commands:=/policy_probe/execution/policy/commands',
    ]
    if args.measure_status:
        if args.backend != 'native':
            parser.error('--measure-status requires native backend')
        command = [
            sys.executable,
            str(Path(__file__).with_name('monitoring_probe.py')),
            'server',
            *command[3:],
        ]
    for key, value in options.items():
        command += ['-p', f'{key}:={value}']
    policy = feeder = node = bag = None
    recorder = recording_id = None
    recorder_next_tick = 0.0
    recorder_error = None
    timestamps, metrics = [], []
    observed_trace_ids = set()
    monitor_children, monitor_events = [], []
    monitor_future = None
    monitor_started = None
    next_monitor = None
    monitor_cutoff = None
    rclpy.init(args=[])
    with args.log.open('x') as log:
        try:
            if args.managed_recorder_root:
                from uuid import uuid4

                from recording_manager.manager import RecordingManager

                def spawn_recording(command, **kwargs):
                    return subprocess.Popen(
                        [command[0], str(args.managed_recorder_worker), *command[3:]], **kwargs
                    )

                recorder = RecordingManager(
                    args.managed_recorder_root, 'private-policy-test', spawn=spawn_recording
                )
                recording_id = recorder.start(str(uuid4()), duration_s=180)['recording_id']
            if args.bag:
                topics = [
                    '/policy_probe/joints',
                    '/policy_probe/imu',
                    '/policy_probe/commands_traced',
                    '/policy_probe/permission',
                    '/policy_probe/metrics',
                    '/rosout',
                ]
                qos = Path(str(args.bag) + '.qos.yaml')
                with qos.open('x') as stream:
                    stream.write(
                        ''.join(
                            f'{topic}:\n  reliability: best_effort\n'
                            '  durability: volatile\n  history: keep_last\n  depth: 200\n'
                            for topic in topics
                        )
                    )
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
                        str(args.bag),
                        *topics,
                    ],
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
            feeder = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).with_name('check_runtime.py')),
                    '--source',
                    json.dumps([
                        config.get_joint_names(),
                        config.get_default_joint_positions().tolist(),
                        False,
                        True,
                        args.sensor_profile,
                    ]),
                ],
                stdin=subprocess.PIPE,
                stdout=log,
                stderr=log,
            )
            policy = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
            node = rclpy.create_node('private_policy_performance_observer')
            consumer = CommandGateway(lambda target, trace: None, [(-10.0, 10.0)] * 12)
            consumer.enabled = True
            candidate_ages = []

            def observe_command(msg):
                if args.gateway:
                    age = (
                        node.get_clock().now().nanoseconds / 1e9
                        - msg.stamp.sec
                        - msg.stamp.nanosec / 1e9
                    )
                    assert consumer.submit(
                        'policy',
                        msg.generation,
                        msg.sequence,
                        msg.data,
                        age=age,
                        trace_id=msg.trace_id,
                    ), consumer.last_rejection
                    candidate_ages.append(age * 1000.0)
                timestamps.append(time.monotonic())
                observed_trace_ids.add(str(msg.trace_id))

            node.create_subscription(
                ExecutorCommand if args.gateway else TracedFloat64MultiArray,
                (
                    '/policy_probe/execution/policy/commands'
                    if args.gateway
                    else '/policy_probe/commands_traced'
                ),
                observe_command,
                qos_profile_sensor_data,
            )
            node.create_subscription(
                String,
                '/policy_probe/metrics',
                lambda msg: metrics.append((time.monotonic(), json.loads(msg.data))),
                20,
            )
            active = node.create_client(SetBool, '/set_active')
            preparation = node.create_client(PrepareExecution, '/prepare_execution')
            observations = node.create_client(Trigger, '/get_policy_observations')

            def spin(seconds):
                nonlocal monitor_future, monitor_started, next_monitor, recorder_next_tick
                end = time.monotonic() + seconds
                while time.monotonic() < end:
                    if recorder is not None and time.monotonic() >= recorder_next_tick:
                        recorder.tick()
                        assert recorder.process is not None, 'Managed recorder exited early'
                        recorder_next_tick = time.monotonic() + 1.0
                    assert policy.poll() is None, f'Policy exited: inspect {args.log}'
                    assert feeder.poll() is None, 'Synthetic sensor source exited'
                    assert bag is None or bag.poll() is None, 'Bag recorder exited'
                    assert all(
                        child.poll() in (None, 0) for child in monitor_children
                    ), 'Monitoring client failed'
                    rclpy.spin_once(node, timeout_sec=min(0.02, max(0.0, end - time.monotonic())))
                    now = time.monotonic()
                    if monitor_future is not None and monitor_future.done():
                        result = monitor_future.result()
                        assert result is not None and result.success, 'Monitoring query failed'
                        monitor_events.append({
                            'elapsed_s': now - start,
                            'round_trip_ms': (now - monitor_started) * 1000,
                        })
                        monitor_future = None
                    if next_monitor is not None and now >= next_monitor:
                        next_monitor = now + 5.0
                        if now >= monitor_cutoff:
                            next_monitor = None
                            continue
                        if args.monitor == 'persistent':
                            assert monitor_future is None, 'Monitoring query exceeded five seconds'
                            monitor_started = now
                            monitor_future = observations.call_async(Trigger.Request())
                        elif args.monitor in {'fresh', 'discovery'}:
                            assert all(
                                child.poll() is not None for child in monitor_children
                            ), 'Previous monitor still running'
                            child = subprocess.Popen(
                                [
                                    sys.executable,
                                    str(Path(__file__).with_name('monitoring_probe.py')),
                                    args.monitor,
                                ],
                                stdout=log,
                                stderr=log,
                            )
                            monitor_children.append(child)
                            monitor_events.append({'elapsed_s': now - start, 'pid': child.pid})

            def call(client, request):
                if args.gateway and client is active and request.data:
                    generation = uuid.uuid4().hex
                    initial = {
                        'joint_names': config.get_joint_names(),
                        'position_unit': 'rad',
                        'model': 'marvin',
                        'sampled_at': time.monotonic(),
                        'joint_age': 0.0,
                        'imu_age': 0.0,
                        'positions': config.get_default_joint_positions().tolist(),
                        'last_targets': [],
                    }
                    call(
                        preparation,
                        PrepareExecution.Request(
                            generation=generation, initial_state_json=json.dumps(initial)
                        ),
                    )
                    consumer.grant('policy', generation)
                assert client.wait_for_service(timeout_sec=15.0), 'Policy service not discovered'
                future = client.call_async(request)
                rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
                assert future.done() and future.result().success, (
                    str(future.result()) if future.done() else 'RPC timeout'
                )
                return future.result()

            assert active.wait_for_service(timeout_sec=20.0), 'Policy startup timeout'
            spin(2.0)
            if args.backend == 'native':
                deadline = time.monotonic() + 10.0
                while True:
                    status = json.loads(call(observations, Trigger.Request()).message)
                    assert not status.get('execution', {}).get('fault'), status.get('execution')
                    if status.get('execution', {}).get('sensors_valid'):
                        break
                    assert time.monotonic() < deadline, 'Native sensor discovery timeout'
                    spin(0.2)
            if args.inactive_only:
                start = time.monotonic()
                monitor_cutoff = start + args.seconds - 6.0
                if args.monitor != 'none':
                    next_monitor = start + 2.0
                spin(args.seconds)
                next_monitor = None
                assert monitor_future is None, 'Monitoring query did not finish'
                assert all(
                    child.poll() == 0 for child in monitor_children
                ), 'Monitoring client did not finish successfully'
                status = json.loads(call(observations, Trigger.Request()).message)
                assert not status.get('execution', {}).get('fault'), status.get('execution')
                assert not timestamps, 'Unexpected commands while inactive'
                print(
                    'NATIVE_INACTIVE_CHECK '
                    + json.dumps({
                        'seconds': time.monotonic() - start,
                        'monitor': args.monitor,
                        'monitor_events': monitor_events,
                        'commands': len(timestamps),
                        'execution': status.get('execution'),
                        'activation_requested': False,
                    })
                )
                return
            call(active, SetBool.Request(data=True))
            warmup_started = time.monotonic()
            spin(5.0)
            assert timestamps, 'No warmup output'
            # Preserve startup evidence before resetting the steady-state window.
            # Pair monotonic receipt times with the recorder's native stream stats.
            warmup_gaps = [
                {'end_monotonic_ns': int(end * 1e9), 'gap_ms': (end - previous) * 1000}
                for previous, end in zip(timestamps, timestamps[1:])
                if end - previous >= 0.02
            ]
            print(
                'NATIVE_WARMUP_CHECK '
                + json.dumps({
                    'start_monotonic_ns': int(warmup_started * 1e9),
                    'commands': len(timestamps),
                    'max_gap_ms': (
                        max((b - a) * 1000 for a, b in zip(timestamps, timestamps[1:]))
                        if len(timestamps) > 1
                        else None
                    ),
                    'gaps_over_20ms': warmup_gaps,
                }),
                flush=True,
            )
            before_cpu = cpu_tree(policy.pid)
            before_observer_cpu = cpu_tree(os.getpid())[os.getpid()]
            candidate_ages.clear()
            before_recorder_cpu = cpu_tree(recorder.process.pid) if recorder is not None else {}
            timestamps.clear()
            metrics.clear()
            start = time.monotonic()
            # Leave time for bounded discovery/query and cleanup before Stop.
            monitor_cutoff = start + args.seconds - (0.0 if args.monitor_during_stop else 6.0)
            if args.monitor != 'none':
                next_monitor = start + 2.0
            spin(args.seconds)
            next_monitor = None
            elapsed = time.monotonic() - start
            assert monitor_future is None, 'Monitoring query did not finish'
            if not args.monitor_during_stop:
                assert all(
                    child.poll() == 0 for child in monitor_children
                ), 'Monitoring client did not finish successfully'
            after_cpu = cpu_tree(policy.pid)
            observer_cpu = (
                100
                * (cpu_tree(os.getpid())[os.getpid()] - before_observer_cpu)
                / os.sysconf('SC_CLK_TCK')
                / elapsed
            )
            age_window = list(candidate_ages)
            after_recorder_cpu = cpu_tree(recorder.process.pid) if recorder is not None else {}
            captures = list(timestamps)
            # Ignore the first partially overlapping metric window.
            windows = [item for received, item in metrics if received - start >= 1.1]
            status = json.loads(call(observations, Trigger.Request()).message)
            assert not status.get('execution', {}).get('fault'), status.get('execution')
            assert status['concurrent_policy']['history_ready']
            call(active, SetBool.Request(data=False))
            spin(1.5)
            count = len(timestamps)
            spin(0.25)
            assert len(timestamps) == count, 'Policy commands continued after deactivation'
            if args.monitor_during_stop:
                deadline = time.monotonic() + 6.0
                while (
                    any(child.poll() is None for child in monitor_children)
                    and time.monotonic() < deadline
                ):
                    spin(0.1)
                assert all(
                    child.poll() == 0 for child in monitor_children
                ), 'Monitoring client failed or did not finish'
                assert len(timestamps) == count, 'Output resumed while waiting for monitor clients'
            stopped_status = json.loads(call(observations, Trigger.Request()).message)
            assert not stopped_status.get('execution', {}).get('fault'), stopped_status.get(
                'execution'
            )
            if args.check_parent_death:
                call(active, SetBool.Request(data=True))
                spin(0.5)
                count = len(timestamps)
                spin(0.2)
                assert len(timestamps) > count + 10, 'No active output before parent-death test'
                children = set(cpu_tree(policy.pid)) - {policy.pid}
                assert children, 'Native helper process was not found'
                killed_at = time.monotonic()
                policy.kill()  # Kill only Python; the helper must stop itself.
                policy.wait(timeout=3.0)
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline:
                    rclpy.spin_once(node, timeout_sec=0.02)
                assert not any(
                    t > killed_at + 0.3 for t in timestamps
                ), 'Native output survived parent loss'
                for pid in children:
                    path = Path(f'/proc/{pid}/stat')
                    if path.exists():
                        stat = path.read_text()
                        assert (
                            stat[stat.rindex(')') + 2:].split()[0] == 'Z'
                        ), 'Native helper remained running'
            cpu = {
                str(pid): (
                    100 * (ticks - before_cpu.get(pid, ticks)) / os.sysconf('SC_CLK_TCK') / elapsed
                )
                for pid, ticks in after_cpu.items()
            }
            intervals = np.diff(captures) * 1000
            assert (
                len(intervals) > 100 and max(intervals) < 100.0
            ), 'Missing output or an active output gap >=100ms'
            rates = {}
            for source in {item['source'] for item in windows}:
                selected = [item for item in windows if item['source'] == source]
                total = sum(item['window_s'] for item in selected)
                keys = {key for item in selected for key in item.get('counts', {})}
                rates[source] = {
                    key: sum(item.get('counts', {}).get(key, 0) for item in selected) / total
                    for key in keys
                }
            durations = {}
            for name in ('inference', 'callback', 'async_result_pickup'):
                values = [
                    item['durations_ms'][name]
                    for item in windows
                    if name in item.get('durations_ms', {})
                ]
                if values:
                    n = sum(item['count'] for item in values)
                    durations[name] = {
                        'mean_ms': sum(item['count'] * item['mean'] for item in values) / n,
                        'max_ms': max(item['max'] for item in values),
                    }
            print(
                'NATIVE_PERFORMANCE_CHECK '
                + json.dumps({
                    'backend': args.backend,
                    'seconds': elapsed,
                    'commands_hz': len(captures) / elapsed,
                    'gateway': args.gateway,
                    'observer_cpu_percent_one_core': observer_cpu,
                    'candidate_age_ms': (
                        {
                            'mean': float(np.mean(age_window)),
                            'max': max(age_window),
                        }
                        if age_window
                        else None
                    ),
                    'sensor_profile': args.sensor_profile,
                    'monitor': args.monitor,
                    'monitor_events': monitor_events,
                    'monitor_during_stop': args.monitor_during_stop,
                    'thread_environment': thread_environment,
                    'command_gap_ms': {
                        'p50': float(np.percentile(intervals, 50)),
                        'p95': float(np.percentile(intervals, 95)),
                        'max': float(max(intervals)),
                    },
                    'policy_tree_cpu_percent_one_core': sum(cpu.values()),
                    'cpu_by_pid': cpu,
                    'recorder_tree_cpu_percent_one_core': sum(
                        100
                        * (ticks - before_recorder_cpu.get(pid, ticks))
                        / os.sysconf('SC_CLK_TCK')
                        / elapsed
                        for pid, ticks in after_recorder_cpu.items()
                    ),
                    'rates_hz': rates,
                    'durations': durations,
                    'output_stopped_after_deactivation': True,
                    'parent_death_stops_output_and_helper': args.check_parent_death,
                })
            )
        finally:
            if recorder is not None:
                try:
                    recorder.close()
                    print('MANAGED_RECORDER_RESULT ' + json.dumps(recorder.status(recording_id)))
                except Exception as error:
                    recorder_error = error
            for child in monitor_children:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=3.0)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=3.0)
            # Stop the recorder first so cleanup of another child cannot leave it running.
            if bag is not None and bag.poll() is None:
                os.killpg(bag.pid, signal.SIGINT)
                try:
                    bag.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    os.killpg(bag.pid, signal.SIGKILL)
                    bag.wait(timeout=3.0)
            if policy is not None and policy.poll() is None:
                os.killpg(policy.pid, signal.SIGINT)
                try:
                    policy.wait(timeout=8.0)
                except subprocess.TimeoutExpired:
                    os.killpg(policy.pid, signal.SIGKILL)
                    policy.wait(timeout=3.0)
            if feeder is not None:
                if feeder.poll() is None:
                    feeder.stdin.write(b'q')
                    feeder.stdin.flush()
                    feeder.wait(timeout=5.0)
                feeder.stdin.close()
            if node is not None:
                node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            if recorder_error is not None:
                raise RuntimeError('Managed recorder cleanup failed') from recorder_error
            if args.diagnostics:
                # Preserve independent observations even when a policy fault fails
                # the check before the successful-run alignment report.
                with Path(str(args.diagnostics) + '.observed-trace-ids.json').open(
                    'x'
                ) as evidence:
                    json.dump(sorted(observed_trace_ids), evidence)
    if args.diagnostics:
        rows = [json.loads(line) for line in args.diagnostics.read_text().splitlines()]
        assert rows[-1]['event'] == 'summary', 'Diagnostic export incomplete'
        assert rows[-1]['dropped'] == 0, 'Diagnostic buffer exhausted'
        results = {row['trace_id'] for row in rows if row['event'] == 'result_submitted'}
        commits = {row['trace_id'] for row in rows if row['event'] == 'publication_commit'}
        assert commits and commits <= results, 'Commit lacks recorded inference result'
        assert commits & observed_trace_ids, 'No diagnostic commit matches observed commands'
        print(
            'NATIVE_DIAGNOSTIC_ALIGNMENT '
            + json.dumps({
                'results': len(results),
                'commits': len(commits),
                'observed_commits': len(commits & observed_trace_ids),
                'unobserved_commits': len(commits - observed_trace_ids),
                'uncommitted_results': len(results - commits),
            })
        )


if __name__ == '__main__':
    main()
