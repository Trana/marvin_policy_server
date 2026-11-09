"""Private-domain integration check. Synthetic sensors, no hardware subscriber."""

import argparse
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time


def isolation():
    if os.environ.get('ROS_DOMAIN_ID') != '184' or os.environ.get('ROS_LOCALHOST_ONLY') != '1':
        raise RuntimeError('Runtime check requires isolated localhost domain 184')


def source(config):
    isolation()
    from control_msgs.msg import DynamicJointState
    from control_msgs.msg import InterfaceValue
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import Imu
    from sensor_msgs.msg import JointState
    from sensor_msgs.msg import Joy
    from std_msgs.msg import Bool

    rclpy.init(args=[])
    node = rclpy.create_node('native_runtime_synthetic_sensors')
    values = json.loads(config)
    names, positions, sim_clock, dynamic = values[:4]
    robot_rates = len(values) == 5 and values[4] == 'robot'
    joint = JointState(name=names, position=positions, velocity=[0.0] * 12)
    if dynamic:
        # Reverse both joint and interface order to check name-based decoding.
        joint = DynamicJointState(
            joint_names=list(reversed(names)),
            interface_values=[
                InterfaceValue(
                    interface_names=['encoder_receipt_monotonic_s', 'velocity', 'position'],
                    values=[time.monotonic(), 0.0, float(position)],
                )
                for position in reversed(positions)
            ],
        )
    imu = Imu()
    imu.orientation.w = 1.0
    imu.linear_acceleration.z = 9.81
    jp = node.create_publisher(
        DynamicJointState if dynamic else JointState,
        '/policy_probe/joints',
        qos_profile_sensor_data,
    )
    ip = node.create_publisher(Imu, '/policy_probe/imu', qos_profile_sensor_data)
    pp = node.create_publisher(Bool, '/policy_probe/permission', 1)
    feedback = JointState(name=names, position=positions, velocity=[0.0] * 12, effort=[0.0] * 12)
    fp = (
        node.create_publisher(JointState, '/policy_probe/joint_states', qos_profile_sensor_data)
        if robot_rates
        else None
    )
    wp = node.create_publisher(Bool, '/policy_probe/watchdog_ready', 1) if robot_rates else None
    op = (
        node.create_publisher(Joy, '/policy_probe/joy', qos_profile_sensor_data)
        if robot_rates
        else None
    )
    tick = 0
    cp = node.create_publisher(Clock, '/clock', 10) if sim_clock else None
    sim_ns, running = 1_000_000_000, True
    allowed = joints = inertial = True
    deadline, end = time.monotonic(), time.monotonic() + 120.0
    try:
        while time.monotonic() < end:
            if select.select([sys.stdin], [], [], 0)[0]:
                command = os.read(sys.stdin.fileno(), 64).decode()
                if not command or 'q' in command:
                    break
                for c in command:
                    if c in 'pP':
                        allowed = c == 'P'
                    if c in 'jJ':
                        joints = c == 'J'
                    if c in 'iI':
                        inertial = c == 'I'
                    if c in 'sS':
                        running = c == 'S'
                    if c == 'r':
                        sim_ns = 0
            if cp is not None:
                if running:
                    sim_ns += 5_000_000
                clock = Clock()
                clock.clock.sec, clock.clock.nanosec = divmod(sim_ns, 1_000_000_000)
                cp.publish(clock)
                joint.header.stamp = imu.header.stamp = clock.clock
            else:
                joint.header.stamp = imu.header.stamp = node.get_clock().now().to_msg()
            if dynamic:
                for row in joint.interface_values:
                    row.values[0] = time.monotonic()
            if joints and running:
                jp.publish(joint)
            if inertial and running and (not robot_rates or tick % 4 == 0):
                ip.publish(imu)
            if not robot_rates or tick % 4 == 0:
                pp.publish(Bool(data=allowed))
            if robot_rates:
                feedback.header.stamp = joint.header.stamp
                if joints and running:
                    fp.publish(feedback)
                if tick % 20 == 0:
                    wp.publish(Bool(data=True))
                if tick % 4 == 0:
                    op.publish(Joy(axes=[0.0] * 8, buttons=[0] * 11))
            tick += 1
            deadline += 0.005
            if deadline < time.monotonic():
                deadline = time.monotonic() + 0.005
            time.sleep(max(0.0, deadline - time.monotonic()))
    finally:
        node.destroy_node()
        rclpy.shutdown()


def check(binary, model, env, sim_clock=False, dynamic=False, gateway=False):
    isolation()
    import numpy as np
    # NumPy loads its testing helpers lazily. Import and warm them before the
    # policy runs, not inside the executor's active 100 ms inference deadline.
    from numpy.testing import assert_allclose
    from numpy.testing import assert_array_equal

    assert_allclose([0.0], [0.0])
    assert_array_equal([0.0], [0.0])
    import uuid

    from main.command_gateway import CommandGateway
    from marvin_policy_interfaces.srv import SetPolicyInputLock
    from marvin_policy_server.native_protocol import Kind
    from marvin_policy_server.native_server import NativePolicyServer
    from marvin_trace_msgs.msg import TracedFloat64MultiArray
    from marvin_trace_msgs.msg import TraceEvent
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile
    from rclpy.qos import ReliabilityPolicy
    from runtime_interfaces.msg import ExecutorCommand
    from runtime_interfaces.srv import PrepareExecution
    from std_msgs.msg import String
    from std_srvs.srv import SetBool
    from std_srvs.srv import Trigger
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    options = {
        'native_io_executable': str(binary),
        'policy_path': str(model),
        'env_path': str(env),
        'async_inference': 'false',
        'compact_estimator': 'true',
        'lean_qos_events': 'true',
        'activation_ramp_duration': '0.2',
        'deactivation_ramp_duration': '0.2',
        'operator_input_required': 'true',
        'joint_states_topic': '/policy_probe/joints',
        'imu_topic': '/policy_probe/imu',
        'command_topic': '/policy_probe/commands',
        'traced_command_topic': '/policy_probe/commands_traced',
        'trace_event_topic': '/policy_probe/events',
        'publish_traced_commands': 'true',
        'publish_legacy_commands': 'false',
        'tracing_mode': 'capture',
        'trace_capture_seconds': '5.0',
        'use_sim_time': str(sim_clock).lower(),
        'execution_gateway': str(gateway).lower(),
    }
    if dynamic:
        options['same_host_dynamic_joint_states_topic'] = '/policy_probe/joints'
    ros_args = [
        '--ros-args',
        '-r',
        '/operator/motion_permitted:=/policy_probe/permission',
        '-r',
        '/trace/metrics:=/policy_probe/metrics',
        '-r',
        'execution/policy/commands:=/policy_probe/execution/policy/commands',
    ]
    for name, value in options.items():
        ros_args += ['-p', f'{name}:={value}']
    rclpy.init(args=ros_args)
    server = observer = executor = feeder = None
    results = {}
    commands, events, metrics, sent = [], [], [], []
    direct, candidate_ages, callback_us = [], [], []
    floor = 10 if gateway else 30
    try:
        server = NativePolicyServer()
        send_original = server._native_channel.send

        def record(packet):
            if packet.kind == Kind.RESULT:
                sent.append(packet)
            return send_original(packet)

        server._native_channel.send = record
        feeder = subprocess.Popen(
            [
                sys.executable,
                __file__,
                '--source',
                json.dumps([server.joint_names, server.default_pos.tolist(), sim_clock, dynamic]),
            ],
            stdin=subprocess.PIPE,
        )
        observer = rclpy.create_node('native_runtime_observer')
        qos = QoSProfile(depth=200, reliability=ReliabilityPolicy.BEST_EFFORT)
        consumer = CommandGateway(
            lambda targets, trace: None, zip(server._safety_min_pos, server._safety_max_pos)
        )
        consumer.enabled = True

        def candidate(msg):
            before = time.perf_counter_ns()
            stamp = msg.stamp.sec + msg.stamp.nanosec / 1e9
            age = observer.get_clock().now().nanoseconds / 1e9 - stamp
            candidate_ages.append(age)
            accepted = consumer.submit(
                'policy', msg.generation, msg.sequence, msg.data, age=age, trace_id=msg.trace_id
            )
            assert accepted, consumer.last_rejection
            commands.append((time.monotonic(), msg))
            callback_us.append((time.perf_counter_ns() - before) / 1000.0)

        if gateway:
            observer.create_subscription(
                ExecutorCommand, '/policy_probe/execution/policy/commands', candidate, qos
            )
        observer.create_subscription(
            TracedFloat64MultiArray,
            '/policy_probe/commands_traced',
            lambda msg: (direct if gateway else commands).append((time.monotonic(), msg)),
            qos,
        )
        observer.create_subscription(
            TraceEvent, '/policy_probe/events', lambda msg: events.append(msg), qos
        )
        observer.create_subscription(
            String, '/policy_probe/metrics', lambda msg: metrics.append(json.loads(msg.data)), 50
        )
        executor = SingleThreadedExecutor()
        executor.add_node(server)
        executor.add_node(observer)
        active = observer.create_client(SetBool, '/set_active')
        preparation = observer.create_client(PrepareExecution, '/prepare_execution')
        observations = observer.create_client(Trigger, '/get_policy_observations')
        locks = observer.create_client(SetPolicyInputLock, '/set_policy_input_lock')

        def spin(seconds):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                executor.spin_once(timeout_sec=min(0.02, max(0.0, deadline - time.monotonic())))

        def call(client, request):
            if gateway and client is active and request.data:
                generation = uuid.uuid4().hex
                initial = {
                    'joint_names': server.joint_names,
                    'position_unit': 'rad',
                    'model': 'marvin',
                    'sampled_at': time.monotonic(),
                    'joint_age': 0.0,
                    'imu_age': 0.0,
                    'positions': server.default_pos.tolist(),
                    'last_targets': server.default_pos.tolist(),
                }
                call(
                    preparation,
                    PrepareExecution.Request(
                        generation=generation, initial_state_json=json.dumps(initial)
                    ),
                )
                consumer.grant('policy', generation)
            assert client.wait_for_service(timeout_sec=3.0), 'Service not discovered'
            future = client.call_async(request)
            executor.spin_until_future_complete(future, timeout_sec=3.0)
            assert future.done(), 'Service request timed out'
            response = future.result()
            assert response.success, str(response)
            return response

        def state():
            return json.loads(call(observations, Trigger.Request()).message)

        def feed(command):
            feeder.stdin.write(command.encode())
            feeder.stdin.flush()

        def stopped(seconds=0.25):
            spin(0.1)  # Drain subscriber messages already queued before the check.
            start = len(commands)
            spin(seconds)
            assert len(commands) == start, 'Commands continued after output was stopped'

        deadline = time.monotonic() + 8.0
        while server._latest_joint_state is None and time.monotonic() < deadline:
            spin(0.05)
        assert server._latest_joint_state is not None, 'Sensor discovery timed out'
        spin(0.2)
        assert state()['execution']['sensors_valid']
        assert not commands
        if gateway:
            assert active.wait_for_service(timeout_sec=3.0)
            future = active.call_async(SetBool.Request(data=True))
            executor.spin_until_future_complete(future, timeout_sec=3.0)
            assert (
                future.done() and not future.result().success
            ), 'Unprepared activation was accepted'
            spin(0.1)
            assert not commands
            results['unprepared_activation_rejected'] = True
        call(active, SetBool.Request(data=True))
        spin(2.2)
        assert len(commands) > (
            60 if gateway else 100
        ), f'Too few policy commands: {len(commands)}'
        assert server._native_fault is None, server._native_fault
        assert sent and server._policy_runner.history.count > 0
        # Do not starve the policy executor with hundreds of per-message NumPy
        # assertions on the Pi. Validate the bounded capture in one array.
        values = np.asarray([msg.data for _, msg in commands])
        assert values.shape[1:] == (12,) and np.isfinite(values).all()
        assert np.all(values >= server._safety_min_pos) and np.all(
            values <= server._safety_max_pos
        )
        results['activation_and_limits'] = True
        if dynamic:
            assert_allclose(server._latest_joint_state.position, server.default_pos)
            encoder_snapshot = server._input_timing.encoder_snapshot
            assert encoder_snapshot is not None
            assert set(encoder_snapshot[1]) == set(server.joint_names)
            assert all(value > 0 for value in encoder_snapshot[1].values())
            results['dynamic_joint_state_name_mapping'] = True

        target = np.clip(server.default_pos, server._safety_min_pos, server._safety_max_pos)
        call(
            locks,
            SetPolicyInputLock.Request(
                input_name='joint_command', locked=True, use_latest=False, values=target.tolist()
            ),
        )
        spin(0.25)
        before = len(commands)
        spin(0.25)
        assert len(commands) > before + (5 if gateway else 10)
        values = np.asarray([msg.data for _, msg in commands[before:]])
        assert np.all(np.abs(values - target) <= 1e-12)
        call(locks, SetPolicyInputLock.Request(input_name='joint_command', locked=False))
        spin(0.25)
        before = len(commands)
        spin(0.25)
        assert any(not np.allclose(msg.data, target) for _, msg in commands[before:])
        results['output_lock_and_release'] = True

        lock = SetPolicyInputLock.Request(
            input_name='command', locked=True, use_latest=False, values=[0.2, 0.1, 0.1]
        )
        call(locks, lock)
        spin(0.2)
        assert_allclose(state()['locks']['command']['values'], [0.2, 0.1, 0.1])
        before = len(commands)
        feed('p')
        spin(0.6)
        assert_array_equal(state()['locks']['command']['values'], [0.0, 0.0, 0.0])
        assert len(commands) > before + floor, 'Permission loss disabled balance output'
        assert server._native_fault is None
        results['permission_loss_preserves_balance'] = True
        feed('P')
        spin(0.2)

        feed('j')
        spin(0.75)
        assert server._latest_joint_state is None
        stopped()
        feed('J')
        spin(0.6)
        before = len(commands)
        spin(0.3)
        assert len(commands) > before + (5 if gateway else 10)
        assert server._native_fault is None
        results['stale_sensor_recovery'] = True

        if sim_clock:
            before_generation = server._native_generation
            feed('s')
            spin(0.7)
            stopped()
            assert server._native_fault is None
            assert state()['execution']['paused']
            feed('S')
            spin(0.5)
            assert server._native_fault is None
            assert server._native_generation > before_generation
            before = len(commands)
            spin(0.3)
            assert len(commands) > before + (5 if gateway else 10)
            before_generation = server._native_generation
            feed('r')
            spin(0.5)
            assert server._native_generation > before_generation
            assert server._native_last_ros < 2_000_000_000
            assert server._native_fault is None
            if gateway:
                stopped()
                call(active, SetBool.Request(data=False))
                spin(0.1)
                call(active, SetBool.Request(data=True))
            before = len(commands)
            spin(0.3)
            assert len(commands) > before + (5 if gateway else 10)
            results['simulation_pause_resume_and_rewind'] = True

        old = sent[-1]
        call(active, SetBool.Request(data=False))
        spin(0.3)
        stopped()
        server._native_channel.send(old)
        stopped()
        assert server._execution_state()['output_released']
        results['deactivation_and_old_reply_rejection'] = True

        call(active, SetBool.Request(data=True))
        spin(0.5)
        original_step = server._policy_runner.step
        baseline_callback_ms = server._native_request_timing['callback_ms']
        target_callback_ms = 65.0
        injected_delay_s = max(0.0, (target_callback_ms - baseline_callback_ms) / 1000.0)

        def bounded_delay(*args, **kwargs):
            time.sleep(injected_delay_s)
            return original_step(*args, **kwargs)

        before = len(commands)
        server._policy_runner.step = bounded_delay
        spin(0.8)
        observed_callback_ms = server._native_request_timing['callback_ms']
        server._policy_runner.step = original_step
        assert server._native_fault is None, server._native_fault
        delayed_commands = commands[before:]
        assert delayed_commands, 'Bounded delayed inference produced no commands'
        delayed_gaps = [
            current[0] - previous[0]
            for previous, current in zip(delayed_commands, delayed_commands[1:])
        ]
        assert not delayed_gaps or max(delayed_gaps) < consumer.max_age
        results['bounded_inflight_delay'] = {
            'target_callback_ms': target_callback_ms,
            'baseline_callback_ms': baseline_callback_ms,
            'injected_delay_ms': injected_delay_s * 1000.0,
            'observed_callback_ms': observed_callback_ms,
            'command_samples': len(delayed_commands),
            'max_command_gap_ms': max(delayed_gaps, default=0.0) * 1000.0,
        }

        def delayed(*args, **kwargs):
            time.sleep(0.16)
            return original_step(*args, **kwargs)

        server._policy_runner.step = delayed
        spin(0.4)
        server._policy_runner.step = original_step
        assert server._native_fault is not None, 'Slow inference did not latch a fault'
        stopped()
        call(active, SetBool.Request(data=True))
        spin(0.5)
        assert server._native_fault is None
        before = len(commands)
        spin(0.3)
        assert len(commands) > before + (5 if gateway else 10)
        results['inference_timeout_and_explicit_reactivation'] = True

        groups = {}
        for event in events:
            groups.setdefault(event.trace_id, set()).add(event.stage)
        assert any(
            {
                'obs_ready',
                'infer_start',
                'infer_end',
                'policy_candidate_published' if gateway else 'command_published',
            }
            <= stages
            for stages in groups.values()
        ), 'Missing end-to-end trace chain'
        spin(0.1)
        recent = [event for event in events if event.source == 'marvin_policy_io']
        count = len(recent)
        spin(0.3)
        assert (
            len([event for event in events if event.source == 'marvin_policy_io']) == count
        ), 'Detailed native capture did not expire'
        results['bounded_end_to_end_tracing'] = True
        native_metrics = [item for item in metrics if item.get('source') == 'policy_native_io']
        assert native_metrics and any(
            item['counts']['commands'] > (30 if gateway else 100) for item in native_metrics
        )
        results['native_metrics'] = True

        if gateway:
            assert not direct, 'Native gateway mode published direct motor commands'
            assert observer.count_publishers('/policy_probe/commands_traced') == 0
            assert observer.count_publishers('/policy_probe/commands') == 0
            assert observer.count_publishers('/policy_probe/execution/policy/commands') == 1
            results['exclusive_gateway_output'] = True
            results['candidate_age_ms'] = {
                'mean': float(np.mean(candidate_ages) * 1000),
                'max': float(max(candidate_ages) * 1000),
            }
            results['gateway_callback_us'] = {
                'mean': float(np.mean(callback_us)),
                'max': float(max(callback_us)),
            }
            assert consumer.rejected == 0
        server._native_child.terminate()
        server._native_child.wait(timeout=3.0)
        spin(0.3)
        assert server._native_fault is not None
        stopped()
        results['native_process_exit_stops_output'] = True
        results['command_samples'] = len(commands)
        results['native_metric_windows'] = native_metrics
        print('NATIVE_RUNTIME_CHECK ' + json.dumps(results, sort_keys=True))
    finally:
        if feeder is not None:
            if feeder.poll() is None:
                try:
                    feeder.stdin.write(b'q')
                    feeder.stdin.flush()
                    feeder.wait(timeout=3.0)
                except (BrokenPipeError, subprocess.TimeoutExpired):
                    feeder.terminate()
                    feeder.wait(timeout=3.0)
            feeder.stdin.close()
        if executor is not None:
            executor.shutdown()
        if server is not None:
            server._close_velocity_estimator_shadow()
            server.destroy_node()
        if observer is not None:
            observer.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source')
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--env', type=Path)
    parser.add_argument('--sim-clock', action='store_true')
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--gateway', action='store_true')
    args = parser.parse_args()
    if args.source:
        source(args.source)
    else:
        if not all((args.binary, args.model, args.env)):
            parser.error('--binary, --model and --env are required')
        check(
            args.binary.resolve(),
            args.model.resolve(),
            args.env.resolve(),
            args.sim_clock,
            args.dynamic,
            args.gateway,
        )
