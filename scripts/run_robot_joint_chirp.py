#!/usr/bin/env python3
"""
Explicit stand test for one joint or a sequential all-joint fitting capture.

Ctrl+C requests cancellation and hardware INACTIVE. Keep the physical disconnect
accessible. Never rely on this process or a network request as an emergency stop.
"""

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import struct
import subprocess
import sys
import time
import uuid
from xml.etree import ElementTree as ET

import yaml

# Two challenge generations can contribute to permission age. Keep margin within
# the unchanged 500 ms lease, including service and update acknowledgement delays.
RENEW_INTERVAL_S = 0.1


def terminal_failure(code, state):
    """Retain the controller evidence before cancellation changes its state."""
    return (
        f'{code}: phase={state.phase}, reason={state.reason_code}, '
        f'tracking_fault={state.tracking_fault_latched}, '
        f'feedback_fault={state.feedback_fault_latched}'
    )


def approved_amplitude(profile, joint):
    profiles = {
        'marvin-stand-chirp-v1-robot-only': ({'FR_thigh_joint'}, 0.005, 0.8),
        'marvin-stand-thigh-chirp-v2-robot-only': (
            {'FR_thigh_joint', 'FL_thigh_joint', 'RR_thigh_joint', 'RL_thigh_joint'},
            0.008726646259971648,
            0.4,
        ),
    }
    allowed, amplitude, end_hz = profiles.get(
        profile.get('joint_test_profile.id'), (set(), 0.0, 0.0)
    )
    if (
        joint not in allowed
        or joint not in profile['joints']
        or profile.get('joint_test_profile.chirp_start_hz') != 0.2
        or profile.get('joint_test_profile.chirp_end_hz') != end_hz
    ):
        raise ValueError('Unapproved profile identity, joint or frequency')
    index = profile['joints'].index(joint)
    if profile['joint_test_profile.max_amplitude_rad'][index] != amplitude:
        raise ValueError('Unapproved amplitude limit')
    return amplitude


def stop_recorder(process, signal_sent=False):
    """Bounded cleanup after motor stop. Never label a forced exit as success."""
    if process is None:
        return
    if process.poll() is None:
        if not signal_sent:
            process.send_signal(signal.SIGINT)
        try:
            result = process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
            raise RuntimeError('Recorder shutdown timed out; capture finalization is unconfirmed')
    else:
        result = process.returncode
    if result:
        raise RuntimeError('Recorder failed; capture finalization is unconfirmed')


def profile_hash(profile):
    def text(value):
        raw = value.encode()
        return struct.pack('>I', len(raw)) + raw

    directions = profile.get('joint_test_profile.simultaneous_directions', [])
    raw = (
        b'joint-test-profile.v3-simultaneous\0' if directions else b'joint-test-profile.v2-chirp\0'
    ) + text(profile['joint_test_profile.id'])
    raw += struct.pack(
        '>dIddI',
        profile['joint_test_profile.max_total_seconds'],
        profile['joint_test_profile.max_cycles'],
        profile['joint_test_profile.chirp_start_hz'],
        profile['joint_test_profile.chirp_end_hz'],
        len(profile['joints']),
    )
    fields = ('lower_rad', 'upper_rad', 'speed_rad_s', 'acceleration_rad_s2')
    for i, name in enumerate(profile['joints']):
        values = [profile['joint_test_profile.' + field][i] for field in fields]
        values += [
            profile['max_handover_error'][i],
            profile['joint_test_profile.max_amplitude_rad'][i],
        ]
        raw += text(name) + struct.pack('>6d', *values)
        if directions:
            raw += struct.pack('>d', directions[i])
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--operator-ready', action='store_true')
    parser.add_argument('--joint', default='FR_thigh_joint')
    parser.add_argument(
        '--all-joints',
        action='store_true',
        help=(
            'Run the bounded fitting profile sequentially; one operator start, abort on any fault'
        ),
    )
    parser.add_argument(
        '--simultaneous',
        action='store_true',
        help='Explicit simultaneous profile; all joints must pass, no skipping',
    )
    parser.add_argument(
        '--stop-only',
        action='store_true',
        help='Request hardware INACTIVE only; never hand over or start',
    )
    parser.add_argument(
        '--wait-for-operator',
        action='store_true',
        help='Connect while hardware is inactive, then wait up to 180 s for ENTER',
    )
    args = parser.parse_args()
    if (
        not args.operator_ready
        or os.environ.get('ROS_DOMAIN_ID') != '0'
        or os.environ.get('ROS_LOCALHOST_ONLY') != '1'
    ):
        parser.error('Requires operator-ready confirmation and local robot ROS domain 0')
    profile = yaml.safe_load(args.profile.read_text())['/joint_test_controller']['ros__parameters']
    if args.simultaneous:
        from make_simultaneous_fit_profile import FAST_SHORT_PROFILE_ID
        from make_simultaneous_fit_profile import FIVE_HZ_PROFILE_ID
        from make_simultaneous_fit_profile import LONG_PROFILE_ID
        from make_simultaneous_fit_profile import PROFILE_ID
        from make_simultaneous_fit_profile import SHORT_PROFILE_ID
        from make_simultaneous_fit_profile import simultaneous_trial
        from make_simultaneous_fit_profile import TEN_SHORT_PROFILE_ID
        from make_simultaneous_fit_profile import WIDE_SHORT_PROFILE_ID

        if args.all_joints or profile['joint_test_profile.id'] not in (
            PROFILE_ID,
            SHORT_PROFILE_ID,
            WIDE_SHORT_PROFILE_ID,
            TEN_SHORT_PROFILE_ID,
            FAST_SHORT_PROFILE_ID,
            FIVE_HZ_PROFILE_ID,
            LONG_PROFILE_ID,
        ):
            parser.error('Simultaneous mode requires its own profile and excludes sequential mode')
    elif args.all_joints:
        from joint_fit_plan import plan_joints
        from joint_fit_plan import PROFILE_ID

        if profile['joint_test_profile.id'] != PROFILE_ID:
            parser.error('All-joint mode requires the fitting profile')
    else:
        try:
            amplitude = approved_amplitude(profile, args.joint)
        except ValueError as error:
            parser.error(str(error))
    if args.output.exists() and not args.stop_only:
        parser.error('Output directory must not exist')
    from controller_manager_msgs.srv import ListControllers
    from controller_manager_msgs.srv import ListHardwareComponents
    from controller_manager_msgs.srv import SetHardwareComponentState
    from controller_manager_msgs.srv import SwitchController
    from lifecycle_msgs.msg import State
    from rcl_interfaces.srv import GetParameters
    import rclpy
    from rclpy.signals import SignalHandlerOptions
    from runtime_interfaces.srv import ControlJointTest
    from runtime_interfaces.srv import GetJointTestState
    from runtime_interfaces.srv import PrepareJointTest

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    node = rclpy.create_node('robot_stand_chirp_client')
    clients = {}
    session = None
    sequence = 0
    recorder = None
    recorder_stop_requested = False
    handed_over = False
    hardware_name = None
    prepared_wait = False
    service_timings = deque(maxlen=16384)
    trials = []

    def call(kind, endpoint, **fields):
        if endpoint not in clients:
            clients[endpoint] = node.create_client(kind, endpoint)
        client = clients[endpoint]
        if not client.wait_for_service(timeout_sec=5):
            raise RuntimeError('Service unavailable: ' + endpoint)
        request_started = time.monotonic()
        future = client.call_async(kind.Request(**fields))
        rclpy.spin_until_future_complete(node, future, timeout_sec=3)
        response = future.result() if future.done() else None
        service_timings.append({
            'endpoint': endpoint,
            'started_monotonic_s': request_started,
            'elapsed_ms': 1000 * (time.monotonic() - request_started),
            'completed': response is not None,
            'action': fields.get('action'),
            'sequence': fields.get('sequence'),
            'accepted': getattr(response, 'accepted', None),
            'phase': getattr(getattr(response, 'state', None), 'phase', None),
            'reason_code': getattr(getattr(response, 'state', None), 'reason_code', None),
        })
        if not future.done() or future.result() is None:
            raise RuntimeError('Outcome unknown: ' + endpoint)
        return future.result()

    def hardware():
        items = call(
            ListHardwareComponents, '/controller_manager/list_hardware_components'
        ).component
        if len(items) != 1:
            raise RuntimeError('Unexpected hardware configuration')
        plugin = getattr(items[0], 'plugin_name', None)
        if plugin is None:
            # Humble does not expose plugin_name in HardwareComponentState.
            values = call(
                GetParameters, '/controller_manager/get_parameters', names=['robot_description']
            ).values
            if len(values) != 1 or values[0].type != 4:
                raise RuntimeError('Cannot verify hardware description')
            systems = ET.fromstring(values[0].string_value).findall('ros2_control')
            if len(systems) != 1 or systems[0].get('name') != items[0].name:
                raise RuntimeError('Hardware description does not match live component')
            plugin = systems[0].findtext('hardware/plugin')
        if plugin != 'odrive_ros2_control_plugin/ODriveHardwareInterface':
            raise RuntimeError('Unexpected hardware plugin')
        return items[0]

    def ownership():
        return {
            c.name: c
            for c in call(ListControllers, '/controller_manager/list_controllers').controller
        }

    def start_recorder(instance, activation, seconds):
        return subprocess.Popen([
            sys.executable,
            str(Path(__file__).with_name('record_joint_test_guarded.py')),
            '--controller',
            '/joint_test_controller',
            '--feedback',
            '/joint_states',
            '--instance',
            instance,
            '--activation',
            str(activation),
            '--output',
            str(args.output),
            '--seconds',
            str(seconds),
            '--joints',
            *profile['joints'],
        ])

    def control(action, challenge=''):
        nonlocal sequence
        sequence += 1
        return call(
            ControlJointTest,
            '/joint_test_controller/joint_test/control',
            action=action,
            controller_instance_id=session.controller_instance_id,
            activation_generation=session.activation_generation,
            session_id=session.session_id,
            operation_id=str(uuid.uuid4()),
            sequence=sequence,
            permit_challenge=challenge,
        )

    try:
        component = hardware()
        hardware_name = component.name
        if args.stop_only:
            return
        if args.wait_for_operator:
            if component.state.label != 'inactive':
                raise RuntimeError('Connection preparation requires inactive hardware')
            prepared_wait = True
            endpoints = (
                (ListControllers, '/controller_manager/list_controllers'),
                (SwitchController, '/controller_manager/switch_controller'),
                (SetHardwareComponentState, '/controller_manager/set_hardware_component_state'),
                (GetJointTestState, '/joint_test_controller/joint_test/state'),
                (PrepareJointTest, '/joint_test_controller/joint_test/prepare'),
                (ControlJointTest, '/joint_test_controller/joint_test/control'),
            )
            for kind, endpoint in endpoints:
                if endpoint not in clients:
                    clients[endpoint] = node.create_client(kind, endpoint)
            for _, endpoint in endpoints:
                if not clients[endpoint].wait_for_service(timeout_sec=5):
                    raise RuntimeError('Service unavailable: ' + endpoint)
            before = call(GetJointTestState, '/joint_test_controller/joint_test/state').state
            owners = ownership()
            if (
                getattr(owners.get('marvin_joint_controller'), 'state', None) != 'active'
                or getattr(owners.get('joint_test_controller'), 'state', None) != 'inactive'
            ):
                raise RuntimeError(
                    'Connection preparation requires normal ownership and inactive test controller'
                )
            # The inactive controller increments activation once at handover.
            recorder = start_recorder(
                before.controller_instance_id,
                before.activation_generation + 1,
                600 if args.all_joints else 360,
            )
            recorder_deadline = time.monotonic() + 5
            while True:
                if recorder.poll() is not None or time.monotonic() >= recorder_deadline:
                    raise RuntimeError('Recorder preparation failed')
                try:
                    health = json.loads((args.output / 'health.json').read_text())
                    if health['running'] and not health['fault'] and health['feedback'] >= 2:
                        break
                except (OSError, ValueError):
                    pass
                rclpy.spin_once(node, timeout_sec=0.05)
            print(
                'CONNECTIONS READY. Enable IK, then send ENTER to proceed. No automatic start.'
                ' Timeout 180 s.',
                flush=True,
            )
            deadline = time.monotonic() + 180
            while not select.select([sys.stdin], [], [], 0)[0]:
                if time.monotonic() >= deadline:
                    raise RuntimeError('Operator wait expired')
                if recorder.poll() is not None:
                    raise RuntimeError('Recorder stopped during operator wait')
                rclpy.spin_once(node, timeout_sec=0.05)
            if sys.stdin.readline() != '\n':
                raise RuntimeError('Operator input cancelled or closed')
            after = call(GetJointTestState, '/joint_test_controller/joint_test/state').state
            if (before.controller_instance_id, before.activation_generation) != (
                after.controller_instance_id,
                after.activation_generation,
            ):
                raise RuntimeError('Controller changed during operator wait; prepare again')
            component = hardware()
        owners = ownership()
        if (
            component.state.label != 'active'
            or getattr(owners.get('marvin_joint_controller'), 'state', None) != 'active'
            or getattr(owners.get('joint_test_controller'), 'state', None) != 'inactive'
        ):
            raise RuntimeError('Requires active IK hardware and an inactive test controller')
        # Mark uncertainty before the request. A lost response still requires stop.
        handed_over = True
        timeout = SwitchController.Request().timeout
        timeout.sec = 2
        call(
            SwitchController,
            '/controller_manager/switch_controller',
            activate_controllers=['joint_test_controller'],
            deactivate_controllers=['marvin_joint_controller'],
            strictness=SwitchController.Request.STRICT,
            timeout=timeout,
        )
        owners = ownership()
        expected = sorted(j + '/position' for j in profile['joints'])
        if (
            owners['joint_test_controller'].state != 'active'
            or sorted(owners['joint_test_controller'].claimed_interfaces) != expected
            or owners['marvin_joint_controller'].state != 'inactive'
            or owners['marvin_joint_controller'].claimed_interfaces
        ):
            raise RuntimeError('Exclusive handover not confirmed')
        time.sleep(0.05)
        state = call(GetJointTestState, '/joint_test_controller/joint_test/state').state
        if (
            not state.hold_applied
            or not state.feedback_receipt_verified
            or state.tracking_fault_latched
            or state.feedback_fault_latched
        ):
            raise RuntimeError('Inherited hold is not ready')
        if recorder is None:
            recorder = start_recorder(
                state.controller_instance_id,
                state.activation_generation,
                (
                    420
                    if args.all_joints
                    else max(35.0, profile['joint_test_profile.max_total_seconds'] + 15.0)
                ),
            )

        def recording_ready():
            if recorder.poll() is not None:
                return False
            try:
                health = json.loads((args.output / 'health.json').read_text())
                return (
                    health['ready']
                    and health['instance'] == state.controller_instance_id
                    and health['activation'] == state.activation_generation
                    and 0 <= time.monotonic() - health['sampled_at_monotonic_s'] <= 0.15
                )
            except (OSError, ValueError):
                return False

        deadline = time.monotonic() + 3
        while not recording_ready():
            if time.monotonic() > deadline:
                raise RuntimeError('Recording did not become healthy')
            time.sleep(0.01)
        (args.output / 'profile.yaml').write_bytes(args.profile.read_bytes())
        if args.simultaneous:
            health = json.loads((args.output / 'health.json').read_text())
            trials = [
                simultaneous_trial(
                    profile, health['target_positions'], health['feedback_positions']
                )
            ]
        elif args.all_joints:
            health = json.loads((args.output / 'health.json').read_text())
            trials = plan_joints(profile, health['target_positions'], health['feedback_positions'])
        else:
            trials = [{'joint': args.joint, 'amplitude_rad': amplitude, 'status': 'ready'}]
        (args.output / 'excitation-plan.json').write_text(json.dumps(trials, indent=2))
        for trial in trials:
            print(json.dumps(trial), flush=True)
        for trial in trials:
            if trial['status'] != 'ready':
                continue
            sequence = 0
            prepared = call(
                PrepareJointTest,
                '/joint_test_controller/joint_test/prepare',
                controller_instance_id=state.controller_instance_id,
                activation_generation=state.activation_generation,
                operation_id=str(uuid.uuid4()),
                profile_id=profile['joint_test_profile.id'],
                profile_sha256=profile_hash(profile),
                joint_name=trial['joint'],
                amplitude_rad=trial['amplitude_rad'],
                transition_duration_s=trial.get('transition_s', 1.0),
                excitation_duration_s=trial.get('excitation_s', 20.0),
                cycles=1,
            )
            if not prepared.accepted:
                raise RuntimeError(prepared.error_code)
            session = prepared.state
            if not recording_ready() or hardware().state.label != 'active':
                raise RuntimeError('Readiness lost before start')
            trial['start_requested_monotonic_s'] = time.monotonic()
            started = control(ControlJointTest.Request.START, session.permit_challenge)
            if not started.accepted:
                raise RuntimeError(started.error_code)
            trial['status'] = 'executing'
            print(
                trial['joint'] + ': CHIRP STARTED. Ctrl+C requests cancel and motor disable.',
                flush=True,
            )
            duration = trial.get('transition_s', 1.0) + trial.get('excitation_s', 20.0)
            deadline, next_renewal = (
                time.monotonic() + duration + 3.0,
                time.monotonic() + RENEW_INTERVAL_S,
            )
            while time.monotonic() < deadline:
                if not recording_ready():
                    raise RuntimeError('Recording lost; no further renewal')
                state = call(GetJointTestState, '/joint_test_controller/joint_test/state').state
                if state.phase == 'execution_hold':
                    if state.reason_code != 'TEST_COMPLETED':
                        raise RuntimeError(state.reason_code)
                    print(trial['joint'] + ': CHIRP COMPLETED', flush=True)
                    trial['status'] = 'completed'
                    trial['completed_monotonic_s'] = time.monotonic()
                    break
                if state.phase not in ('start_pending', 'executing', 'renewal_pending'):
                    raise RuntimeError('Unexpected test phase: ' + state.phase)
                if (
                    time.monotonic() >= next_renewal
                    and state.permit_challenge
                    and state.applied_sequence == sequence
                ):
                    renewal_started = time.monotonic()
                    if hardware().state.label != 'active' or not recording_ready():
                        raise RuntimeError('Hardware or recording readiness lost')
                    result = control(ControlJointTest.Request.RENEW, state.permit_challenge)
                    if not result.accepted:
                        terminal = call(
                            GetJointTestState, '/joint_test_controller/joint_test/state'
                        ).state
                        if not (
                            result.error_code == 'NOT_READY'
                            and terminal.controller_instance_id == session.controller_instance_id
                            and terminal.activation_generation == session.activation_generation
                            and terminal.session_id == session.session_id
                            and terminal.phase == 'execution_hold'
                            and terminal.reason_code == 'TEST_COMPLETED'
                        ):
                            raise RuntimeError(terminal_failure(result.error_code, terminal))
                        # Completion can occur between state read and renewal.
                        # Do not retry renewal or start another motion here.
                        trial['status'] = 'completed'
                        trial['completed_monotonic_s'] = time.monotonic()
                        print(trial['joint'] + ': CHIRP COMPLETED', flush=True)
                        break
                    next_renewal = renewal_started + RENEW_INTERVAL_S
                time.sleep(0.02)
            else:
                raise RuntimeError('Test completion deadline exceeded')
            # A completed session must be acknowledged cancelled before replacement.
            cancelled = control(ControlJointTest.Request.CANCEL)
            if not cancelled.accepted:
                raise RuntimeError(cancelled.error_code)
            cancel_deadline = time.monotonic() + 1
            while True:
                state = call(GetJointTestState, '/joint_test_controller/joint_test/state').state
                if state.phase == 'cancelled':
                    break
                if time.monotonic() >= cancel_deadline:
                    raise RuntimeError('Cancellation acknowledgement missing')
                time.sleep(0.01)
            (args.output / 'excitation-plan.json').write_text(json.dumps(trials, indent=2))
        # Stop capture before the hardware shutdown tail. Signal only: never
        # wait for disk/ROS cleanup before issuing hardware INACTIVE below.
        if recorder is not None and recorder.poll() is None:
            recorder.send_signal(signal.SIGINT)
            recorder_stop_requested = True
    except BaseException as error:
        for trial in trials:
            if trial['status'] == 'executing':
                trial['status'] = 'failed'
                trial['error'] = str(error)
                trial['failed_monotonic_s'] = time.monotonic()
        raise
    finally:
        # Keep ROS alive during cleanup. A second interrupt must not skip stop.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            # End capture on failure too. Motor shutdown can pause the manager
            # and cause telemetry bursts. Never wait for recorder disk cleanup
            # here, and never let a signal error prevent motor shutdown.
            if recorder is not None and not recorder_stop_requested:
                try:
                    if recorder.poll() is None:
                        recorder.send_signal(signal.SIGINT)
                        recorder_stop_requested = True
                except Exception as error:
                    print('Recorder stop request failed:', error, file=sys.stderr)
            if session is not None:
                try:
                    control(ControlJointTest.Request.CANCEL)
                except Exception as error:
                    print('Cancellation unconfirmed:', error, file=sys.stderr)
            if hardware_name is not None and (handed_over or args.stop_only or prepared_wait):
                try:
                    result = call(
                        SetHardwareComponentState,
                        '/controller_manager/set_hardware_component_state',
                        name=hardware_name,
                        target_state=State(id=State.PRIMARY_STATE_INACTIVE, label='inactive'),
                    )
                    if not result.ok or hardware().state.label != 'inactive':
                        raise RuntimeError('Hardware is not confirmed inactive')
                except Exception as error:
                    raise RuntimeError(
                        'MOTOR STOP UNCONFIRMED. Use physical disconnect.'
                    ) from error
                print(
                    'Hardware INACTIVE confirmed. Normal controller was not restored.', flush=True
                )
        finally:
            primary_error = sys.exc_info()[1]
            try:
                try:
                    stop_recorder(recorder, recorder_stop_requested)
                except Exception as error:
                    if primary_error is None:
                        raise
                    print('Additional recorder cleanup error:', error, file=sys.stderr)
            finally:
                # Export even if recorder cleanup failed. No tokens are recorded.
                try:
                    if args.output.is_dir():
                        try:
                            if trials:
                                (args.output / 'excitation-plan.json').write_text(
                                    json.dumps(trials, indent=2)
                                )
                            (args.output / 'service-timings.json').write_text(
                                json.dumps(
                                    {
                                        'schema': 'joint-test-service-timing.v1',
                                        'records': list(service_timings),
                                    },
                                    indent=2,
                                )
                            )
                        except OSError as error:
                            print('Could not save service timings:', error, file=sys.stderr)
                finally:
                    try:
                        node.destroy_node()
                    finally:
                        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
