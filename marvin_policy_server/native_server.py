"""Opt-in native ROS I/O with the existing Python policy management API.

The ROS executor owns Python policy state. A bounded IPC reader wakes it for
sensor snapshots and publication commits. C++ alone publishes joint commands.
Physical use requires an explicit, constrained deployment profile.
"""

from dataclasses import replace
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.time import Time

from .execution_outcome import ExecutionOutcome
from .inference_schedule import InferenceSchedule
from .marvin_policy_server import MarvinPolicyServer
from .native_channel import NativeChannel
from .native_diagnostics import NativeDiagnostics
from .policy_gc_lifecycle import PolicyGcLifecycle
from .native_protocol import Flags, Kind, Packet
from .native_profile import validate_environment, validate_profile
from .sensor_snapshot import SensorSnapshot


class SnapshotSchedule(InferenceSchedule):
    """Native I/O supplies the only inference deadline."""

    def due(self, now_ns):
        super().due(now_ns)
        return True


class FrameClock:
    def __init__(self, nanoseconds):
        self.nanoseconds = nanoseconds

    def now(self):
        return Time(nanoseconds=self.nanoseconds)


class NativePolicyServer(MarvinPolicyServer):
    def __init__(self):
        self._policy_gc = PolicyGcLifecycle()
        self._native_profile = os.environ.get('MARVIN_POLICY_NATIVE_PROFILE', 'test')
        validate_environment(os.environ.get('ROS_DOMAIN_ID'),
                             os.environ.get('ROS_LOCALHOST_ONLY'), self._native_profile)
        self._native_channel = None
        self._native_child = None
        self._frame = None
        self._native_generation = 0
        self._native_fault = None
        self._native_request_timing = None
        self._native_status_flags = 0
        self._native_status_at = 0.
        self._native_status_generation = -1
        self._native_last_ros = None
        self._pending_result = None
        self._history_requested = False
        self._closing = False
        self._native_diagnostics = None
        super().__init__(native_io=True)
        try:
            self.declare_parameter('native_io_executable', '')
            self._validate_native_profile()
            self.declare_parameter('native_diagnostic_path', '')
            diagnostic_path = str(self.get_parameter('native_diagnostic_path').value)
            if diagnostic_path:
                self._native_diagnostics = NativeDiagnostics(diagnostic_path, self.joint_names)
            binary = Path(str(self.get_parameter('native_io_executable').value))
            if not binary.is_absolute() or not binary.is_file() or not os.access(binary, os.X_OK):
                raise ValueError('native_io_executable must name an existing absolute executable path')
            self._inference_schedule = SnapshotSchedule(self._policy_cadence.periods_ns)
            self._expected_tick_ns = 20_000_000
            # This subscription must not overwrite permission attached to a snapshot.
            self.destroy_subscription(self._operator_permission_subscription)
            self._operator_permission_subscription = None
            self._native_guard = self.create_guard_condition(self._drain_native)
            parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            try:
                self._native_child = subprocess.Popen(
                    self._native_arguments(binary, child.fileno()), pass_fds=(child.fileno(),))
                self._native_channel = NativeChannel(parent, self._native_guard.trigger)
            except Exception:
                parent.close()
                raise
            finally:
                child.close()
            config = tuple(self.default_pos) + tuple(self._safety_min_pos) + tuple(self._safety_max_pos) + (
                float(self._action_scale), self._activation_mgr.cfg.activation_ramp_duration,
                self._activation_mgr.cfg.deactivation_ramp_duration,
                float(self._action_output_mapping == 'smooth_bounded'),
                float(self.get_parameter('inference_max_age_ms').value),
                float(self.get_parameter('publish_period_ms').value),
                float(self.get_parameter('trace_capture_seconds').value))
            flags = (int(self._publish_legacy_commands) | (int(self._publish_traced_commands) << 1) |
                     (int(self._timing.mode != 'off') << 2) |
                     (int(self._timing.mode == 'capture' and self._publish_trace_events) << 3))
            if self._execution_gateway:
                flags = (flags & ~3) | 16
            prefix = (self._trace_source_id << 56) | (self._trace_instance_id << 40)
            ack = self._native_channel.control(Packet(Kind.CONFIGURE, flags=flags, trace_id=prefix, values=config), timeout=5.)
            if not ack.flags & Flags.SUCCESS:
                raise RuntimeError('Native configuration rejected')
            self._native_generation = ack.generation
            self._logger.info('Native ROS I/O connected. Python management remains available; policy is inactive.')
        except Exception:
            self.close_native()
            raise

    def _validate_native_profile(self):
        domain = os.environ.get('ROS_DOMAIN_ID')
        validate_profile(domain, self._native_profile, self.get_namespace(),
                         bool(self.get_parameter('use_sim_time').value),
                         self._operator_permission.required, self.joint_names, self._native_topics())
        if self._native_profile == 'marvin-v1':
            if self.get_parameter('inference_max_age_ms').value != 100.0:
                raise ValueError('marvin-v1 requires inference_max_age_ms=100')
            if self._publish_legacy_commands or not self._publish_traced_commands:
                raise ValueError('marvin-v1 requires traced commands only')
            if self.resolve_topic_name(str(self.get_parameter('joy_topic').value)) != '/joy':
                raise ValueError('marvin-v1 requires the robot-local /joy receiver')
        if len(self.joint_names) != 12 or not self._safety_limits_enabled or not self._safety_limits_loaded:
            raise ValueError('Native I/O requires twelve joints and enabled, loaded position limits')
        if not np.isfinite(self._safety_min_pos).all() or not np.isfinite(self._safety_max_pos).all():
            raise ValueError('Native I/O requires finite limits for every joint')
        if self.get_parameter('publish_period_ms').value != 5:
            raise ValueError('Native I/O currently requires publish_period_ms=5')

    def _native_topics(self):
        resolve = self.resolve_topic_name
        return {
            'joint_states_topic': resolve(self._joint_states_topic),
            'dynamic_joint_states_topic': resolve(self._dynamic_joint_states_topic) if self._dynamic_joint_states_topic else '',
            'imu_topic': resolve(self._imu_topic),
            'permission_topic': resolve('/operator/motion_permitted'),
            'command_topic': resolve(self._command_topic),
            'traced_command_topic': resolve(self._traced_command_topic),
            'trace_event_topic': resolve(self._trace_event_topic),
            'metrics_topic': resolve('/trace/metrics'),
            'candidate_topic': resolve('execution/policy/commands'),
        }

    def _native_arguments(self, binary, fd):
        params = dict(self._native_topics(),
                      use_sim_time=bool(self.get_parameter('use_sim_time').value),
                      joint_names=list(self.joint_names),
                      operator_input_required=self._operator_permission.required)
        args = [str(binary), str(fd), '--ros-args', '-r', f'__ns:={self.get_namespace()}']
        for name, value in params.items():
            args += ['-p', f'{name}:={json.dumps(value)}']
        return args

    def get_clock(self):
        frame = getattr(self, '_frame', None)
        return FrameClock(frame.ros_ns) if frame is not None else super().get_clock()

    def _next_trace_id(self):
        return self._frame.trace_id if self._frame is not None else super()._next_trace_id()

    def prepare_execution(self, generation: str, initial_state_json: str) -> ExecutionOutcome:
        outcome = super().prepare_execution(generation, initial_state_json)
        if not outcome.success:
            return outcome
        try:
            preparation = self._execution_preparation
            ack = self._native_channel.control(Packet(
                Kind.PREPARE, token=preparation.generation,
                requested_ns=int(preparation.expires * 1e9), values=preparation.targets))
            if not ack.flags & Flags.SUCCESS:
                raise ValueError('Native policy preparation rejected')
            self._observe_generation(ack.generation)
        except Exception as error:
            self._execution_preparation.generation = None
            outcome.success, outcome.message = False, str(error)
        return outcome

    def request_activation(self, active: bool) -> ExecutionOutcome:
        outcome = ExecutionOutcome()
        if self._native_channel is None:
            outcome.success, outcome.message = False, 'Native I/O is not ready'
            return outcome
        if not active and self._native_output_process_exited():
            self._fail_native('Native output process exited', notify=False, output_stopped=True)
            outcome.success, outcome.message = True, 'Native output process exited; output released'
            return outcome
        if active and self._activation_mgr.state.active:
            fresh = (time.monotonic() - self._native_status_at < .5 and
                     self._native_status_generation == self._native_generation)
            outcome.success = bool(fresh and self._native_status_flags & Flags.ACTIVE)
            outcome.message = 'Already active' if outcome.success else 'Native activation state is not confirmed'
            return outcome
        try:
            # Validate preparation and reset inference before granting native output.
            # This operation and inference share the same ROS executor thread.
            outcome = super().request_activation(active)
            if not outcome.success:
                return outcome
            if active:
                self._policy_gc.enter_output_window()
            previous_action = self.action.copy()
            ack = self._native_channel.control(Packet(Kind.ACTIVATE if active else Kind.DEACTIVATE))
            if not ack.flags & Flags.SUCCESS:
                if active:
                    # A rejected activation never grants native output.
                    self._policy_gc.confirm_output_stopped()
                raise ValueError('Native activation rejected: sensor inputs or preparation are not ready')
            self._observe_generation(ack.generation)
            if not active:
                self._policy_gc.request_restore()
                self.action = previous_action
                if self._execution_gateway:
                    self._execution_preparation.generation = None
                    self._activation_mgr.state.deactivation_start_time = None
                    self._activation_mgr.state.deactivation_start_action = None
                    outcome.message = 'Native policy output released'
            self._native_fault = None
            if active:
                self._activation_mgr.state.activation_start_time = ack.ros_ns / 1e9
            return outcome
        except Exception as exc:
            self._fail_native(str(exc))
            outcome.success, outcome.message = False, f'Native policy operation failed: {exc}'
            return outcome

    def _apply_policy_input_lock(self, input_name, locked, values=None):
        result = super()._apply_policy_input_lock(input_name, locked, values)
        if result[0] and self._native_channel is not None:
            try:
                ack = self._native_channel.control(Packet(Kind.INVALIDATE))
                if not ack.flags & Flags.SUCCESS:
                    raise RuntimeError('Native override invalidation rejected')
                self._observe_generation(ack.generation)
            except Exception as exc:
                self._fail_native(str(exc))
                return False, f'Override updated locally, but native output stopped: {exc}'
        return result

    def _native_output_process_exited(self):
        child = getattr(self, '_native_child', None)
        return child is not None and child.poll() is not None

    def _execution_state(self):
        # A socket fault alone is not proof. Reaping the sole command publisher
        # proves that it cannot publish more commands, even without a final ACK.
        if self._native_output_process_exited():
            return {'active': False, 'output_released': True}
        fresh = (time.monotonic() - self._native_status_at < .5 and
                 self._native_status_generation == self._native_generation)
        return {'active': bool(self._native_status_flags & Flags.ACTIVE),
                'output_released': bool(fresh and self._native_status_flags & Flags.OUTPUT_RELEASED)}

    def _observe_generation(self, generation):
        if generation <= self._native_generation:
            return
        self._native_generation = generation
        self._pending_result = None
        self._history_requested = False
        self._policy_runner.reset()
        self._inference_schedule.reset()
        self.action = np.zeros(len(self.joint_names))
        self._latest_estimated_velocity = None
        if self._velocity_estimator is not None:
            self._velocity_estimator.reset()

    def _fail_native(self, message, *, notify=True, output_stopped=False):
        if self._native_fault != message:
            self._logger.error(f'Native policy output stopped: {message}. Explicit reactivation is required.')
            timing = getattr(self, '_native_request_timing', None)
            if timing is not None:
                self._logger.error('NATIVE_PYTHON_FAULT ' + json.dumps(timing, separators=(',', ':')))
        self._native_fault = message
        self._pending_result = None
        self._activation_mgr.state.active = False
        self._activation_mgr.state.deactivation_start_time = None
        self._activation_mgr.state.deactivation_start_action = None
        if output_stopped:
            self._policy_gc.confirm_output_stopped()
        else:
            # Sending a fault is not proof that the native output process has
            # handled it. Keep GC suppressed until status or child exit proves
            # that output stopped.
            self._policy_gc.request_restore()
        if notify and self._native_channel is not None:
            try:
                self._native_channel.send(Packet(Kind.FAULT))
            except RuntimeError:
                pass

    def _drain_native(self):
        if self._closing or self._native_channel is None:
            return
        try:
            for packet in self._native_channel.drain():
                if packet.generation < self._native_generation:
                    continue
                self._observe_generation(packet.generation)
                if packet.kind == Kind.COMMIT:
                    self._commit(packet)
                    continue
                if packet.kind not in {Kind.STATUS, Kind.SNAPSHOT} or len(packet.values) != 58:
                    raise ValueError('Invalid native sensor/status packet')
                self._native_status_flags = packet.flags
                self._native_status_at = time.monotonic()
                self._native_status_generation = packet.generation
                self._policy_gc.observe_native_state(
                    active=bool(packet.flags & Flags.ACTIVE),
                    output_released=bool(packet.flags & Flags.OUTPUT_RELEASED))
                if packet.flags & Flags.FAULTED:
                    reasons = {1: 'Inference deadline exceeded', 2: 'Invalid native result',
                               3: 'Python/native transport heartbeat expired'}
                    self._fail_native(reasons.get(packet.trace_id, 'Native lifecycle fault'),
                                      notify=False, output_stopped=True)
                if self._native_last_ros is not None and packet.ros_ns < self._native_last_ros:
                    if self._activation_mgr.is_active():
                        self._activation_mgr.state.activation_start_time = packet.ros_ns / 1e9
                    if self._joint_position_filter is not None:
                        self._joint_position_filter.reset()
                self._native_last_ros = packet.ros_ns
                self._operator_permission.update(bool(packet.flags & Flags.PERMITTED))
                if packet.flags & Flags.SENSORS_VALID:
                    sensors = SensorSnapshot.from_values(self.joint_names, packet.joint_ns, packet.imu_ns, packet.values[:34])
                    self._joint_state_cb(sensors.joint)
                    self._imu_cb(sensors.imu)
                    if packet.flags & Flags.ENCODER_TIMING:
                        self._input_timing.received_encoders(sensors.joint, dict(zip(self.joint_names, packet.values[46:58])))
                else:
                    self._latest_joint_state = self._latest_imu = None
                if packet.kind == Kind.STATUS:
                    self._native_channel.send(Packet(Kind.STATUS))
                    continue
                if not self._activation_mgr.is_active() or self._native_fault:
                    continue
                started_ns = time.monotonic_ns()
                started_cpu_ns = time.thread_time_ns()
                self._native_request_timing = {
                    'sequence': packet.sequence, 'generation': packet.generation,
                    'request_to_callback_ms': (started_ns - packet.requested_ns) / 1e6,
                    'callback_ms': None, 'request_to_result_send_ms': None,
                    'callback_thread_cpu_ms': None,
                }
                if started_ns - packet.requested_ns > self._inference_max_age_ns:
                    raise RuntimeError('Native snapshot expired before Python inference')
                self._frame = packet
                self._pending_result = None
                try:
                    self._timer_tick()
                    if self._pending_result is None:
                        raise RuntimeError('Policy computation produced no native result')
                finally:
                    self._native_request_timing['callback_ms'] = (time.monotonic_ns() - started_ns) / 1e6
                    self._native_request_timing['callback_thread_cpu_ms'] = (
                        time.thread_time_ns() - started_cpu_ns) / 1e6
                    self._frame = None
        except Exception as exc:
            self._frame = None
            self._fail_native(str(exc))

    def _publish_joint_command(self, command_values, trace_id=None, **kwargs):
        if self._frame is None:
            raise RuntimeError('Native publication requires an owned sensor request')
        policy = np.asarray(self._apply_safety_position_limits(command_values), dtype=np.float64)
        self._latest_policy_joint_command = policy.copy()
        flags = 0
        if self._nan_obs_warned:
            values = tuple(policy)
            flags |= Flags.DIRECT_OUTPUT
        else:
            values = tuple(self.action)
        locked = np.zeros(12)
        if self._joint_command_output_locked and self._locked_joint_command is not None:
            locked = self._locked_joint_command
            flags |= Flags.OUTPUT_LOCKED
        result = replace(self._frame, kind=Kind.RESULT, flags=int(flags), values=values + tuple(locked))
        self._native_channel.send(result)
        timing = getattr(self, '_native_request_timing', None)
        if timing is not None:
            timing['request_to_result_send_ms'] = (time.monotonic_ns() - self._frame.requested_ns) / 1e6
        self._pending_result = result
        self._history_requested = False
        diagnostics = getattr(self, '_native_diagnostics', None)
        if diagnostics is not None:
            # The observation builder stores the effective command after locks
            # and permission overrides. Do not substitute the raw joystick value.
            effective_command = None
            if not flags & Flags.DIRECT_OUTPUT:
                candidate = self._obs_state.velocity_commands_history[-1]
                if np.isfinite(candidate).all():
                    effective_command = candidate
            diagnostics.record('result_submitted', result,
                               estimated_velocity=self._latest_estimated_velocity,
                               sensor_values=self._frame.values[:34],
                               effective_velocity_command=effective_command)
        return result.trace_id

    def _advance_action_history(self):
        # Commit only after C++ confirms the first publication of this result.
        self._history_requested = True

    def _commit(self, packet):
        pending = self._pending_result
        if pending is None or (packet.sequence, packet.generation) != (pending.sequence, pending.generation):
            return
        if len(packet.values) != 24 or packet.trace_id != pending.trace_id:
            raise ValueError('Invalid native publication commit')
        diagnostics = getattr(self, '_native_diagnostics', None)
        if diagnostics is not None:
            diagnostics.record('publication_commit', packet)
        self._latest_policy_joint_command = np.asarray(packet.values[:12])
        self._last_published_joint_command = np.asarray(packet.values[12:])
        if self._history_requested:
            super()._advance_action_history()
        self._pending_result = None
        self._history_requested = False
        self._timing.count('native_result_commits')

    def _observation_status_payload(self):
        payload = super()._observation_status_payload()
        payload['execution'] = {
            'io_backend': 'native', 'profile': self._native_profile, 'fault': self._native_fault,
            'generation': self._native_generation,
            'native_active': bool(self._native_status_flags & Flags.ACTIVE),
            'sensors_valid': bool(self._native_status_flags & Flags.SENSORS_VALID),
            'paused': bool(self._native_status_flags & Flags.PAUSED),
        }
        return payload

    def close_native(self):
        self._closing = True
        child_stopped = self._native_child is None
        try:
            if self._native_channel is not None:
                self._native_channel.close()
            if self._native_child is not None:
                try:
                    self._native_child.wait(timeout=2.)
                    child_stopped = True
                except subprocess.TimeoutExpired:
                    self._native_child.terminate()
                    try:
                        self._native_child.wait(timeout=2.)
                        child_stopped = True
                    except subprocess.TimeoutExpired:
                        self._native_child.kill()
                        self._native_child.wait(timeout=2.)
                        child_stopped = True
        finally:
            if child_stopped:
                self._policy_gc.confirm_output_stopped()

        diagnostics = getattr(self, '_native_diagnostics', None)
        if diagnostics is not None:
            diagnostics.close()
            if diagnostics.error:
                self._logger.error(f'Native diagnostic output incomplete: {diagnostics.error}')

    def destroy_node(self):
        self.close_native()
        return super().destroy_node()


def main(args=None):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rclpy.init(args=args)
    node = None
    try:
        node = NativePolicyServer()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node._close_velocity_estimator_shadow()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
