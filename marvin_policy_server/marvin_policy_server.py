#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
import torch
import numpy as np
import io
import json
import os
import time
from array import array
from pathlib import Path
from .execution_outcome import ExecutionOutcome
from .artifact_identity import artifact_identity
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState, Imu 
from std_msgs.msg import Float64MultiArray, String, Bool
from std_srvs.srv import SetBool, Trigger
from marvin_policy_interfaces.srv import SetPolicyInputLock, PolicyControl as PolicyControlService
from marvin_trace_msgs.msg import TracedFloat64MultiArray, TraceEvent
from runtime_interfaces.msg import ExecutorCommand
from runtime_interfaces.srv import PrepareExecution
from .execution_preparation import PolicyPreparation
# from message_filters import Subscriber, TimeSynchronizer, ApproximateTimeSynchronizer
from sensor_msgs.msg import Joy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.qos import qos_profile_sensor_data

# New modular imports
from .activation_manager import ActivationManager, ActivationConfig, ActivationState
from .action_history_feedback import (
    VALID_ACTION_HISTORY_FEEDBACK_MODES,
    action_history_feedback,
)
from .action_output_mapping import SmoothBoundedActionMapper
from .concurrent_policy import ConcurrentPolicyRunner, load_concurrent_policy_manifest
from .observation import ObservationBuilder, ObservationConfig, ObservationState, quat_to_rot_matrix  # noqa: F401 (quat reuse)
from .policy_runner import PolicyRunner, PolicyRunnerConfig
from .env_config_loader import EnvConfigLoader
from .sim_timing import simulation_gate
from .joy_command_recording import policy_velocity_command_from_axes
from .joint_position_filter import ExponentialJointPositionFilter
from .velocity_estimator import VelocityEstimator
from .velocity_estimator_diagnostics import VelocityEstimatorDiagnosticRecorder
from .velocity_estimator_features import VelocityEstimatorFeatureBuilder
from .velocity_estimator_shadow import VelocityEstimatorShadowRecorder
from .policy_control import PolicyControl, PolicyControlError
from .timing import Timing
from .input_timing import InputTiming
from .inference_schedule import InferenceSchedule
from .inference_inputs import runner_inputs
from .inference_process import InferenceProcess
from .async_policy import create_runner_task
from .command_clip_cache import CommandClipCache
from .policy_node import PolicyNode
from .concurrent_policy.optimized import optimize_policy
from copy import deepcopy
from .encoder_snapshot import decode_snapshot
from .operator_permission import OperatorPermission



class MarvinPolicyServer(PolicyNode):
    """Fullbody controller for Marvin robot.
    
    This ROS 2 node subscribes to velocity commands and synchronized joint/IMU
    data, processes the data through a neural network policy, and publishes
    joint commands for controlling the Marvin robot's movements.
    """

    def __init__(self, *, native_io=False):
        """Initialize the marvin controller node."""
        super().__init__('marvin_policy_server')
        # Only the native adapter supplies this internal option. Normal launch
        # still owns its ROS sensors, publication timer and command publishers.
        self._native_io = bool(native_io)
        self.declare_parameter('execution_gateway', False)
        self._execution_gateway = bool(self.get_parameter('execution_gateway').value)
        self._policy_control = PolicyControl(self._observation_status_payload, self._apply_policy_input_lock)
        self._policy_control_service = self.create_service(
            PolicyControlService, "policy_control", self._policy_control_cb)

        # Declare and set parameters
        self.declare_parameter('publish_period_ms', 5)
        # Opt-in until matching simulation and physical acceptance checks pass.
        self.declare_parameter('async_inference', False)
        self.declare_parameter('inference_max_age_ms', 100.0)
        self.declare_parameter('compact_estimator', False)
        self.declare_parameter('policy_path', 'policy/policy.pt')
        self.declare_parameter('env_path', 'policy/env.yaml')
        self.declare_parameter('activation_ramp_duration', 1.0)    # seconds to smoothly ramp in
        self.declare_parameter('deactivation_ramp_duration', 1.0)  # seconds to smoothly ramp out
        self.declare_parameter('release_after_deactivate', True)   # if True, stop publishing after ramp-down
        self.declare_parameter('joy_topic', '/joy')
        self.declare_parameter('operator_input_required', False)
        self._operator_permission = OperatorPermission(
            required=bool(self.get_parameter('operator_input_required').value))
        self._operator_permission_subscription = self.create_subscription(
            Bool, '/operator/motion_permitted',
            lambda msg: self._operator_permission.update(msg.data), 1)
        self.declare_parameter('joint_states_topic', '/joint_states')
        # Explicit same-host/boot diagnostic opt-in; empty preserves normal input.
        self.declare_parameter('same_host_dynamic_joint_states_topic', '')
        self.declare_parameter('imu_topic', '/imu')
        self.declare_parameter('base_lin_vel_source', 'imu_integration')
        self.declare_parameter('odometry_topic', '/isaac_odometry')
        self.declare_parameter('odometry_child_frame_id', 'trunk')
        self.declare_parameter('base_lin_vel_timeout_sec', 0.25)
        default_velocity_estimator_path = str(
            Path(get_package_share_directory('marvin_policy_server'))
            / 'policy'
            / 'velocity_estimator.pt'
        )
        self.declare_parameter('velocity_estimator_shadow_enabled', False)
        self.declare_parameter('velocity_estimator_checkpoint', default_velocity_estimator_path)
        self.declare_parameter('velocity_estimator_shadow_topic', 'velocity_estimator_shadow')
        self.declare_parameter('velocity_estimator_shadow_log_dir', '')
        self.declare_parameter('velocity_estimator_shadow_summary_interval_samples', 250)
        self.declare_parameter('velocity_estimator_diagnostic_log_dir', '')
        self.declare_parameter('velocity_estimator_add_gravity_to_imu_acceleration', False)
        self.declare_parameter('velocity_estimator_imu_acceleration_scale', 1.0)
        self.declare_parameter('velocity_estimator_gravity_magnitude', 9.81)
        self.declare_parameter('command_topic', 'marvin_joint_controller/commands')
        self.declare_parameter('traced_command_topic', 'marvin_joint_controller/commands_traced')
        self.declare_parameter('publish_legacy_commands', False)
        self.declare_parameter('tracing_mode', os.environ.get('MARVIN_TRACING_MODE', 'metrics'))
        self.declare_parameter('trace_capture_seconds', float(os.environ.get('MARVIN_TRACE_CAPTURE_SECONDS', '30')))
        self.declare_parameter('publish_traced_commands', True)
        self.declare_parameter('trace_event_topic', '/trace/events')
        self.declare_parameter('publish_trace_events', True)
        self.declare_parameter('trace_source_id', 1)
        self.declare_parameter('trace_instance_id', -1)
        self.declare_parameter('set_active_service_name', 'set_active')
        self.declare_parameter('set_policy_input_lock_service_name', 'set_policy_input_lock')
        self.declare_parameter('safety_limits_enabled', True)
        default_safety_limits_path = str(
            Path(get_package_share_directory('description'))
            / 'models'
            / 'limits'
            / 'marvin_joint_limits.yaml'
        )
        self.declare_parameter('safety_limits_path', default_safety_limits_path)
        self.declare_parameter('action_output_mapping', 'linear')
        self.declare_parameter('actor_action_history_mode', 'raw')
        self.declare_parameter('joint_position_filter_tau_sec', 0.0)

        self._logger = self.get_logger()
        self._joy_topic = str(self.get_parameter('joy_topic').value)
        self._joint_states_topic = str(self.get_parameter('joint_states_topic').value)
        self._dynamic_joint_states_topic = str(
            self.get_parameter('same_host_dynamic_joint_states_topic').value)
        self._imu_topic = str(self.get_parameter('imu_topic').value)
        self._base_lin_vel_source = str(self.get_parameter('base_lin_vel_source').value).strip().lower()
        self._odometry_topic = str(self.get_parameter('odometry_topic').value).strip()
        self._odometry_child_frame_id = str(self.get_parameter('odometry_child_frame_id').value).strip()
        self._base_lin_vel_timeout_sec = float(self.get_parameter('base_lin_vel_timeout_sec').value)
        self._velocity_estimator_shadow_enabled = self._coerce_bool(
            self.get_parameter('velocity_estimator_shadow_enabled').value
        )
        self._velocity_estimator_checkpoint_raw = str(
            self.get_parameter('velocity_estimator_checkpoint').value
        ).strip()
        self._velocity_estimator_shadow_topic = str(
            self.get_parameter('velocity_estimator_shadow_topic').value
        ).strip()
        self._velocity_estimator_shadow_log_dir = str(
            self.get_parameter('velocity_estimator_shadow_log_dir').value
        ).strip()
        self._velocity_estimator_shadow_summary_interval_samples = int(
            self.get_parameter('velocity_estimator_shadow_summary_interval_samples').value
        )
        self._velocity_estimator_diagnostic_log_dir = str(
            self.get_parameter('velocity_estimator_diagnostic_log_dir').value
        ).strip()
        self._velocity_estimator_add_gravity_to_imu_acceleration = self._coerce_bool(
            self.get_parameter('velocity_estimator_add_gravity_to_imu_acceleration').value
        )
        self._velocity_estimator_gravity_magnitude = float(
            self.get_parameter('velocity_estimator_gravity_magnitude').value
        )
        self._velocity_estimator_imu_acceleration_scale = float(
            self.get_parameter('velocity_estimator_imu_acceleration_scale').value
        )
        if self._base_lin_vel_source not in {"imu_integration", "odometry", "estimator"}:
            raise ValueError(
                "base_lin_vel_source must be one of: imu_integration, odometry, estimator; "
                f"got {self._base_lin_vel_source!r}"
            )
        if not self._odometry_topic:
            raise ValueError("odometry_topic must not be empty")
        if self._base_lin_vel_timeout_sec <= 0.0:
            raise ValueError("base_lin_vel_timeout_sec must be greater than zero")
        if self._velocity_estimator_shadow_enabled and not self._velocity_estimator_shadow_topic:
            raise ValueError("velocity_estimator_shadow_topic must not be empty when shadow mode is enabled")
        if self._velocity_estimator_shadow_summary_interval_samples <= 0:
            raise ValueError("velocity_estimator_shadow_summary_interval_samples must be greater than zero")
        if (
            not np.isfinite(self._velocity_estimator_gravity_magnitude)
            or self._velocity_estimator_gravity_magnitude <= 0.0
        ):
            raise ValueError("velocity_estimator_gravity_magnitude must be finite and greater than zero")
        self._env_path_raw = str(self.get_parameter('env_path').value).strip()
        self._command_topic = str(self.get_parameter('command_topic').value)
        self._traced_command_topic = str(self.get_parameter('traced_command_topic').value)
        self._publish_legacy_commands = bool(self.get_parameter('publish_legacy_commands').value)
        self._publish_traced_commands = bool(self.get_parameter('publish_traced_commands').value)
        self._trace_event_topic = str(self.get_parameter('trace_event_topic').value)
        self._publish_trace_events = bool(self.get_parameter('publish_trace_events').value)
        self._trace_source_id = int(self.get_parameter('trace_source_id').value)
        raw_trace_instance_id = int(self.get_parameter('trace_instance_id').value)
        self._set_active_service_name = str(self.get_parameter('set_active_service_name').value)
        self._set_policy_input_lock_service_name = str(
            self.get_parameter('set_policy_input_lock_service_name').value
        )
        self._safety_limits_enabled = self._coerce_bool(self.get_parameter('safety_limits_enabled').value)
        self._safety_limits_path_raw = str(self.get_parameter('safety_limits_path').value).strip()
        self._action_output_mapping = str(
            self.get_parameter('action_output_mapping').value
        ).strip().lower()
        if self._action_output_mapping not in {'linear', 'smooth_bounded'}:
            raise ValueError(
                "action_output_mapping must be one of: linear, smooth_bounded; "
                f"got {self._action_output_mapping!r}"
            )
        self._actor_action_history_mode = str(
            self.get_parameter('actor_action_history_mode').value
        ).strip().lower()
        if (
            self._actor_action_history_mode
            not in VALID_ACTION_HISTORY_FEEDBACK_MODES
        ):
            choices = ', '.join(
                sorted(VALID_ACTION_HISTORY_FEEDBACK_MODES)
            )
            raise ValueError(
                'actor_action_history_mode must be one of: '
                f'{choices}; got {self._actor_action_history_mode!r}'
            )
        self._joint_position_filter_tau_sec = float(
            self.get_parameter('joint_position_filter_tau_sec').value
        )
        if (
            not np.isfinite(self._joint_position_filter_tau_sec)
            or self._joint_position_filter_tau_sec < 0.0
        ):
            raise ValueError(
                'joint_position_filter_tau_sec must be finite and '
                'non-negative'
            )
        if not self._publish_legacy_commands and not self._publish_traced_commands:
            self._logger.warn("Both publish_legacy_commands and publish_traced_commands are false. Enabling legacy publishing.")
            self._publish_legacy_commands = True
        if self._trace_source_id < 0 or self._trace_source_id > 255:
            self._logger.warn(
                f"trace_source_id={self._trace_source_id} out of range [0,255]. Falling back to 1.")
            self._trace_source_id = 1
        if raw_trace_instance_id < 0:
            self._trace_instance_id = os.getpid() & 0xFFFF
        else:
            self._trace_instance_id = raw_trace_instance_id & 0xFFFF
        self._trace_counter = 0
        
        # Sensor QoS (BestEffort, depth=1)
        sensor_qos = qos_profile_sensor_data  # built-in: BestEffort + KeepLast(10); we’ll shrink depth below
        sensor_qos.depth = 1

        # Command QoS: BestEffort, depth=1
        cmd_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        trace_event_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
        )

        # Create subscription for velocity commands
        # self._cmd_vel_subscription = self.create_subscription(
        #     Twist,
        #     '/cmd_vel',
        #     self._cmd_vel_callback,
        #     qos_profile=10)

        self._joy_subscription = self.create_subscription(
            Joy,
            self._joy_topic,
            self._joy_callback,
            qos_profile=10
        )

        sim_qos_profile = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            history=rclpy.qos.HistoryPolicy.KEEP_ALL,
            # depth=1
        )

        # Publisher (use cmd_qos)
        self._joint_publisher = None
        self._joint_publisher_traced = None
        self._trace_event_publisher = None
        self._velocity_estimator_shadow_publisher = None
        if self._publish_legacy_commands and not self._native_io and not self._execution_gateway:
            self._joint_publisher = self.create_publisher(Float64MultiArray, self._command_topic, qos_profile=cmd_qos)
        if self._publish_traced_commands and not self._native_io and not self._execution_gateway:
            self._joint_publisher_traced = self.create_publisher(
                TracedFloat64MultiArray, self._traced_command_topic, qos_profile=cmd_qos
            )
        if self._publish_trace_events and self.get_parameter('tracing_mode').value == 'capture':
            self._trace_event_publisher = self.create_publisher(
                TraceEvent, self._trace_event_topic, qos_profile=trace_event_qos
            )
        if self._velocity_estimator_shadow_enabled:
            self._velocity_estimator_shadow_publisher = self.create_publisher(
                Float64MultiArray,
                self._velocity_estimator_shadow_topic,
                qos_profile=cmd_qos,
            )
        # self._joint_publisher = self.create_publisher(JointState, 'isaac_joint_commands', qos_profile=sim_qos_profile)

        # Subscriptions (direct, store latest messages)
        self._latest_joint_state = None
        self._latest_imu = None
        self._latest_odometry_velocity = None
        self._latest_odometry_received_monotonic = None
        self._latest_odometry_stamp_sec = None
        self._odometry_validation_error = None
        self._base_lin_vel_warned = False

        self._joint_state_sub = None
        self._imu_sub = None
        if self._native_io:
            pass  # The native adapter supplies coherent sensor snapshots.
        elif self._dynamic_joint_states_topic:
            from control_msgs.msg import DynamicJointState
            self._joint_state_sub = self.create_subscription(
                DynamicJointState, self._dynamic_joint_states_topic,
                self._dynamic_joint_state_cb, qos_profile=sensor_qos)
        else:
            self._joint_state_sub = self.create_subscription(
                JointState, self._joint_states_topic,
                self._joint_state_cb, qos_profile=sensor_qos)
        if not self._native_io:
            self._imu_sub = self.create_subscription(
                Imu,
                self._imu_topic,
                self._imu_cb,
                qos_profile=sensor_qos
            )
        self._odometry_sub = None
        if (
            self._base_lin_vel_source == "odometry"
            or self._velocity_estimator_shadow_enabled
            or self._velocity_estimator_diagnostic_log_dir
        ):
            self._odometry_sub = self.create_subscription(
                Odometry,
                self._odometry_topic,
                self._odometry_cb,
                qos_profile=sensor_qos,
            )

        # Timer for control loop
        publish_period_ms = self.get_parameter('publish_period_ms').value
        self._timer = None
        if not self._native_io:
            self._timer = self.create_timer(publish_period_ms / 1000.0, self._timer_tick)

        # Load neural network policy
        self.policy_path = self.get_parameter('policy_path').value
        self._loaded_artifacts = {}
        self.load_policy()
        self._concurrent_policy_manifest = load_concurrent_policy_manifest(Path(self.policy_path))
        self._concurrent_policy_enabled = self._concurrent_policy_manifest is not None
        if self.get_parameter('compact_estimator').value and not self._concurrent_policy_enabled:
            raise ValueError('compact_estimator requires an embedded-estimator artifact')
        if self._concurrent_policy_enabled:
            self._loaded_artifacts['metadata.json'] = self._concurrent_policy_manifest.loaded_artifact
            if self.get_parameter('compact_estimator').value:
                # An explicit request fails startup if numerical validation fails.
                self.policy = optimize_policy(self.policy, self._concurrent_policy_manifest)
                self._logger.info('Compact estimator validated against the original artifact.')
        self._policy_input_dim = (
            self._concurrent_policy_manifest.base_observation_count
            if self._concurrent_policy_manifest is not None
            else self._infer_policy_input_dim(self.policy)
        )
        self._env_path = self._resolve_env_path(self._env_path_raw)

        # Initialize state variables (original kept; structured below)
        self._joint_state = JointState()
        # self._joint_command = JointState()
        self._cmd_vel = Twist()
        self._imu = Imu()
        self._command_input_locked = False
        self._imu_input_locked = False
        self._joint_position_input_locked = False
        self._velocity_input_locked = False
        self._joint_command_output_locked = False
        self._estimator_velocity_override_mode = "live"
        self._locked_command_input = None
        self._locked_imu_input = None
        self._locked_joint_position_relative = None
        self._locked_joint_velocity = None
        self._locked_joint_command = None
        self._locked_estimator_velocity = None
        self._locked_estimator_fk = None
        self._locked_estimator_foot_height = None
        self._locked_estimator_contact = None
        self._latest_policy_joint_command = None
        self._last_published_joint_command = None
        self._action_scale = 0.25
        self._stale_warn_count = 0
        
        env_loader = EnvConfigLoader(self._env_path)
        manifest = self._concurrent_policy_manifest
        self._policy_cadence = env_loader.get_policy_cadence(manifest.training if manifest else None)
        profile = env_loader.get_deployment_timing()
        delay_max_ns = int(max(profile['inference_delay_ms']) * 1e6) if profile and self.get_parameter('use_sim_time').value else 0
        self._policy_cadence.validate_backend(int(publish_period_ms * 1e6),
            native=self._native_io, delay_max_ns=delay_max_ns)
        self.joint_names = env_loader.get_joint_names()
        self._execution_preparation = PolicyPreparation(self.joint_names)
        self._execution_candidate_publisher = (self.create_publisher(
            ExecutorCommand, 'execution/policy/commands', qos_profile=cmd_qos)
            if self._execution_gateway and not self._native_io else None)
        self.create_service(PrepareExecution, 'prepare_execution', self._prepare_execution)
        self.default_pos = env_loader.get_default_joint_positions()
        self._sim_release = simulation_gate(env_loader.get_deployment_timing(), self.default_pos,
            use_sim_time=bool(self.get_parameter('use_sim_time').value), native_io=self._native_io,
            expected_hz=self._policy_cadence.mean_hz)
        if self._sim_release is not None:
            # Timed fitted scenes publish specific force. Readiness validates the
            # scene contract before inference. Legacy and physical inputs retain
            # their configured conversion. Both estimator builders use this value.
            self._velocity_estimator_add_gravity_to_imu_acceleration = False
            self._sim_identity_subscription = self.create_subscription(String, '/marvin_sim/timing_profile',
                self._sim_identity, QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
            self._sim_release_publisher = self.create_publisher(String, 'timing/action_release', 10)
        self._safety_min_pos = np.full(len(self.joint_names), -np.inf, dtype=np.float64)
        self._safety_max_pos = np.full(len(self.joint_names), np.inf, dtype=np.float64)
        self._safety_limits_path = self._resolve_package_path(
            self._safety_limits_path_raw,
            default_relative=default_safety_limits_path,
        )
        self._safety_limits_loaded = False
        self._last_safety_clip_warn_time = 0.0
        self._load_safety_position_limits()
        self._smooth_bounded_action_mapper = None
        if self._action_output_mapping == 'smooth_bounded':
            if not self._safety_limits_enabled or not self._safety_limits_loaded:
                raise RuntimeError(
                    "smooth_bounded action output requires enabled, fully specified safety limits"
                )
            try:
                self._smooth_bounded_action_mapper = SmoothBoundedActionMapper(
                    self.default_pos,
                    self._safety_min_pos,
                    self._safety_max_pos,
                    local_action_scale=self._action_scale,
                )
            except ValueError as exc:
                raise RuntimeError(f"Invalid smooth_bounded action output configuration: {exc}") from exc
            self._logger.warn(
                "Experimental smooth_bounded action output is enabled. "
                "The selected policy was not trained with this mapping."
            )
        self._policy_observation_schema = env_loader.get_policy_observation_schema()
        policy_terms = {item["key"] for item in self._policy_observation_schema}
        self._base_lin_vel_required = "base_lin_vel" in policy_terms
        history_lengths = env_loader.get_policy_term_history_lengths(
            (
                "base_lin_vel",
                "base_ang_vel",
                "projected_gravity",
                "velocity_commands",
                "joint_pos",
                "joint_vel",
                "actions",
            )
        )
        if "base_lin_vel" not in policy_terms:
            history_lengths["base_lin_vel"] = 0

        # --- Modular components ---
        obs_cfg = ObservationConfig(
            base_lin_vel_history_len=history_lengths["base_lin_vel"],
            base_ang_vel_history_len=history_lengths["base_ang_vel"],
            projected_gravity_history_len=history_lengths["projected_gravity"],
            velocity_commands_history_len=history_lengths["velocity_commands"],
            joint_pos_history_len=history_lengths["joint_pos"],
            joint_vel_history_len=history_lengths["joint_vel"],
            actions_history_len=history_lengths["actions"],
        )
        self._obs_cfg = obs_cfg
        self._obs_builder = ObservationBuilder(self.joint_names, obs_cfg)
        self._joint_position_filter = (
            ExponentialJointPositionFilter(
                self._joint_position_filter_tau_sec
            )
            if self._joint_position_filter_tau_sec > 0.0
            else None
        )
        action_dim = len(self.joint_names)
        action_history = np.zeros((obs_cfg.actions_history_len, action_dim), dtype=np.float64)
        self._obs_state = ObservationState(
            lin_vel_b=np.zeros(3),
            action_history=action_history,
            default_pos=self.default_pos.copy(),
            base_lin_vel_history=np.full((obs_cfg.base_lin_vel_history_len, 3), np.nan, dtype=np.float64),
            base_ang_vel_history=np.full((obs_cfg.base_ang_vel_history_len, 3), np.nan, dtype=np.float64),
            projected_gravity_history=np.full((obs_cfg.projected_gravity_history_len, 3), np.nan, dtype=np.float64),
            velocity_commands_history=np.full((obs_cfg.velocity_commands_history_len, 3), np.nan, dtype=np.float64),
            joint_pos_history=np.full((obs_cfg.joint_pos_history_len, action_dim), np.nan, dtype=np.float64),
            joint_vel_history=np.full((obs_cfg.joint_vel_history_len, action_dim), np.nan, dtype=np.float64),
        )
        self._velocity_estimator = None
        self._velocity_estimator_feature_builder = None
        self._velocity_estimator_shadow_recorder = None
        self._velocity_estimator_diagnostic_recorder = None
        self._pending_velocity_estimator_diagnostic = None
        self._latest_estimated_velocity = None
        self._velocity_estimator_error = None
        self._velocity_estimator_shadow_warned = False
        self._configure_velocity_estimator()
        self._loaded_artifacts['env.yaml'] = env_loader.loaded_artifact
        if self._velocity_estimator is not None:
            self._loaded_artifacts['velocity_estimator.pt'] = self._velocity_estimator.loaded_artifact
        self._concurrent_policy_feature_builder = None
        if self._concurrent_policy_enabled:
            self._concurrent_policy_feature_builder = VelocityEstimatorFeatureBuilder(
                self.joint_names,
                self.default_pos,
                add_gravity_to_linear_acceleration=(
                    self._velocity_estimator_add_gravity_to_imu_acceleration
                ),
                gravity_magnitude=self._velocity_estimator_gravity_magnitude,
                acceleration_scale=self._velocity_estimator_imu_acceleration_scale,
            )
        self._expected_obs_dim = self._obs_builder.expected_obs_dim
        if self._policy_input_dim is not None and self._policy_input_dim != self._expected_obs_dim:
            raise RuntimeError(
                "Policy/observation mismatch: "
                f"policy_input_dim={self._policy_input_dim}, expected_obs_dim={self._expected_obs_dim}, "
                f"policy_path={self.policy_path}, env_path={self._env_path}"
            )
        decimation = 4  # Legacy runner fallback; node uses explicit deadline decisions.
        self._inference_schedule = InferenceSchedule(self._policy_cadence.periods_ns)
        if self._concurrent_policy_manifest is None:
            self._policy_runner = PolicyRunner(
                self.policy,
                len(self.joint_names),
                PolicyRunnerConfig(decimation=decimation),
            )
        else:
            if self._concurrent_policy_manifest.action_count != len(self.joint_names):
                raise RuntimeError(
                    "Concurrent policy/action mismatch: "
                    f"policy={self._concurrent_policy_manifest.action_count}, joints={len(self.joint_names)}"
                )
            self._policy_runner = ConcurrentPolicyRunner(
                self.policy,
                self._concurrent_policy_manifest,
                decimation=decimation,
            )
        self._async_worker = None
        self._async_pending_ns = None
        self._async_fault = None
        self._async_has_action = False
        self._async_last_clock_ns = None
        self._async_permission = None
        self._inference_max_age_ns = int(float(self.get_parameter('inference_max_age_ms').value) * 1e6)
        if self._inference_max_age_ns <= 0:
            raise ValueError('inference_max_age_ms must be positive')
        if self.get_parameter('async_inference').value:
            if self._native_io:
                raise ValueError('Native I/O requires async_inference=false; native I/O already separates publication from inference')
            if self._velocity_estimator is not None:
                raise ValueError('Async inference currently requires an embedded estimator or no external estimator')
            self._async_worker = InferenceProcess(create_runner_task, {
                'policy_path': str(self.policy_path),
                'policy_sha256': self._loaded_artifacts['policy.pt']['sha256'],
                'metadata_sha256': self._loaded_artifacts.get('metadata.json', {}).get('sha256'),
                'compact_estimator': bool(self.get_parameter('compact_estimator').value),
                'action_count': len(self.joint_names),
                'observation_count': self._expected_obs_dim,
            })
        self._activation_mgr = ActivationManager(ActivationConfig(
            activation_ramp_duration=float(self.get_parameter('activation_ramp_duration').value),
            deactivation_ramp_duration=float(self.get_parameter('deactivation_ramp_duration').value),
            release_after_deactivate=bool(self.get_parameter('release_after_deactivate').value),
        ), logger=self.get_logger())
        self.create_service(Trigger, "get_policy_execution_state", self._execution_state_cb)
        self._set_active_srv = self.create_service(SetBool, self._set_active_service_name, self._set_active_cb)
        self._policy_input_lock_srv = self.create_service(
            SetPolicyInputLock,
            self._set_policy_input_lock_service_name,
            self._set_policy_input_lock_cb,
        )
        self._policy_observations_srv = self.create_service(
            Trigger,
            "get_policy_observations",
            self._get_policy_observations_cb,
        )
        self._timing = Timing(
            str(self.get_parameter('tracing_mode').value),
            float(self.get_parameter('trace_capture_seconds').value),
        )
        self._input_timing = InputTiming(self._timing)
        self._expected_tick_ns = int(self.get_parameter('publish_period_ms').value * 1e6)
        if self._timing.mode != 'off':
            self._timing_publisher = self.create_publisher(String, '/trace/metrics', 10)
            self._timing_timer = self.create_timer(
                1.0, self._publish_timing, clock=Clock(clock_type=ClockType.STEADY_TIME)
            )
        self.action = np.zeros(len(self.joint_names))
        self._nan_action_warned = False
        self._nan_obs_warned = False
        if self._concurrent_policy_enabled:
            self._logger.info(
                "The combined artifact supplies base_lin_vel from its embedded estimator; "
                f"the external {self._base_lin_vel_source} selection is not used by the actor."
            )
        elif self._base_lin_vel_required and self._base_lin_vel_source == "imu_integration":
            self._logger.warn(
                "The selected policy requires base_lin_vel and is configured for imu_integration. "
                "This compatibility source can drift; use odometry in Isaac Sim or a validated estimator on hardware."
            )
        elif self._base_lin_vel_required:
            self._logger.info(
                f"The selected policy requires base_lin_vel; waiting for {self._base_lin_vel_source} source readiness."
            )
        else:
            self._logger.info("The selected policy does not require a base_lin_vel observation.")
        self._logger.info(
            "Initializing MarvinController (inactive by default). "
            f"io_backend={'native' if self._native_io else 'python'}, "
            f"async_inference={self._async_worker is not None}, "
            f"compact_estimator={self.get_parameter('compact_estimator').value}, "
            f"lean_qos_events={self.get_parameter('lean_qos_events').value}, "
            f"joy_topic={self._joy_topic}, joint_states_topic={self._joint_states_topic}, "
            f"imu_topic={self._imu_topic}, command_topic={self._command_topic}, "
            f"base_lin_vel_required={self._base_lin_vel_required}, "
            f"base_lin_vel_source={self._base_lin_vel_source}, "
            f"concurrent_policy_enabled={self._concurrent_policy_enabled}, "
            f"odometry_topic={self._odometry_topic}, "
            f"odometry_child_frame_id={self._odometry_child_frame_id}, "
            f"base_lin_vel_timeout_sec={self._base_lin_vel_timeout_sec}, "
            f"velocity_estimator_shadow_enabled={self._velocity_estimator_shadow_enabled}, "
            f"velocity_estimator_shadow_topic={self._velocity_estimator_shadow_topic}, "
            f"traced_command_topic={self._traced_command_topic}, "
            f"trace_event_topic={self._trace_event_topic}, "
            f"publish_legacy_commands={self._publish_legacy_commands}, "
            f"publish_traced_commands={self._publish_traced_commands}, "
            f"publish_trace_events={self._publish_trace_events}, "
            f"trace_source_id={self._trace_source_id}, "
            f"trace_instance_id={self._trace_instance_id}, "
            f"set_active_service_name={self._set_active_service_name}, "
            f"policy_input_lock_service={self._set_policy_input_lock_service_name}, "
            "policy_observations_service=get_policy_observations, "
            f"safety_limits_enabled={self._safety_limits_enabled}, "
            f"safety_limits_path={self._safety_limits_path}, "
            f"safety_limits_loaded={self._safety_limits_loaded}, "
            f"action_output_mapping={self._action_output_mapping}, "
            f"actor_action_history_mode={self._actor_action_history_mode}, "
            "joint_position_filter_tau_sec="
            f"{self._joint_position_filter_tau_sec}, "
            f"env_path={self._env_path}, expected_obs_dim={self._expected_obs_dim}, "
            f"policy_input_dim={self._policy_input_dim}"
        )

        self._reload_adapter = None
        if self._native_io and self._execution_gateway and os.environ.get('MARVIN_POLICY_RELOAD_ENABLED') == '1':
            from .reload_adapter import ReloadAdapter
            self._reload_adapter = ReloadAdapter(self)

    def _next_trace_id(self) -> int:
        # trace_id layout: [8 bits source_id][16 bits instance_id][40 bits monotonic counter]
        self._trace_counter = (self._trace_counter + 1) & ((1 << 40) - 1)
        if self._trace_counter == 0:
            self._trace_counter = 1
        return ((self._trace_source_id & 0xFF) << 56) | ((self._trace_instance_id & 0xFFFF) << 40) | self._trace_counter

    def _publish_trace_event(
        self,
        trace_id: int,
        stage: str,
        source: str = "marvin_policy_server",
        stamp_ns: int | None = None,
    ) -> None:
        timing = getattr(self, '_timing', None)
        if timing is not None and not timing.capture_enabled():
            return
        if not self._publish_trace_events or self._trace_event_publisher is None:
            return
        if trace_id <= 0:
            return
        msg = TraceEvent()
        if stamp_ns is None:
            msg.stamp = self.get_clock().now().to_msg()
        else:
            ns = int(stamp_ns)
            msg.stamp.sec = int(ns // 1_000_000_000)
            msg.stamp.nanosec = int(ns % 1_000_000_000)
        msg.trace_id = int(trace_id)
        msg.stage = stage
        msg.source = source
        self._trace_event_publisher.publish(msg)

    def _publish_joint_command(
        self,
        command_values: list[float],
        trace_id: int | None = None,
        emit_trace_event: bool = False,
        command_published_stamp_ns: int | None = None,
    ) -> int:
        if trace_id is None:
            trace_id = self._next_trace_id()
        trace_id = int(trace_id)
        timing = getattr(self, '_timing', None)
        if timing is not None and timing.mode == 'off':
            timing = None
        section_started = timing.clock() if timing is not None else 0
        policy_command_values = self._apply_safety_position_limits(command_values)
        if timing is not None:
            section_started = timing.checkpoint('publish_safety_limits', section_started)
        self._latest_policy_joint_command = np.array(policy_command_values, dtype=np.float64, copy=True)
        if self._joint_command_output_locked and self._locked_joint_command is not None:
            command_values = self._locked_joint_command.tolist()
        else:
            command_values = policy_command_values
        if getattr(self, '_execution_gateway', False):
            prepared = self._execution_preparation.command(command_values)
            if prepared is not None:
                command_values = list(prepared)
                candidate = ExecutorCommand()
                candidate.stamp = self.get_clock().now().to_msg()
                candidate.generation = self._execution_preparation.generation
                candidate.sequence = self._execution_preparation.sequence
                candidate.trace_id = trace_id
                candidate.data = command_values
                self._execution_candidate_publisher.publish(candidate)
        self._last_published_joint_command = np.array(command_values, dtype=np.float64, copy=True)
        if timing is not None:
            timing.count('commands')
            section_started = timing.checkpoint('publish_array_copies', section_started)
        if self._publish_legacy_commands and self._joint_publisher is not None:
            legacy_msg = Float64MultiArray()
            legacy_msg.data = array('d', command_values)
            if timing is not None:
                section_started = timing.checkpoint('publish_legacy_build', section_started)
            self._joint_publisher.publish(legacy_msg)
            if timing is not None:
                section_started = timing.checkpoint('publish_legacy_ros', section_started)
        if self._publish_traced_commands and self._joint_publisher_traced is not None:
            traced_msg = TracedFloat64MultiArray()
            traced_msg.stamp = self.get_clock().now().to_msg()
            traced_msg.trace_id = int(trace_id)
            traced_msg.data = array('d', command_values)
            if timing is not None:
                section_started = timing.checkpoint('publish_traced_build', section_started)
            self._joint_publisher_traced.publish(traced_msg)
            if timing is not None:
                section_started = timing.checkpoint('publish_traced_ros', section_started)
        if emit_trace_event:
            self._publish_trace_event(int(trace_id), "command_published", stamp_ns=command_published_stamp_ns)
            if timing is not None:
                timing.checkpoint('publish_trace_event', section_started)
        return int(trace_id)

    def _prepare_execution(self, request, response):
        outcome = self.prepare_execution(request.generation, request.initial_state_json)
        response.success, response.message = outcome.success, outcome.message
        return response

    def _set_active_cb(self, request, response):
        outcome = self.request_activation(request.data)
        response.success, response.message = outcome.success, outcome.message
        return response

    def prepare_execution(self, generation: str, initial_state_json: str) -> ExecutionOutcome:
        outcome = ExecutionOutcome()
        try:
            if getattr(self, '_reload_adapter', None) and self._reload_adapter.manager.blocked:
                raise ValueError('Policy reload pending or failed')
            if not self._execution_gateway:
                raise ValueError('Policy is not configured for gateway output')
            ready, reason = self._base_lin_vel_readiness()
            if not ready:
                raise ValueError(reason)
            self._execution_preparation.prepare(generation, initial_state_json,
                                                released=self._execution_state()['output_released'])
            outcome.success, outcome.message = True, 'Policy prepared; output authority not granted'
        except (ValueError, KeyError, TypeError) as error:
            outcome.success, outcome.message = False, str(error)
        return outcome

    def request_activation(self, active: bool) -> ExecutionOutcome:
        """Change activation and reset inference state on the existing executor thread."""
        outcome = ExecutionOutcome()
        if active and getattr(self, '_reload_adapter', None) and self._reload_adapter.manager.blocked:
            outcome.message = 'Policy reload pending or failed; load successfully before activation'
            return outcome
        if active and getattr(self, '_execution_gateway', False):
            try:
                self._execution_preparation.start()
            except ValueError as error:
                outcome.success, outcome.message = False, str(error)
                return outcome
        if active:
            ready, readiness_message = self._base_lin_vel_readiness()
            if not ready:
                outcome.success = False
                outcome.message = f"Policy activation rejected: {readiness_message}"
                self._logger.warn(outcome.message)
                return outcome
        now = self.get_clock().now().nanoseconds * 1e-9
        self._invalidate_async_inference()
        if active:
            success, msg = self._activation_mgr.request_activate(now)
            outcome.success = success
            outcome.message = msg
            if success:
                self._async_fault = None
            if success and hasattr(self, '_inference_schedule'):
                self._inference_schedule.reset()
            if success and self._velocity_estimator is not None:
                self._velocity_estimator.reset()
                self._latest_estimated_velocity = None
                self._velocity_estimator_error = None
            if success and getattr(self, "_concurrent_policy_enabled", False):
                self._policy_runner.reset()
                self.action = np.zeros(len(self.joint_names))
                self._latest_estimated_velocity = None
            if success and getattr(self, '_sim_release', None) is not None:
                self._sim_release.reset(self.default_pos)
            if success and self._joint_position_filter is not None:
                self._joint_position_filter.reset()
        else:
            # Provide current action (unscaled raw) for ramp start
            success, msg = self._activation_mgr.request_deactivate(now, self.action)
            outcome.success = success
            outcome.message = msg
        return outcome

    def _set_policy_input_lock_cb(self, request, response):
        values = None if request.use_latest else list(request.values)
        success, message = self._set_policy_input_lock(
            str(request.input_name),
            bool(request.locked),
            values,
        )
        status_name = (
            "estimator_velocity"
            if str(request.input_name) == "estimator_velocity_odometry"
            else str(request.input_name)
        )
        lock_state = self._observation_status_payload()["locks"].get(status_name, {})
        response.success = success
        response.message = message
        response.locked = bool(lock_state.get("locked", False))
        response.values = [float(value) for value in lock_state.get("values", [])]
        response.live_values = [float(value) for value in lock_state.get("live_values", [])]
        return response

    def _set_policy_input_lock(self, input_name: str, locked: bool, values=None) -> tuple[bool, str]:
        result = self._apply_policy_input_lock(input_name, locked, values)
        if result[0] and hasattr(self, "_policy_control"):
            self._policy_control.changed(input_name, locked, values)
        return result

    def _policy_control_cb(self, request, response):
        try:
            if request.action == "state":
                result = self._policy_control.state()
            elif request.action == "operation":
                result = self._policy_control.get(request.session_id, request.command_id)
            elif request.action == "submit":
                result = self._policy_control.submit({
                    "command_id": request.command_id, "session_id": request.session_id,
                    "expected_revision": request.expected_revision, "control": request.control,
                    "mode": request.mode, "values": list(request.values),
                })
            else:
                raise PolicyControlError("POLICY_INVALID_REQUEST", "Unsupported policy action")
            response.result_json = json.dumps(result, separators=(",", ":"), allow_nan=False)
            response.success = True
        except PolicyControlError as exc:
            response.success, response.error_code, response.error_message = False, exc.code, str(exc)
        except Exception as exc:
            response.success = False
            response.error_code = "POLICY_UNAVAILABLE"
            response.error_message = str(exc)
        return response

    def _apply_policy_input_lock(self, input_name: str, locked: bool, values=None) -> tuple[bool, str]:
        if getattr(self, '_reload_adapter', None) and self._reload_adapter.manager.blocked:
            return False, 'Policy reload pending or failed; overrides are unavailable'
        result = self._apply_policy_input_lock_impl(input_name, locked, values)
        if result[0]:
            self._invalidate_async_inference()
        return result

    def _apply_policy_input_lock_impl(self, input_name: str, locked: bool, values=None) -> tuple[bool, str]:
        if input_name == "estimator_velocity_odometry":
            if not locked:
                return self._apply_policy_input_lock("estimator_velocity", False, None)
            if values is not None:
                return False, "Odometry estimator-velocity mode does not accept explicit values."
            if not self._estimator_velocity_override_available():
                return False, "Estimator velocity override is available only for compatible concurrent policies."
            if not self._estimator_velocity_odometry_available():
                return False, f"No fresh body-frame odometry is available on {self._odometry_topic}."
            odometry = self._latest_odometry_velocity
            self._locked_estimator_velocity = np.asarray(odometry, dtype=np.float64).copy()
            self._estimator_velocity_override_mode = "odometry"
            return True, "Actor velocity now follows body-frame odometry."

        expected_sizes = {
            "command": 3,
            "imu": 9,
            "joint_position": len(self.joint_names),
            "velocity": len(self.joint_names),
            "joint_command": len(self.joint_names),
            "estimator_velocity": 3,
            "estimator_fk": 12,
            "estimator_foot_height": 4,
            "estimator_contact": 4,
        }
        if input_name not in expected_sizes:
            return False, f"Unknown policy input lock: {input_name}."

        if not locked:
            if input_name == "command":
                self._command_input_locked = False
                self._locked_command_input = None
            elif input_name == "imu":
                self._imu_input_locked = False
                self._locked_imu_input = None
            elif input_name == "joint_position":
                self._joint_position_input_locked = False
                self._locked_joint_position_relative = None
            elif input_name == "velocity":
                self._velocity_input_locked = False
                self._locked_joint_velocity = None
            elif input_name == "joint_command":
                self._joint_command_output_locked = False
                self._locked_joint_command = None
                return True, "Policy joint command output released."
            elif input_name == "estimator_velocity":
                self._estimator_velocity_override_mode = "live"
                self._locked_estimator_velocity = None
                return True, "Actor velocity returned to the live embedded estimator."
            elif input_name == "estimator_fk":
                self._locked_estimator_fk = None
                return True, "Actor FK foot positions returned to live values."
            elif input_name == "estimator_foot_height":
                self._locked_estimator_foot_height = None
                return True, "Actor foot heights returned to live estimates."
            else:
                self._locked_estimator_contact = None
                return True, "Actor contacts returned to live estimates."
            return True, f"Policy {input_name} input released."

        estimator_outputs = {
            "estimator_velocity",
            "estimator_fk",
            "estimator_foot_height",
            "estimator_contact",
        }
        if (
            input_name in estimator_outputs
            and not self._estimator_velocity_override_available()
        ):
            return False, "Estimator velocity override is available only for compatible concurrent policies."

        if values is None:
            if input_name == "estimator_velocity":
                values_array = (
                    np.asarray(self._latest_estimated_velocity, dtype=np.float64).copy()
                    if self._latest_estimated_velocity is not None
                    else np.zeros(3, dtype=np.float64)
                )
            elif input_name in {
                "estimator_fk",
                "estimator_foot_height",
                "estimator_contact",
            }:
                age = self._estimator_snapshot_age()
                if age is None or age > 0.1:
                    return False, "A fresh estimator result is required to hold this input."
                if input_name == "estimator_fk" and getattr(self._policy_runner, 'latest_foot_positions', None) is None:
                    return False, "Foot positions are not available yet."
                values_array = self._current_estimator_actor_input(
                    input_name
                )
            else:
                values_array = self._current_lock_group_values(input_name)
        else:
            values_array = np.asarray(values, dtype=np.float64).reshape(-1)
        expected_size = expected_sizes[input_name]
        if values_array.size != expected_size:
            return False, f"Policy {input_name} lock requires {expected_size} values, got {values_array.size}."
        if not np.isfinite(values_array).all():
            return False, f"Policy {input_name} lock values must all be finite."

        if input_name == "command":
            self._locked_command_input = values_array.copy()
            self._command_input_locked = True
        elif input_name == "imu":
            self._locked_imu_input = {
                "base_lin_vel": values_array[0:3].copy(),
                "base_ang_vel": values_array[3:6].copy(),
                "projected_gravity": values_array[6:9].copy(),
            }
            self._imu_input_locked = True
        elif input_name == "joint_position":
            self._locked_joint_position_relative = values_array.copy()
            self._joint_position_input_locked = True
        elif input_name == "velocity":
            self._locked_joint_velocity = values_array.copy()
            self._velocity_input_locked = True
        elif input_name == "joint_command":
            outside_limits = (
                np.logical_or(
                    values_array < self._safety_min_pos,
                    values_array > self._safety_max_pos,
                )
                if self._safety_limits_enabled and self._safety_limits_loaded
                else np.zeros(values_array.shape, dtype=bool)
            )
            if np.any(outside_limits):
                return False, "Policy joint command output contains values outside the configured safety limits."
            self._locked_joint_command = values_array.copy()
            self._joint_command_output_locked = True
            return True, f"Policy joint command output held with {expected_size} values."
        elif input_name == "estimator_velocity":
            self._locked_estimator_velocity = values_array.copy()
            self._estimator_velocity_override_mode = (
                "zero" if np.allclose(values_array, 0.0, rtol=0.0, atol=0.0) else "hold"
            )
            return True, (
                "Actor velocity fixed at zero."
                if self._estimator_velocity_override_mode == "zero"
                else "Actor velocity held at the selected embedded-estimator value."
            )
        elif input_name == "estimator_fk":
            self._locked_estimator_fk = values_array.copy()
            return True, "Actor FK foot positions held."
        elif input_name == "estimator_foot_height":
            self._locked_estimator_foot_height = values_array.copy()
            return True, "Actor estimated foot heights held."
        else:
            self._locked_estimator_contact = values_array.copy()
            return True, "Actor estimated contacts held."
        return True, f"Policy {input_name} input locked with {expected_size} values."

    def _estimator_velocity_override_available(self) -> bool:
        return bool(
            getattr(self, "_concurrent_policy_enabled", False)
            and getattr(self, "_policy_runner", None) is not None
            and self._policy_runner.supports_velocity_override
        )

    def _estimator_velocity_odometry_available(self) -> bool:
        if self._latest_odometry_velocity is None or self._latest_odometry_received_monotonic is None:
            return False
        values = np.asarray(self._latest_odometry_velocity, dtype=np.float64).reshape(-1)
        age = max(0.0, time.monotonic() - self._latest_odometry_received_monotonic)
        return values.shape == (3,) and np.isfinite(values).all() and age <= self._base_lin_vel_timeout_sec

    def _effective_estimator_velocity_override(self) -> np.ndarray | None:
        mode = self._estimator_velocity_override_mode
        if mode == "live":
            return None
        if mode == "odometry" and self._latest_odometry_velocity is not None:
            candidate = np.asarray(self._latest_odometry_velocity, dtype=np.float64).reshape(-1)
            if candidate.shape == (3,) and np.isfinite(candidate).all():
                self._locked_estimator_velocity = candidate.copy()
        if self._locked_estimator_velocity is None:
            return np.zeros(3, dtype=np.float64)
        return self._locked_estimator_velocity.copy()

    def _estimator_snapshot_age(self) -> float | None:
        stamp = getattr(getattr(self, '_policy_runner', None), 'latest_result_monotonic', None)
        return None if stamp is None else max(0.0, time.monotonic() - stamp)

    def _current_estimator_actor_input(
        self, input_name: str
    ) -> np.ndarray:
        """Return one live estimator-to-actor channel for a hold snapshot."""
        if input_name == "estimator_fk":
            cached = getattr(
                self._policy_runner,
                "latest_foot_positions",
                None,
            )
            values = (
                np.asarray(cached, dtype=np.float64).copy()
                if cached is not None
                else None
            )
            return (
                values
                if values is not None
                else np.zeros(12, dtype=np.float64)
            )
        estimated_state = self._policy_runner.latest_estimated_state
        if estimated_state is None:
            width = 4 if input_name != "estimator_velocity" else 3
            return np.zeros(width, dtype=np.float64)
        state = np.asarray(estimated_state, dtype=np.float64)
        if input_name == "estimator_velocity":
            return state[0:3].copy()
        if input_name == "estimator_foot_height":
            return state[3:7].copy()
        if input_name == "estimator_contact":
            return state[7:11].copy()
        raise ValueError(f"Unknown estimator actor input: {input_name}")

    @staticmethod
    def _effective_estimator_actor_input(
        live: np.ndarray,
        locked: np.ndarray | None,
    ) -> np.ndarray:
        """Select a held estimator channel or its live value."""
        return live if locked is None else locked.copy()

    def _current_lock_group_values(self, input_name: str) -> np.ndarray:
        if input_name == "command":
            return np.array(
                [self._cmd_vel.linear.x, self._cmd_vel.linear.y, self._cmd_vel.angular.z],
                dtype=np.float64,
            )
        if input_name == "imu":
            imu = self._latest_imu
            angular_velocity = np.zeros(3, dtype=np.float64)
            projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float64)
            if imu is not None:
                angular_velocity[:] = [imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z]
                quat = np.array([imu.orientation.w, imu.orientation.x, imu.orientation.y, imu.orientation.z])
                projected_gravity = quat_to_rot_matrix(quat).T @ np.array([0.0, 0.0, -1.0])
            return np.concatenate((self._current_base_lin_vel(), angular_velocity, projected_gravity))
        if input_name == "joint_command":
            if self._last_published_joint_command is not None:
                return self._last_published_joint_command.copy()
            if self._latest_policy_joint_command is not None:
                return self._latest_policy_joint_command.copy()
            return self.default_pos.copy()
        if input_name == "joint_position":
            return self._latest_history_value(
                self._obs_state.joint_pos_history,
                self._joint_position_relative_vector(),
            )
        if input_name == "estimator_velocity":
            override = self._effective_estimator_velocity_override()
            if override is not None:
                return override
            if self._latest_estimated_velocity is not None:
                return np.asarray(self._latest_estimated_velocity, dtype=np.float64).copy()
            return np.zeros(3, dtype=np.float64)
        if self._latest_joint_state is None:
            return np.zeros(len(self.joint_names), dtype=np.float64)
        return self._joint_velocity_vector(self._latest_joint_state)

    @staticmethod
    def _latest_history_value(history: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        if history.shape[0] > 0 and np.isfinite(history[-1]).all():
            return history[-1].copy()
        return np.asarray(fallback, dtype=np.float64).copy()

    def _joint_position_relative_vector(self) -> np.ndarray:
        positions = np.zeros(len(self.joint_names), dtype=np.float64)
        joint_state = self._latest_joint_state
        if joint_state is None:
            return positions
        name_to_index = {name: index for index, name in enumerate(joint_state.name)}
        for index, name in enumerate(self.joint_names):
            source_index = name_to_index.get(name)
            if source_index is not None and source_index < len(joint_state.position):
                positions[index] = joint_state.position[source_index] - self.default_pos[index]
        return positions

    def _execution_state(self):
        state = self._activation_mgr.state
        return {'active': state.active,
                'output_released': not state.active and state.deactivation_start_time is None}

    def _execution_state_cb(self, request, response):
        response.success = True
        response.message = json.dumps(self._execution_state())
        return response

    def _timing_status(self):
        cadence = getattr(self, '_policy_cadence', None)
        if cadence is None:
            return None
        return dict(schema='marvin.policy-timing.v1', cadence=cadence.projection(),
            environment_sha256=self._loaded_artifacts['env.yaml']['sha256'], source=cadence.source,
            scheduler='native-fixed' if self._native_io else 'python-cyclic',
            observed=self._inference_schedule.observed())

    def _observation_status_payload(self) -> dict[str, object]:
        base_lin_vel_ready, base_lin_vel_message = self._base_lin_vel_readiness()
        command_live = self._current_lock_group_values("command")
        imu_live = self._current_lock_group_values("imu")
        joint_position_live = self._joint_position_relative_vector()
        velocity_live = self._current_lock_group_values("velocity")
        joint_command_snapshot = self._current_lock_group_values("joint_command")
        joint_command_live = (
            self._latest_policy_joint_command.copy()
            if self._latest_policy_joint_command is not None
            else joint_command_snapshot
        )
        estimator_velocity_live = (
            np.asarray(self._latest_estimated_velocity, dtype=np.float64).copy()
            if self._latest_estimated_velocity is not None
            else np.zeros(3, dtype=np.float64)
        )
        estimator_velocity_override = self._effective_estimator_velocity_override()
        estimator_velocity_effective = (
            estimator_velocity_live
            if estimator_velocity_override is None
            else estimator_velocity_override
        )
        estimator_outputs_available = self._estimator_velocity_override_available()
        if estimator_outputs_available:
            estimator_fk_live = self._current_estimator_actor_input(
                "estimator_fk"
            )
            estimator_foot_height_live = self._current_estimator_actor_input(
                "estimator_foot_height"
            )
            estimator_contact_live = self._current_estimator_actor_input(
                "estimator_contact"
            )
        else:
            estimator_fk_live = np.zeros(12, dtype=np.float64)
            estimator_foot_height_live = np.zeros(4, dtype=np.float64)
            estimator_contact_live = np.zeros(4, dtype=np.float64)
        estimator_fk_effective = self._effective_estimator_actor_input(
            estimator_fk_live, self._locked_estimator_fk
        )
        estimator_foot_height_effective = (
            self._effective_estimator_actor_input(
                estimator_foot_height_live,
                self._locked_estimator_foot_height,
            )
        )
        estimator_contact_effective = self._effective_estimator_actor_input(
            estimator_contact_live, self._locked_estimator_contact
        )
        odometry_velocity = (
            np.asarray(self._latest_odometry_velocity, dtype=np.float64).copy()
            if self._latest_odometry_velocity is not None
            else np.zeros(3, dtype=np.float64)
        )
        command_effective = self._locked_command_input if self._command_input_locked else command_live
        if not self._operator_permission.allowed():
            command_effective = np.zeros(3)
        imu_effective = (
            np.concatenate(tuple(self._locked_imu_input.values()))
            if self._imu_input_locked and self._locked_imu_input is not None
            else imu_live
        )
        velocity_effective = self._locked_joint_velocity if self._velocity_input_locked else velocity_live
        joint_position_effective = (
            self._locked_joint_position_relative
            if self._joint_position_input_locked
            and self._locked_joint_position_relative is not None
            else self._latest_history_value(
                self._obs_state.joint_pos_history,
                joint_position_live,
            )
        )
        joint_command_effective = (
            self._locked_joint_command
            if self._joint_command_output_locked and self._locked_joint_command is not None
            else joint_command_live
        )
        joint_labels = list(self.joint_names)

        observations = [
            {"key": "base_lin_vel", "label": "Base Linear Velocity", "labels": ["x", "y", "z"], "values": imu_effective[0:3].tolist()},
            {"key": "base_ang_vel", "label": "Base Angular Velocity", "labels": ["x", "y", "z"], "values": imu_effective[3:6].tolist()},
            {"key": "projected_gravity", "label": "Projected Gravity", "labels": ["x", "y", "z"], "values": imu_effective[6:9].tolist()},
            {"key": "velocity_commands", "label": "Velocity Command", "labels": ["x", "y", "yaw"], "values": command_effective.tolist()},
            {"key": "joint_pos", "label": "Joint Position (relative)", "labels": joint_labels, "values": joint_position_effective.tolist()},
            {"key": "joint_vel", "label": "Joint Velocity", "labels": joint_labels, "values": velocity_effective.tolist()},
            {"key": "actions", "label": "Policy Action", "labels": joint_labels, "values": np.asarray(self.action, dtype=np.float64).tolist()},
        ]
        observations_by_key = {item["key"]: item for item in observations}
        observations = [
            observations_by_key.get(
                term["key"],
                {
                    "key": term["key"],
                    "label": term["key"].replace("_", " ").title(),
                    "labels": [],
                    "values": [],
                    "available": False,
                },
            )
            | {
                "func": term["func"],
                "history_length": term["history_length"],
                "available": term["key"] in observations_by_key,
            }
            for term in self._policy_observation_schema
        ]
        return {
            "policy_path": str(self.policy_path),
            "env_path": str(self._env_path),
            "loaded_artifacts": getattr(self, '_loaded_artifacts', None),
            "timing": self._timing_status(),
            "base_lin_vel_source": {
                "required": self._base_lin_vel_required,
                "configured": self._base_lin_vel_source,
                "owned_by_concurrent_policy": getattr(self, "_concurrent_policy_enabled", False),
                "ready": base_lin_vel_ready,
                "message": base_lin_vel_message,
                "odometry_topic": self._odometry_topic,
                "expected_child_frame_id": self._odometry_child_frame_id,
                "timeout_sec": self._base_lin_vel_timeout_sec,
            },
            "concurrent_policy": {
                "snapshot_age_seconds": self._estimator_snapshot_age(),
                "snapshot_available": self._estimator_snapshot_age() is not None,
                "enabled": getattr(self, "_concurrent_policy_enabled", False),
                "manifest": (
                    str(self._concurrent_policy_manifest.path)
                    if getattr(self, "_concurrent_policy_manifest", None) is not None
                    else None
                ),
                "history_count": (
                    self._policy_runner.history.count
                    if getattr(self, "_concurrent_policy_enabled", False)
                    else 0
                ),
                "history_ready": (
                    self._policy_runner.history.ready
                    if getattr(self, "_concurrent_policy_enabled", False)
                    else False
                ),
                "estimated_state": (
                    self._policy_runner.latest_estimated_state.tolist()
                    if getattr(self, "_concurrent_policy_enabled", False)
                    and self._policy_runner.latest_estimated_state is not None
                    else None
                ),
            },
            "velocity_estimator_shadow": {
                "enabled": getattr(self, "_velocity_estimator_shadow_enabled", False),
                "checkpoint": (
                    str(self._velocity_estimator.checkpoint_path)
                    if getattr(self, "_velocity_estimator", None) is not None
                    else getattr(self, "_velocity_estimator_checkpoint_raw", "")
                ),
                "topic": getattr(self, "_velocity_estimator_shadow_topic", ""),
                "add_gravity_to_imu_acceleration": getattr(
                    self,
                    "_velocity_estimator_add_gravity_to_imu_acceleration",
                    False,
                ),
                "imu_acceleration_scale": getattr(
                    self, "_velocity_estimator_imu_acceleration_scale", 1.0
                ),
                "summary": (
                    self._velocity_estimator_shadow_recorder.summary()
                    if getattr(self, "_velocity_estimator_shadow_recorder", None) is not None
                    else None
                ),
            },
            "observations": observations,
            "locks": {
                "command": {"locked": self._command_input_locked, "labels": ["x", "y", "yaw"], "values": command_effective.tolist(), "live_values": command_live.tolist()},
                "imu": {"locked": self._imu_input_locked, "labels": ["lin x", "lin y", "lin z", "ang x", "ang y", "ang z", "gravity x", "gravity y", "gravity z"], "values": imu_effective.tolist(), "live_values": imu_live.tolist()},
                "joint_position": {
                    "locked": self._joint_position_input_locked,
                    "labels": joint_labels,
                    "values": joint_position_effective.tolist(),
                    "live_values": joint_position_live.tolist(),
                },
                "velocity": {"locked": self._velocity_input_locked, "labels": joint_labels, "values": velocity_effective.tolist(), "live_values": velocity_live.tolist()},
                "joint_command": {
                    "locked": self._joint_command_output_locked,
                    "labels": joint_labels,
                    "values": joint_command_effective.tolist(),
                    "live_values": joint_command_live.tolist(),
                },
                "estimator_velocity": {
                    "locked": self._estimator_velocity_override_mode != "live",
                    "mode": self._estimator_velocity_override_mode,
                    "available": self._estimator_velocity_override_available(),
                    "reference_available": self._estimator_velocity_odometry_available(),
                    "labels": ["x", "y", "z"],
                    "values": estimator_velocity_effective.tolist(),
                    "live_values": estimator_velocity_live.tolist(),
                    "reference_values": odometry_velocity.tolist(),
                },
                "estimator_fk": {
                    "locked": self._locked_estimator_fk is not None,
                    "available": estimator_outputs_available and getattr(self._policy_runner, 'latest_foot_positions', None) is not None,
                    "labels": [
                        f"{foot} {axis}"
                        for foot in ("FL", "RL", "FR", "RR")
                        for axis in ("x", "y", "z")
                    ],
                    "values": estimator_fk_effective.tolist(),
                    "live_values": estimator_fk_live.tolist(),
                },
                "estimator_foot_height": {
                    "locked": (
                        self._locked_estimator_foot_height is not None
                    ),
                    "available": estimator_outputs_available,
                    "labels": ["FL", "RL", "FR", "RR"],
                    "values": estimator_foot_height_effective.tolist(),
                    "live_values": estimator_foot_height_live.tolist(),
                },
                "estimator_contact": {
                    "locked": self._locked_estimator_contact is not None,
                    "available": estimator_outputs_available,
                    "labels": ["FL", "RL", "FR", "RR"],
                    "values": estimator_contact_effective.tolist(),
                    "live_values": estimator_contact_live.tolist(),
                },
            },
        }

    def _get_policy_observations_cb(self, request, response):
        response.success = True
        response.message = json.dumps(self._observation_status_payload(), separators=(",", ":"))
        return response

    def _joint_velocity_vector(self, joint_state: JointState) -> np.ndarray:
        velocities = np.zeros(len(self.joint_names), dtype=np.float64)
        name_to_index = {name: index for index, name in enumerate(joint_state.name)}
        for index, name in enumerate(self.joint_names):
            source_index = name_to_index.get(name)
            if source_index is not None and source_index < len(joint_state.velocity):
                velocities[index] = joint_state.velocity[source_index]
        return velocities

    def _joy_callback(self, msg):
        twist = Twist()
        linear_x, linear_y, angular_z = policy_velocity_command_from_axes(msg.axes)
        twist.linear.x = linear_x
        twist.linear.y = linear_y
        twist.angular.z = angular_z
        self._cmd_vel = twist
        # self._logger.info(
        #     f"Joy->Twist: lin=({twist.linear.x}, {twist.linear.y}, {twist.linear.z}), "
        #     f"ang=({twist.angular.x}, {twist.angular.y}, {twist.angular.z})"
        # )

    def _cmd_vel_callback(self, msg):
        """Store the latest velocity command."""
        self._cmd_vel = msg

    def _dynamic_joint_state_cb(self, msg):
        try:
            state, receipts = decode_snapshot(msg, self.joint_names, JointState)
        except ValueError:
            self._latest_joint_state = None
            self._timing.count('encoder_snapshot_invalid')
            return
        self._input_timing.received_encoders(state, receipts)
        self._joint_state_cb(state)

    def _joint_state_cb(self, msg: JointState):
        # Store latest joint state
        self._latest_joint_state = msg
        if hasattr(self, '_input_timing'):
            self._input_timing.received('joint')

    def _imu_cb(self, msg: Imu):
        # Store latest imu
        self._latest_imu = msg
        if hasattr(self, '_input_timing'):
            self._input_timing.received('imu')

    def _odometry_cb(self, msg: Odometry):
        expected_frame = self._odometry_child_frame_id.lstrip("/")
        actual_frame = str(msg.child_frame_id or "").strip().lstrip("/")
        if expected_frame and actual_frame != expected_frame:
            error = (
                f"odometry child_frame_id must be {self._odometry_child_frame_id!r}, "
                f"got {msg.child_frame_id!r} on {self._odometry_topic}"
            )
            self._invalidate_odometry(error)
            return

        velocity = np.array(
            [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z],
            dtype=np.float64,
        )
        if not np.isfinite(velocity).all():
            self._invalidate_odometry(f"non-finite linear velocity received on {self._odometry_topic}")
            return

        self._latest_odometry_velocity = velocity
        self._latest_odometry_received_monotonic = time.monotonic()
        self._latest_odometry_stamp_sec = self.header_time_in_seconds(msg.header)
        self._odometry_validation_error = None

    def _invalidate_odometry(self, error: str) -> None:
        changed = error != self._odometry_validation_error
        self._latest_odometry_velocity = None
        self._latest_odometry_received_monotonic = None
        self._latest_odometry_stamp_sec = None
        self._odometry_validation_error = error
        if changed:
            self._logger.warn(error)

    def _configure_velocity_estimator(self) -> None:
        standalone_policy_source = (
            self._base_lin_vel_source == "estimator"
            and not getattr(self, "_concurrent_policy_enabled", False)
        )
        if (
            not standalone_policy_source
            and not self._velocity_estimator_shadow_enabled
            and not self._velocity_estimator_diagnostic_log_dir
        ):
            return
        checkpoint_path = self._resolve_package_path(
            self._velocity_estimator_checkpoint_raw,
            default_relative="policy/velocity_estimator.pt",
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Velocity-estimator checkpoint does not exist: {checkpoint_path}"
            )
        self._velocity_estimator = VelocityEstimator(checkpoint_path)
        self._velocity_estimator_feature_builder = VelocityEstimatorFeatureBuilder(
            self.joint_names,
            self.default_pos,
            add_gravity_to_linear_acceleration=(
                self._velocity_estimator_add_gravity_to_imu_acceleration
            ),
            gravity_magnitude=self._velocity_estimator_gravity_magnitude,
            acceleration_scale=self._velocity_estimator_imu_acceleration_scale,
        )
        if self._velocity_estimator_shadow_enabled:
            self._velocity_estimator_shadow_recorder = VelocityEstimatorShadowRecorder(
                self._velocity_estimator_shadow_log_dir
            )
        if self._velocity_estimator_diagnostic_log_dir:
            self._velocity_estimator_diagnostic_recorder = VelocityEstimatorDiagnosticRecorder(
                self._velocity_estimator_diagnostic_log_dir,
                joint_names=self.joint_names,
                velocity_source=self._base_lin_vel_source,
            )
        log_path = (
            self._velocity_estimator_shadow_recorder.log_path
            if self._velocity_estimator_shadow_recorder is not None
            else None
        )
        self._logger.info(
            "Velocity estimator enabled: "
            f"policy_velocity_source={self._base_lin_vel_source}, "
            f"shadow_enabled={self._velocity_estimator_shadow_enabled}, "
            f"checkpoint={checkpoint_path}, topic={self._velocity_estimator_shadow_topic}, "
            f"add_gravity_to_imu_acceleration="
            f"{self._velocity_estimator_add_gravity_to_imu_acceleration}, log_path={log_path}, "
            f"diagnostic_log_path="
            f"{getattr(self._velocity_estimator_diagnostic_recorder, 'log_path', None)}"
        )

    def _update_velocity_estimator(
        self,
        joint_state: JointState,
        imu: Imu,
        ros_time_sec: float,
    ) -> None:
        """Infer velocity and, when enabled, compare it with fresh odometry."""

        if (
            self._velocity_estimator is None
            or self._velocity_estimator_feature_builder is None
        ):
            return
        self._pending_velocity_estimator_diagnostic = None
        try:
            features = self._velocity_estimator_feature_builder.build(
                joint_state,
                imu,
                self.action,
            )
            inference_started_ns = time.perf_counter_ns()
            estimated_velocity = self._velocity_estimator.predict(features)
            inference_duration_ms = (time.perf_counter_ns() - inference_started_ns) / 1.0e6
            self._latest_estimated_velocity = estimated_velocity
            self._velocity_estimator_error = None

            if (
                self._latest_odometry_velocity is None
                or self._latest_odometry_received_monotonic is None
                or self._latest_odometry_stamp_sec is None
                or time.monotonic() - self._latest_odometry_received_monotonic
                > self._base_lin_vel_timeout_sec
            ):
                self._velocity_estimator_shadow_warned = False
                return

            odometry_velocity = self._latest_odometry_velocity.copy()
            command = (
                self._locked_command_input.copy()
                if self._command_input_locked and self._locked_command_input is not None
                else np.array(
                    [self._cmd_vel.linear.x, self._cmd_vel.linear.y, self._cmd_vel.angular.z],
                    dtype=np.float64,
                )
            )
            if not self._operator_permission.allowed():
                command = np.zeros(3)
            if self._velocity_estimator_diagnostic_recorder is not None:
                self._pending_velocity_estimator_diagnostic = {
                    "ros_time_sec": ros_time_sec,
                    "joint_state_time_sec": self.header_time_in_seconds(joint_state.header),
                    "imu_time_sec": self.header_time_in_seconds(imu.header),
                    "odometry_time_sec": self._latest_odometry_stamp_sec,
                    "history_count": self._velocity_estimator.history_count,
                    "history_ready": self._velocity_estimator.history_ready,
                    "command": command.copy(),
                    "features": features.copy(),
                    "estimated_velocity": estimated_velocity.copy(),
                    "odometry_velocity": odometry_velocity.copy(),
                }
            if self._velocity_estimator_shadow_recorder is None:
                return
            error = self._velocity_estimator_shadow_recorder.record(
                ros_time_sec=ros_time_sec,
                joint_state_time_sec=self.header_time_in_seconds(joint_state.header),
                imu_time_sec=self.header_time_in_seconds(imu.header),
                odometry_time_sec=self._latest_odometry_stamp_sec,
                command=command,
                inference_duration_ms=inference_duration_ms,
                estimated_velocity=estimated_velocity,
                odometry_velocity=odometry_velocity,
                history_count=self._velocity_estimator.history_count,
                history_ready=self._velocity_estimator.history_ready,
            )
            if self._velocity_estimator_shadow_publisher is not None:
                message = Float64MultiArray()
                message.data = [
                    *estimated_velocity.tolist(),
                    *odometry_velocity.tolist(),
                    *error.tolist(),
                    float(self._velocity_estimator.history_count),
                    float(self._velocity_estimator.history_ready),
                ]
                self._velocity_estimator_shadow_publisher.publish(message)

            count = self._velocity_estimator_shadow_recorder.count
            if count % self._velocity_estimator_shadow_summary_interval_samples == 0:
                ready = self._velocity_estimator_shadow_recorder.summary()["history_ready"]
                self._logger.info(
                    "Velocity-estimator shadow summary: "
                    f"samples={count}, ready_samples={ready['count']}, "
                    f"rmse_xyz={np.array2string(np.asarray(ready['rmse_xyz']), precision=4)}, "
                    f"vector_rmse={ready['vector_rmse']:.4f} m/s"
                )
            self._velocity_estimator_shadow_warned = False
        except Exception as exc:
            self._latest_estimated_velocity = None
            self._velocity_estimator_error = str(exc)
            if not self._velocity_estimator_shadow_warned:
                self._logger.warn(
                    f"Velocity-estimator inference failed: {exc}"
                )
                self._velocity_estimator_shadow_warned = True

    def _record_velocity_estimator_diagnostic(self, policy_action: np.ndarray) -> None:
        pending = self._pending_velocity_estimator_diagnostic
        self._pending_velocity_estimator_diagnostic = None
        if self._velocity_estimator_diagnostic_recorder is None or pending is None:
            return
        self._velocity_estimator_diagnostic_recorder.record(
            **pending,
            policy_action=policy_action,
        )

    def _close_velocity_estimator_shadow(self) -> None:
        if self._velocity_estimator_shadow_recorder is not None:
            summary = self._velocity_estimator_shadow_recorder.summary()
            self._velocity_estimator_shadow_recorder.close()
            if rclpy.ok():
                self._logger.info(f"Final velocity-estimator shadow summary: {json.dumps(summary)}")
        if self._velocity_estimator_diagnostic_recorder is not None:
            path = self._velocity_estimator_diagnostic_recorder.log_path
            count = self._velocity_estimator_diagnostic_recorder.count
            self._velocity_estimator_diagnostic_recorder.close()
            if rclpy.ok():
                self._logger.info(
                    f"Final velocity-estimator diagnostic trace: samples={count}, path={path}"
                )

    def _current_base_lin_vel(self) -> np.ndarray:
        if self._imu_input_locked and self._locked_imu_input is not None:
            return np.asarray(self._locked_imu_input["base_lin_vel"], dtype=np.float64).copy()
        if self._base_lin_vel_source == "odometry" and self._latest_odometry_velocity is not None:
            return self._latest_odometry_velocity.copy()
        if self._base_lin_vel_source == "estimator" and self._latest_estimated_velocity is not None:
            return self._latest_estimated_velocity.copy()
        return np.asarray(self._obs_state.lin_vel_b, dtype=np.float64).copy()

    def _base_lin_vel_override(self) -> np.ndarray | None:
        if getattr(self, "_concurrent_policy_enabled", False):
            # The exported model replaces this placeholder with its estimate.
            return np.zeros(3, dtype=np.float64)
        if self._imu_input_locked and self._locked_imu_input is not None:
            return np.asarray(self._locked_imu_input["base_lin_vel"], dtype=np.float64)
        if self._base_lin_vel_source == "odometry" and self._latest_odometry_velocity is not None:
            return self._latest_odometry_velocity
        if self._base_lin_vel_source == "estimator" and self._latest_estimated_velocity is not None:
            return self._latest_estimated_velocity
        return None

    def _base_lin_vel_readiness(self, monotonic_now: float | None = None) -> tuple[bool, str]:
        gate = getattr(self, '_sim_release', None)
        if gate is not None:
            ready, reason = gate.readiness(self.get_clock().now().nanoseconds)
            if not ready:
                return ready, reason
        if not self._base_lin_vel_required:
            return True, "The selected policy does not require base_lin_vel."
        if getattr(self, "_concurrent_policy_enabled", False):
            return True, "base_lin_vel is supplied inside the concurrent estimator-policy artifact."
        if self._imu_input_locked and self._locked_imu_input is not None:
            values = np.asarray(self._locked_imu_input["base_lin_vel"], dtype=np.float64)
            if values.shape == (3,) and np.isfinite(values).all():
                return True, "base_lin_vel is supplied by the active policy input lock."
            return False, "the locked base_lin_vel value is invalid"
        if self._base_lin_vel_source == "imu_integration":
            if self._latest_imu is None:
                return False, f"no IMU sample has been received on {self._imu_topic}"
            acceleration = np.array(
                [
                    self._latest_imu.linear_acceleration.x,
                    self._latest_imu.linear_acceleration.y,
                    self._latest_imu.linear_acceleration.z,
                ],
                dtype=np.float64,
            )
            if not np.isfinite(acceleration).all():
                return False, f"the IMU sample on {self._imu_topic} contains non-finite acceleration"
            return True, "base_lin_vel is supplied by IMU acceleration integration."

        if self._base_lin_vel_source == "estimator":
            if self._velocity_estimator is None or self._velocity_estimator_feature_builder is None:
                return False, "the configured velocity estimator is not loaded"
            if self._velocity_estimator_error:
                return False, f"velocity-estimator inference failed: {self._velocity_estimator_error}"
            if self._latest_estimated_velocity is None:
                return True, "the velocity estimator is loaded and will initialize on the next policy inference."
            values = np.asarray(self._latest_estimated_velocity, dtype=np.float64)
            if values.shape != (3,) or not np.isfinite(values).all():
                return False, "the velocity estimator produced an invalid base_lin_vel value"
            return True, "base_lin_vel is supplied by the learned velocity estimator."

        if self._latest_odometry_velocity is None or self._latest_odometry_received_monotonic is None:
            detail = self._odometry_validation_error or f"no odometry sample has been received on {self._odometry_topic}"
            return False, detail
        now = time.monotonic() if monotonic_now is None else float(monotonic_now)
        age = max(0.0, now - self._latest_odometry_received_monotonic)
        if age > self._base_lin_vel_timeout_sec:
            return False, (
                f"odometry on {self._odometry_topic} is stale "
                f"({age:.3f}s > {self._base_lin_vel_timeout_sec:.3f}s)"
            )
        if not np.isfinite(self._latest_odometry_velocity).all():
            return False, f"odometry on {self._odometry_topic} contains non-finite linear velocity"
        return True, f"base_lin_vel is supplied by fresh body-frame odometry on {self._odometry_topic}."

    def _publish_timing(self):
        self._timing_publisher.publish(String(data=json.dumps(self._timing.snapshot())))

    def _sim_identity(self, msg):
        try:
            changed = self._sim_release.identity(msg.data)
        except (ValueError, KeyError, TypeError) as error:
            self._sim_release.error = str(error)
            changed = True
        if changed or self._sim_release.error:
            self._invalidate_async_inference()
            self._inference_schedule.reset()
            # A new scene cannot inherit action history or command ownership.
            self._sim_release.reset(self.default_pos)
            self.action = self._sim_release.last_raw.copy()
            self._activation_mgr.state = ActivationState(released=True)
            self._policy_runner.reset()
            self._obs_state.action_history.fill(0.)
            self._execution_preparation = PolicyPreparation(self.joint_names)

    def _release_sim_action(self, now_ns):
        gate = self._sim_release
        result = gate.take(now_ns)
        if result is not None:
            self.action, target, trace_id = result
            self._publish_joint_command(target.tolist(), trace_id=trace_id,
                emit_trace_event=True, command_published_stamp_ns=now_ns)
            self._advance_action_history()
            gate.last_release['joint_names'] = self.joint_names
            gate.last_release['target'] = self._last_published_joint_command.tolist()
            self._sim_release_publisher.publish(String(data=json.dumps(gate.last_release)))
            warning = gate.cadence_warning(now_ns)
            if warning:
                self._logger.warn(warning)
        return result is not None

    def _invalidate_async_inference(self):
        gate = getattr(self, '_sim_release', None)
        if gate is not None:
            gate.clear()
            self.action = gate.last_raw.copy()
        worker = getattr(self, '_async_worker', None)
        if worker is not None:
            worker.invalidate()
            self._async_pending_ns = None
            self._async_has_action = False

    def _poll_async_inference(self, now_ns):
        """Accept only current-generation, timely results on the ROS thread."""
        worker = self._async_worker
        try:
            result = worker.poll()
        except (RuntimeError, EOFError, BrokenPipeError, OSError) as exc:
            self._fault_async_inference(str(exc))
            return None
        if result is None:
            if (self._async_pending_ns is not None and
                    time.monotonic_ns() - self._async_pending_ns > self._inference_max_age_ns):
                self._fault_async_inference('Inference deadline exceeded')
            return None
        self._async_pending_ns = None
        observation_ns, trace_id = result.context
        self._result_observation_ns = observation_ns
        if (result.error is not None or
                result.finished_ns - result.submitted_ns > self._inference_max_age_ns or
                not 0 <= now_ns - observation_ns <= self._inference_max_age_ns):
            self._fault_async_inference(str(result.error) if result.error else 'Inference result expired')
            return None
        action, state = result.value
        candidate = np.asarray(action)
        if candidate.shape != self.action.shape or not np.isfinite(candidate).all():
            self._fault_async_inference('Inference returned invalid action')
            return None
        self.action = np.clip(candidate, -10., 10.)
        for name, value in state.items():
            setattr(self._policy_runner, name, value)
        self._policy_runner.current_action = self.action.copy()
        self._async_has_action = True
        estimated = state.get('latest_estimated_state')
        if estimated is not None:
            self._latest_estimated_velocity = estimated[:3].copy()
        self._publish_trace_event(trace_id, 'infer_end', stamp_ns=now_ns)
        timing = getattr(self, '_timing', None)
        if timing is not None:
            timing.count('inferences')
            timing.duration('inference', result.finished_ns - result.submitted_ns)
            timing.duration('inference_result_wait', time.monotonic_ns() - result.finished_ns)
        return trace_id

    def _fault_async_inference(self, message):
        self._invalidate_async_inference()
        self._async_fault = message
        self._logger.error(f'{message}; policy command publication stopped. Explicit reactivation is required.')

    def _advance_action_history(self):
        if self._obs_state.action_history.shape[0] > 0:
            frame = action_history_feedback(
                self._actor_action_history_mode, self.action,
                self._latest_policy_joint_command, self.default_pos, self._action_scale)
            self._obs_state.action_history[:-1] = self._obs_state.action_history[1:]
            self._obs_state.action_history[-1] = frame

    def _timer_tick(self):
        timing = getattr(self, '_timing', None)
        if timing is None or timing.mode == 'off':
            return self._timer_tick_impl()
        started = timing.tick(self._expected_tick_ns)
        try:
            return self._timer_tick_impl()
        finally:
            timing.finish_callback(started, self._expected_tick_ns)

    def _timer_tick_impl(self):
        timing = getattr(self, '_timing', None)
        if timing is not None and timing.mode == 'off':
            timing = None
        # Need both messages before proceeding
        if self._latest_joint_state is None or self._latest_imu is None:
            self._inference_schedule.reset()
            self._invalidate_async_inference()
            if timing is not None:
                timing.count('ticks_missing_inputs')
            return

        observation_ns = self.get_clock().now().nanoseconds
        now_time = observation_ns * 1e-9

        joint_state = self._latest_joint_state
        imu = self._latest_imu

        # Data freshness check (original logic retained)
        js_age = now_time - (joint_state.header.stamp.sec + joint_state.header.stamp.nanosec * 1e-9 if joint_state.header.stamp else now_time)
        imu_age = now_time - (imu.header.stamp.sec + imu.header.stamp.nanosec * 1e-9 if imu.header.stamp else now_time)
        if js_age > 0.5 or imu_age > 0.5:
            self._inference_schedule.reset()
            self._invalidate_async_inference()
            if timing is not None:
                timing.count('ticks_stale_inputs')
            if self._stale_warn_count < 10:
                self._stale_warn_count += 1
                self._logger.warn(f"Stale sensor data detected (joint_state age: {js_age:.3f}s, imu age: {imu_age:.3f}s); skipping control tick")
            return

        self._stale_warn_count = 0  # reset counter on fresh data

        # If not active, keep publishing default stance (so downstream controllers hold posture)
        if not self._activation_mgr.is_active():
            self._inference_schedule.reset()
            self._invalidate_async_inference()
            if hasattr(self, '_input_timing'):
                self._input_timing.reset_interval()
            if timing is not None:
                timing.count('ticks_inactive')
            blended = self._activation_mgr.compute_deactivation_blend(now_time)
            if blended is not None:
                # Use ramp-down blending (blended already scaled raw action; apply scale here)
                action_pos = self._map_action_to_joint_positions(blended)
                self._publish_joint_command(action_pos.tolist())
                return
            return

        gate = getattr(self, '_sim_release', None)
        if gate is not None:
            if gate.last_clock is not None and observation_ns < gate.last_clock:
                gate.error = 'Waiting for a new scene identity after simulation clock reset'
            if gate.observe(observation_ns, self._operator_permission.allowed()):
                self._invalidate_async_inference()
                self._inference_schedule.reset()
            if not self._base_lin_vel_readiness()[0]:
                self._invalidate_async_inference()
                self._inference_schedule.reset()
                return
            self._release_sim_action(observation_ns)
            if gate.pending is not None:
                self._publish_joint_command(gate.last_target.tolist())
                return
        worker = getattr(self, '_async_worker', None)
        completed_trace_id = None
        if worker is not None:
            permission = self._operator_permission.allowed()
            if ((self._async_last_clock_ns is not None and observation_ns < self._async_last_clock_ns)
                    or permission != self._async_permission):
                self._invalidate_async_inference()
                self._inference_schedule.reset()
            self._async_last_clock_ns = observation_ns
            self._async_permission = permission
            if self._async_fault is not None:
                return
            if not self._base_lin_vel_readiness()[0]:
                self._invalidate_async_inference()
                self._inference_schedule.reset()
                return
            completed_trace_id = self._poll_async_inference(observation_ns)
            if self._async_fault is not None:
                return
            if completed_trace_id is not None:
                # Publish/commit this result before building another observation.
                # This avoids an extra timer-period delay without stale action history.
                ramp = self._activation_mgr.compute_activation_factor(now_time)
                position = self._map_action_to_joint_positions(self.action * ramp)
                if gate is not None:
                    gate.submit(self.action, position, self._result_observation_ns, observation_ns, completed_trace_id)
                    self.action = gate.last_raw.copy()
                    if not self._release_sim_action(observation_ns):
                        self._publish_joint_command(gate.last_target.tolist())
                    return
                self._publish_joint_command(position.tolist(), trace_id=completed_trace_id,
                                            emit_trace_event=True, command_published_stamp_ns=observation_ns)
                self._advance_action_history()
        should_infer = self._inference_schedule.due(observation_ns) and (worker is None or worker.ready)
        applied_inference = completed_trace_id is not None
        if timing is not None:
            timing.count('inference_ticks' if should_infer else 'held_action_ticks')
            timing.stage('input_checks')
        if should_infer:
            self._update_velocity_estimator(joint_state, imu, now_time)
        if timing is not None:
            timing.stage('velocity_estimator')

        base_lin_vel_ready, base_lin_vel_message = self._base_lin_vel_readiness()
        if not base_lin_vel_ready:
            self._inference_schedule.reset()
            self._invalidate_async_inference()
            if timing is not None:
                timing.count('ticks_velocity_unready')
            if not self._base_lin_vel_warned:
                self._logger.warn(f"Skipping active policy ticks: {base_lin_vel_message}")
                self._base_lin_vel_warned = True
            return
        self._base_lin_vel_warned = False

        # Build observation via modular builder
        trace_id = self._next_trace_id()
        capture = timing is not None and timing.capture_enabled()
        base_ros_ns = int(self.get_clock().now().nanoseconds) if capture else 0
        base_perf_ns = time.perf_counter_ns() if capture else 0
        last_stage_ns = base_ros_ns

        def _next_stage_stamp_ns() -> int:
            nonlocal last_stage_ns
            if not capture:
                return 0
            candidate = base_ros_ns + max(0, int(time.perf_counter_ns() - base_perf_ns))
            if candidate <= last_stage_ns:
                candidate = last_stage_ns + 1
            last_stage_ns = candidate
            return candidate

        if timing is not None:
            timing.stage('readiness_and_trace_setup')
        if should_infer:
            command_override = self._locked_command_input if self._command_input_locked else None
            if not self._operator_permission.allowed():
                command_override = np.zeros(3)
            imu_override = self._locked_imu_input if self._imu_input_locked else None
            base_lin_vel_override = self._base_lin_vel_override()
            filtered_joint_position = None
            if self._joint_position_filter is not None:
                filtered_joint_position = self._joint_position_filter.update(
                    self._joint_position_relative_vector(),
                    now_time,
                )
            joint_position_override = (
                self._locked_joint_position_relative
                if self._joint_position_input_locked
                else filtered_joint_position
            )
            joint_velocity_override = self._locked_joint_velocity if self._velocity_input_locked else None
            obs = self._obs_builder.build(
                joint_state,
                imu,
                self._cmd_vel,
                self._inference_schedule.observation_dt(observation_ns),
                self._obs_state,
                command_override=command_override,
                imu_override=imu_override,
                base_lin_vel_override=base_lin_vel_override,
                joint_position_relative_override=joint_position_override,
                joint_velocity_override=joint_velocity_override,
            )
            if timing is not None:
                timing.stage('observation_build')
            if not np.isfinite(obs).all():
                self._invalidate_async_inference()
                if not self._nan_obs_warned:
                    self._logger.warn(
                        "Non-finite values detected in policy observation; publishing default stance until inputs recover."
                    )
                    self._nan_obs_warned = True
                self._publish_joint_command(
                    self.default_pos.tolist(),
                    trace_id=trace_id,
                    emit_trace_event=True,
                    command_published_stamp_ns=_next_stage_stamp_ns(),
                )
                return
            self._nan_obs_warned = False
            self._publish_trace_event(trace_id, "obs_ready", stamp_ns=_next_stage_stamp_ns())

            self._publish_trace_event(trace_id, "infer_start", stamp_ns=_next_stage_stamp_ns())
            if timing is not None:
                timing.stage('observation_validation_and_trace')
                input_timing = getattr(self, '_input_timing', None)
                if input_timing is not None:
                    overridden = []
                    if self._imu_input_locked:
                        overridden.append('imu')
                    if self._joint_position_input_locked or self._velocity_input_locked:
                        overridden.append('joint')
                    input_timing.inference(
                        self.get_clock().now().nanoseconds,
                        {'joint': joint_state, 'imu': imu}, overridden=overridden,
                    )
                    if (getattr(self, '_dynamic_joint_states_topic', '')
                            and 'joint' not in overridden):
                        input_timing.encoder_inference(joint_state)
            inference_started = time.monotonic_ns()
            overrides = {}
            if self._concurrent_policy_feature_builder is not None:
                overrides = {
                    'estimated_velocity_override': self._effective_estimator_velocity_override(),
                    'estimated_foot_positions_override': self._locked_estimator_fk,
                    'estimated_foot_height_override': self._locked_estimator_foot_height,
                    'estimated_contact_override': self._locked_estimator_contact,
                }
            args, kwargs = runner_inputs(
                obs, feature_builder=self._concurrent_policy_feature_builder,
                joint_state=joint_state, imu=imu,
                previous_action=(self.action if self._concurrent_policy_feature_builder is not None else None),
                filtered_joint_position=filtered_joint_position, **overrides)
            if worker is not None:
                self._async_pending_ns = time.monotonic_ns()
                try:
                    submitted = worker.submit(deepcopy((args, kwargs)), (observation_ns, trace_id))
                except (RuntimeError, EOFError, BrokenPipeError, OSError) as exc:
                    self._fault_async_inference(str(exc))
                    return
                if not submitted:
                    self._async_pending_ns = None
                else:
                    # Reserve the next ROS-clock slot at submission, not at polling.
                    # No backlog is possible while the worker is busy.
                    self._inference_schedule.complete(observation_ns, observation_ns)
                trace_id = self._next_trace_id()  # Held output is not the pending inference result.
            else:
                self.action = self._policy_runner.step(*args, **kwargs)
                if self._concurrent_policy_feature_builder is not None:
                    estimated_state = self._policy_runner.latest_estimated_state
                    if estimated_state is not None:
                        self._latest_estimated_velocity = estimated_state[:3].copy()
            if worker is None:
                applied_inference = True
                self._publish_trace_event(trace_id, "infer_end", stamp_ns=_next_stage_stamp_ns())
            if timing is not None and worker is None:
                timing.count('inferences')
                timing.duration('inference', time.monotonic_ns() - inference_started)
                timing.stage('inference_stage')
            self.action = np.clip(self.action, -10.0, 10.0)
            if not np.isfinite(self.action).all():
                if not self._nan_action_warned:
                    self._logger.warn(
                        "Policy returned non-finite action values; zeroing action for safety until policy outputs recover."
                    )
                    self._nan_action_warned = True
                self.action = np.zeros_like(self.action)
            else:
                self._nan_action_warned = False
            self._record_velocity_estimator_diagnostic(self.action)
            # self._logger.info(f"Policy action: {self.action}")
        elif worker is None and gate is None:
            self.action = self._policy_runner.step(None, infer=False)
        if timing is not None:
            timing.stage('action_postprocess')

        if worker is not None and applied_inference:
            return  # Already published and committed before the next submission.
        if worker is not None and not self._async_has_action:
            return  # Never publish an action from before activation/invalidation.

        if gate is not None:
            if applied_inference:
                ramp = self._activation_mgr.compute_activation_factor(now_time)
                target = self._map_action_to_joint_positions(self.action * ramp)
                completed_ns = self.get_clock().now().nanoseconds
                gate.submit(self.action, target, observation_ns, completed_ns, trace_id)
                self.action = gate.last_raw.copy()
                self._inference_schedule.complete(observation_ns, completed_ns)
            if not self._release_sim_action(self.get_clock().now().nanoseconds):
                self._publish_joint_command(gate.last_target.tolist())
            return

        # Compute ramp factor if within activation ramp window
        ramp_factor = self._activation_mgr.compute_activation_factor(now_time)

        # Blend action with default stance using ramp_factor
        action_pos = self._map_action_to_joint_positions(self.action * ramp_factor)
        # Default stance position
        # action_pos = np.array([
        #     0.013983699157926743,
        #     0.014070058297082522,
        #     0.010710449401688749,
        #     0.010222955727580363,
        #     0.7223747663654485,
        #     0.8646544758427479,
        #     -0.6927793343187066,
        #     -0.8130560657618886,            
        #     1.4660535165153326,
        #     1.4227150784029632,
        #     -1.4433106484754676,
        #     -1.3653020549923698,            
        # ], dtype=float)

        # output of policy before scaling and offset

        # [ 5.9300490e-02  1.8793666e-01  1.6632198e-01  1.2081476e-01
        #   -1.2849021e-01  2.3038940e-01 -1.1642768e-01  3.7860721e-03
        #    1.8053133e+01 -1.1327438e+01 -1.0141122e+01  1.8795408e+01]
        # self._joint_command = JointState()
        # self._joint_command.header.stamp = self.get_clock().now().to_msg()
        # self._joint_command.name = self.joint_names
        
        # self._joint_command.position = action_pos.tolist()
        # self._joint_command.velocity = np.zeros(len(self.joint_names)).tolist()
        # self._joint_command.effort = np.zeros(len(self.joint_names)).tolist()
        # self._joint_publisher.publish(self._joint_command)
 
        if timing is not None:
            timing.stage('action_mapping')
        self._publish_joint_command(
            action_pos.tolist(),
            trace_id=trace_id,
            emit_trace_event=True,
            command_published_stamp_ns=_next_stage_stamp_ns(),
        )
        if timing is not None:
            timing.stage('command_publish')
        if applied_inference:
            self._advance_action_history()
        if timing is not None:
            timing.stage('action_history')
        if should_infer and worker is None:
            self._inference_schedule.complete(observation_ns, self.get_clock().now().nanoseconds)
    # Legacy methods (_compute_observation, _compute_action, forward, quat_to_rot_matrix)
    # are now handled by modular components but retained above as commented history.
    def destroy_node(self):
        adapter = getattr(self, '_reload_adapter', None)
        if adapter is not None:
            adapter.manager.close()
        return super().destroy_node()

    def load_policy(self):
        """Load the neural network policy from the specified path."""
        # Load policy from file to io.BytesIO object
        with open(self.policy_path, 'rb') as f:
            content = f.read()
            buffer = io.BytesIO(content)
        # Load TorchScript model from buffer
        self.policy = torch.jit.load(buffer)
        self._loaded_artifacts['policy.pt'] = artifact_identity(self.policy_path, content)

    def _load_safety_position_limits(self) -> None:
        """Load optional per-joint position clamps from a YAML safety envelope."""
        if not self._safety_limits_enabled:
            self._logger.info("Policy output safety position limits are disabled.")
            return
        if not self._safety_limits_path.exists():
            self._logger.warn(
                f"Safety limits enabled but file does not exist: {self._safety_limits_path}. "
                "Policy commands will be published unclamped."
            )
            return
        try:
            with self._safety_limits_path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
        except Exception as exc:
            self._logger.warn(
                f"Failed to load safety limits from {self._safety_limits_path}: {exc}. "
                "Policy commands will be published unclamped."
            )
            return

        limits = data.get("limits", {})
        loaded = 0
        for idx, name in enumerate(self.joint_names):
            joint_limit = limits.get(name)
            if not isinstance(joint_limit, dict) or not bool(joint_limit.get("enabled", True)):
                continue
            try:
                min_position = joint_limit.get("min_position")
                max_position = joint_limit.get("max_position")
                if min_position is not None:
                    self._safety_min_pos[idx] = float(min_position)
                if max_position is not None:
                    self._safety_max_pos[idx] = float(max_position)
            except (TypeError, ValueError) as exc:
                self._logger.warn(f"Ignoring invalid safety limit for {name}: {exc}")
                continue
            loaded += 1

        self._safety_limits_loaded = loaded > 0
        if self._safety_limits_loaded:
            self._logger.info(f"Loaded safety position limits for {loaded}/{len(self.joint_names)} joints.")
        else:
            self._logger.warn(
                f"Safety limits file contained no usable joint position limits: {self._safety_limits_path}. "
                "Policy commands will be published unclamped."
            )

    def _map_action_to_joint_positions(self, action: np.ndarray) -> np.ndarray:
        """Apply the selected policy-action output mapping before the safety guard."""
        if self._smooth_bounded_action_mapper is not None:
            return self._smooth_bounded_action_mapper.map_action(action)
        return self.default_pos + np.asarray(action, dtype=np.float64) * self._action_scale

    def _apply_safety_position_limits(self, command_values: list[float]) -> list[float]:
        if not self._safety_limits_enabled or not self._safety_limits_loaded:
            return command_values
        command = np.asarray(command_values, dtype=np.float64)
        if command.shape != self._safety_min_pos.shape:
            self._logger.warn(
                f"Safety limit shape mismatch: command_shape={command.shape}, "
                f"limit_shape={self._safety_min_pos.shape}. Publishing unclamped command."
            )
            return command_values

        cache = getattr(self, '_command_clip_cache', None)
        if cache is None:
            cache = self._command_clip_cache = CommandClipCache()
        clipped, clipped_indices = cache.apply(
            command, self._safety_min_pos, self._safety_max_pos)
        if clipped_indices:
            now = self.get_clock().now().nanoseconds * 1e-9
            if now - self._last_safety_clip_warn_time >= 1.0:
                self._last_safety_clip_warn_time = now
                clipped_joints = [
                    f"{self.joint_names[i]}:{command[i]:.3f}->{clipped[i]:.3f}"
                    for i in clipped_indices
                ]
                self._logger.warn("Safety position limits clipped policy command: " + ", ".join(clipped_joints))
        return clipped

    @staticmethod
    def _infer_policy_input_dim(policy: torch.jit.ScriptModule) -> int | None:
        """Infer expected policy input width from first 2D weight tensor."""
        try:
            for name, tensor in policy.state_dict().items():
                if "weight" in name and tensor.ndim == 2:
                    return int(tensor.shape[1])
        except Exception:
            return None
        return None

    @staticmethod
    def _coerce_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _resolve_env_path(self, env_path_raw: str) -> Path:
        """Resolve env yaml path from parameter value."""
        return self._resolve_package_path(env_path_raw, default_relative="policy/env.yaml")

    def _resolve_package_path(self, raw_path: str, *, default_relative: str) -> Path:
        raw = (raw_path or "").strip() or default_relative
        resolved = Path(raw)
        if not resolved.is_absolute():
            resolved = (Path(__file__).resolve().parent.parent / resolved).resolve()
        return resolved

    def _get_stamp_prefix(self) -> str:
        """Create a timestamp prefix for logging with both system and ROS time.
        
        Returns:
            str: Formatted timestamp string with system and ROS time
        """
        now = time.time()
        now_ros = self.get_clock().now().nanoseconds / 1e9
        return f'[{now}][{now_ros}]'

    def header_time_in_seconds(self, header) -> float:
        """Convert a ROS message header timestamp to seconds.
        
        Args:
            header: ROS message header containing timestamp
            
        Returns:
            float: Time in seconds
        """
        return header.stamp.sec + header.stamp.nanosec * 1e-9


def main(args=None):
    """Main function to initialize and run the Marvin policy server node."""
    rclpy.init(args=args)
    node = None
    try:
        node = MarvinPolicyServer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info("KeyboardInterrupt received, shutting down policy server.")
    except ExternalShutdownException:
        # Launch has already shut down the shared ROS context.
        pass
    finally:
        if node is not None:
            worker = getattr(node, '_async_worker', None)
            if worker is not None:
                worker.close()
            node._close_velocity_estimator_shadow()
            try:
                node.destroy_node()
            except (KeyboardInterrupt, ExternalShutdownException):
                pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
