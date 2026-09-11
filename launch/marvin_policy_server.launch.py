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

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    policy_path = os.path.join(
        get_package_share_directory('marvin_policy_server'),
        'policy/policy.pt'
    )
    env_path = os.path.join(
        get_package_share_directory('marvin_policy_server'),
        'policy/env.yaml'
    )
    velocity_estimator_path = os.path.join(
        get_package_share_directory('marvin_policy_server'),
        'policy/velocity_estimator.pt'
    )
    safety_limits_path = os.path.join(
        get_package_share_directory('description'),
        'models/limits/marvin_joint_limits.yaml'
    )
    return LaunchDescription([
        DeclareLaunchArgument(
            "publish_period_ms",
            default_value="5",
            description="publishing dt in milliseconds"),
        DeclareLaunchArgument(
            "policy_path",
            default_value=policy_path,
            description="path to the policy file"),
        DeclareLaunchArgument(
            "env_path",
            default_value=env_path,
            description="path to the env yaml used for observation layout"),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="True",
            description="Use simulation (Omniverse Isaac Sim) clock if true"),
        DeclareLaunchArgument(
            "namespace",
            default_value="",
            description="Optional namespace for the policy server node"),
        DeclareLaunchArgument(
            "joy_topic",
            default_value="/joy",
            description="Joy topic used by policy server"),
        DeclareLaunchArgument(
            "joint_states_topic",
            default_value="/joint_states",
            description="Joint states topic used by policy server"),
        DeclareLaunchArgument(
            "imu_topic",
            default_value="/imu",
            description="IMU topic used by policy server"),
        DeclareLaunchArgument(
            "base_lin_vel_source",
            default_value="imu_integration",
            description="Source for base_lin_vel: imu_integration, odometry, or estimator"),
        DeclareLaunchArgument(
            "odometry_topic",
            default_value="/isaac_odometry",
            description="Body-frame nav_msgs/Odometry source used when base_lin_vel_source=odometry"),
        DeclareLaunchArgument(
            "odometry_child_frame_id",
            default_value="trunk",
            description="Required child_frame_id for body-frame odometry"),
        DeclareLaunchArgument(
            "base_lin_vel_timeout_sec",
            default_value="0.25",
            description="Maximum wall-clock age of odometry accepted for policy activation and inference"),
        DeclareLaunchArgument(
            "velocity_estimator_shadow_enabled",
            default_value="False",
            description="Record estimator-versus-odometry metrics independently of the selected policy source"),
        DeclareLaunchArgument(
            "velocity_estimator_checkpoint",
            default_value=velocity_estimator_path,
            description="Velocity-estimator training checkpoint used as a policy source or in shadow mode"),
        DeclareLaunchArgument(
            "velocity_estimator_shadow_topic",
            default_value="velocity_estimator_shadow",
            description="Eleven-value estimator/reference/error/history shadow topic"),
        DeclareLaunchArgument(
            "velocity_estimator_shadow_log_dir",
            default_value="",
            description="Optional directory for timestamped shadow CSV files"),
        DeclareLaunchArgument(
            "velocity_estimator_shadow_summary_interval_samples",
            default_value="250",
            description="Number of 50 Hz shadow samples between metric summaries"),
        DeclareLaunchArgument(
            "velocity_estimator_diagnostic_log_dir",
            default_value="",
            description="Optional directory for detailed estimator feature/action CSV traces"),
        DeclareLaunchArgument(
            "velocity_estimator_add_gravity_to_imu_acceleration",
            default_value="False",
            description="Convert gravity-free ROS acceleration to Isaac Lab IMU specific force"),
        DeclareLaunchArgument(
            "velocity_estimator_gravity_magnitude",
            default_value="9.81",
            description="Gravity magnitude used by the optional acceleration conversion"),
        DeclareLaunchArgument(
            "command_topic",
            default_value="marvin_joint_controller/commands",
            description="Output joint command topic"),
        DeclareLaunchArgument(
            "traced_command_topic",
            default_value="marvin_joint_controller/commands_traced",
            description="Output traced joint command topic"),
        DeclareLaunchArgument(
            "publish_legacy_commands",
            default_value="True",
            description="Publish std_msgs/Float64MultiArray command topic"),
        DeclareLaunchArgument(
            "publish_traced_commands",
            default_value="True",
            description="Publish marvin_trace_msgs/TracedFloat64MultiArray command topic"),
        DeclareLaunchArgument(
            "trace_event_topic",
            default_value="/trace/events",
            description="Shared trace event topic"),
        DeclareLaunchArgument(
            "publish_trace_events",
            default_value="True",
            description="Publish TraceEvent stage markers"),
        DeclareLaunchArgument(
            "trace_source_id",
            default_value="1",
            description="8-bit source namespace for trace_id high bits"),
        DeclareLaunchArgument(
            "trace_instance_id",
            default_value="-1",
            description="16-bit trace instance id (-1 uses process pid low bits)"),
        DeclareLaunchArgument(
            "set_active_service_name",
            default_value="set_active",
            description="SetBool service name to activate/deactivate policy"),
        DeclareLaunchArgument(
            "set_policy_input_lock_service_name",
            default_value="set_policy_input_lock",
            description="Typed service used to lock policy observation inputs"),
        DeclareLaunchArgument(
            "safety_limits_enabled",
            default_value="True",
            description="Enable per-joint policy output position clamps"),
        DeclareLaunchArgument(
            "safety_limits_path",
            default_value=safety_limits_path,
            description="Canonical YAML containing Marvin's soft command limits"),
        DeclareLaunchArgument(
            "action_output_mapping",
            default_value="linear",
            description="Policy action mapping: linear or experimental smooth_bounded"),
        DeclareLaunchArgument(
            "actor_action_history_mode",
            default_value="raw",
            description="Actor action history: raw, zero, or applied target equivalent"),
        DeclareLaunchArgument(
            "joint_position_filter_tau_sec",
            default_value="0.0",
            description=(
                "Optional first-order filter time constant for joint-position "
                "inputs; zero disables filtering"
            )),
        Node(
            package='marvin_policy_server',
            executable='marvin_policy_server',
            name='marvin_policy_server',
            namespace=LaunchConfiguration('namespace'),
            prefix='/home/trana/venv_ros/bin/python',
            output="screen",
            parameters=[{
                'publish_period_ms': LaunchConfiguration('publish_period_ms'),
                'policy_path': LaunchConfiguration('policy_path'),
                'env_path': LaunchConfiguration('env_path'),
                "use_sim_time": LaunchConfiguration('use_sim_time'),
                "joy_topic": LaunchConfiguration('joy_topic'),
                "joint_states_topic": LaunchConfiguration('joint_states_topic'),
                "imu_topic": LaunchConfiguration('imu_topic'),
                "base_lin_vel_source": LaunchConfiguration('base_lin_vel_source'),
                "odometry_topic": LaunchConfiguration('odometry_topic'),
                "odometry_child_frame_id": LaunchConfiguration('odometry_child_frame_id'),
                "base_lin_vel_timeout_sec": LaunchConfiguration('base_lin_vel_timeout_sec'),
                "velocity_estimator_shadow_enabled": LaunchConfiguration('velocity_estimator_shadow_enabled'),
                "velocity_estimator_checkpoint": LaunchConfiguration('velocity_estimator_checkpoint'),
                "velocity_estimator_shadow_topic": LaunchConfiguration('velocity_estimator_shadow_topic'),
                "velocity_estimator_shadow_log_dir": LaunchConfiguration('velocity_estimator_shadow_log_dir'),
                "velocity_estimator_shadow_summary_interval_samples": LaunchConfiguration(
                    'velocity_estimator_shadow_summary_interval_samples'
                ),
                "velocity_estimator_diagnostic_log_dir": LaunchConfiguration(
                    'velocity_estimator_diagnostic_log_dir'
                ),
                "velocity_estimator_add_gravity_to_imu_acceleration": LaunchConfiguration(
                    'velocity_estimator_add_gravity_to_imu_acceleration'
                ),
                "velocity_estimator_gravity_magnitude": LaunchConfiguration(
                    'velocity_estimator_gravity_magnitude'
                ),
                "command_topic": LaunchConfiguration('command_topic'),
                "traced_command_topic": LaunchConfiguration('traced_command_topic'),
                "publish_legacy_commands": LaunchConfiguration('publish_legacy_commands'),
                "publish_traced_commands": LaunchConfiguration('publish_traced_commands'),
                "trace_event_topic": LaunchConfiguration('trace_event_topic'),
                "publish_trace_events": LaunchConfiguration('publish_trace_events'),
                "trace_source_id": LaunchConfiguration('trace_source_id'),
                "trace_instance_id": LaunchConfiguration('trace_instance_id'),
                "set_active_service_name": LaunchConfiguration('set_active_service_name'),
                "set_policy_input_lock_service_name": LaunchConfiguration('set_policy_input_lock_service_name'),
                "safety_limits_enabled": LaunchConfiguration('safety_limits_enabled'),
                "safety_limits_path": LaunchConfiguration('safety_limits_path'),
                "action_output_mapping": LaunchConfiguration('action_output_mapping'),
                "actor_action_history_mode": LaunchConfiguration(
                    'actor_action_history_mode'
                ),
                "joint_position_filter_tau_sec": LaunchConfiguration(
                    'joint_position_filter_tau_sec'
                ),
            }]
            
        ),
    ])
