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
            "command_topic",
            default_value="marvin_joint_controller/commands",
            description="Output joint command topic"),
        DeclareLaunchArgument(
            "set_active_service_name",
            default_value="set_active",
            description="SetBool service name to activate/deactivate policy"),
        Node(
            package='marvin_policy_server',
            executable='marvin_policy_server',
            name='marvin_policy_server',
            namespace=LaunchConfiguration('namespace'),
            output="screen",
            parameters=[{
                'publish_period_ms': LaunchConfiguration('publish_period_ms'),
                'policy_path': LaunchConfiguration('policy_path'),
                "use_sim_time": LaunchConfiguration('use_sim_time'),
                "joy_topic": LaunchConfiguration('joy_topic'),
                "joint_states_topic": LaunchConfiguration('joint_states_topic'),
                "imu_topic": LaunchConfiguration('imu_topic'),
                "command_topic": LaunchConfiguration('command_topic'),
                "set_active_service_name": LaunchConfiguration('set_active_service_name'),
            }]
            
        ),
    ])
