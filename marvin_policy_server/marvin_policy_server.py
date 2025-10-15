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
import time
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Imu 
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import SetBool
# from message_filters import Subscriber, TimeSynchronizer, ApproximateTimeSynchronizer
from sensor_msgs.msg import Joy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.qos import qos_profile_sensor_data

# New modular imports
from .activation_manager import ActivationManager, ActivationConfig
from .observation import ObservationBuilder, ObservationState, quat_to_rot_matrix  # noqa: F401 (quat reuse)
from .policy_runner import PolicyRunner, PolicyRunnerConfig



class MarvinPolicyServer(Node):
    """Fullbody controller for Marvin robot.
    
    This ROS 2 node subscribes to velocity commands and synchronized joint/IMU
    data, processes the data through a neural network policy, and publishes
    joint commands for controlling the Marvin robot's movements.
    """

    def __init__(self):
        """Initialize the marvin controller node."""
        super().__init__('marvin_policy_server')

        # Declare and set parameters
        self.declare_parameter('publish_period_ms', 5)
        self.declare_parameter('policy_path', 'policy/marvin_policy.pt')
        self.declare_parameter('activation_ramp_duration', 1.0)    # seconds to smoothly ramp in
        self.declare_parameter('deactivation_ramp_duration', 1.0)  # seconds to smoothly ramp out
        self.declare_parameter('release_after_deactivate', True)   # if True, stop publishing after ramp-down

        self._logger = self.get_logger()
        
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

        # Create subscription for velocity commands
        # self._cmd_vel_subscription = self.create_subscription(
        #     Twist,
        #     '/cmd_vel',
        #     self._cmd_vel_callback,
        #     qos_profile=10)

        self._joy_subscription = self.create_subscription(
            Joy,
            '/joy',
            self._joy_callback,
            qos_profile=10
        )

        sim_qos_profile = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            history=rclpy.qos.HistoryPolicy.KEEP_ALL,
        )

        # Publisher (use cmd_qos)
        self._joint_publisher = self.create_publisher(Float64MultiArray, 'marvin_joint_controller/commands', qos_profile=cmd_qos)
        # self._joint_publisher = self.create_publisher(JointState, 'isaac_joint_commands', qos_profile=sim_qos_profile)

        # Subscriptions (direct, store latest messages)
        self._latest_joint_state = None
        self._latest_imu = None

        self._joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_cb,
            qos_profile=sensor_qos
        )
        self._imu_sub = self.create_subscription(
            Imu,
            '/imu',
            self._imu_cb,
            qos_profile=sensor_qos
        )

        # Timer for control loop
        publish_period_ms = self.get_parameter('publish_period_ms').value
        self._timer = self.create_timer(publish_period_ms / 1000.0, self._timer_tick)

        # Load neural network policy
        self.policy_path = self.get_parameter('policy_path').value
        self.load_policy()

        # Initialize state variables (original kept; structured below)
        self._joint_state = JointState()
        self._joint_command = Float64MultiArray()
        self._cmd_vel = Twist()
        self._imu = Imu()
        self._action_scale = 0.25
        self._last_tick_time = self.get_clock().now().nanoseconds * 1e-9
        self._dt = 0.0
        
        # Default joint positions representing the nominal stance
        self.default_pos = np.array([0, 0, 0, 0, 0.7853981633974483, -0.7853981633974483, -0.7853981633974483, 0.7853981633974483, 1.2217304763960306, -1.2217304763960306, -1.2217304763960306, 1.2217304763960306])

        # Joint names in the order expected by the policy
        self.joint_names = [
            'FL_hip_joint',
            'FR_hip_joint',
            'RL_hip_joint',
            'RR_hip_joint',
            'FL_thigh_joint',
            'FR_thigh_joint',
            'RR_thigh_joint',
            'RL_thigh_joint',
            'FL_calf_joint',
            'FR_calf_joint',
            'RR_calf_joint',
            'RL_calf_joint'
        ]

        # --- Modular components ---
        self._obs_builder = ObservationBuilder(self.joint_names)
        self._obs_state = ObservationState(
            lin_vel_b=np.zeros(3),
            previous_action=np.zeros(len(self.joint_names)),
            default_pos=self.default_pos.copy(),
        )
        decimation = 4  # original value
        self._policy_runner = PolicyRunner(self.policy, len(self.joint_names), PolicyRunnerConfig(decimation=decimation))
        self._activation_mgr = ActivationManager(ActivationConfig(
            activation_ramp_duration=float(self.get_parameter('activation_ramp_duration').value),
            deactivation_ramp_duration=float(self.get_parameter('deactivation_ramp_duration').value),
            release_after_deactivate=bool(self.get_parameter('release_after_deactivate').value),
        ), logger=self.get_logger())
        self._set_active_srv = self.create_service(SetBool, 'set_active', self._set_active_cb)
        self.action = np.zeros(len(self.joint_names))
        self._previous_action = self._obs_state.previous_action
        self._logger.info("Initializing MarvinController (inactive by default; call /marvin_controller/set_active to enable)")

    def _set_active_cb(self, request, response):
        """Handle SetBool to enable/disable policy output.

        When activating: start ramp timer. When deactivating: reset actions and publish default stance once.
        """
        now = self.get_clock().now().nanoseconds * 1e-9
        if request.data:
            success, msg = self._activation_mgr.request_activate(now)
            response.success = success
            response.message = msg
        else:
            # Provide current action (unscaled raw) for ramp start
            success, msg = self._activation_mgr.request_deactivate(now, self.action)
            response.success = success
            response.message = msg
        return response

    def _joy_callback(self, msg):
        twist = Twist()
        # Map axes to Twist fields based on your config
        # Apply deadband: if abs(value) < 0.06, set to zero
        linear_x = msg.axes[1] * 2 if msg.axes[1] > 0 else msg.axes[1]
        angular_z = msg.axes[0] * 1.5

# if abs(linear_x) >= 0.06 else 0.0
#         if abs(angular_z) >= 0.06 else 0.0
        twist.linear.x = linear_x 
        twist.angular.z = angular_z 

        # twist.linear.x = 0
        # twist.linear.z = 0


        twist.linear.y = msg.axes[0] 
        # if abs(msg.axes[0]) >= 0.06 else 0.0  # Left stick left/right
        twist.linear.y = 0
        # twist.linear.z = msg.axes[7]    # Cross up/down
        # twist.angular.x = msg.axes[6]   # Cross left/right (roll)
        # twist.angular.y = msg.axes[4]   # Right stick up/down (pitch)
        twist.linear.y = msg.axes[3]   # Right stick left/right (yaw)
        self._cmd_vel = twist
        # self._logger.info(
        #     f"Joy->Twist: lin=({twist.linear.x}, {twist.linear.y}, {twist.linear.z}), "
        #     f"ang=({twist.angular.x}, {twist.angular.y}, {twist.angular.z})"
        # )

    def _cmd_vel_callback(self, msg):
        """Store the latest velocity command."""
        self._cmd_vel = msg

    def _joint_state_cb(self, msg: JointState):
        # Store latest joint state
        self._latest_joint_state = msg

    def _imu_cb(self, msg: Imu):
        # Store latest imu
        self._latest_imu = msg

    def _timer_tick(self):
        
        # Need both messages before proceeding
        if self._latest_joint_state is None or self._latest_imu is None:
            return

        now_time = self.get_clock().now().nanoseconds * 1e-9
        # Compute dt
        self._dt = max(1e-4, now_time - self._last_tick_time)
        self._last_tick_time = now_time

        joint_state = self._latest_joint_state
        imu = self._latest_imu

        # Data freshness check (original logic retained)
        js_age = now_time - (joint_state.header.stamp.sec + joint_state.header.stamp.nanosec * 1e-9 if joint_state.header.stamp else now_time)
        imu_age = now_time - (imu.header.stamp.sec + imu.header.stamp.nanosec * 1e-9 if imu.header.stamp else now_time)
        if js_age > 0.5 or imu_age > 0.5:
            return

        # If not active, keep publishing default stance (so downstream controllers hold posture)
        if not self._activation_mgr.is_active():
            blended = self._activation_mgr.compute_deactivation_blend(now_time)
            if blended is not None:
                # Use ramp-down blending (blended already scaled raw action; apply scale here)
                action_pos = self.default_pos + blended * self._action_scale
                self._joint_command.data = action_pos.tolist()
                self._joint_publisher.publish(self._joint_command)
                return
            return

        # Build observation via modular builder
        obs = self._obs_builder.build(joint_state, imu, self._cmd_vel, self._dt, self._obs_state)
        # Run policy (decimated)
        self.action = self._policy_runner.step(obs)
        # Link previous action reference for legacy observation method compatibility
        self._obs_state.previous_action = self._policy_runner.previous_action

        # Compute ramp factor if within activation ramp window
        ramp_factor = self._activation_mgr.compute_activation_factor(now_time)

        # Blend action with default stance using ramp_factor
        action_pos = self.default_pos + (self.action * self._action_scale * ramp_factor)
        # Default stance position
        # action_pos = np.array([
        #     0.013983699157926743,
        #     0.014070058297082522,
        #     0.010710449401688749,
        #     0.010222955727580363,
        #     0.7223747663654485,
        #     -0.6927793343187066,
        #     -0.8130560657618886,
        #     0.8646544758427479,
        #     1.4660535165153326,
        #     -1.4433106484754676,
        #     -1.3653020549923698,
        #     1.4227150784029632
        # ], dtype=float)
        self._joint_command.data = action_pos.tolist()
        self._joint_publisher.publish(self._joint_command)
    # Legacy methods (_compute_observation, _compute_action, forward, quat_to_rot_matrix)
    # are now handled by modular components but retained above as commented history.
    def load_policy(self):
        """Load the neural network policy from the specified path."""
        # Load policy from file to io.BytesIO object
        with open(self.policy_path, 'rb') as f:
            buffer = io.BytesIO(f.read())
        # Load TorchScript model from buffer
        self.policy = torch.jit.load(buffer)

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
    node = MarvinPolicyServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()