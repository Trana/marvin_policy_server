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
from message_filters import Subscriber, TimeSynchronizer, ApproximateTimeSynchronizer
from sensor_msgs.msg import Joy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.qos import qos_profile_sensor_data



class MarvinPolicyServer(Node):
    """Fullbody controller for Marvin robot.
    
    This ROS 2 node subscribes to velocity commands and synchronized joint/IMU
    data, processes the data through a neural network policy, and publishes
    joint commands for controlling the Marvin robot's movements.
    """

    def __init__(self):
        """Initialize the marvin controller node."""
        super().__init__('marvin_controller')

        # Declare and set parameters
        self.declare_parameter('publish_period_ms', 5)
        self.declare_parameter('policy_path', 'policy/marvin_policy.pt')
        self.set_parameters(
            [rclpy.parameter.Parameter(
                'use_sim_time', 
                rclpy.Parameter.Type.BOOL, 
                True
            )]
        )

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

        # Subscriptions (use sensor_qos)
        self._imu_sub_filter = Subscriber(self, Imu, '/imu', qos_profile=sensor_qos)
        self._joint_states_sub_filter = Subscriber(self, JointState, '/joint_states', qos_profile=sensor_qos)



        # self.sync = ApproximateTimeSynchronizer([self._joint_states_sub_filter, self._imu_sub_filter],
        #                                 queue_size=5,
        #                                 slop=0.003,
        #                                 allow_headerless=False)

        self.sync = TimeSynchronizer([self._joint_states_sub_filter, self._imu_sub_filter], queue_size=2)

        def log_and_tick(joint_state, imu):
            # self._logger.info("Received synchronized JointState and Imu messages")
            self._tick(joint_state, imu)

        self.sync.registerCallback(log_and_tick)

        # Load neural network policy
        self.policy_path = self.get_parameter('policy_path').value
        self.load_policy()

        # Initialize state variables
        self._joint_state = JointState()
        self._joint_command = Float64MultiArray()
        # self._joint_command = JointState()
        self._cmd_vel = Twist()
        self._imu = Imu()
        # Same as in extension
        self._action_scale = 0.25  # Scale factor for policy output
        self._previous_action = np.zeros(12)
        self._policy_counter = 0
        self._decimation = 4  # Run policy every 4 ticks to reduce computation same as in extension
        self._last_tick_time = self.get_clock().now().nanoseconds * 1e-9
        self._lin_vel_b = np.zeros(3)  # Linear velocity in body frame
        self._dt = 0.0  # Time delta between ticks
        
        # Default joint positions representing the nominal stance
        self.default_pos = np.array([0, 0, 0, 0, 0.7853981633974483, -0.7853981633974483, -0.7853981633974483, 0.7853981633974483, 1.2217304763960306, -1.2217304763960306, -1.2217304763960306, 1.2217304763960306])
         
        # used in isaac lambda
        # joint_pos={
        #     "F[L,R]_hip_joint": 0,
        #     "R[L,R]_hip_joint": 0,
        #     ".*L_thigh_joint": radians(45),
        #     ".*R_thigh_joint": radians(-45),
        #     ".*L_calf_joint": radians(70),
        #     ".*R_calf_joint": radians(-70),
            
        # },

        # Joint list output from isaac sim:
        #  Joint names: ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 'FL_thigh_joint', 'FR_thigh_joint', 'RR_thigh_joint', 'RL_thigh_joint', 'FL_calf_joint', 'FR_calf_joint', 'RR_calf_joint', 'RL_calf_joint']


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

        # self.joint_names = [
        #     'FR_hip_joint',
        #     'FR_thigh_joint',
        #     'FR_calf_joint',

        #     'FL_hip_joint',
        #     'FL_thigh_joint',
        #     'FL_calf_joint',
            
        #     'RR_hip_joint',
        #     'RR_thigh_joint',
        #     'RR_calf_joint',
            
        #     'RL_hip_joint',
        #     'RL_thigh_joint',
        #     'RL_calf_joint'
        # ]

        # Update default positions to match the new joint count/order
        # self.default_pos = np.zeros(len(self.joint_names))
        self._previous_action = np.zeros(len(self.joint_names))



        self._logger.info("Initializing MarvinController")

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


        twist.linear.y = msg.axes[0] if abs(msg.axes[0]) >= 0.06 else 0.0  # Left stick left/right
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

    def _tick(self, joint_state: JointState, imu: Imu):
        """Process synchronized joint state and IMU data to generate robot commands.
        
        This method is called whenever new joint state and IMU data are available.
        It computes the policy's action and publishes the resulting joint
        commands.
        
        Args:
            joint_state: Current joint positions and velocities
            imu: Current IMU data (orientation, angular velocity, acceleration)
        """
        # Reset if time jumped backwards (most likely due to sim time reset)
        # stamp_js  = joint_state.header.stamp.sec + joint_state.header.stamp.nanosec * 1e-9
        stamp_imu = imu.header.stamp.sec        + imu.header.stamp.nanosec        * 1e-9
        msg_time  = self.get_clock().now().nanoseconds * 1e-9

        if msg_time < self._last_tick_time:
            self._logger.error(
                f'{self._get_stamp_prefix()} Time jumped backwards. Resetting.'
            )

        # self._logger.error("Tick")
        # Calculate time delta since last tick
        self._dt = max(1e-4, msg_time - self._last_tick_time)
        self._last_tick_time = msg_time

        # Run the control policy
        self.forward(joint_state, imu)

        # Prepare and publish the joint command message
        # self._joint_command.header.stamp = self.get_clock().now().to_msg()
        # self._joint_command.name = self.joint_names
        
        # Compute final joint positions by adding scaled actions to default positions
        action_pos = self.default_pos + (self.action * self._action_scale)
        self._joint_command.data = action_pos.tolist()
        # self._joint_command.position = action_pos.tolist()
        # self._joint_command.velocity = np.zeros(len(self.joint_names)).tolist()
        # self._joint_command.effort = np.zeros(len(self.joint_names)).tolist()
        self._joint_publisher.publish(self._joint_command)


    def _compute_observation(self, joint_state: JointState, imu: Imu):
        """
        Compute the observation vector for the policy

        Argument:
        command (np.ndarray) -- the robot command (v_x, v_y, w_z)

        Returns:
        np.ndarray -- The observation vector.

        """

        # Extract quaternion orientation from IMU
        quat_I = imu.orientation
        quat_array = np.array([quat_I.w, quat_I.x, quat_I.y, quat_I.z])

        # Convert quaternion to rotation matrix
        # (transpose for body to inertial frame)
        R_BI = self.quat_to_rot_matrix(quat_array).T

        # Extract linear acceleration and integrate to estimate velocity
        lin_acc_b = np.array([
            imu.linear_acceleration.x,
            imu.linear_acceleration.y,
            imu.linear_acceleration.z
        ])
        
        # Simple integration to estimate velocity
        self._lin_vel_b = lin_acc_b * self._dt + self._lin_vel_b
        
        # Extract angular velocity
        ang_vel_b = np.array([
            imu.angular_velocity.x,
            imu.angular_velocity.y,
            imu.angular_velocity.z
        ])
        
        # Calculate gravity direction in body frame
        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))


        # Prepare command vector
        cmd_vel = [
            self._cmd_vel.linear.x,
            self._cmd_vel.linear.y,
            self._cmd_vel.angular.z
        ]
        
        # self._lin_vel_b = v_b_est
        obs = np.zeros(48)
        # Base lin vel
        obs[:3] = self._lin_vel_b
        # Base ang vel
        obs[3:6] = ang_vel_b
        # Gravity
        obs[6:9] = gravity_b
        # Command
        obs[9:12] = cmd_vel

        # Joint states
        # Joint states (12 positions + 12 velocities)
        current_joint_pos = np.zeros(12)
        current_joint_vel = np.zeros(12)

        # Map joint states from message to our ordered arrays
        for i, name in enumerate(self.joint_names):
            if name in joint_state.name:
                idx = joint_state.name.index(name)
                current_joint_pos[i] = joint_state.position[idx]
                current_joint_vel[i] = joint_state.velocity[idx]

        # current_joint_pos = joint_state.position
        # current_joint_vel = joint_state.velocity
        
        obs[12:24] = current_joint_pos - self.default_pos
        obs[24:36] = current_joint_vel
        obs[36:48] = self._previous_action

        # Previous Action
        # working observations
        workingObs = np.array([
            -5.63125957e-04,  1.19583442e-04,  1.90295313e-02, -8.12328851e-04,
            2.62006618e-04,  1.10231055e-05,  1.14435471e-02, -5.79303097e-03,
            -9.99917740e-01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
            1.92985162e-02, -3.61246429e-02,  7.96489790e-03,  1.64176393e-02,
            5.54808597e-02, -3.99552802e-02, -1.43656949e-02, -5.39752622e-02,
            1.74683684e-01, -1.77788133e-01, -1.75175303e-01,  1.78479785e-01,
            1.65282330e-03,  3.07996734e-03, -2.40746490e-03, -1.70999649e-03,
            -7.07001314e-02,  7.76687935e-02,  7.22212270e-02, -7.14234561e-02,
            -1.19785279e-01,  1.21698461e-01,  1.24407470e-01, -1.18479051e-01,
            8.63728598e-02,  1.41581148e-01,  6.21595085e-02,  2.89009869e-01,
            1.64836347e-01, -8.35022405e-02, -1.44479156e-01, -9.60966051e-02,
            1.97437706e+01, -1.02578382e+01, -9.58512211e+00,  1.97399521e+01
        ])

        # obs = workingObs
        # self._logger.info(f"Observation: {obs}")
        return obs
    def _compute_action(self, obs):
        """Run the neural network policy to compute an action from the observation.
        
        Args:
            obs: Observation vector containing robot state information
            
        Returns:
            np.ndarray: Action vector containing joint position adjustments
        """
        # Run inference with the PyTorch policy
        with torch.no_grad():
            obs = torch.from_numpy(obs).view(1, -1).float()
            action = self.policy(obs).detach().view(-1).numpy()


        # self._logger.info(f"Policy action: {action}")
        return action

    def forward(self, joint_state: JointState, imu: Imu):
        """Process sensor data and compute control actions.
        
        This combines observation computation and policy evaluation.
        The policy is run at a reduced rate (decimation) to save computation.
        
        Args:
            joint_state: Current joint positions and velocities
            imu: Current IMU data
        """
        # Compute observation from current state
        obs = self._compute_observation(joint_state, imu)

        # Run policy at reduced frequency (every _decimation ticks)
        if self._policy_counter % self._decimation == 0:
            self.action = self._compute_action(obs)
            self._previous_action = self.action.copy()
        self._policy_counter += 1

    def quat_to_rot_matrix(self, quat: np.ndarray) -> np.ndarray:
        """Convert input quaternion to rotation matrix.

        Args:
            quat (np.ndarray): Input quaternion (w, x, y, z).

        Returns:
            np.ndarray: A 3x3 rotation matrix.
        """
        q = np.array(quat, dtype=np.float64, copy=True)
        nq = np.dot(q, q)
        if nq < 1e-10:
            return np.identity(3)
        q *= np.sqrt(2.0 / nq)
        q = np.outer(q, q)
        return np.array(
            (
                (1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]),
                (q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]),
                (q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]),
            ),
            dtype=np.float64,
        )

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