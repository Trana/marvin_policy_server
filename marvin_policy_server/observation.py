"""Observation construction utilities for Marvin policy server.

This module factors out the quaternion conversion and observation vector
computation originally embedded in `marvin_policy_server.py`.
Original logic is preserved; only structured into functions/classes.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence
import numpy as np
from sensor_msgs.msg import JointState, Imu  # type: ignore


def quat_to_rot_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert input quaternion (w, x, y, z) to 3x3 rotation matrix.
    (Same implementation as original code.)"""
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


@dataclass
class ObservationState:
    lin_vel_b: np.ndarray  # shape (3,)
    previous_action: np.ndarray  # shape (12,)
    default_pos: np.ndarray  # shape (12,)


class ObservationBuilder:
    """Creates policy observations from ROS messages.

    The integration of linear acceleration into velocity mirrors the original
    simple approach (velocity += acc * dt) used in `marvin_policy_server.py`.
    """
    def __init__(self, joint_names: Sequence[str]):
        self.joint_names = list(joint_names)

    def build(self, joint_state: JointState, imu: Imu, cmd_vel, dt: float, obs_state: ObservationState) -> np.ndarray:
        # Quaternion extraction
        quat_I = imu.orientation
        quat_array = np.array([quat_I.w, quat_I.x, quat_I.y, quat_I.z])
        R_BI = quat_to_rot_matrix(quat_array).T

        # Linear acceleration (body)
        lin_acc_b = np.array([
            imu.linear_acceleration.x,
            imu.linear_acceleration.y,
            imu.linear_acceleration.z,
        ])
        # Integrate velocity in-place
        obs_state.lin_vel_b[:] = lin_acc_b * dt + obs_state.lin_vel_b

        ang_vel_b = np.array([
            imu.angular_velocity.x,
            imu.angular_velocity.y,
            imu.angular_velocity.z,
        ])

        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))

        cmd_vec = [cmd_vel.linear.x, cmd_vel.linear.y, cmd_vel.angular.z]

        obs = np.zeros(48)
        obs[:3] = obs_state.lin_vel_b
        obs[3:6] = ang_vel_b
        obs[6:9] = gravity_b
        obs[9:12] = cmd_vec

        current_joint_pos = np.zeros(12)
        current_joint_vel = np.zeros(12)
        for i, name in enumerate(self.joint_names):
            if name in joint_state.name:
                idx = joint_state.name.index(name)
                current_joint_pos[i] = joint_state.position[idx]
                current_joint_vel[i] = joint_state.velocity[idx]

        obs[12:24] = current_joint_pos - obs_state.default_pos
        obs[24:36] = current_joint_vel
        obs[36:48] = obs_state.previous_action
        return obs

__all__ = [
    'quat_to_rot_matrix',
    'ObservationBuilder',
    'ObservationState'
]
