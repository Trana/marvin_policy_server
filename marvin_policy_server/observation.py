"""Observation construction utilities for Marvin policy server.

This module factors out the quaternion conversion and observation vector
computation originally embedded in `marvin_policy_server.py`.
Original logic is preserved; only structured into functions/classes.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence
import numpy as np
from rclpy.logging import get_logger
from sensor_msgs.msg import JointState, Imu  # type: ignore


def quat_to_rot_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert input quaternion (w, x, y, z) to 3x3 rotation matrix."""
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
    action_history: np.ndarray  # shape (history_len, action_dim)
    default_pos: np.ndarray  # shape (12,)


class ObservationBuilder:
    """Creates policy observations from ROS messages.

    The integration of linear acceleration (velocity += acc * dt).
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
        logger = get_logger(__name__)
        
        prev_lin_vel = obs_state.lin_vel_b.copy()
        obs_state.lin_vel_b[:] = lin_acc_b * dt + obs_state.lin_vel_b
        # logger.info(
        #     f"lin_acc_b={np.array2string(lin_acc_b, precision=6)}, "
        #     f"dt={dt}, "
        #     f"prev_lin_vel_b={np.array2string(prev_lin_vel, precision=6)}, "
        #     f"new_lin_vel_b={np.array2string(obs_state.lin_vel_b, precision=6)}"
        # )

        # Zero out small velocity components (magnitude < 0.2)
        # mask = np.abs(obs_state.lin_vel_b) < 0.2
        # if np.any(mask):
        #     logger.debug(
        #         "Zeroing lin_vel_b components below 0.2: indices=%s, values=%s",
        #         np.array2string(obs_state.lin_vel_b[mask], precision=6),
        #     )
        # obs_state.lin_vel_b[mask] = 0.0
       
        # obs_state.lin_vel_b[:] = np.array(
        #     [-1.59406548e-04, -2.59802181e-04,  1.87091297e-02],
        #     dtype=np.float64,
        # )
        # logger.info('obs: %s' %obs_state.lin_vel_b)
        # obs_state.lin_vel_b[:] = np.array(
        #     [0.0, 0.0, 0.0],
        #     dtype=np.float64,
        # )
        

        ang_vel_b = np.array([
            imu.angular_velocity.x,
            imu.angular_velocity.y,
            imu.angular_velocity.z,
        ])
        # ang_vel_b = np.array([0.0, 0.0, 0.0])        
        # ang_vel_b = np.array([imu.angular_velocity.x,
        #     imu.angular_velocity.y,
        #     0.0,
        # ])

        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))

        cmd_vec = [cmd_vel.linear.x, cmd_vel.linear.y, cmd_vel.angular.z]
        # cmd_vec = np.where(np.abs(cmd_vec) < 0.2, 0.0, cmd_vec)
        # cmd_vec = [0.0, 0.0, 0.0]


        action_dim = len(self.joint_names)
        history_len = obs_state.action_history.shape[0]
        obs = np.zeros(33 + action_dim * history_len)
        # IMPORTANT ZEROING OUT LIN VELOCITY BECAUSE OF DRIFT
        # obs[:3] = obs_state.lin_vel_b #[0.0, 0.0, 0.0]  # obs_state.lin_vel_b
        # Linear acceleration (body) as observation
        # obs[3:6] = lin_acc_b #[0.0, 0.0, 0.0] 
        obs[:3] = ang_vel_b
        obs[3:6] = gravity_b
        obs[6:9] = cmd_vec

        current_joint_pos = np.zeros(12)
        current_joint_vel = np.zeros(12)
        for i, name in enumerate(self.joint_names):
            if name in joint_state.name:
                idx = joint_state.name.index(name)
                current_joint_pos[i] = joint_state.position[idx]
                current_joint_vel[i] = joint_state.velocity[idx]

        obs[9:21] = current_joint_pos - obs_state.default_pos
        # diff = current_joint_pos - obs_state.default_pos
        # print('pos diff:', np.array2string(diff, precision=6, separator=', '))
        obs[21:33] = current_joint_vel
        obs[33:33 + action_dim * history_len] = obs_state.action_history.reshape(-1)
        
        # ang_vel_b_str = np.array2string(ang_vel_b, precision=4, suppress_small=True)
        # logger.info('obs: %s' % obs)
        
        # Example observation vectors for reference/debugging:
        # static_obs = np.array([
        #     -3.18336813e-03, -2.36710650e-04,  5.55757375e-04, -3.07860186e-02,
        #     -2.73388228e-02, -9.99152045e-01,  0.00000000e+00,  0.00000000e+00,
        #      0.00000000e+00,  5.45000000e-02, -2.63600000e-01, -1.41000000e-01,
        #      2.96300000e-01, -2.20105361e-01,  3.14952516e-02,  1.43105361e-01,
        #     -1.94395252e-01,  1.98802449e-01,  1.78202449e-01, -2.43202449e-01,
        #     -8.73024488e-02, -9.20000000e-03,  2.26000000e-02,  6.44000000e-02,
        #     -4.11000000e-02, -6.13000000e-02, -1.15800000e-01,  5.22000000e-02,
        #      9.72000000e-02, -1.45900000e-01, -1.66000000e-01,  1.28100000e-01,
        #      1.58000000e-01, -4.05929424e-02, -5.45067608e-01, -3.89900237e-01,
        #      5.50662816e-01, -4.14984345e-01,  1.20059617e-01,  2.54735425e-02,
        #     -8.30087289e-02, -7.74564892e-02,  5.88614494e-03, -7.19503760e-02,
        #     -1.18519031e-01,
        # ])
        # if static_obs.shape[0] == obs.shape[0]:
        #     obs = static_obs.copy()

        return obs

__all__ = [
    'quat_to_rot_matrix',
    'ObservationBuilder',
    'ObservationState'
]
