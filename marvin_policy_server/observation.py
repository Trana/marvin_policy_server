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
    previous_action: np.ndarray  # shape (12,)
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
        logger.info(
            f"lin_acc_b={np.array2string(lin_acc_b, precision=6)}, "
            f"dt={dt}, "
            f"prev_lin_vel_b={np.array2string(prev_lin_vel, precision=6)}, "
            f"new_lin_vel_b={np.array2string(obs_state.lin_vel_b, precision=6)}"
        )

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
        logger.info('obs: %s' %obs_state.lin_vel_b)
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


        obs = np.zeros(48)
        # IMPORTANT ZEROING OUT LIN VELOCITY BECAUSE OF DRIFT
        obs[:3] = [0.0, 0.0, 0.0]  # obs_state.lin_vel_b
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
        # diff = current_joint_pos - obs_state.default_pos
        # print('pos diff:', np.array2string(diff, precision=6, separator=', '))
        obs[24:36] = current_joint_vel
        obs[36:48] = obs_state.previous_action
        
        # ang_vel_b_str = np.array2string(ang_vel_b, precision=4, suppress_small=True)
        # logger.info('obs: %s' % obs)
        
        # Example observation vectors for reference/debugging:
        # obs = np.array([
        #     -2.27122939e-04, -1.04240797e-03,  1.86668151e-02,  1.30820591e-04,
        #     4.06018859e-04, -6.54161420e-04,  2.87281836e-03,  5.19567449e-02,
        #     -9.98645204e-01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #     4.34763022e-02, -6.54107407e-02,  1.80343583e-01,  9.78033915e-02,
        #     9.99135594e-02,  2.13553152e-01, -2.04890988e-01, -1.72359071e-01,
        #     3.38269943e-01,  3.88945335e-01, -3.54829544e-01, -3.02865499e-01,
        #     1.14134280e-03,  7.39401160e-03, -1.10237226e-02, -6.63060695e-03,
        #     -6.28539175e-02, -6.91853985e-02,  7.30465949e-02,  7.41629899e-02,
        #     -1.11090131e-01, -1.12651236e-01,  1.17409497e-01,  1.21343993e-01,
        #     -3.59977305e-01,  7.28558004e-03,  8.13645363e-01,  4.62237269e-01,
        #     1.21130264e+00,  5.87532163e-01,  5.06762683e-01, -1.03605735e+00,
        #     -2.25508377e-01,  1.44387960e-01, -4.36299890e-01, -6.08238056e-02,
        # ])

        # obs = np.array([
        #     -1.59406548e-04, -2.59802181e-04,  1.87091297e-02, -7.99350006e-04,
        #      7.33400934e-04, -1.76898812e-04, -1.12909066e-02,  1.41172010e-02,
        #     -9.99836597e-01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #      3.39348763e-02,  1.26848114e-03, -2.30620033e-03,  4.23359834e-02,
        #     -1.34518350e-02, -3.22518668e-02, -9.34725201e-02, -1.72445455e-02,
        #      1.11695999e-01,  1.28602737e-01, -1.06356019e-01, -7.82326402e-02,
        #      6.48760120e-04,  3.39452899e-03,  2.24570627e-03, -8.37928557e-04,
        #     -6.21068738e-02, -8.76936167e-02,  7.12144077e-02,  1.05473384e-01,
        #     -1.15727656e-01, -1.36240005e-01,  1.22187212e-01,  1.46895304e-01,
        #     -8.28574747e-02, -7.22911730e-02,  8.63198936e-02,  3.34160626e-01,
        #      2.54687201e-02, -3.45593803e-02,  3.19603980e-01,  1.68863952e-01,
        #     -8.14546943e-01, -8.76762331e-01,  7.57133424e-01,  4.21751499e-01,
        # ])

        return obs

__all__ = [
    'quat_to_rot_matrix',
    'ObservationBuilder',
    'ObservationState'
]
