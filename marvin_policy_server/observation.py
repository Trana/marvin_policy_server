"""Observation construction utilities for Marvin policy server.

This module factors out the quaternion conversion and observation vector
computation originally embedded in `marvin_policy_server.py`.
Original logic is preserved; only structured into functions/classes.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
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
    base_lin_vel_history: np.ndarray  # shape (H_lin, 3)
    base_ang_vel_history: np.ndarray  # shape (H_ang, 3)
    projected_gravity_history: np.ndarray  # shape (H_grav, 3)
    velocity_commands_history: np.ndarray  # shape (H_cmd, 3)
    joint_pos_history: np.ndarray  # shape (H_pos, 12)
    joint_vel_history: np.ndarray  # shape (H_vel, 12)


@dataclass
class ObservationConfig:
    base_lin_vel_history_len: int
    base_ang_vel_history_len: int
    projected_gravity_history_len: int
    velocity_commands_history_len: int
    joint_pos_history_len: int
    joint_vel_history_len: int
    actions_history_len: int


class ObservationBuilder:
    """Creates policy observations from ROS messages.

    The integration of linear acceleration (velocity += acc * dt).
    """
    def __init__(self, joint_names: Sequence[str], cfg: ObservationConfig):
        self.joint_names = list(joint_names)
        self.cfg = cfg
        self.expected_obs_dim = (
            3 * self.cfg.base_lin_vel_history_len
            + 3 * self.cfg.base_ang_vel_history_len
            + 3 * self.cfg.projected_gravity_history_len
            + 3 * self.cfg.velocity_commands_history_len
            + len(self.joint_names) * self.cfg.joint_pos_history_len
            + len(self.joint_names) * self.cfg.joint_vel_history_len
            + len(self.joint_names) * self.cfg.actions_history_len
        )

    @staticmethod
    def _push_history(history: np.ndarray, value: np.ndarray) -> None:
        """Push a single vector into a fixed-size history [oldest ... newest]."""
        if history.shape[0] <= 1:
            history[0] = value
            return
        if np.isnan(history).all():
            history[:] = value
            return
        history[:-1] = history[1:]
        history[-1] = value

    def build(
        self,
        joint_state: JointState,
        imu: Imu,
        cmd_vel,
        dt: float,
        obs_state: ObservationState,
        *,
        command_override: np.ndarray | None = None,
        imu_override: dict[str, np.ndarray] | None = None,
        base_lin_vel_override: np.ndarray | None = None,
        joint_position_relative_override: np.ndarray | None = None,
        joint_velocity_override: np.ndarray | None = None,
    ) -> np.ndarray:

        # Quaternion extraction
        quat_I = imu.orientation
        quat_array = np.array([quat_I.w, quat_I.x, quat_I.y, quat_I.z])
        R_BI = quat_to_rot_matrix(quat_array).T

        lin_acc_b = np.array([
            imu.linear_acceleration.x,
            imu.linear_acceleration.y,
            imu.linear_acceleration.z,
        ])
        if base_lin_vel_override is not None:
            base_lin_vel = np.asarray(base_lin_vel_override, dtype=np.float64)
            if base_lin_vel.shape != (3,):
                raise ValueError(f"base_lin_vel_override must have shape (3,), got {base_lin_vel.shape}")
            obs_state.lin_vel_b[:] = base_lin_vel
        elif imu_override is not None:
            base_lin_vel = np.asarray(imu_override["base_lin_vel"], dtype=np.float64)
            obs_state.lin_vel_b[:] = base_lin_vel
        else:
            obs_state.lin_vel_b[:] = lin_acc_b * dt + obs_state.lin_vel_b
            base_lin_vel = obs_state.lin_vel_b
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
        if obs_state.base_lin_vel_history.shape[0] > 0:
            self._push_history(obs_state.base_lin_vel_history, base_lin_vel)
        

        ang_vel_b = (
            np.array([
                imu.angular_velocity.x,
                imu.angular_velocity.y,
                imu.angular_velocity.z,
            ])
            if imu_override is None
            else np.asarray(imu_override["base_ang_vel"], dtype=np.float64)
        )
        # ang_vel_b = np.array([0.0, 0.0, 0.0])        
        # ang_vel_b = np.array([imu.angular_velocity.x,
        #     imu.angular_velocity.y,
        #     0.0,
        # ])

        gravity_b = (
            np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))
            if imu_override is None
            else np.asarray(imu_override["projected_gravity"], dtype=np.float64)
        )

        cmd_vec = (
            np.array([cmd_vel.linear.x, cmd_vel.linear.y, cmd_vel.angular.z], dtype=np.float64)
            if command_override is None
            else np.asarray(command_override, dtype=np.float64)
        )
        # cmd_vec = np.where(np.abs(cmd_vec) < 0.2, 0.0, cmd_vec)
        # cmd_vec = [0.0, 0.0, 0.0]


        current_joint_pos = np.zeros(len(self.joint_names), dtype=np.float64)
        current_joint_vel = np.zeros(len(self.joint_names), dtype=np.float64)
        for i, name in enumerate(self.joint_names):
            if name in joint_state.name:
                idx = joint_state.name.index(name)
                current_joint_pos[i] = joint_state.position[idx]
                current_joint_vel[i] = joint_state.velocity[idx]
        if joint_velocity_override is not None:
            override = np.asarray(joint_velocity_override, dtype=np.float64)
            if override.shape != current_joint_vel.shape:
                raise ValueError(
                    f"joint_velocity_override shape {override.shape} does not match {current_joint_vel.shape}"
                )
            current_joint_vel[:] = override

        joint_pos_rel = current_joint_pos - obs_state.default_pos
        if joint_position_relative_override is not None:
            override = np.asarray(
                joint_position_relative_override,
                dtype=np.float64,
            )
            if override.shape != joint_pos_rel.shape:
                raise ValueError(
                    'joint_position_relative_override shape '
                    f'{override.shape} does not match {joint_pos_rel.shape}'
                )
            joint_pos_rel[:] = override

        # Keep history order aligned with Isaac Lab flattening:
        # [oldest ... newest] for each term.
        self._push_history(obs_state.base_ang_vel_history, ang_vel_b)
        self._push_history(obs_state.projected_gravity_history, gravity_b)
        self._push_history(obs_state.velocity_commands_history, cmd_vec)
        self._push_history(obs_state.joint_pos_history, joint_pos_rel)
        self._push_history(obs_state.joint_vel_history, current_joint_vel)

        obs = np.concatenate(
            [array.reshape(-1) for array in (
                obs_state.base_lin_vel_history,
                obs_state.base_ang_vel_history,
                obs_state.projected_gravity_history,
                obs_state.velocity_commands_history,
                obs_state.joint_pos_history,
                obs_state.joint_vel_history,
                obs_state.action_history,
            ) if array.size > 0]
        )
        return obs

__all__ = [
    'quat_to_rot_matrix',
    'ObservationBuilder',
    'ObservationState',
    'ObservationConfig',
]
