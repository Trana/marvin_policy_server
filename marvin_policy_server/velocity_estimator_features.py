"""Construct velocity-estimator features from deployable ROS inputs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sensor_msgs.msg import Imu, JointState

from .observation import quat_to_rot_matrix


class VelocityEstimatorFeatureBuilder:
    """Build the 45 values recorded by the Isaac Lab data collector."""

    def __init__(
        self,
        joint_names: Sequence[str],
        default_positions: np.ndarray,
        *,
        add_gravity_to_linear_acceleration: bool = False,
        gravity_magnitude: float = 9.81,
    ):
        self.joint_names = tuple(joint_names)
        self.default_positions = np.asarray(default_positions, dtype=np.float64).reshape(-1)
        self.add_gravity_to_linear_acceleration = bool(add_gravity_to_linear_acceleration)
        self.gravity_magnitude = float(gravity_magnitude)
        if len(self.joint_names) != 12 or self.default_positions.shape != (12,):
            raise ValueError("The baseline velocity estimator requires Marvin's 12 motor joints")
        if not np.isfinite(self.gravity_magnitude) or self.gravity_magnitude <= 0.0:
            raise ValueError("gravity_magnitude must be finite and greater than zero")

    def build(
        self,
        joint_state: JointState,
        imu: Imu,
        previous_policy_action: np.ndarray,
        joint_position_relative_override: np.ndarray | None = None,
    ) -> np.ndarray:
        name_to_index = {name: index for index, name in enumerate(joint_state.name)}
        missing = [name for name in self.joint_names if name not in name_to_index]
        if missing:
            raise ValueError(f"JointState is missing estimator joints: {missing}")

        joint_position = np.empty(12, dtype=np.float64)
        joint_velocity = np.empty(12, dtype=np.float64)
        for output_index, name in enumerate(self.joint_names):
            source_index = name_to_index[name]
            if source_index >= len(joint_state.position) or source_index >= len(joint_state.velocity):
                raise ValueError(f"JointState arrays do not contain position and velocity for {name}")
            joint_position[output_index] = joint_state.position[source_index]
            joint_velocity[output_index] = joint_state.velocity[source_index]

        quaternion = np.array(
            [imu.orientation.w, imu.orientation.x, imu.orientation.y, imu.orientation.z],
            dtype=np.float64,
        )
        projected_gravity = quat_to_rot_matrix(quaternion).T @ np.array(
            [0.0, 0.0, -1.0], dtype=np.float64
        )
        linear_acceleration = np.array(
            [
                imu.linear_acceleration.x,
                imu.linear_acceleration.y,
                imu.linear_acceleration.z,
            ],
            dtype=np.float64,
        )
        if self.add_gravity_to_linear_acceleration:
            # Isaac's ROS graph publishes kinematic acceleration (approximately
            # zero at rest). Isaac Lab ImuCfg, used for estimator training,
            # publishes specific force (approximately +g on body Z at rest).
            linear_acceleration -= self.gravity_magnitude * projected_gravity
        angular_velocity = np.array(
            [imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z],
            dtype=np.float64,
        )
        previous_action = np.asarray(previous_policy_action, dtype=np.float64).reshape(-1)
        if previous_action.shape != (12,):
            raise ValueError(f"Expected 12 previous policy actions, got {previous_action.shape}")
        joint_position_relative = joint_position - self.default_positions
        if joint_position_relative_override is not None:
            override = np.asarray(
                joint_position_relative_override,
                dtype=np.float64,
            ).reshape(-1)
            if override.shape != (12,) or not np.isfinite(override).all():
                raise ValueError(
                    'joint_position_relative_override must contain 12 '
                    'finite values'
                )
            joint_position_relative = override

        features = np.concatenate(
            (
                linear_acceleration,
                angular_velocity,
                projected_gravity,
                joint_position_relative,
                joint_velocity,
                previous_action,
            )
        )
        if features.shape != (45,) or not np.isfinite(features).all():
            raise ValueError("Velocity-estimator feature frame is invalid")
        return features


__all__ = ["VelocityEstimatorFeatureBuilder"]
