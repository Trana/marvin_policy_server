"""Detailed, opt-in CSV traces for velocity-estimator runtime diagnosis."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Sequence

import numpy as np


FEATURE_NAMES = (
    "imu_accel_x",
    "imu_accel_y",
    "imu_accel_z",
    "imu_gyro_x",
    "imu_gyro_y",
    "imu_gyro_z",
    "projected_gravity_x",
    "projected_gravity_y",
    "projected_gravity_z",
    *(f"joint_position_relative_{index}" for index in range(12)),
    *(f"joint_velocity_{index}" for index in range(12)),
    *(f"previous_policy_action_{index}" for index in range(12)),
)


class VelocityEstimatorDiagnosticRecorder:
    """Record estimator features, reference velocity, and resulting actor action."""

    def __init__(self, log_dir: str, *, joint_names: Sequence[str], velocity_source: str):
        directory = Path(log_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.log_path = directory / f"diagnostic_{stamp}_{os.getpid()}.csv"
        self.velocity_source = str(velocity_source)
        self.joint_names = tuple(joint_names)
        if len(self.joint_names) != 12:
            raise ValueError("Velocity-estimator diagnostics require 12 policy joints")
        self.header = (
            "sample_index",
            "velocity_source",
            "ros_time_sec",
            "joint_state_time_sec",
            "imu_time_sec",
            "odometry_time_sec",
            "joint_state_age_sec",
            "imu_age_sec",
            "odometry_age_sec",
            "history_count",
            "history_ready",
            "command_x",
            "command_y",
            "command_yaw",
            *FEATURE_NAMES,
            "estimated_vx",
            "estimated_vy",
            "estimated_vz",
            "odometry_vx",
            "odometry_vy",
            "odometry_vz",
            *(f"policy_action_{name}" for name in self.joint_names),
        )
        self._stream = self.log_path.open("x", encoding="utf-8", newline="")
        self._writer = csv.writer(self._stream)
        self._writer.writerow(self.header)
        self._stream.flush()
        self.count = 0

    def record(
        self,
        *,
        ros_time_sec: float,
        joint_state_time_sec: float,
        imu_time_sec: float,
        odometry_time_sec: float,
        history_count: int,
        history_ready: bool,
        command: np.ndarray,
        features: np.ndarray,
        estimated_velocity: np.ndarray,
        odometry_velocity: np.ndarray,
        policy_action: np.ndarray,
    ) -> None:
        command = np.asarray(command, dtype=np.float64).reshape(3)
        features = np.asarray(features, dtype=np.float64).reshape(45)
        estimated = np.asarray(estimated_velocity, dtype=np.float64).reshape(3)
        odometry = np.asarray(odometry_velocity, dtype=np.float64).reshape(3)
        action = np.asarray(policy_action, dtype=np.float64).reshape(12)
        values = np.concatenate((command, features, estimated, odometry, action))
        if not np.isfinite(values).all():
            raise ValueError("Velocity-estimator diagnostic row contains non-finite values")
        self._writer.writerow(
            (
                self.count,
                self.velocity_source,
                f"{ros_time_sec:.9f}",
                f"{joint_state_time_sec:.9f}",
                f"{imu_time_sec:.9f}",
                f"{odometry_time_sec:.9f}",
                f"{max(0.0, ros_time_sec - joint_state_time_sec):.9f}",
                f"{max(0.0, ros_time_sec - imu_time_sec):.9f}",
                f"{max(0.0, ros_time_sec - odometry_time_sec):.9f}",
                int(history_count),
                int(history_ready),
                *values.tolist(),
            )
        )
        self.count += 1
        if self.count % 50 == 0:
            self._stream.flush()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.flush()
            self._stream.close()
            self._stream = None
            self._writer = None


__all__ = ["FEATURE_NAMES", "VelocityEstimatorDiagnosticRecorder"]
