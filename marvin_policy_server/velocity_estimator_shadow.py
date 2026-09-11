"""Metrics and CSV recording for non-controlling velocity-estimator inference."""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


class VelocityEstimatorShadowRecorder:
    """Record estimator predictions while odometry remains in control."""

    CSV_HEADER = (
        "ros_time_sec",
        "joint_state_time_sec",
        "imu_time_sec",
        "odometry_time_sec",
        "command_x",
        "command_y",
        "command_yaw",
        "estimator_inference_ms",
        "history_count",
        "history_ready",
        "estimated_vx",
        "estimated_vy",
        "estimated_vz",
        "odometry_vx",
        "odometry_vy",
        "odometry_vz",
        "error_vx",
        "error_vy",
        "error_vz",
    )

    def __init__(self, log_dir: str = ""):
        self.count = 0
        self.ready_count = 0
        self._error_sum = np.zeros(3, dtype=np.float64)
        self._error_squared_sum = np.zeros(3, dtype=np.float64)
        self._ready_error_sum = np.zeros(3, dtype=np.float64)
        self._ready_error_squared_sum = np.zeros(3, dtype=np.float64)
        self._inference_duration_sum_ms = 0.0
        self._inference_duration_max_ms = 0.0
        self.log_path: Path | None = None
        self._stream = None
        self._writer = None
        if log_dir.strip():
            directory = Path(log_dir).expanduser().resolve()
            directory.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            self.log_path = directory / f"shadow_{stamp}_{os.getpid()}.csv"
            self._stream = self.log_path.open("x", encoding="utf-8", newline="")
            self._writer = csv.writer(self._stream)
            self._writer.writerow(self.CSV_HEADER)
            self._stream.flush()

    def record(
        self,
        *,
        ros_time_sec: float,
        joint_state_time_sec: float,
        imu_time_sec: float,
        odometry_time_sec: float,
        command: np.ndarray,
        inference_duration_ms: float,
        estimated_velocity: np.ndarray,
        odometry_velocity: np.ndarray,
        history_count: int,
        history_ready: bool,
    ) -> np.ndarray:
        estimated = np.asarray(estimated_velocity, dtype=np.float64).reshape(3)
        odometry = np.asarray(odometry_velocity, dtype=np.float64).reshape(3)
        command_values = np.asarray(command, dtype=np.float64).reshape(3)
        error = estimated - odometry
        self.count += 1
        self._error_sum += error
        self._error_squared_sum += error * error
        self._inference_duration_sum_ms += float(inference_duration_ms)
        self._inference_duration_max_ms = max(
            self._inference_duration_max_ms,
            float(inference_duration_ms),
        )
        if history_ready:
            self.ready_count += 1
            self._ready_error_sum += error
            self._ready_error_squared_sum += error * error

        if self._writer is not None:
            self._writer.writerow(
                (
                    f"{ros_time_sec:.9f}",
                    f"{joint_state_time_sec:.9f}",
                    f"{imu_time_sec:.9f}",
                    f"{odometry_time_sec:.9f}",
                    *command_values.tolist(),
                    float(inference_duration_ms),
                    int(history_count),
                    int(history_ready),
                    *estimated.tolist(),
                    *odometry.tolist(),
                    *error.tolist(),
                )
            )
            if self.count % 50 == 0:
                self._stream.flush()
        return error

    @staticmethod
    def _metrics(count: int, error_sum: np.ndarray, squared_sum: np.ndarray) -> dict:
        if count == 0:
            return {
                "count": 0,
                "bias_xyz": [float("nan")] * 3,
                "rmse_xyz": [float("nan")] * 3,
                "vector_rmse": float("nan"),
            }
        bias = error_sum / count
        mean_squared = squared_sum / count
        return {
            "count": count,
            "bias_xyz": bias.tolist(),
            "rmse_xyz": np.sqrt(mean_squared).tolist(),
            "vector_rmse": float(np.sqrt(mean_squared.sum())),
        }

    def summary(self) -> dict:
        return {
            "all": self._metrics(self.count, self._error_sum, self._error_squared_sum),
            "history_ready": self._metrics(
                self.ready_count,
                self._ready_error_sum,
                self._ready_error_squared_sum,
            ),
            "inference_duration_ms": {
                "mean": (
                    self._inference_duration_sum_ms / self.count
                    if self.count
                    else float("nan")
                ),
                "max": self._inference_duration_max_ms if self.count else float("nan"),
            },
            "log_path": str(self.log_path) if self.log_path is not None else None,
        }

    def close(self) -> None:
        if self._stream is not None:
            self._stream.flush()
            self._stream.close()
            self._stream = None
            self._writer = None


__all__ = ["VelocityEstimatorShadowRecorder"]
