#!/usr/bin/env python3
"""Summarize a matched ROS velocity-estimator runtime sweep.

Expected input layout::

    ROOT/<friction>_<source>/diagnostic/diagnostic_*.csv

The motion window is relative to the first active policy sample.  The replay
tool currently uses two seconds of active neutral history before ten seconds of
recorded Joy commands, hence the defaults of 2 and 12 seconds.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _columns(rows: list[dict[str, str]], names: tuple[str, ...]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in names] for row in rows])


def _rmse(values: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    result = np.sqrt(np.mean(np.square(values), axis=axis))
    return float(result) if np.ndim(result) == 0 else result


def _high_frequency_fraction(
    values: np.ndarray,
    rate_hz: float,
    cutoff_hz: float,
) -> float:
    centered = values - np.mean(values, axis=0, keepdims=True)
    spectrum = np.fft.rfft(centered, axis=0)
    power = np.square(np.abs(spectrum))
    frequencies = np.fft.rfftfreq(len(values), d=1.0 / rate_hz)
    total = float(np.sum(power[1:]))
    if total == 0.0:
        return 0.0
    return float(np.sum(power[frequencies >= cutoff_hz]) / total)


def analyze_csv(
    path: Path,
    *,
    motion_start_sec: float,
    motion_end_sec: float,
    sample_rate_hz: float,
    high_frequency_cutoff_hz: float,
) -> dict[str, object]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No diagnostic samples in {path}")

    time = _columns(rows, ("ros_time_sec",)).reshape(-1)
    relative_time = time - time[0]
    ready = _columns(rows, ("history_ready",)).reshape(-1).astype(bool)
    motion = (relative_time >= motion_start_sec) & (relative_time < motion_end_sec)
    if not np.any(motion):
        raise ValueError(f"No samples in motion window for {path}")

    estimate = _columns(rows, ("estimated_vx", "estimated_vy", "estimated_vz"))
    odometry = _columns(rows, ("odometry_vx", "odometry_vy", "odometry_vz"))
    command = _columns(rows, ("command_x", "command_y", "command_yaw"))
    gravity = _columns(
        rows,
        ("projected_gravity_x", "projected_gravity_y", "projected_gravity_z"),
    )
    action_names = tuple(name for name in rows[0] if name.startswith("policy_action_"))
    actions = _columns(rows, action_names)

    velocity_error = estimate - odometry
    selected_tracking_error = np.linalg.norm(
        command[motion, :2] - odometry[motion, :2],
        axis=1,
    )
    selected_actions = actions[motion]
    action_delta = np.diff(selected_actions, axis=0)
    action_second_difference = np.diff(selected_actions, n=2, axis=0)
    gravity_z = gravity[motion, 2]

    def velocity_metrics(mask: np.ndarray) -> dict[str, object]:
        error = velocity_error[mask]
        per_axis = _rmse(error, axis=0)
        return {
            "samples": int(np.sum(mask)),
            "per_axis_mps": per_axis.tolist(),
            "planar_vector_mps": float(_rmse(np.linalg.norm(error[:, :2], axis=1))),
            "three_axis_vector_mps": float(_rmse(np.linalg.norm(error, axis=1))),
        }

    return {
        "csv": str(path.resolve()),
        "velocity_source": rows[0]["velocity_source"],
        "sample_count": len(rows),
        "duration_sec": float(relative_time[-1]),
        "history_ready_count": int(np.sum(ready)),
        "motion_window_sec": [motion_start_sec, motion_end_sec],
        "motion_sample_count": int(np.sum(motion)),
        "estimator_error_history_ready": velocity_metrics(ready),
        "estimator_error_motion": velocity_metrics(motion),
        "planar_tracking_error_motion": {
            "mean_mps": float(np.mean(selected_tracking_error)),
            "rms_mps": float(_rmse(selected_tracking_error)),
            "p95_mps": float(np.percentile(selected_tracking_error, 95)),
        },
        "policy_action_motion": {
            "mean_delta": float(np.mean(np.linalg.norm(action_delta, axis=1))),
            "mean_second_difference": float(
                np.mean(np.linalg.norm(action_second_difference, axis=1))
            ),
            "high_frequency_fraction": _high_frequency_fraction(
                selected_actions,
                sample_rate_hz,
                high_frequency_cutoff_hz,
            ),
            "high_frequency_cutoff_hz": high_frequency_cutoff_hz,
        },
        "projected_gravity_z_motion": {
            "minimum": float(np.min(gravity_z)),
            "maximum": float(np.max(gravity_z)),
            "fallen_sample_count": int(np.sum(gravity_z > -0.5)),
        },
        "median_input_age_ms": {
            name.removesuffix("_age_sec"): float(
                np.median(_columns(rows, (name,)).reshape(-1)) * 1000.0
            )
            for name in ("joint_state_age_sec", "imu_age_sec", "odometry_age_sec")
        },
    }


def _percent_change(new: float, baseline: float) -> float:
    return (new / baseline - 1.0) * 100.0


def add_comparisons(cases: dict[str, dict[str, object]]) -> dict[str, object]:
    def metric(case: str, section: str, key: str) -> float:
        return float(cases[case][section][key])  # type: ignore[index]

    pairs = {
        "high_vs_low_odometry": ("s090_d070_odometry", "s060_d050_odometry"),
        "low_estimator_vs_odometry": ("s060_d050_estimator", "s060_d050_odometry"),
        "high_estimator_vs_odometry": ("s090_d070_estimator", "s090_d070_odometry"),
        "high_vs_low_estimator": ("s090_d070_estimator", "s060_d050_estimator"),
    }
    comparisons: dict[str, object] = {}
    for label, (new, baseline) in pairs.items():
        if new not in cases or baseline not in cases:
            continue
        comparisons[label] = {
            "new": new,
            "baseline": baseline,
            "planar_tracking_mean_percent": _percent_change(
                metric(new, "planar_tracking_error_motion", "mean_mps"),
                metric(baseline, "planar_tracking_error_motion", "mean_mps"),
            ),
            "policy_action_delta_percent": _percent_change(
                metric(new, "policy_action_motion", "mean_delta"),
                metric(baseline, "policy_action_motion", "mean_delta"),
            ),
            "policy_action_second_difference_percent": _percent_change(
                metric(new, "policy_action_motion", "mean_second_difference"),
                metric(baseline, "policy_action_motion", "mean_second_difference"),
            ),
            "policy_action_high_frequency_percent": _percent_change(
                metric(new, "policy_action_motion", "high_frequency_fraction"),
                metric(baseline, "policy_action_motion", "high_frequency_fraction"),
            ),
        }
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--motion-start-sec", type=float, default=2.0)
    parser.add_argument("--motion-end-sec", type=float, default=12.0)
    parser.add_argument("--sample-rate-hz", type=float, default=50.0)
    parser.add_argument("--high-frequency-cutoff-hz", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    cases: dict[str, dict[str, object]] = {}
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        matches = sorted((case_dir / "diagnostic").glob("diagnostic_*.csv"))
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one diagnostic CSV for {case_dir.name}, found {len(matches)}"
            )
        cases[case_dir.name] = analyze_csv(
            matches[0],
            motion_start_sec=args.motion_start_sec,
            motion_end_sec=args.motion_end_sec,
            sample_rate_hz=args.sample_rate_hz,
            high_frequency_cutoff_hz=args.high_frequency_cutoff_hz,
        )

    result = {
        "schema": "marvin.velocity_estimator_runtime_sweep_summary.v1",
        "root": str(root),
        "cases": cases,
        "comparisons": add_comparisons(cases),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output = args.output or root / "summary.json"
    output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
