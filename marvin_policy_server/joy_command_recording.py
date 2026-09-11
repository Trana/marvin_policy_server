"""Replay-friendly recording for Marvin joystick command sessions."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Sequence


SCHEMA_VERSION = 'marvin.joy_commands.v1'
CSV_FIELDS = (
    'sample_index',
    'elapsed_sec',
    'ros_time_sec',
    'source_stamp_sec',
    'command_linear_x',
    'command_linear_y',
    'command_angular_z',
    'axes_json',
    'buttons_json',
)


def _axis(axes: Sequence[float], index: int) -> float:
    return float(axes[index]) if index < len(axes) else 0.0


def _deadband(value: float, threshold: float) -> float:
    return value if abs(value) >= threshold else 0.0


def _scale_axis(value: float, negative_max: float, positive_max: float) -> float:
    return positive_max * value if value >= 0.0 else abs(negative_max) * value


def policy_velocity_command_from_axes(axes: Sequence[float]) -> tuple[float, float, float]:
    """Apply the policy server's Joy-to-body-command mapping.

    Returns forward velocity, lateral velocity, and yaw rate.
    """
    linear_x = _scale_axis(_deadband(_axis(axes, 1), 0.06), -1.0, 1.5)
    linear_y = _scale_axis(_deadband(_axis(axes, 3), 0.13), -1.0, 1.0)
    angular_z = _scale_axis(_deadband(_axis(axes, 0), 0.06), -1.0, 1.0)
    return linear_x, linear_y, angular_z


class JoyCommandSessionWriter:
    """Write one timestamped recording folder containing metadata and samples."""

    def __init__(
        self,
        output_root: str | Path,
        *,
        joy_topic: str,
        flush_every_samples: int = 50,
    ) -> None:
        self.output_root = Path(output_root).expanduser().resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.joy_topic = str(joy_topic)
        self.flush_every_samples = max(1, int(flush_every_samples))
        self.started_at = datetime.now(timezone.utc)
        self.started_monotonic = time.monotonic()
        self.session_dir = self._create_session_dir()
        self.csv_path = self.session_dir / 'joy_commands.csv'
        self.metadata_path = self.session_dir / 'metadata.json'
        self._stream = self.csv_path.open('x', encoding='utf-8', newline='')
        self._writer = csv.DictWriter(self._stream, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        self.sample_count = 0
        self._last_elapsed_sec = 0.0
        self._closed = False
        self._write_metadata('recording')

    def _create_session_dir(self) -> Path:
        stem = f"joy_{self.started_at.strftime('%Y%m%dT%H%M%SZ')}_{os.getpid()}"
        for suffix in range(1000):
            name = stem if suffix == 0 else f'{stem}_{suffix}'
            candidate = self.output_root / name
            try:
                candidate.mkdir()
                return candidate
            except FileExistsError:
                continue
        raise RuntimeError(
            f'Unable to allocate a Joy recording directory under {self.output_root}'
        )

    def _metadata(self, status: str) -> dict[str, object]:
        payload: dict[str, object] = {
            'schema_version': SCHEMA_VERSION,
            'status': status,
            'started_at_utc': self.started_at.isoformat(),
            'joy_topic': self.joy_topic,
            'sample_count': self.sample_count,
            'duration_sec': round(self._last_elapsed_sec, 9),
            'sample_file': self.csv_path.name,
            'command_frame': 'body',
            'command_units': {
                'command_linear_x': 'm/s',
                'command_linear_y': 'm/s',
                'command_angular_z': 'rad/s',
            },
            'command_mapping': {
                'linear_x': (
                    'axis[1], deadband=0.06, negative_scale=1.0, positive_scale=1.5'
                ),
                'linear_y': (
                    'axis[3], deadband=0.13, negative_scale=1.0, positive_scale=1.0'
                ),
                'angular_z': (
                    'axis[0], deadband=0.06, negative_scale=1.0, positive_scale=1.0'
                ),
            },
            'columns': list(CSV_FIELDS),
        }
        if status == 'complete':
            payload['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
        return payload

    def _write_metadata(self, status: str) -> None:
        temporary_path = self.metadata_path.with_suffix('.json.tmp')
        temporary_path.write_text(
            json.dumps(self._metadata(status), indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
        temporary_path.replace(self.metadata_path)

    def record(
        self,
        *,
        axes: Sequence[float],
        buttons: Sequence[int],
        ros_time_sec: float,
        source_stamp_sec: float,
        monotonic_time_sec: float | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError('Cannot append to a closed Joy recording')
        current_monotonic = (
            time.monotonic()
            if monotonic_time_sec is None
            else float(monotonic_time_sec)
        )
        elapsed_sec = max(0.0, current_monotonic - self.started_monotonic)
        linear_x, linear_y, angular_z = policy_velocity_command_from_axes(axes)
        self._writer.writerow({
            'sample_index': self.sample_count,
            'elapsed_sec': f'{elapsed_sec:.9f}',
            'ros_time_sec': f'{float(ros_time_sec):.9f}',
            'source_stamp_sec': f'{float(source_stamp_sec):.9f}',
            'command_linear_x': f'{linear_x:.9f}',
            'command_linear_y': f'{linear_y:.9f}',
            'command_angular_z': f'{angular_z:.9f}',
            'axes_json': json.dumps(
                [float(value) for value in axes], separators=(',', ':')
            ),
            'buttons_json': json.dumps(
                [int(value) for value in buttons], separators=(',', ':')
            ),
        })
        self.sample_count += 1
        self._last_elapsed_sec = elapsed_sec
        if self.sample_count % self.flush_every_samples == 0:
            self._stream.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stream.flush()
        self._stream.close()
        self._write_metadata('complete')


__all__ = [
    'CSV_FIELDS',
    'JoyCommandSessionWriter',
    'SCHEMA_VERSION',
    'policy_velocity_command_from_axes',
]
