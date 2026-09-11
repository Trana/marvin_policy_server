import csv
import json

from marvin_policy_server.joy_command_recording import (
    JoyCommandSessionWriter,
    policy_velocity_command_from_axes,
)
import numpy as np


def test_policy_velocity_mapping_matches_runtime_axes_and_deadbands() -> None:
    np.testing.assert_allclose(
        policy_velocity_command_from_axes([0.5, 0.5, 0.0, -0.5]),
        [0.75, -0.5, 0.5],
    )
    np.testing.assert_allclose(
        policy_velocity_command_from_axes([0.05, -0.05, 0.0, 0.12]),
        [0.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(policy_velocity_command_from_axes([]), [0.0, 0.0, 0.0])


def test_session_writer_stores_raw_joy_and_derived_commands(tmp_path) -> None:
    writer = JoyCommandSessionWriter(tmp_path, joy_topic='/joy', flush_every_samples=1)
    writer.record(
        axes=[-0.25, 0.8, 0.0, 0.4],
        buttons=[0, 1, 0],
        ros_time_sec=12.5,
        source_stamp_sec=12.45,
        monotonic_time_sec=writer.started_monotonic + 0.2,
    )
    writer.close()
    writer.close()

    with writer.csv_path.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]['sample_index'] == '0'
    assert float(rows[0]['elapsed_sec']) == 0.2
    assert float(rows[0]['command_linear_x']) == 1.2
    assert float(rows[0]['command_linear_y']) == 0.4
    assert float(rows[0]['command_angular_z']) == -0.25
    assert json.loads(rows[0]['axes_json']) == [-0.25, 0.8, 0.0, 0.4]
    assert json.loads(rows[0]['buttons_json']) == [0, 1, 0]

    metadata = json.loads(writer.metadata_path.read_text(encoding='utf-8'))
    assert metadata['schema_version'] == 'marvin.joy_commands.v1'
    assert metadata['status'] == 'complete'
    assert metadata['sample_count'] == 1
    assert metadata['duration_sec'] == 0.2
    assert metadata['sample_file'] == 'joy_commands.csv'
