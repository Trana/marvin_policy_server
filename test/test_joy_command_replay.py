import csv

import numpy as np

from marvin_policy_server.joy_command_recording import policy_velocity_command_from_axes
from marvin_policy_server.joy_command_replay_node import command_to_axes, load_commands


def test_command_to_axes_round_trips_deployed_mapping() -> None:
    for command in ((1.2, -0.4, 0.3), (-0.8, 0.5, -0.6), (0.0, 0.0, 0.0)):
        np.testing.assert_allclose(
            policy_velocity_command_from_axes(command_to_axes(command)), command
        )


def test_load_commands_reads_chopped_episode(tmp_path) -> None:
    path = tmp_path / "episode.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "step_index",
                "command_linear_x",
                "command_linear_y",
                "command_angular_z",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "step_index": 0,
                "command_linear_x": 0.5,
                "command_linear_y": -0.2,
                "command_angular_z": 0.1,
            }
        )

    assert load_commands(path) == ((0.5, -0.2, 0.1),)
