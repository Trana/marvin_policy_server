# Joy command recording

Isaac ROS bringup can start a separate `joy_command_recorder` node with:

```text
record_joy_commands:=true
```

The Training UI exposes this as **Record Joy Commands** under Sim -> ROS2 Sim
Runtime. Recording begins with the ROS2 runtime and finishes when that runtime is
stopped.

Each run creates a timestamped directory under:

```text
~/.ros/marvin_joy_recordings/joy_<UTC timestamp>_<pid>/
```

The directory contains:

- `metadata.json`: schema version, topic, status, sample count, duration, units,
  and the Joy-to-command mapping;
- `joy_commands.csv`: one row per received `/joy` message.

Each CSV row stores elapsed monotonic time, ROS receipt time, Joy source stamp,
the raw axes/buttons as JSON arrays, and the body-frame command values
`command_linear_x`, `command_linear_y`, and `command_angular_z`. The derived
values use the same shared mapping function as `marvin_policy_server`, so they
match the command presented to the policy.

This ROS package records only. The separate Isaac Lab estimator tooling consumes
this versioned `marvin.joy_commands.v1` format, resamples it with zero-order hold,
and chops named source ranges into fixed replay episodes. See:

```text
/home/trana/Development/isaac/lab/marvin_isaaclab/marvin/
  scripts/velocity_estimator/joy_replay/README.md
```
