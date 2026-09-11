# Velocity-estimator runtime and shadow mode

The learned body-velocity estimator can either supply the actor's
`base_lin_vel` observation or run in shadow mode beside another source. Select
the active source with `base_lin_vel_source=estimator`; shadow recording remains
independently controlled by `velocity_estimator_shadow_enabled`.

## File boundaries

- `velocity_estimator.py`: checkpoint loading, normalization, MLP inference, and
  ten-frame history.
- `velocity_estimator_features.py`: exact 45-value deployable feature frame from
  ROS IMU, joint state, and previous policy action.
- `velocity_estimator_shadow.py`: aggregate error metrics and timestamped CSV
  output.
- `velocity_estimator_diagnostics.py`: opt-in full feature/action/timestamp CSV
  traces for matched runtime experiments.
- `joy_command_replay_node.py`: deterministic `50 Hz` replay of one chopped Joy
  episode through the deployed `/joy` and policy-activation interfaces.
- `tools/analyze_velocity_estimator_runtime_sweep.py`: offline comparison of
  matched friction/source diagnostic runs.
- `marvin_policy_server.py`: ROS parameters, source selection, topic publication,
  and the call before each decimated policy inference tick.

The packaged estimator artifact is `policy/velocity_estimator.pt`; it remains
separate from the actor's `policy.pt`.

## Output

Isaac bringup enables shadow mode by default. It publishes best-effort
`std_msgs/msg/Float64MultiArray` samples on
`/sim/velocity_estimator_shadow` in this order:

1. estimated `vx`, `vy`, `vz`;
2. odometry `vx`, `vy`, `vz`;
3. estimate-minus-odometry error `vx`, `vy`, `vz`;
4. history count;
5. history-ready flag (`0` or `1`).

CSV files contain the same values plus the policy-tick time, joint/IMU/odometry
source timestamps, effective velocity command, and estimator inference duration.
They are created under
`~/.ros/marvin_velocity_estimator_shadow/`. Metrics are reported separately for
all samples and for samples after the ten-frame history has filled.

When `base_lin_vel_source=estimator`, each estimate is written into the same
policy observation that immediately follows it. The estimator uses IMU, joint
position/velocity, and the previously applied policy action; it does not consume
odometry. With shadow mode enabled, `/isaac_odometry` remains subscribed only as
the reference used for metrics and CSV output. Estimator history is reset on
each successful policy activation.

## Detailed matched-runtime diagnostics

Set `velocity_estimator_diagnostic_log_dir` only for a diagnostic run. The
policy server then writes one CSV row per policy inference containing source
timestamps and ages, history readiness, effective command, the exact 45-value
estimator feature frame, estimate, odometry reference, and the resulting 12 raw
actor actions. Odometry is subscribed for this reference even when the
estimator supplies the actor. The default empty value disables this recorder.

`joy_command_replay` reads one chopped Joy episode and publishes the equivalent
raw Joy axes. It waits for simulation time, sends neutral commands before
activation, fills the ten-frame estimator history during a two-second active
neutral warmup, replays the episode at `50 Hz`, returns to neutral, and
deactivates the policy. This exercises the same launch, topic, mapping,
estimator, observation, actor, and safety path as interactive control.

For a sweep laid out as `<root>/<friction>_<source>/diagnostic/*.csv`, run:

```bash
/home/trana/venv_ros/bin/python \
  tools/analyze_velocity_estimator_runtime_sweep.py <root>
```

The analyzer writes `<root>/summary.json`. Its default motion window is seconds
`2..12`, corresponding to the replay tool's active warmup followed by one
ten-second clip. It reports estimator RMSE, planar tracking error, action first
and second differences, action energy at or above `10 Hz`, projected-gravity
upright checks, and median sensor ages.

## Sensor convention

The training collector's Isaac Lab `ImuCfg` reports specific force and reads
approximately `+9.81 m/s^2` on body Z at rest. Isaac's ROS IMU graph reports
approximately gravity-free kinematic acceleration. Isaac bringup therefore sets
`velocity_estimator_add_gravity_to_imu_acceleration=true`, which converts the ROS
sample with:

```text
training_linear_acceleration = ros_linear_acceleration - 9.81 * projected_gravity
```

The generic policy-server launch leaves this conversion disabled. Do not enable
it for the physical BNO055 until a stationary hardware trace confirms the sensor's
sign and gravity convention.
