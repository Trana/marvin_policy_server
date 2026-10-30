# Estimator acceleration contract

The estimator consumes SI specific force. For gravity-free input, the feature
adapter reconstructs this as:

```python
specific_force = input_acceleration * acceleration_scale - gravity_magnitude * projected_gravity
```

The gravity term is conditional on
`velocity_estimator_add_gravity_to_imu_acceleration`. The scale is configured by
`velocity_estimator_imu_acceleration_scale` and must be finite and positive.
Defaults remain scale 1.0 and no reconstruction in the generic policy launch.
Both embedded and separate estimator feature builders use the same conversion.
These settings do not modify ROS IMU messages or the IK input.

Isaac bringup already enables reconstruction and retains scale 1.0. Prepared
Marvin robot bringup now enables reconstruction and defaults
`policy_imu_acceleration_scale` to 0.980665. This specific scale applies to the
currently measured BNO055 driver: UNIT_SEL 0x83 selects mg and the driver divides
the register value by 100. Thus each reported unit corresponds to 100 mg, or
0.980665 m/s². Do not apply this scale to a driver that already publishes SI units.
Change the robot launch scale to 1.0 if the driver is corrected separately.

The IMU-only check found filtered acceleration norm 0.416 and raw norm 9.569 in
driver units. Restoring gravity addresses the input convention, not all bias,
mounting, calibration or fusion-delay errors. Do not subtract the measured bias
without a controlled calibration procedure.

Status exposes the scale beside the existing acceleration-conversion flag under
`velocity_estimator_shadow` (also used for the embedded estimator configuration).
This historical field name does not mean the conversion is shadow-only.

Deployment status: installed on the Pi on October 11 with runtime stopped. Nine
feature tests passed against the installed package. Managed inactive startup
confirmed reconstruction enabled and scale 0.980665 through runtime status.
Sensors became valid, native policy stayed inactive, and hardware stayed inactive.
Managed stop succeeded afterward. Physical mounting and motion checks remain pending.
Original files remain in
`/home/mikael/Development/imu-feature-correction.0omjMp/original` for rollback.
The selected model, sensor configuration and safety limits are unchanged.

Live feature verification: 650 filtered IMU samples produced mean corrected
acceleration `[0.092518, -0.045743, 9.402711]` m/s². Independently scaled raw mean
was `[0.091363, -0.036274, 9.383608]`. This confirms approximate consistency, not
calibrated accuracy or physical mounting. The selected env explicitly specifies
training IMU gravity bias `[0, 0, 9.81]`. No inference or motor commands were used.
