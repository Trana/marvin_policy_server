# Native policy I/O: physical rollout preparation

Latest checkpoint, October 11: physical stand testing reached about 48.5 Hz
inference and 197 Hz commands, but an earlier activation hit the 100 ms deadline.
Returning to IK also produced calculation errors. Managed Stop succeeded. Physical
acceptance has not passed. The sections below retain the preparation procedure.

## Deadline diagnostics

`NATIVE_DEADLINE_FAULT` is emitted once per native transition into expiry. It
records the branch (`pending`, `accepted`, or `arriving_result`), the pre-fault
generation, pending/accepted sequence numbers and both clock ages, the configured
limit, and the previous C++ tick gap. Missing requests are `null`. Ages are measured
from request creation, not model completion. The snapshot survives ownership
invalidation and is cleared by explicit activation or stop.

`NATIVE_PYTHON_FAULT` records the last processed request's sequence/generation,
request-to-callback age, total callback time and request-to-result-send age.
The callback includes observation work and publication bookkeeping, not just
model inference. A null duration means that stage did not finish. Match sequence
and generation before relating it to the native record: it may describe an older
request if Python did not receive the expired request. No sensor/action arrays
are logged. Diagnostics do not alter the deadline, scheduling order, or restart
rules. Recorders should start before activation.

Validated on workstation and Pi: 264 Python tests and three C++ tests passed.
Private-domain delayed-inference injection produced both diagnostic records and
passed explicit recovery. Pi staging is `native-deadline-diagnostics.UJttHi`.
The physical service still selects `native-policy-rollout.Ey4Nh0`; staging alone
does not enable the new diagnostics in physical tests.

## Safety and scope

The USB-C preparation does not start the physical runtime or activate motors.
It builds an independent overlay, checks private ROS domain 184, and prepares a
systemd drop-in outside `/etc`. The installed Python runtime remains unchanged.

The new code supports the explicit `marvin-v1` profile. This is configuration
validation, not physical acceptance. Do not start the physical native runtime
until the robot is supported and the operator is nearby.

## Deployment contract

Python remains the default backend. Native selection requires both variables:

```text
MARVIN_POLICY_IO_BACKEND=native
MARVIN_POLICY_NATIVE_PROFILE=marvin-v1
```

Both Python and C++ independently check these physical profile requirements:

- `ROS_DOMAIN_ID=0` and `ROS_LOCALHOST_ONLY=1`.
- Root namespace `/`, wall time, and required operator permission.
- Exact physical controller order: FL, RL, FR, RR hips, then thighs, then calves.
- Fixed input topics `/joint_states`, `/imu`, and `/operator/motion_permitted`.
- Fixed command topics `/marvin_joint_controller/commands` and `commands_traced`
  under the same controller namespace.
- Traced command publication only, matching the deployed controller.
- Fixed `/trace/events` and `/trace/metrics` topics.
- No dynamic joint input in this initial physical profile.
- 5 ms publication period and 100 ms inference expiry.

Python also checks the robot-local `/joy` topic and enabled, loaded finite limits.
C++ checks finite defaults, limits and action mapping before creating command
publishers. The profile checks topic names after ROS remapping. A private-domain
profile cannot select physical topic names.

The test profile remains the default. It accepts only simulation domain 10 with
`/sim` and simulation time, or private domain 184 with `/policy_probe/` topics.
It cannot authorize physical output. No profile automatically activates policy.
Faults do not switch to Python or reactivate policy.

The profile is a configuration guard, not a security boundary or motor-stop device.
It does not identify a physical host cryptographically. The runtime manager still
owns startup, the inference reservation and the selected policy bundle. The
controller and ODrive watchdog contracts remain unchanged.

## Prepared Pi overlay

October 11 staging directory:

```text
/home/mikael/Development/native-policy-rollout.Ey4Nh0
```

This overlay contains only the new policy package. Its colcon setup chains the
existing `policy-execution-rollout.f2Lx2z` underlay. It preserves the deployed
bringup, runtime runner, policy selection, inference reservation and CAN lock.
The renderer creates `review/zzzzz-native-policy.conf` in this directory. It
does not copy the file to `/etc`, run systemctl, or start any process.

To prepare another review file after building a new overlay:

```bash
source /home/mikael/Development/native-policy-rollout.Ey4Nh0/install/setup.bash
/home/mikael/Development/robotdog_ws/.venv-policy/bin/python \
  -m marvin_policy_server.native_rollout \
  --setup /home/mikael/Development/native-policy-rollout.Ey4Nh0/install/setup.bash \
  --runner /home/mikael/Development/policy-selection-rollout.RxGxmD/src/robot_control_ros2/scripts/run-motion-runtime \
  --output-directory /home/mikael/Development/native-policy-rollout.Ey4Nh0/review
```

The renderer refuses to overwrite an existing file. It rejects temporary,
relative, substituted or external paths. It is a preparation tool, not an installer.

## USB-C checks

`native_io/check_startup.py` runs two native lifecycle cycles and one Python
fallback cycle. It uses a private domain and no sensor publisher. It checks
the loaded policy/env hashes, management responses, native activation rejection,
zero output commands, graceful shutdown and native child cleanup.

`test/test_native_profile.py` passes the same accepted/rejected configuration
matrix to Python and C++. Its C++ checker does not initialize ROS or open IPC.
`check_runtime.py` additionally checks synthetic input, sensor loss, permission,
overrides, expiry and clock resets. These checks do not prove physical behavior.

## Later installation and rollback

Do not perform this procedure during unattended USB-C preparation.

1. Check robot support and operator presence before physical tests.
2. Stop the runtime through the existing control plane.
3. Prevent new Start requests during the configuration change.
4. Check that `robot-runtime.service` is inactive and `MainPID=0`.
5. Save the effective unit configuration and existing drop-in inventory.
6. Review the prepared drop-in and overlay package paths.
7. Install only `zzzzz-native-policy.conf` into the runtime service's drop-in directory.
8. Reload systemd configuration.
9. Check the effective command and environment before any Start request.
10. Restore runtime management, without starting the motion runtime automatically.

Rollback uses the same inactive-runtime and closed-admission conditions. Move
only `zzzzz-native-policy.conf` out of the active drop-in directory. Reload
systemd. The preceding `zzzz-policy-execution.conf` restores the old Python
overlay and its async/compact/lean settings. Do not remove either overlay until
acceptance is complete. Do not use automatic fallback while policy is active.

## Remaining physical acceptance

- Inactive startup, model identity, service readiness and shutdown.
- Supported IK → policy → IK transitions, Y/Stop and controller input loss.
- Loaded CPU, inference, command rates, IMU age and joint feedback age.
- Supervised walking only after the supported checks pass.

Private tests cannot prove motor response, loaded CAN behavior, watchdog response,
or gait quality. Those checks require motor power and the operator.

## October 11 USB-C evidence

- Jazzy and Humble/Pi builds pass with C++ warnings treated as errors.
- All 263 Python functional tests and three C++ CTest programs pass on both hosts.
  The Python group includes the cross-language admission matrix. Existing style
  tests are excluded from this functional count.
- Repeated private startup/Stop/shutdown checks pass on both hosts. They load
  the October 3 embedded-estimator policy and compare loaded artifact hashes.
- The isolated Pi dynamic-input and simulated-clock check passes sensor loss,
  permission loss, output locks, old replies, expiry, explicit reactivation,
  clock pause/rewind, tracing and native process exit.
- The selected bundle's joint order matches the physical controller profile.
- The staged package prefix resolves to the new overlay. Bringup still resolves
  to the existing `policy-selection-rollout.RxGxmD` overlay.

The workstation and Pi policy/env bytes match. Metadata hashes differ because
the transfer changes `combined_policy.pt` to `policy.pt` and adds the matching
hash alias. A sorted JSON comparison found no other metadata difference. Each
startup check compares the loaded metadata against that host's actual file.

A separate-process observer measured this final 60-second synthetic-input window:

| Measurement | Native I/O |
| --- | ---: |
| Command rate | 199.91 Hz |
| Inference rate | 50.00 Hz |
| Policy process-tree CPU, one core = 100% | 109.18% |
| Mean / maximum inference | 9.90 / 19.67 ms |
| Command gap p95 / maximum | 5.86 / 14.16 ms |

Detailed tracing was off. One-second metrics remained enabled. Synthetic joint
and IMU input ran at 200 Hz. The observer and sensor process CPU are excluded.
No inference expiry fault occurred during this window. Output stopped after
deactivation. Killing only the Python parent stopped output and the C++ helper.
The Pi reported `throttled=0x0` and 43.8°C during the check.

This longer window supports the earlier timing results. It is not a physical
runtime measurement or a hard deadline guarantee. One Python callback exceeded
20 ms, while the separate C++ command path continued.

Evidence remains in the Pi overlay's `performance-60s.log`, `startup-check/` and
`startup-check-final/`. Workstation result summaries are in
`/tmp/native-usb-performance-60s-result.log` and
`/tmp/native-usb-startup-pi-result.log`. These logs are disposable. This document
retains the results and the versioned harnesses reproduce the checks.
