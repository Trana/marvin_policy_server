# Native policy I/O integration

For the current supervisor output contract, see [native_gateway_output.md](native_gateway_output.md).
The historical direct-output results below do not describe gateway publication rates.

Status: authorized simulation and isolated-Pi integration complete on October 11.
Not deployed or enabled on the physical robot. Physical acceptance remains separate.

Subsequent USB-C-only preparation adds the explicit `marvin-v1` profile and a
separate Pi overlay. See [native_robot_rollout.md](native_robot_rollout.md).
The default test profile still rejects physical domains. The entries below
describe the earlier integration goal and its original restrictions.

## Authorized scope

Implement an opt-in C++ ROS I/O path with Python inference and retain the current
Python runtime. Validate in Isaac Sim and isolated Pi tests. The Pi is available
on USB-C power. This does not authorize physical motion, robot service changes,
or replacement of the deployed robot runtime. Do not commit or push.

## Completion checks

- [x] ROS-independent Python computation with recorded-input equivalence tests.
- [x] Versioned, bounded local protocol with generation and sequence validation.
- [x] Native sensor reception and command publication with explicit ownership.
- [x] Management APIs and observation/override status remain available.
- [x] Activation/deactivation ramps, limits and output locks retain their meaning.
- [x] Operator permission loss zeros requested velocity, not balance inference.
- [x] Stop, worker death, expiry and stale input tests reject old results.
- [x] Simulation pause, resume and clock rewind tests.
- [x] Metrics and bounded tracing identify the native and Python stages.
- [x] Opt-in launch path with an unchanged default Python fallback.
- [x] Isaac startup, policy/IK transitions, reset and shutdown validation.
- [x] Same-model comparison of behavior and timing against the Python path.
- [x] Isolated Humble/Pi build and performance validation without hardware output.
- [x] Wiki, architecture and final results updated.

The progress entries below are chronological. Earlier pending items are superseded
by the final acceptance record at the end of this document.

## Decisions

1. Retain the diagnostic as a benchmark. Do not remove its private-domain guards
   to turn it into a runtime.
2. Preserve the model, observation order, histories, overrides and action mapping.
   Performance work must not silently change policy semantics.
3. Separate movement permission from inference validity. The existing
   `OperatorPermission` suppresses requested velocity during input loss. It does
   not disable balance inference. The prototype's boolean output gate is not a
   suitable replacement for that contract.
4. Use monotonic time for process/transport expiry and ROS time for simulation
   scheduling and ramps. Invalidate pending results across clock discontinuities.
5. Keep one inference outstanding. Never queue obsolete sensor snapshots.
6. Distinguish a requested policy deactivation ramp from immediate transport
   failure. A reply from an earlier generation cannot restore output ownership.
7. Preserve private isolated Pi tests. Simulator integration must use an explicit
   simulation profile and must not route operator input to the physical robot.

## Progress

### 2026-10-11 — contract audit

Created the persistent Goal. Inspected the existing node, diagnostic, activation
manager, inference schedule and operator permission implementation. Existing
worktrees contain substantial earlier work and will be preserved. The prototype
omits management processing and differs in permission and ramp behavior. These
gaps must be resolved before the simulator can use the new path.

### Numeric and transport foundations

- The observation and estimator-feature builders now import ROS message types
  only for static type checking. Runtime calculations work without ROS imports.
- `SensorSnapshot` provides immutable, validated numeric input. Replay tests
  compare reordered ROS messages and snapshots, including histories and locks.
- `runner_inputs` supplies the same actor/estimator arguments to both existing
  execution paths. All 200 functional tests pass locally and in isolated Humble
  staging on the Pi. The local package build passes.
- `native_protocol.py` and `native_io/protocol.hpp` define a bounded v1 envelope.
  Python/C++ byte agreement, malformed inputs and socket truncation are tested.
- `native_io/lifecycle.hpp` tracks one request, epochs, expiry and clock changes.
  Permission changes invalidate velocity-conditioned results without disabling
  balance inference. These are unit-tested contracts, not connected ROS behavior.
- Native output mapping matches Python linear/smooth mapping and clipping in 240
  randomized cases. Native ramp tests cover activation, deactivation and Stop.
- Three local CTest programs pass. Pi staging is
  `/tmp/native-policy-integration.xbhKqJ`; it is not a deployed overlay.

The following checkpoints describe subsequent integration. Complete artifact-level
equivalence and Isaac tests before considering physical deployment.

### Management boundary checkpoint

- `NativeChannel` has a bounded reader queue and an independent acknowledgement
  condition. A Stop service can wait for C++ acknowledgement without requiring
  its own executor to process that acknowledgement. Transport errors close the
  channel and wake the executor. Eight unit tests cover this contract.
- The existing node accepts an internal `native_io=True` construction option.
  This omits its joint/IMU subscribers, high-rate timer and command publishers.
  It retains management services and the shared numerical state. Normal launch
  does not select this option. Native plus async inference is rejected explicitly.
- All 209 functional tests pass on the workstation and in isolated Pi staging.
  Three native CTest programs also pass on both architectures.
- A private-domain constructor/service check loaded the exact October 3 robot
  artifact locally and on the Pi. It confirmed no high-rate Python endpoints and a responding
  observation service. This is not an integrated native runtime test.

Next implementation constraints:

1. A dedicated native entrypoint must use the constructor option. Keep normal
   Python launch unchanged until that adapter has passing integration tests.
2. Wake the Python executor with a guard condition for snapshots. Do not add a
   fast Python polling timer that recreates the original dispatch overhead.
3. Keep ROS service callbacks and numerical state changes on the executor thread.
   Only the IPC reader and acknowledgement condition run on the reader thread.
4. Have the native owner acknowledge ownership changes before reporting success.
   Sequence/generation checks must reject queued results after Stop or overrides.
5. Use the native ramp/mapping at publication rate. Keep Python status and action
   history consistent with the actual accepted output, including output locks.
6. Preserve controller permission semantics, clock resets, metrics and bounded
   trace IDs. The integration must distinguish fault states from normal inactivity.

### Connected runtime and packaging checkpoint

- `native_server.py` retains Python management and model inference. Its bounded
  socket reader wakes the ROS executor for native snapshots and output commits.
- `native_io/main.cpp` owns sensor reception, 50 Hz inference requests and
  phase-aligned 200 Hz command publication. Python does not publish commands.
- Native output commits update Python action history only after publication.
  No next inference request starts before that commit. Generations reject old
  results after Stop, overrides, permission changes and clock discontinuities.
- Python and C++ exchange control acknowledgements through the inherited socket.
  Service success requires native acknowledgement. Transport or inference faults
  stop output and require explicit reactivation. Movement-permission loss instead
  zeros requested velocity while preserving balance inference.
- Packaging uses `ament_cmake` plus `ament_cmake_python` to install both processes.
  The existing Python executable and launch arguments remain available.
  `io_backend:=native` or `MARVIN_POLICY_IO_BACKEND=native` selects the new path.
  The default remains `python`. Native mode disables the separate async worker.
  `native_io_executable` / `MARVIN_POLICY_NATIVE_IO_EXECUTABLE` can select a staged
  helper. The default helper resides in the package's installed `lib` directory.
- Both processes reject physical domains. They require localhost-only domain 184
  with private `/policy_probe/` topics, or domain 10 with `/sim` and simulation time.
  These guards remain until a separate physical-deployment review.

Validation on October 11:

- Temporary package build and all four CTest groups passed: 215 Python tests plus
  native protocol, lifecycle and output tests. Nine additional adapter tests passed.
- `native_io/check_runtime.py` passed locally and on the isolated Humble Pi with
  the October 3 robot artifact. It checks management calls, activation, limits,
  permission loss, stale-sensor recovery, deactivation, expired inference, old
  replies, explicit reactivation, bounded tracing and native-process exit.
- The local simulated-clock variant also passed pause, resume and rewind checks.
- Pi steady windows included 50 accepted results / 200 published commands per
  second. This test includes a Python observer and intentional failure injection.
  Its other windows are not steady-state performance measurements. Reported
  maximum gaps include deliberate output stops and inactive intervals.
- Local evidence: `/tmp/native-clock-validation.log` and
  `/tmp/native-pi-validation.log`. Temporary install:
  `/tmp/marvin-native-package-install`. Pi staging remains
  `/tmp/native-policy-integration.xbhKqJ`. None replaces the deployed runtime.

Remaining: full same-model replay, dynamic joint-state integration, output-lock
checks, Isaac behavior/lifecycle comparison, minimally instrumented Pi performance,
and final architecture/results documentation. Completion boxes remain open until
the combined acceptance checks cover each contract.

### Dynamic inputs and Isaac comparison

- Added `--dynamic` to the isolated runtime check. It sends reversed joint names
  and reversed interface order, including encoder receipt timestamps. The check
  verifies name-based decoding and receipt propagation. It also applies an output
  lock, checks actual published values and confirms release restores policy output.
- These checks passed locally and on the Pi. The first Pi attempt faulted near
  the override check. The observer performed hundreds of NumPy assertions without
  servicing the policy executor. Batched validation removed that avoidable pause.
  The repeat passed without increasing the 100 ms inference deadline. Evidence:
  `/tmp/native-pi-dynamic-validation-2.log`. Longer performance checks remain needed.
- Restarted the stale API-owned Isaac instance. Its launcher remained present,
  but its child had exited and `/clock` was absent. The simulation runtime was
  stopped through the simulation control plane before test launches.
- Both backends passed the same bounded Isaac sequence with the October 3 model:
  Stand, policy standing, forward, neutral, return to IK, Rest, Stand again, Rest.
  Both used domain 10 and verified `topic_based_ros2_control/TopicBasedSystem`.
  Workstation operator selection was disconnected. No physical domain was used.
- Python used the previously accepted async/compact/lean configuration. Native
  used compact/lean and selected its own scheduler. Both used the same temporary
  package install, model, environment, safety limits and reset backlash scene.

| Forward phase | Python async | Native I/O |
| --- | ---: | ---: |
| Wall duration | 8 s | 8 s |
| Simulated duration | 6.840 s | 6.740 s |
| Horizontal displacement | 1.669 m | 1.630 m |
| Commands per simulated second | 199.85 | 199.85 |
| End-of-phase body-up Z | 0.99985 | 0.99966 |

This is functional equivalence evidence, not a deterministic replay or a gait
quality guarantee. Different physics runs need not produce identical trajectories.
Evidence: `/tmp/native-sim-{python,native}-{launch,check}.log`. The temporary
simulation launches and API-owned Isaac process were stopped after final Rest.
No simulation environment defaults or robot services changed.

Next: recorded-input artifact replay, live simulator-reset behavior, minimally
instrumented paired Pi measurements, and final architecture/documentation review.

### Artifact replay and separate-observer measurements

`native_io/check_replay.py` replays 256 deterministic synthetic input frames through
ROS-message and serialized-snapshot branches using the exact robot artifact.
Observations, estimator features, estimator histories, observation histories,
estimated states and actions match exactly within each machine. This passed on
Jazzy and Humble/Pi. The replay includes changed joint ordering, command overrides,
estimator overrides and a state reset. It is not a recorded physical trajectory.
Architecture-dependent floating-point input generation produced different stream
hashes across hosts, so this does not assert bitwise cross-architecture equality.

The replay also passes its artifact actions through native linear and smooth
mapping, including ramps and limits. All 512 mapped outputs match Python within
`1e-13`. Maximum absolute error was `1.11e-16` locally and `2.22e-16` on the Pi.
Evidence: `/tmp/native-artifact-replay.log` and `/tmp/native-pi-artifact-replay.log`.

`native_io/check_performance.py` runs policy, synthetic sensors and observer in
separate processes on private localhost domain 184. It measures only the policy
process tree's CPU consumption. Sensor generation and observation CPU are excluded
equally for both backends. Detailed trace events are disabled. One-second metrics
remain enabled. Each measurement follows a five-second active warmup and uses
200 Hz dynamic joint/IMU input. The script verifies deactivation stops output.

First 20-second Pi pair:

| Measurement | Python async | Native I/O |
| --- | ---: | ---: |
| Observed command rate | 188.49 Hz | 199.84 Hz |
| Inference rate | 49.80 Hz | 50.00 Hz |
| Policy-tree CPU, one core = 100% | 164.09% | 108.10% |
| Mean inference duration | 11.69 ms | 9.56 ms |
| Maximum inference duration | 22.78 ms | 13.26 ms |
| Observed command gap p95 | 7.73 ms | 5.89 ms |
| Maximum observed command gap | 31.09 ms | 14.86 ms |

The Pi reported `throttled=0x0`, approximately 42°C and 1.8 GHz between runs.
These are isolated USB-C tests, not loaded physical-runtime measurements. Evidence:
`/tmp/native-pi-perf-{python,native}-1.log`.
The first local performance attempt requested activation before sensor discovery
completed. The harness now waits for native sensor readiness before activation.

The reverse-order repeat also passed (`/tmp/native-pi-perf-pair-2.log`). Native
measured 194.64 Hz commands, 50.00 Hz inference, 110.45% CPU and 9.91 ms mean
inference. Python measured 191.04 Hz commands, 49.95 Hz inference, 163.69% CPU and
11.40 ms mean inference. Native therefore reduced measured policy-tree CPU by
approximately 33–34% across these pairs. Command rate varied between runs and did
not remain exactly 200 Hz. No inference deadline faults occurred in these windows.
Do not interpret these short checks as a real-time guarantee. Live simulator-reset
and parent-process-death checks remain before the final completion audit.

## Final acceptance record — October 11

| Requirement | Inspected evidence |
| --- | --- |
| Numeric and recorded-input equivalence | `test_sensor_snapshot.py`, `check_replay.py`; 256 synthetic and 256 recorded Isaac frames per host. Exact branch observations, features, histories, estimates and actions. |
| Native mapping and ramps | `action_output_test.cpp`, `test_native_action_output.py`; 240 randomized cases plus 512 cases per artifact replay, tolerance `1e-13`. |
| Protocol, bounded queues and ownership | `test_native_protocol.py`, `test_native_channel.py`, `protocol_test.cpp`, `lifecycle_test.cpp`, `test_native_server.py`; malformed packets, queue overflow, ACK expiry, epochs and duplicate commits. |
| Services, permissions and output locks | `check_runtime.py`; observation RPC, activation ACK, velocity-zero permission semantics, actual fixed output, release and continued balance output. Existing typed-control unit tests also pass. |
| Stale inputs and inference failure | `check_runtime.py`; stopped sensor feed, delayed inference, explicit reactivation, native exit and old-result rejection after deactivation. |
| Python process loss | `check_performance.py --check-parent-death`; kill only the test Python parent while output is active. C++ stops output and exits independently. |
| Simulation clock behavior | Combined dynamic/sim-clock check from the installed Pi package; pause, resume and rewind pass. |
| Isaac lifecycle and reset | Paired Python/native walking check, then `check_sim_reset.py`; stop/restart actual Isaac while policy is active, observe output stop and clock rewind, explicitly authorize Stand/policy again, finish in inactive Rest. |
| Trace and metric continuity | Connected trace IDs span observation, inference and native publication. Capture expires while output continues. Separate native/Python metrics remain available. |
| Default fallback and packaging | Launch-selection tests, live Python baseline and native opt-in. Jazzy colcon build and Humble CMake build/install pass. Python remains the launch default. |
| Pi performance | Two reverse-order, 20-second pairs after warmup. About 50 Hz inference on both, native 194.64–199.84 Hz commands and 33–34% lower measured policy-tree CPU. |
| Documentation and scope | This record, wiki control/trace page, simulation workflow, app overview and Canvas. No physical runtime launch, service change, deployment replacement, commit or push. |

Final test totals: **224 Python functional tests and three C++ CTest programs pass
on both Jazzy and Humble/Pi**, with no skips in the Python functional group. Legacy
copyright/style test files are excluded from this functional group, not claimed as
passing. C++ builds use `-Wall -Wextra -Wpedantic -Werror`.

The final Pi integration check imports the temporary installed Python package and
executes its installed C++ binary. Evidence: `/tmp/native-pi-installed-check-3.log`.
It passes dynamic input, encoder timestamps, output locks, permission loss, clock
pause/resume/rewind, stale input, delayed inference, old replies, traces and exit.
The test harness warms NumPy testing helpers before activation. Their first lazy
import took 95.3 ms on the Pi versus 0.56 ms afterward, enough to affect a 100 ms
deadline when done inside the policy executor. The runtime deadline was not relaxed.

Recorded-input evidence: `/tmp/native-isaac-input-replay.json`, SHA-256
`1deff0043eb57ed21d2bc45c283936592de0097e5f72ca3b7fe4a16b41a5c91b`.
Both hosts replayed the same packet-stream hash
`154dfc1a1e307b22f5b0284f7fd298134a92a7d5fc5f0139833f97d028cfd624`.
Logs: `/tmp/native-recorded-replay.log` and `/tmp/native-pi-recorded-replay.log`.
Maximum mapping difference was `2.22e-16`. These frames came from passive Isaac
sensors, not physical robot motion. The earlier synthetic replay remains useful
for deliberately varied inputs and overrides.

Other final evidence: `/tmp/native-parent-death.log`, `/tmp/native-sim-reset-check.log`,
`/tmp/native-sim-reset-launch.log`, and `/tmp/native-pi-package-build.log`.
Actual Isaac restart advanced native generation from 4 to 11. Before/after policy
body-up Z was 0.99959 / 0.99948. Final Rest confirmed inactive simulated hardware.
Temporary test processes and API-owned Isaac were stopped afterward.

### Build and selection

The package requires `ament_cmake`, `ament_cmake_python` and `rclcpp` in addition
to its existing ROS/Python dependencies. No new inference framework is required.
For an isolated workstation build from `robotdog_ws`:

```bash
colcon build --packages-select marvin_policy_server \
  --build-base /tmp/marvin-native-package-build \
  --install-base /tmp/marvin-native-package-install \
  --cmake-args -DCMAKE_BUILD_TYPE=Release \
  -DPOLICY_TEST_PYTHON=/home/trana/venv_ros/bin/python
```

Source the temporary install after the existing workspace. For simulation bringup,
set `MARVIN_POLICY_IO_BACKEND=native` only in that launch environment. The direct
policy launch also accepts `io_backend:=native`. Keep the selected policy/env pair,
`ROS_DOMAIN_ID=10`, `ROS_LOCALHOST_ONLY=1`, `/sim` namespace and simulation time.
The exact tested model is recorded above. Selecting `python` restores the existing
backend. Native faults never cause an automatic fallback or reactivation.

The Pi package prefix is `/tmp/native-policy-integration.xbhKqJ/package-install`.
Its direct CMake install has a package-level setup script at
`share/marvin_policy_server/local_setup.bash`, not a colcon workspace-level script.
The deployed robot overlay remains unchanged.

### Limitations and next authorization

- At completion of this integration goal, physical domains remained rejected.
  Subsequent explicit-profile preparation is recorded in `native_robot_rollout.md`.
  Physical deployment and acceptance remain separate supervised work.
- These are bounded tests. They do not prove long-duration reliability, a fixed
  200 Hz deadline, loaded CAN behavior, physical gait quality or universal model support.
- The native profile currently requires twelve joints, finite enabled limits and
  a 5 ms publish period. The validated artifact uses raw action history.
- Maximum gap metrics include inactive and deliberately stopped periods. Use
  separate-observer steady windows for performance comparisons.
- Temporary `/tmp` builds and raw logs are disposable. The scripts, measured
  results and architecture decisions are retained in the repository and wiki.


## Native policy status and release recovery (2026-10-28)

A physical default-pose request failed before activation because the inactive
policy transport closed during startup. Five STATUS packets exceeded the former
four-packet queue. The supervisor could not confirm release and reported
"Handover deadline expired". Motor hardware remained inactive.

The native policy channel now retains only the newest pending STATUS snapshot.
It preserves the receive order of inference SNAPSHOT and publication COMMIT
packets, with a separate four-packet work bound. Control ACKs retain their
independent delivery path. Status remains 20 Hz.

The 300 ms parent heartbeat deadline applies while native output is active or
ramping down. Inactive startup continues to report status without a heartbeat
fault. Configuration timeout, socket EOF handling, inference expiry and explicit
reactivation requirements remain enforced.

Python can report output released after confirming that its sole native command
publisher process exited. A socket error or an unresponsive live child does not
prove release. A dead publisher still requires runtime restart before policy
activation; this does not automatically activate any other executor.

Workstation checks: 88 focused Python tests, three C++ suites and two native
process tests passed. The latter use synthetic sensors in localhost ROS domain
184; no physical hardware. The isolated Pi build passed, with 39 Python/native-process tests and three
C++ suites. The robot runtime service now selects this overlay on its next start.
Physical default-pose validation remains pending. Staging: /home/mikael/Development/policy-status-20261028.
Sources: marvin_policy_server/native_channel.py, native_server.py,
native_io/main.cpp, test/test_native_transport_diagnostics.py.
