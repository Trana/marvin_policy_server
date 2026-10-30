# Robot-local joint response measurement

Status: manual read-only recorder and offline PACE export tested. Physical
excitation is not enabled by these tools.

## Manual recording with the existing IK controller

This is the first measurement path. It does not need the experimental controller.
Use `--target robot --topics-only` to record existing commanded positions and
encoder feedback. This option makes no status service requests. It does not start
the runtime, send commands, or stop motors when recording ends.

The scripts are staged on the Pi at
`/home/mikael/Development/recording-tools-fixed.6kFBvW`. No production source or service
was replaced. Source versions remain in this repository's `scripts/` directory.

Before a physical test, support the robot and check independent stop behavior.
Use the existing runtime and IK controls. Keep policy control disabled. Do not
start motion until the recorder log reports subscriptions to both command and
feedback topics. Missing publishers do not prevent a bag from being finalized.

Prepare `session.json` with these fields:

```json
{
  "joint_names": ["REPLACE_WITH_ALL_JOINTS_IN_COMMAND_ARRAY_ORDER"],
  "position_units": "rad",
  "support": "stand, feet clear of the floor",
  "control_mode": "IK",
  "imu_stabilization": "record the actual setting",
  "model_identity": "record the actual model/configuration revision",
  "notes": "record initial pose, payload and test description"
}
```

Do not guess the joint order. Take it from the active controller configuration.
Supply that configuration and an ODrive settings snapshot as artifacts. Include
control/input modes, position and velocity gains, filter bandwidth, limits,
encoder feedback rate, gear ratios, direction signs and joint zero offsets.
The recorder does not query ODrive settings automatically or verify applied
settings. Unknown settings must remain explicitly unknown, not invented defaults.
Avoid supplying files that contain credentials. Artifacts are copied in full.

On the Pi, source the ROS workspace and select localhost domain 0:

```bash
source /home/mikael/Development/robotdog_ws/install/setup.bash
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=1
python3 /home/mikael/Development/recording-tools-fixed.6kFBvW/capture_reference_run.py \
  --target robot --topics-only --seconds 60 \
  --session-config /absolute/path/session.json \
  --artifact /absolute/path/controller-config.yaml \
  --artifact /absolute/path/odrive-settings.json \
  --output /home/mikael/Development/measurements/ik-stand-001
```

Replace example input paths with real files. Select a new output directory for
each run. Start with a stationary baseline, then small controlled movements at
different speeds with direction reversals. Keep startup/shutdown outside the
selected fitting window. Record another independent trial for validation.

The recorder stops after the selected duration (1–300 seconds). For a clean early
stop, run the following on the capture host:

```bash
python3 /home/mikael/Development/recording-tools-fixed.6kFBvW/capture_control.py /home/mikael/Development/measurements/ik-stand-001 --stop
```

Omit `--stop` to read status. This reads only `progress.json`, never the database.
The recorder updates status once per second. Check its timestamp for freshness.
Subscriptions and file size do not prove message arrival. A stop request is not
confirmation; wait for `complete.json` and a finalized state. The completion
receipt records `operator_request` for a clean early stop. It does not stop motors.

Never query or analyze a live bag database, even with SQLite `mode=ro`. Read locks
can block its writer. Analyze only after the recorder exits and files are closed.
Start recording while motors are disabled. Let discovery settle before enabling
IK. Recording adds load and this startup order does not guarantee fault-free motion.

Interrupted or
failed captures do not receive `complete.json` and return an error. A completion
receipt means recording finalized, not that all required data arrived.
Do not change files inside a completed capture. Store later notes separately.

From the workstation, fetch the whole directory into a new local destination:

```bash
scp -r marvin-robot:/home/mikael/Development/measurements/ik-stand-001 ./ik-stand-001
```

## Offline PACE export

Keep the raw bag. In a sourced ROS environment, run
`scripts/export_pace_capture.py --help`. Supply the capture directory, a new
output directory, exact joint names, trace source ID, clean source-clock start
and end timestamps, maximum accepted gap, and PACE environment step interval.
Use `--confirm-common-clock` only after checking that both streams share a clock.
Select the IK producer from the recorded trace IDs; do not assume the policy ID.

The exporter checks checksums, the complete file inventory, session metadata,
joint mapping, finite values, clock ordering, time coverage and stream gaps.
It writes `data.json` and `report.json`. It holds commands constant between
samples and linearly interpolates feedback. Resampling does not create new
sensor information. Feedback timestamps can describe publication rather than
encoder acquisition. Commands are requested targets, not verified CAN writes.

In the PACE/PyTorch environment, convert the exported data:

```bash
python scripts/pace_json_to_tensor.py /absolute/path/export/data.json /absolute/path/export/data.pt
```

Keep `report.json` beside the tensor file. The tensor keys are `time`,
`des_dof_pos` and `dof_pos`, with positions ordered as specified in the report.
The sample interval must match the PACE environment step interval. Upstream
[fit.py](https://github.com/leggedrobotics/pace-sim2real/blob/main/scripts/pace/fit.py)
advances one environment step per data row. The exporter does not configure a
Marvin PACE task or run optimization. Match stand support, model, initial pose,
command semantics and joint order before fitting. Fitted values are candidates,
not automatically approved ODrive settings.

Verification: 58 offline checks passed (one unrelated ROS-dependent check skipped),
two private-domain ROS capture tests passed, and four PyTorch conversion tests
passed. The synthetic end-to-end test covers topics-only capture and PACE JSON
export. A passive Humble recording on the Pi finalized and transferred with all
six checksums verified. Runtime remained inactive. This was not a physical
response trial. Evidence: `marvin-measurements/recorder-verification.5ReQu8`.

Later physical attempt: `marvin-measurements/ik-stand.M6w8Jl/ik-stand-001`
failed with SQLite error 5 (`database is locked`) during a live database query.
The file passes offline integrity checking, but has 1,220 joint-state messages
and zero traced commands. It is not usable for fitting. The earlier short capture
also coincided with a CAN `batch_deadline` fault. Causation is not proven.
The replacement status/stop path passed 60 offline tests (one skipped), three
isolated ROS tests, and a clean early-stop/receipt check on the Pi in domain 231.
No physical retry occurred during that verification.

## Future direct-controller recording

Use the same finite trajectory recipe as the isolated simulation test. Record
on the robot. Transfer the completed capture directory to the workstation for
comparison. Do not stream full-rate data over Zenoh for the first measurement.

## First trial

Use a supported robot, one joint and a stationary baseline. The simulation
reference uses FL thigh, one second of inherited hold, and a four-second
windowed sine with one cycle. Its amplitude parameter is 0.005 rad. A windowed
sine does not reach the full amplitude parameter at every peak.

These are reference settings, not approved physical limits. Review the physical
pose, clearance, direction, tracking error, velocity, acceleration and ODrive
settings before selecting the robot profile. Do not copy the simulation hip
tracking tolerance to the robot. Ground tests and simultaneous starts are deferred.

## Required measurement sources

- Test-controller target values after a successful interface write, with joint
  names, controller identity, activation/session, update sequence and sample time.
- `/joint_states` and `/dynamic_joint_states`, including encoder receipt timing
  when the deployed hardware and broadcaster expose it.
- Typed test state, including completion/fault reason and applied sequence.
- Supplied profile, test plan, baseline pose and ODrive configuration snapshots.
  Hash these files with repeated `--artifact` arguments. A file hash does not
  prove that the runtime applied its contents.

The experimental controller currently writes interfaces directly. Its old IK
command subscription is not a record of these writes. A robot command-telemetry
publisher now exists behind `enable_joint_test_target_telemetry=false` by default.
When enabled, `<controller>/joint_test/applied_targets` carries JointState targets
after successful interface writes. It does not confirm CAN delivery or motor motion.
The publisher is nonblocking and best effort. Its stamp is controller-manager update
time. Confirm the clock domain before comparison. The companion
`<controller>/joint_test/target_samples` stream adds identity, activation, update
sequence and monotonic time. This typed stream uses a 256-sample fixed queue and
a separate reliable ROS publisher. Queue overflow and publication errors are
visible in controller status. The control loop never waits for that publisher.
The recorder requires continuous sequences and fresh timestamps despite buffering.
The recorder now includes that topic. Do not substitute the old
`/marvin_joint_controller/commands` or `commands_traced` topic.

## Prepared recorder

`scripts/capture_reference_run.py` now accepts `robot_joint_test`. It requires
localhost ROS domain 0, an artifact, an explicit controller namespace and an
explicit command-telemetry topic. It creates a new directory and refuses overwrite.
It records a SQLite ROS bag and samples the read-only typed state service at 5 Hz.
It never activates hardware, switches controllers, prepares a trajectory, or
renews a movement permit. Stopping the recorder does not stop motors.

The endpoint arguments are required because no physical test launch is approved
yet. They select subscriptions, not verified publisher identity. Missing topics
can produce an incomplete bag. Review `reference_capture_quality.py` after the
recorder stops. Do not interpret absent data as zero movement.

Each capture includes raw source timestamps, bag receipt timestamps, host identity,
artifact hashes and status request/response times. Movement challenges are omitted
from typed snapshots. Command/feedback time comparisons must use a known common
clock. Never subtract simulator clock time from robot wall time.

## Deployment gates

1. Build the controller and interfaces for Humble in an isolated overlay.
2. Test ownership, cancellation, lease expiry and feedback loss with mock hardware.
3. Implement and verify the command-telemetry source without adding blocking work
   to the controller update loop. Report sequence gaps instead of hiding them.
4. Review a robot-specific profile and receipt-age configuration.
5. Verify independent stop behavior and explicit recovery after a failed switch.
6. Confirm operator presence, mechanical support and power before enabling motors.
7. Start recording before motion. Check command and feedback arrival locally.
8. Run one explicitly approved test. Confirm completion and hold before transfer.

Do not enable physical start through the existing simulation gateway. Its domain,
target and profile restrictions remain unchanged. A later robot-local supervisor
must tie recording health to test admission/lease renewal. The read-only recorder
alone does not provide that safety relationship.

## Guarded chirp preparation, October 16

`record_joint_test_guarded.py` is a separate local JSONL recorder. It flushes and
fsyncs samples before publishing a healthy `health.json`. It records targets and
joint feedback with source and receipt timestamps. It rejects changed identity,
target sequence gaps and invalid values. Health expires when data or commits stop.
It does not send motion commands. Stopping it does not itself disable motors.
ROS callbacks now submit records to a bounded 1,024-record disk queue. A separate
writer performs flush/fsync. Receive history is depth 64, reliable for typed
targets and best effort for joint feedback. Overflow
is an explicit failure, not silent record eviction. Health includes synchronized
progress, disk-sync duration and queue depth. Only synchronized sample timestamps
can satisfy durability readiness. A recent fsync does not make old samples fresh.
Live target source/receipt and feedback receipt freshness remain bounded at 150 ms.
Saved-data progress has a separate two-second bound. Both checks must pass for START
or RENEW. Health exposes `live_ready`, `disk_ready`, and both limits. A fresh sync
cannot make old saved samples fresh. Power loss can still lose the unsynchronized
tail; buffered data is not reported as durable. Missing samples and writer errors
remain faults. Long disk stalls can still stop the test with no sample loss.
The storage allowance accommodates observed durable-progress delays near 0.7 s.
It is not a change to motor/CAN safety limits.
Use `probe_guarded_recorder.py --output NEW_DIRECTORY` only with localhost-only
domain 231 to validate without CAN or motor access. This does not validate the
experimental controller publisher or physical stop behaviour.

`run_robot_joint_chirp.py` is an explicit stand-test client. It requires operator
confirmation before motion. Renewal requests are scheduled every 100 ms from
the start of the preceding renewal checks, not 200 ms after their completion.
The controller lease remains 500 ms. Its expiry uses server-issued challenge time,
so service delay and challenge age consume the available margin. All existing
hardware, recorder, identity and applied-sequence checks remain in place.
The client keeps at most 4,096 service timing records in memory. It writes
`service-timings.json` after motor-stop cleanup. Records include endpoint, elapsed
time, action, sequence, accepted flag, phase and reason, but no challenge tokens.
These timings help distinguish delayed calls from controller-side renewal failures.
Recorder SIGINT/SIGTERM handlers only request loop exit. ROS stays active until
the disk worker drains and synchronizes queued samples. Final health includes
`running=false`, `finalized=true`, and the durable sample counts. A fault field
still marks an invalid recording, even if its files were finalized.
The parent waits eight seconds after SIGINT. On timeout, it requests termination,
then forces exit if necessary. Forced cleanup is reported as unconfirmed, not
success. Service timing export and ROS cleanup run even if recorder shutdown fails.

The client requires operator
confirmation, local domain 0, active IK hardware and a preloaded inactive test
controller. The initial candidate permits only FR_thigh_joint, amplitude 0.005 rad,
one-second inherited hold and a 20-second 0.2–0.8 Hz chirp. It checks actual
interface ownership after handover. Humble can return success for a conflicting
switch without actually activating the requested controller.

The client requires healthy recording before START and every RENEW. Renewals are
0.1 seconds apart under the 0.5-second lease. The bounded control ledger now holds
4096 ordinary operations, plus 128 reserved first-cancellation outcomes.
On exit, the client requests cancellation and hardware INACTIVE. It never enables
hardware itself or automatically restores IK. Ctrl+C keeps ROS available for stop
requests. An unconfirmed stop requires the physical disconnect.

Do not treat simulation or inactive startup as proof of physical stop behaviour.
First verify motor disable on the supported robot. Do not use joystick Y as the
only stop until its behaviour under test ownership is physically checked.

The guarded JSONL format uses `export_guarded_joint_fit.py` for PACE export.
The separate `export_pace_capture.py` still reads finalized ROS bags.
Preserve the raw JSONL and profile. Explicitly confirm a common ROS source clock
before export. Monotonic receipt timestamps are not source timestamps.

## Comparison of completed recordings

Keep the complete raw directory. Compare joint names, baseline-relative targets,
feedback excursion, following error, overshoot, settling and command-to-feedback
delay. Check record gaps and controller outcomes first. Separate constant loaded
offset from dynamic response. ODrive encoder feedback cannot measure downstream
belt backlash or foot position by itself.

Use local ROS bags for both targets when comparing timing. The existing browser
recording is useful evidence, but it is not equivalent to full-rate local capture.
# Preparing chirp connections before enabling IK

Use `run_robot_joint_chirp.py --wait-for-operator` with the existing explicit
profile, output path and operator-ready arguments. Hardware must be inactive.
The client creates its service connections and starts the recorder before it
prints `CONNECTIONS READY`. Enable IK only after this message. Send ENTER to
release the prepared client within 180 seconds. Do not restart the runtime.
The client never enables hardware. Timeout, EOF or changed controller identity
aborts and requests hardware INACTIVE. Existing recorder, tracking, feedback,
lease and CAN deadline checks remain in force. Keep the disconnect accessible.

## One-session recording for all joints

Use `--all-joints` with `robot_joint_test_fit_all.yaml`. This is an explicit
diagnostic profile, not a normal runtime default. Load it into the inactive test
controller before enabling IK. Keep the robot supported with every leg clear.
Keep the physical disconnect accessible. The client does not enable motors.

The client inherits the current command pose. It moves one joint at a time and
holds the other eleven joints unchanged. Each joint gets a one-second hold and
a 20-second 0.05–0.15 Hz chirp. The complete sequence takes approximately 4½ minutes.
Only one operator release is needed, not one per joint.

The planner selects amplitude from position clearance, measured initial error,
tracking margin, speed and acceleration bounds. The requested ceiling is ±5°.
The unchanged 3° tracking limit reduces actual amplitude to at most ±2.75°,
minus initial error. It also reserves 0.25° from each position limit.
It never moves the baseline to gain clearance. Joints with less than 0.25°
available amplitude are marked skipped. Export rejects an incomplete all-joint plan.

Any recording, control or hardware fault aborts the sequence and requests hardware
INACTIVE. There is no automatic retry. A renewal rejected at normal completion is
accepted as completion only after a fresh state confirms the same session completed.
On success, capture receives its stop signal before hardware shutdown begins.
The client does not wait for disk cleanup before requesting hardware INACTIVE.

Keep `excitation-plan.json`, `samples.jsonl`, `health.json`, `metadata.json`,
`profile.yaml` and `service-timings.json` together. After checking the capture:

```bash
python3 scripts/export_guarded_joint_fit.py CAPTURE_DIRECTORY \
  --output NEW_EXPORT_DIRECTORY --confirm-common-clock \
  --dt-ns 5000000 --max-gap-ms 50
```

The example uses a 5 ms fitting step. Match this to the actual fitting task.
The exporter rejects recording faults, sequence gaps, count mismatches and unfinished
joints. It uses held commands and interpolated feedback on the common source clock.
It preserves source hashes and actual amplitudes. Check response ranges before fitting.
This slow first dataset does not prove that delay, friction and all gains are identifiable.

### Temporary simultaneous fitting mode

`make_simultaneous_fit_profile.py` creates a separate opt-in profile from canonical
`description/models/limits/marvin_joint_limits.yaml`. Use `--template`, `--limits`
and a new `--output` path. The client option is `--simultaneous`, not `--all-joints`.
Normal runtime and policy settings remain unchanged.

The profile requests all twelve joints together: ±5°, a 0.1–2 Hz chirp, one second
of inherited hold and twenty seconds of excitation. Directions mirror left/right
joints and oppose calf/thigh changes. Speed is bounded at 1.2 rad/s and acceleration
at 15 rad/s². Tracking is bounded at 5° for this temporary profile, feedback at 100 ms, lease at 500 ms,
and maximum update gap at 50 ms. Initial tracking error is checked separately
from amplitude, not subtracted from permitted movement.
The operator approved the 5° tracking bound after capture 17 exceeded 3°.
Reload the controller with the regenerated profile before the next capture.

Every joint must qualify. The inherited command pose must leave five degrees of
clearance inside the policy limits. The previous IK standing pose does not qualify:
front thighs near 37.65° lack clearance and rear thighs near ±52° exceed the ±39°
policy envelope. This mode does not automatically reposition the robot.

For the operator-approved standing test, explicitly add `--standing-pose-envelope`
when generating a new profile. This changes only FL thigh upper to +44°, FR thigh
lower to −44°, RL thigh upper to +58°, and RR thigh lower to −58°. All other bounds
and tracking/stop checks remain unchanged. These are temporary test bounds, not
measured mechanical stops. The historical ±44° mechanical-stop note conflicts with
recorded rear-thigh motion. Numerical bounds do not establish collision clearance
or physical tracking. Normal policy limits remain unchanged.

The `__all__` selector requires a profile with `simultaneous_directions`. Its v3 hash
includes every direction. Legacy single-joint profiles and hashes remain unchanged.
PACE export requires a completed session with the exact recorded joint order.
Successful cleanup sends one recorder stop signal, not a second SIGINT during exit.
Failure cleanup also requests recorder stop before hardware shutdown, without
waiting for disk cleanup. Failed trials retain the primary error and timestamp.
Renewal rejection reports terminal tracking/feedback flags before cancellation
changes the state. Additional recorder cleanup failures do not replace the primary
test error or an unconfirmed motor stop. Partial failed captures remain ineligible
for fitting export.

### Short test without dynamic tracking abort

The operator-approved short diagnostic uses generator flag `--short-report-only`
with `--standing-pose-envelope`. Its reserved profile identity is
`marvin-stand-simultaneous-report-only-8s-v1`. This identity selects an eight-second
sweep plus one-second hold, with no dynamic tracking-error abort after execution
starts. Initial tracking checks and nonfinite-position rejection remain active.
Command limits, feedback freshness, lease, cancellation and update-gap checks remain
active. Raw command and position data still allow tracking-error analysis.
This profile is for a supported robot with an operator at the power disconnect.
It does not change normal IK, policy or other test profiles. The controller rejects
this identity with duration above nine seconds, amplitude above 5°, multiple cycles,
missing simultaneous directions or frequencies other than 0.1–2 Hz.

### Short seven-degree test

The later operator-approved `--short-7deg-3hz` profile also requires
`--standing-pose-envelope`. It uses identity
`marvin-stand-simultaneous-7deg-3hz-report-only-8s-v1`, an eight-second sweep plus
one-second hold, ±7° amplitude and 0.1–3 Hz. Only this profile expands speed to
2.5 rad/s, acceleration to 50 rad/s², front outward thigh bounds to ±46°, and
rear outward thigh bounds to ±60°. Opposite-side, hip and calf bounds are unchanged.
These operator-approved test bounds are not newly measured mechanical stops.
The validator's conservative derivative bounds are 2.351 rad/s and 45.534 rad/s².
Numerical sampling of the windowed waveform gives peaks near 1.283 rad/s and
15.147 rad/s². The configured bounds cover the conservative validation formula.
Initial tracking remains bounded at 5°. Dynamic tracking abort remains disabled
after execution starts. Other stop checks remain active. The original short
profile still has its original 5°/2 Hz limits. Do not reuse either profile for
normal robot operation or extend its duration implicitly.

### Short ten-degree test

After the operator confirmed physical clearance, the separate
`--short-10deg-3hz --standing-pose-envelope` profile increases amplitude to ±10°
but retains 0.1–3 Hz and eight seconds of excitation plus one-second hold.
Its identity is `marvin-stand-simultaneous-10deg-3hz-report-only-8s-v1`.
Temporary speed and acceleration bounds are 3.5 rad/s and 70 rad/s².
Outward thigh bounds are ±49° front and ±63° rear. The requested trajectory from
the recorded standing pose reaches approximately ±48° front and ±62° rear.
The operator confirmed clearance, not newly measured mechanical stops.
Initial tracking remains bounded at 5°, with dynamic tracking abort omitted after
execution starts. All other stop checks and normal profiles remain unchanged.

### Short four-Hz variant

After capture 23, the operator requested 0.1–4 Hz and reported good physical motion.
`--short-10deg-4hz --standing-pose-envelope` selects separate identity
`marvin-stand-simultaneous-10deg-4hz-report-only-8s-v1`. Amplitude remains ±10°,
excitation eight seconds, and hold one second. Position bounds remain unchanged.
Only temporary speed/acceleration bounds increase to 4.6 rad/s and 120 rad/s²,
covering conservative bounds of 4.455 rad/s and 114.279 rad/s². Numerical waveform
peaks are approximately 2.425 rad/s and 38.027 rad/s². Initial tracking remains 5°.
Dynamic tracking abort is omitted after execution starts. Other stop checks remain
active. This does not extend the duration or change normal robot profiles.

### Short five-Hz variant

After capture 24, the operator requested preparation for 0.1–5 Hz.
`--short-10deg-5hz --standing-pose-envelope` selects the separate identity
`marvin-stand-simultaneous-10deg-5hz-report-only-8s-v1`.
Amplitude remains ±10°, with eight seconds of excitation and one second of hold.
Position bounds remain unchanged. Temporary speed and acceleration bounds are
5.7 rad/s and 180 rad/s². These cover conservative admission requirements of
5.552 rad/s and 177.289 rad/s², not predicted physical motion.
Numerical waveform peaks are approximately 3.026 rad/s and 59.110 rad/s².
Initial tracking remains 5°. Dynamic tracking abort remains omitted after execution
starts. Other stop checks remain active. Preparation does not authorize a physical sweep.

### Sixty-second five-Hz variant

`--long-10deg-5hz --standing-pose-envelope` selects
`marvin-stand-simultaneous-10deg-5hz-report-only-60s-v1`.
It retains the short five-Hz profile's amplitude, position bounds, frequency range,
speed bounds, acceleration bounds and initial tracking checks. Only the identity
and total duration change: one-second hold plus sixty-second excitation, one cycle.
The controller caps this reserved profile at 61 seconds. Other short profiles remain
capped at nine seconds. Dynamic tracking remains report-only after execution starts.
Feedback freshness, lease, update-gap and cancellation checks remain active.
The client completion deadline follows the planned duration plus three seconds.
Recorder allowance covers operator waiting and the longer sweep.
The Hann envelope remains unchanged. High-frequency amplitude still fades near the end.
Longer operation increases exposure and heating. Require supervised stand testing
with an accessible disconnect. Do not infer clearance from a successful software test.

### Recorder storage and callback timing

During capture, `health.json` links to a private directory in `/dev/shm`.
Live health writes therefore do not wait for SD-card filesystem writes.
The sample writer still flushes and synchronizes `samples.jsonl` on disk.
The live freshness gate remains 150 ms. Disk progress must stay within two seconds.
The 1024-record queue covers approximately 2.56 seconds at two 200 Hz streams.
Overflow or a writer error still fails the recording gate immediately.
Finalization replaces the link with a regular on-disk health report, then removes
the private RAM directory. Copy complete captures only after finalization.
A power loss can leave a broken health link. Such a capture is not finalized
and must not pass the export checks.

`loop_timing` reports maximum callback, executor and health-write durations, plus
at most 32 slow events. The recorder copies the fixed message fields directly.
A schema comparison test checks compatibility with generic ROS conversion.

