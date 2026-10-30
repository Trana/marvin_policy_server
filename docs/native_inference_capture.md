# Native inference capture

This diagnostic is disabled by default. The native server accepts the ROS parameter
`native_diagnostic_path`, which must name a new file in an existing directory.
An existing file causes startup to fail rather than overwrite previous evidence.
No runtime profile enables this parameter automatically.

When enabled, the server buffers up to 6000 event records in memory. At 50 inferences
per second, with one result and one commit each, this covers approximately 60 seconds.
Later records are counted as dropped. The buffer retains the beginning, not the end.
It does not automatically activate policy or start a test.

Each result includes sequence, generation, trace ID, request/sensor timestamps,
estimated velocity if available, result values and the request sensor snapshot.
Snapshot layout is 12 positions, 12 velocities, XYZW quaternion, XYZ angular
velocity and XYZ acceleration. Joint order appears in the metadata.
Result records now optionally include `effective_velocity_command`: forward and
lateral velocity in m/s, followed by yaw rate in rad/s. It copies the newest
observation command after permission and input-lock overrides, not the raw joystick
input. Direct-output fallback and records without this field provide no command
evidence. Commit rows use null. This additive field is prepared and tested in a
temporary Pi package, but not yet installed in the physical runtime.
One 15-second synthetic off/on pair with the added field passed shutdown and
fault checks. Inference rates were 49.94/49.85 Hz, mean inference 10.33/10.59 ms,
and mean callbacks 12.37/12.82 ms. Process-tree CPU was 113.93/115.92% of one core.
All 998 enabled-capture results contained the expected zero effective command and
matched observed publication commits. The export had 1996 records and zero drops.
The reverse on/off pair also passed at 49.92/49.77 Hz. Mean inference was
10.54/10.51 ms, mean callback 12.84/12.60 ms and CPU 118.46/112.99% of one core.
It exported 1000 results and 999 observed commits without drops. Only the last
result lacked a commit, so it is not counted as a published action. These short
synthetic trials do not establish overhead under a loaded physical runtime.
Result values follow the native protocol, including DIRECT_OUTPUT flags.
Commit values contain 12 policy targets followed by 12 published targets.

Only a validated COMMIT creates a publication record. Submission alone does not
prove publication. COMMIT acknowledges first ROS command publication, not CAN
transmission or physical motor response. Match trace IDs to the recorded command
topic and use joint feedback to investigate tracking.
The COMMIT packet retains request timestamps. Its `ros_ns` is not the publication
time. `received_monotonic_ns` measures Python receipt of the acknowledgement, which
can be later than publication. Use the matched command message for publication time.

File writes occur at initialization and orderly native shutdown, not inside the
control callback. Kill signals and power loss can lose buffered rows. Check the
final summary before accepting a capture. This tradeoff avoids disk stalls in the
inference callback. Memory-copy overhead still needs measurement under load.

The implementation is prepared but not enabled in the physical runtime. Do not
enable it for a floor trial before validating overhead and artifact completeness.

For synthetic overhead measurement, run `scripts/benchmark_native_diagnostics.py`
with the package on PYTHONPATH. It does not use ROS. For the existing private-domain
policy check, add `--diagnostics /new/output.jsonl` to
`native_io/check_performance.py --backend native ...`. Parent-death tests cannot
use this option because abrupt termination deliberately prevents buffered export.

The performance harness explicitly uses OMP/MKL single-thread environment settings,
matching the physical launcher. Initial tests without those variables hit deadlines
even with diagnostics disabled. Matched-setting trials passed and independently
matched 1000 commits to observed commands. A single pair is not enough to establish
overhead: observed command rate was 199.99/s off versus 196.52/s on. Repeat with
reversed order before enabling physical capture.

Add `--bag /new/bag-directory` to the private performance check to measure combined
recording load. The offline join checks commit IDs and command values:

```bash
python3 scripts/analyze_native_capture.py /path/to/diagnostics.jsonl /path/to/bag \
  --topic /policy_probe/commands_traced
```

Source ROS and the custom trace messages before running the join. It supports
uncompressed SQLite bags. Missing or ambiguous commands do not produce timing
samples. Timestamp differences use ROS time, not mixed wall/monotonic clocks.
In the combined Pi test all 989 commits matched saved commands. Mean inference
increased to 11.85 ms, so retain the capture configuration in trial comparisons.

The report includes linear-interpolated p50, p95 and p99 timing percentiles.
Empty distributions report null, not zero. `complete_export` means only that the
file ends with a summary. `export_quality` reports declared and observed record
counts, dropped records and `lossless_export`. A lossless diagnostic export does
not prove bag completeness, physical sensor accuracy or successful locomotion.
Missing counts cannot establish a lossless export.

## Joint command limit counts

Add `--joint-limits /path/to/marvin_joint_limits.yaml` to report near-limit counts
and command ranges by joint name. The default margin is 0.01 rad. Change it with
`--limit-margin-rad`. Only unique commits with matching recorded command values
contribute. Missing, ambiguous or mismatched commands are excluded and counted.
Counts describe first publications, not time at a limit, clipping events or actual
motor positions. Ramps and overrides can affect these commands.

For a grounded interval, supply `--limit-start-ros-ns` and `--limit-end-ros-ns`.
These are absolute command-publication ROS timestamps, with an inclusive start and
exclusive end. They restrict only the limit report; the timing/export report still
covers the whole capture. Use independently identified grounded/lifted boundaries,
not clipping itself. Do not pass wall-clock or monotonic operator-event timestamps
without verifying their mapping to ROS time. Do not combine simulation clock epochs
across a reset. A selected window is not automatic proof of ground contact.

## Joint target tracking

Reference capture uses default-launch endpoints for each target. With `--target sim`,
the status service is `/sim/get_policy_observations`, and traced controller commands
are `/sim/marvin_joint_controller/commands_traced`. The recorder also includes
`/isaac_joint_commands`, `/isaac_joint_states`, `/isaac_imu`, `/isaac_odometry` and
`/clock`. Check recorded topic counts before analysis. Custom namespaces are not
discovered automatically. Older captures do not gain missing streams retroactively.

`scripts/analyze_joint_tracking.py` reads a completed uncompressed SQLite bag.
It compares every valid feedback sample with the latest preceding stamped target.
Feedback is matched by joint name. Command arrays require an explicit joint order
from the captured runtime configuration. Do not assume JointState array order.
Source ROS and `marvin_trace_msgs` before running the script.

Required arguments are the bag directory, `--joint-names` followed by the exact
command joint names, `--start-ros-ns`, and `--end-ros-ns`. The window is inclusive
at the start and exclusive at the end. Select independently identified grounded
intervals for fall analysis. Stand/lift periods need separate reports.

Defaults are `/marvin_joint_controller/commands_traced` and `/joint_states`.
Use `--command-topic` and `--feedback-topic` for other captured sources.
For mixed-producer sim captures, use `--source-id 1` only after confirming that
source 1 identifies the policy producer. The analyzer reports excluded commands
and still rejects duplicate or regressing stamps. This selects a producer, not a
clock conversion. Equal simulation stamps do not prove causal application order
or zero delay through the scene buffer and actuator.
The script reports signed mean error, RMSE and maximum absolute error in radians.
Results are sample-weighted, not duration-weighted. Empty results use null.
Missing preceding targets, stale targets and invalid feedback have separate counts.
The default `--max-command-age-ms 50` is an analysis cutoff, not a safety limit.
Record it with each comparison and inspect excluded counts.

The reader rejects nonpositive, duplicate or regressing header timestamps. Do not
sort away a clock reset or merge clock epochs. A malformed command also stops
analysis rather than allowing an older command to stand in for it.
Headers must share the same ROS clock. JointState headers can describe publication
time rather than encoder sampling time. Command publication is not CAN receipt.
The report therefore measures observed target/feedback differences, not isolated
motor latency. It does not interpolate, estimate a best-fitting lag, or infer
ground contact. A small error does not establish successful locomotion.

The tests include reversed feedback order, stale/missing data, invalid values,
clock regressions and split SQLite files with actual serialized ROS messages.
No physical tracking result exists yet for this tool. Use the report with capture
quality checks, policy/configuration provenance and synchronized video.
