# Isolated monitoring comparison

`native_io/check_performance.py` can compare monitoring overhead with synthetic
sensors on the Pi. It requires localhost-only ROS domain 184. Its native output
topics must use `/policy_probe/`. This test does not operate the robot hardware.

Stop the physical runtime before testing. Keep the model, environment, native
binary, thread settings, and duration identical between trials. Repeat modes in
reverse order to expose run-to-run variation.

Add these options to the existing private performance command:

```
--seconds 15 --measure-status --monitor none
--seconds 15 --measure-status --monitor persistent
--seconds 15 --measure-status --monitor fresh
--seconds 15 --measure-status --monitor discovery
```

- `none`: no additional status reads during the measured interval. The existing
  harness still observes commands and metrics and checks status before and after.
- `persistent`: use the harness's existing ROS client for status reads, starting
  two seconds into measurement and repeating every five seconds. No new request
  starts in the final six seconds, so client work can finish before deactivation.
- `fresh`: start a new Python/ROS client at the same times. Each client discovers
  the observation service, reads status once, then exits.
- `discovery`: start the same client, discover the service, then exit without a
  status request.

`--measure-status` uses `monitoring_probe.py` as the test server entry point.
It wraps the existing observation callback without changing the installed files.
It records callback elapsed time and thread CPU time in memory, with a 1,000-record
limit. It prints `MONITOR_CALLBACKS` on orderly exit. This instrumentation is
present in all compared modes and has its own small overhead.

`MONITOR_CLIENT` records startup/discovery duration and query duration for each
fresh client. `NATIVE_PERFORMANCE_CHECK` reports policy inference, publication,
command gaps and policy-process-tree CPU. That CPU number excludes the observer,
synthetic source, monitoring clients and system services. A 100% value means one
CPU core, not the whole Pi.

The persistent round-trip time includes observer scheduling. It is not the server
callback duration. Startup tests reuse the same SSH session and therefore do not
measure SSH authentication overhead. Synthetic sensors do not reproduce IMU UART,
CAN traffic, or all physical-runtime CPU load. No fault in this test does not prove
that monitoring is harmless during physical motion.

Review child-client errors and the complete policy logs as well as summary rates.
The harness checks client completion before deactivation and checks policy faults
after deactivation. Earlier comparisons without these checks could print a passing
rate summary despite a transport fault during shutdown. Do not treat those summaries
alone as a lifecycle pass.
Do not increase the inference deadline to make a comparison pass.

## Inactive startup check

Add `--inactive-only --backend native` to observe startup for the selected duration
without requesting activation. The harness checks for zero commands and no reported
native fault. It supports the same monitoring modes. Parent-death and stop-overlap
tests are incompatible because they require activation.

The normal active test also rejects a native fault during sensor discovery, before
activation can clear that fault. A passing active interval must not hide a failed
startup. These checks still use synthetic sensors and cannot establish physical
sensor health or reproduce all physical-runtime load.

## Cache-only estimator observations

Observation requests no longer evaluate the estimator. Each successful inference
stores its foot positions and a monotonic completion timestamp. Decimated hold
ticks retain that result. Reset clears both the result and timestamp.

The compact policy exports `inference_snapshot`, which returns the existing action,
estimated state and foot positions from one calculation. Startup warms this exact
batch-one method before activation and checks numerical agreement with the original
artifact. Warming only `forward` is insufficient because TorchScript can compile
exported methods separately. Legacy split artifacts warm their estimator/actor path
before activation. Async workers copy cache values and the timestamp with the result.

The raw observation response includes `concurrent_policy.snapshot_available` and
`snapshot_age_seconds`. These fields describe cached data, not sensor freshness or
permission to move. The typed control-plane UI does not yet display those fields.
Missing foot-position data is marked unavailable. Holding current foot positions,
foot heights or contacts requires a result no older than 100 ms. Explicit fixed
values do not require a snapshot. No read request triggers a fallback inference.

Use the existing persistent gateway or a prestarted observer during physical motion.
These cache changes do not eliminate ROS discovery overhead from new clients.

## Discovery overlapping with Stop

Use `--monitor fresh --monitor-during-stop` only in the isolated harness to permit
new clients during the final six seconds. The harness deactivates the policy while
a client may still be discovering services. It then waits at most six seconds for
remaining clients and checks that commands did not resume and no fault appeared.
The option does not guarantee overlap on every host. Inspect the client timestamps.

Native queue overflow errors now include the queue limit, queued packet types,
sequences, generations, flags, and source ages. The incoming packet is described
separately. Formatting happens only on failure. Source age includes transport and
queue time since the native packet timestamp. It is not a measured executor stall
or Python queue residence time. Missing or invalid timestamps produce `null`.

The queue limit remains four. Overflow still closes the transport and stops output.
No packet is silently discarded. Control acknowledgements still use their separate
path. Do not increase queue capacity or discard status transitions merely to make
the test pass.

## Native heartbeat fault report

`NATIVE_TRANSPORT_FAULT` records the generation, 300 ms limit, age of the last
processed peer packet, native tick gap and age of the last status send. It appears
only when the native fault changes to transport failure. A missing status send is
`null`. These native-loop ages do not identify a Python callback or prove why
heartbeat processing stopped. The watchdog behavior and deadlines are unchanged.

Run the no-activation fault-injection test with a built binary:

```
MARVIN_TEST_NATIVE_BINARY=/absolute/path/to/marvin_policy_io \
  python3 -m pytest -q test/test_native_transport_diagnostics.py
```

The test forces localhost-only domain 184 and the test profile. It sends configuration
but no activation or sensor data. It withholds heartbeat replies, checks the fault
report, then requests shutdown. Without the binary variable, pytest skips this test.
