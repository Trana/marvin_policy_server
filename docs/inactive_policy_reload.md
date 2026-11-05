# Load a policy while IK owns output

Ticket: https://github.com/Trana/marvin_wiki/issues/18

This implementation is not deployed or physically qualified. Reload defaults off.
Set `MARVIN_POLICY_RELOAD_ENABLED=1` in both the controller and policy process environments only for an authorized verification profile. The policy process requires native I/O and execution-gateway mode. The installed-policy authority must be configured in its environment. The supervisor gateway uses its existing explicit mutation enablement.

A normal runtime start can enter Policy directly. IK Stand is required only to replace the loaded policy.

Switch to IK Stand. Wait for confirmed IK ownership and policy output release. In the robot Policy workspace, use **Load policy** for an installed, CPU-validated bundle. After success, switch to Policy explicitly. Loading does not change next-start selection or activate an executor.

The supervisor serializes reload with policy activation. It validates its instance, command generation, request deadline, fresh feedback, active IK Stand and policy release. It sends a bounded start request to the existing policy process. A worker loads and warms the candidate without touching ROS or live policy state. The supervisor rechecks ownership before commit. Stop invalidates preparation and remains available. A commit already in flight can complete an inactive replacement, but cannot activate policy after Stop.

Version 1 requires identical environment bytes and identical combined-model semantic metadata, plus the same model family, estimator features and output dimensions. Different weights are allowed. Known export timestamps, paths, hashes and the optional training-task label do not define compatibility. Bundle validation checks artifact hashes separately. Unknown metadata fields remain part of the compatibility check. A different environment or semantic metadata reports **Runtime restart required**. This conservative boundary preserves the native mapping, timing, joint references and observation semantics. It does not qualify motion quality.

Commit replaces the model/runner and optional estimator, clears histories and filters, invalidates pending results, resets preparation, clears overrides and rotates the policy-control session. A normal later executor transition performs the existing startup preparation. Synthetic warmup samples are discarded. Reload does not introduce a new walking-transition or history-priming recipe.

The target store pins the requested authority incarnation and copy revision before reading. The robot inference reservation remains held. A loaded replacement retains its reference until another successful replacement or observed runtime teardown. Uncertain references do not expire.

Requests use UUIDs. Identical retries return the original result. Different arguments conflict. The supervisor and node journals are bounded per process. After a process restart, unknown operations are never automatically replayed. The browser retains its original request and queries it after reload. Explicit retry uses the same UUID and payload.

Failures keep Policy blocked while IK remains controlled by its existing executor. A new successful load can restore Policy availability. Do not resume the previous policy automatically. Supervisor telemetry emits a `policy_reload` event with loaded identities and old/new sessions. This event describes inactive replacement, not accepted motion or physical tracking.

## Required timing gate

An in-process thread cannot terminate a stuck Torch or filesystem call. The 90-second operation deadline cancels authorization but cannot guarantee bounded process teardown. Shutdown waits for loader cleanup so file references remain valid. Measure loading, warmup, commit, old-model disposal and garbage collection on the real Pi with mock motors and IMU. Include IK command continuity, memory use, heartbeat response, motion Stop and full runtime Stop. If limits fail, isolate compute before physical use.

Offline fixtures do not satisfy Pi or physical acceptance. Keep reload disabled on the physical profile until the separately authorized gates pass.
