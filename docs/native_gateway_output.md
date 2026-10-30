# Native policy output through the supervisor

The native backend supports `execution_gateway=true`. Physical bringup selects
this mode. The Python policy backend remains the default for timed Isaac scenes.

## Ownership and timing

The native process receives sensors and runs lifecycle checks at 200 Hz. It
requests inference at 50 Hz. In gateway mode it publishes `ExecutorCommand` at
50 Hz on `execution/policy/commands`. The supervisor validates those candidates
and passes accepted targets to its native command controller. The policy process
creates no legacy or traced motor-command publisher in this mode.

Candidate publication replaces the old 200 Hz direct command stream. The gateway
still has a Python validation step. Native inference, sensor handling, expiry and
mapping remain unchanged. Performance measurements must include that downstream
step before claiming a total runtime CPU improvement.

`PrepareExecution` validates the supervisor generation and initial joint targets.
A version-2 local IPC PREPARE message transfers the opaque UTF-8 token (at most
256 bytes), twelve targets, and the absolute monotonic preparation expiry. The
same-host expiry does not restart when the packet arrives. Numeric native epochs
remain separate from supervisor generations.

Python validates activation before sending native ACTIVATE. C++ requires an
unexpired preparation, binds its token to the new epoch, and blends from the
initial targets over two ROS seconds. This follows the existing Python gateway
handover, after policy mapping and output locking. The raw-action activation ramp
still applies before mapping. There is no duplicate Python candidate publisher.

Stop ends candidate publication immediately and invalidates preparation. It does
not publish the legacy return-to-default ramp. The supervisor owns hold and the
next transition. A fresh native status or confirmed child exit proves output
release. A Stop acknowledgement alone is not a motor-disable confirmation.

Input-lock, permission and sensor invalidation discard old inference results but
retain the current supervisor generation. Permission loss still zeros requested
velocity and allows balance inference. Faults clear native candidate authority.
A fresh preparation can clear an inactive native fault. A simulation clock rewind
releases candidate authority and requires a new preparation and activation.

`policy_candidate_published` means submission to the supervisor. COMMIT confirms
that native publication, not actuator application. The existing inference expiry
prevents old inference from being republished indefinitely with fresh stamps.

## Verification

- `native_io/gateway_output_test.cpp`: preparation expiry, handover, epoch rebinding,
  and stop fencing without ROS.
- `test/test_native_protocol.py`: Python/C++ wire agreement and malformed envelopes.
- `test/test_native_server.py`: validation before native activation and failed-ACK
  rollback, plus existing state/history checks.
- `native_io/check_runtime.py --gateway`: isolated domain 184, actual model inference,
  synthetic sensors, real CommandGateway validation, publisher graph assertions,
  input-lock/permission changes, sensor loss, injected deadline fault and Stop.
- `native_io/check_performance.py --gateway`: isolated comparison with an external
  observer. Reports policy-tree CPU and observer CPU separately. The observer uses
  real command validation but does not drive a native hardware writer.

Both scripts require `ROS_DOMAIN_ID=184`, `ROS_LOCALHOST_ONLY=1`, and the test
native profile. They cannot prove physical handover or locomotion. For the native
simulated-clock fixture, use an explicitly non-timed environment: timed simulation
remains a Python-backend contract and is not silently disabled by product code.

Evidence: `marvin-measurements/native-policy-gateway-20261029/`.
