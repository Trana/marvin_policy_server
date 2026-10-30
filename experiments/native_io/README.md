# Isolated native policy I/O experiment

This is a benchmark prototype, not a robot runtime. Do not connect it to hardware.
It requires `ROS_DOMAIN_ID=184` and `ROS_LOCALHOST_ONLY=1`. C++ topics are fixed
under `/policy_probe/`. Global ROS remapping is disabled in the C++ node.
There is no CAN interface, deployment installer or systemd integration.

## Architecture

The C++ node receives synthetic joint/IMU data and publishes held positions at
200 Hz. It requests the latest sensor snapshot at nominal 50 Hz. An inherited
AF_UNIX sequence-packet socket connects it to Python. Only one request can be
outstanding. There is no network socket, filesystem socket path or request queue.

Python reuses the existing policy node's loader, observation builder, embedded
estimator, action mapping and clipping. It calls the synchronous policy tick for
each snapshot. It does not spin that node's ROS executor or publish motor commands.
The diagnostic reports actual inference calls separately from worker replies.
This is a temporary reuse of application code, not a proposed production API.
C++ is the sole request scheduler. Python infers once per fresh request, retaining
the existing observation/history calculations but not a second 50 Hz deadline.

The native gate starts disabled. A diagnostic control packet explicitly enables
it. Fresh synthetic permission and sensor timestamps are required. Loss of either,
an expired result, a malformed result or a 100 ms inference outage latches it off.
Stop increments the generation and clears the held command. Old/duplicate results
cannot restore it. Every native publication has a fresh timestamp. Position limits
are copied from the loaded Python profile and applied again in C++.

## Important differences from production

- Inputs are synthetic. No walking-quality conclusion is possible.
- The permission topic is a diagnostic boolean, not the real operator lease.
- Python management services are not serviced. No UI, joystick, locks or runtime
  manager operation can be tested through this executable.
- Ramps update when Python returns a target, not at every 200 Hz publication.
- The gate uses host monotonic time and does not implement simulation-clock rewind.
- C++ limit configuration is fixed for a run. Live configuration is not supported.
- There is no hardware subscriber or CAN bus. Publication rate is not motor rate.
- Both processes require the same little-endian host and fixed packet layout.

Production integration requires explicit lifecycle/ownership design, equivalent
operator permission handling, clock/reset behavior, dynamic joint-state support,
end-to-end subscriber timing, and sim validation. Do not deploy this prototype.

## Build and test

First source the ROS installation and the workspace with `marvin_trace_msgs`.
Run these commands from this directory. Use a new temporary build directory.

```bash
build_dir=$(mktemp -d /tmp/native-policy-io-build.XXXXXX)
cmake -S . -B "$build_dir" -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_dir" -j2
ctest --test-dir "$build_dir" --output-on-failure
```

For a bounded diagnostic, use the validated Python/PyTorch environment and a
verified policy bundle containing `policy.pt`, `metadata.json` and `env.yaml`.
The Python script loads the model before it starts the isolated C++ node.

```bash
ROS_DOMAIN_ID=184 ROS_LOCALHOST_ONLY=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python probe.py --binary "$build_dir/policy_io_probe" --bundle /path/to/bundle
```

The test warms the policy, measures eight seconds, checks Stop while active, then
explicitly re-enables before stopping responses to inference requests. It checks
that expiry latches publication off, old replies cannot resume output, and malformed
packets latch the gate off. The C++ process has a 30-second
hard lifetime. The synthetic source has a 25-second hard lifetime. Neither is a
service. Cleanup stops only child processes created by the probe.
