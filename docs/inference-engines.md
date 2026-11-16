# Policy inference engines

`inference_engine` defaults to `onnxruntime`. Select `torchscript` to use the retained
PyTorch implementation. Both engines load the same trusted TorchScript source bundle.
The source artifact, metadata, observation history, cadence and safety limits stay unchanged.
Engine selection applies at startup and carries through inactive reload and async workers.
Changing the engine requires a process restart. It does not change policy selection.

Install the default engine in the policy Python environment with Python 3.11 or newer:

```bash
python -m pip install -r requirements-robot.txt -r requirements-onnx.txt
```

ONNX Runtime uses only `CPUExecutionProvider`, sequential execution and one thread.
The server derives fixed batch-one graphs in memory before readiness. It checks the graphs,
compares deterministic nonzero inputs against Torch and warms each supported operation.
It preserves combined snapshots, estimator overrides and separate velocity estimation.
No generated graph is written into the immutable policy bundle. Conversion, dependency,
shape or numerical failures reject preparation. There is no automatic engine fallback.
Torch remains installed for source loading, compact-estimator preparation and runner tensors.
This change does not remove PyTorch from the dependency footprint.

For direct ROS launch, use either command in a sourced workspace:

```bash
ros2 launch marvin_policy_server marvin_policy_server.launch.py inference_engine:=onnxruntime
ros2 launch marvin_policy_server marvin_policy_server.launch.py inference_engine:=torchscript
```

For managed runtime, set `MARVIN_POLICY_INFERENCE_ENGINE=onnxruntime` or `torchscript`
in the reviewed environment of the runtime, runtime manager and policy manager.
Run CPU validation again after an engine or compact-estimator change. Compatibility records
include the selected engine, compact setting and checker hash. Old validation is insufficient.
On Pi 5, the three services share `/etc/default/marvin-pi5-runtime`.
Update this pinned file through the existing profile transaction while inference is stopped.
Do not edit it outside that transaction or restart shared services to change the engine.

Explicit TorchScript installations can omit `requirements-onnx.txt`, including the old
Python 3.10/Humble environment. Select `torchscript` consistently for admission and execution.
Startup logs and timing status identify the engine, provider, version and derived graph hashes.
These hashes supplement the original source artifact identities.

Offline and mock checks do not establish physical tracking or motion safety.
Pi 5 deployment and remaining hardware acceptance belong to
[marvin_wiki#59](https://github.com/Trana/marvin_wiki/issues/59).
