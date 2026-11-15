# Marvin policy server

Supported platform: **Ubuntu 24.04 / ROS 2 Jazzy**. The Pi 5 migration makes Jazzy
the supported target when merged to main. Humble import compatibility is removed.
Dated Pi 4/Humble reports remain historical evidence. Keep that installation
separate for rollback.

ONNX Runtime remains the default CPU inference engine. Explicit TorchScript
selection remains supported. This platform migration does not change policy
artifacts, timing, runner state or safety checks.

See [inference engines](docs/inference-engines.md) and
[wiki #59](https://github.com/Trana/marvin_wiki/issues/59) for implementation,
deployment and required physical verification. A source merge is not deployment proof.
