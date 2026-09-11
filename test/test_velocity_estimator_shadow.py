from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sensor_msgs.msg import Imu, JointState

from marvin_policy_server.velocity_estimator import VelocityEstimator
from marvin_policy_server.velocity_estimator_diagnostics import (
    FEATURE_NAMES,
    VelocityEstimatorDiagnosticRecorder,
)
from marvin_policy_server.velocity_estimator_features import VelocityEstimatorFeatureBuilder
from marvin_policy_server.velocity_estimator_shadow import VelocityEstimatorShadowRecorder
from marvin_policy_server.marvin_policy_server import MarvinPolicyServer


JOINT_NAMES = [
    "FL_hip_joint",
    "RL_hip_joint",
    "FR_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "RL_thigh_joint",
    "FR_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "RL_calf_joint",
    "FR_calf_joint",
    "RR_calf_joint",
]


def test_feature_builder_matches_collection_order() -> None:
    defaults = np.linspace(-0.5, 0.5, 12)
    builder = VelocityEstimatorFeatureBuilder(JOINT_NAMES, defaults)
    joint_state = JointState()
    joint_state.name = list(reversed(JOINT_NAMES))
    by_name_position = {name: float(index + 1) for index, name in enumerate(JOINT_NAMES)}
    by_name_velocity = {name: float(index + 101) for index, name in enumerate(JOINT_NAMES)}
    joint_state.position = [by_name_position[name] for name in joint_state.name]
    joint_state.velocity = [by_name_velocity[name] for name in joint_state.name]

    imu = Imu()
    imu.orientation.w = 1.0
    imu.linear_acceleration.x = 1.0
    imu.linear_acceleration.y = 2.0
    imu.linear_acceleration.z = 3.0
    imu.angular_velocity.x = 4.0
    imu.angular_velocity.y = 5.0
    imu.angular_velocity.z = 6.0
    previous_action = np.arange(12, dtype=np.float64) + 201.0

    features = builder.build(joint_state, imu, previous_action)

    np.testing.assert_allclose(features[0:3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(features[3:6], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(features[6:9], [0.0, 0.0, -1.0])
    np.testing.assert_allclose(
        features[9:21],
        [by_name_position[name] for name in JOINT_NAMES] - defaults,
    )
    np.testing.assert_allclose(features[21:33], [by_name_velocity[name] for name in JOINT_NAMES])
    np.testing.assert_allclose(features[33:45], previous_action)


def test_feature_builder_can_override_relative_joint_position() -> None:
    """Use one conditioned motor-position signal for estimator and actor."""
    builder = VelocityEstimatorFeatureBuilder(
        JOINT_NAMES, np.zeros(12, dtype=np.float64)
    )
    joint_state = JointState()
    joint_state.name = list(JOINT_NAMES)
    joint_state.position = [9.0] * 12
    joint_state.velocity = [0.0] * 12
    imu = Imu()
    imu.orientation.w = 1.0

    features = builder.build(
        joint_state,
        imu,
        np.zeros(12),
        joint_position_relative_override=np.arange(12),
    )

    np.testing.assert_allclose(features[9:21], np.arange(12))


def test_feature_builder_can_restore_specific_force_for_isaac_ros() -> None:
    builder = VelocityEstimatorFeatureBuilder(
        JOINT_NAMES,
        np.zeros(12),
        add_gravity_to_linear_acceleration=True,
        gravity_magnitude=9.81,
    )
    joint_state = JointState()
    joint_state.name = JOINT_NAMES
    joint_state.position = [0.0] * 12
    joint_state.velocity = [0.0] * 12
    imu = Imu()
    imu.orientation.w = 1.0

    features = builder.build(joint_state, imu, np.zeros(12))

    np.testing.assert_allclose(features[0:3], [0.0, 0.0, 9.81])
    np.testing.assert_allclose(features[6:9], [0.0, 0.0, -1.0])


def test_packaged_estimator_tracks_history_readiness() -> None:
    checkpoint = Path(__file__).resolve().parents[1] / "policy" / "velocity_estimator.pt"
    estimator = VelocityEstimator(checkpoint)
    frame = np.zeros(estimator.raw_feature_count, dtype=np.float32)

    for _ in range(estimator.config.history_length - 1):
        prediction = estimator.predict(frame)
        assert prediction.shape == (3,)
        assert np.isfinite(prediction).all()
        assert not estimator.history_ready

    estimator.predict(frame)
    assert estimator.history_ready
    assert estimator.history_count == estimator.config.history_length


def test_estimator_prediction_is_available_without_shadow_odometry() -> None:
    server = object.__new__(MarvinPolicyServer)
    server._velocity_estimator = SimpleNamespace(
        predict=lambda _features: np.array([0.4, -0.1, 0.02]),
    )
    server._velocity_estimator_feature_builder = SimpleNamespace(
        build=lambda _joint_state, _imu, _action: np.zeros(45),
    )
    server._velocity_estimator_shadow_recorder = None
    server._latest_estimated_velocity = None
    server._velocity_estimator_error = None
    server._latest_odometry_velocity = None
    server._latest_odometry_received_monotonic = None
    server._latest_odometry_stamp_sec = None
    server._base_lin_vel_timeout_sec = 0.25
    server._velocity_estimator_shadow_warned = False
    server._logger = SimpleNamespace(warn=lambda _message: None)
    server.action = np.zeros(12)

    server._update_velocity_estimator(JointState(), Imu(), 1.0)

    np.testing.assert_allclose(server._latest_estimated_velocity, [0.4, -0.1, 0.02])


def test_shadow_recorder_separates_history_warmup(tmp_path: Path) -> None:
    recorder = VelocityEstimatorShadowRecorder(str(tmp_path))
    recorder.record(
        ros_time_sec=1.0,
        joint_state_time_sec=0.99,
        imu_time_sec=0.99,
        odometry_time_sec=0.99,
        command=np.zeros(3),
        inference_duration_ms=0.2,
        estimated_velocity=np.array([1.0, 2.0, 3.0]),
        odometry_velocity=np.array([0.0, 0.0, 0.0]),
        history_count=1,
        history_ready=False,
    )
    recorder.record(
        ros_time_sec=1.02,
        joint_state_time_sec=1.01,
        imu_time_sec=1.01,
        odometry_time_sec=1.01,
        command=np.array([0.5, 0.0, 0.0]),
        inference_duration_ms=0.4,
        estimated_velocity=np.array([0.1, 0.2, 0.3]),
        odometry_velocity=np.array([0.0, 0.0, 0.0]),
        history_count=10,
        history_ready=True,
    )
    summary = recorder.summary()
    recorder.close()

    assert summary["all"]["count"] == 2
    assert summary["history_ready"]["count"] == 1
    np.testing.assert_allclose(summary["history_ready"]["rmse_xyz"], [0.1, 0.2, 0.3])
    assert np.isclose(summary["inference_duration_ms"]["mean"], 0.3)
    assert np.isclose(summary["inference_duration_ms"]["max"], 0.4)
    assert recorder.log_path is not None
    assert recorder.log_path.read_text(encoding="utf-8").count("\n") == 3


def test_diagnostic_recorder_aligns_features_and_policy_action(tmp_path: Path) -> None:
    recorder = VelocityEstimatorDiagnosticRecorder(
        str(tmp_path), joint_names=JOINT_NAMES, velocity_source="odometry"
    )
    recorder.record(
        ros_time_sec=2.0,
        joint_state_time_sec=1.98,
        imu_time_sec=1.99,
        odometry_time_sec=1.97,
        history_count=10,
        history_ready=True,
        command=np.array([0.5, -0.2, 0.1]),
        features=np.arange(45, dtype=np.float64),
        estimated_velocity=np.array([0.4, -0.1, 0.0]),
        odometry_velocity=np.array([0.45, -0.15, 0.01]),
        policy_action=np.arange(12, dtype=np.float64) + 100.0,
    )
    recorder.close()

    rows = recorder.log_path.read_text(encoding="utf-8").splitlines()
    assert len(FEATURE_NAMES) == 45
    assert len(rows) == 2
    header = rows[0].split(",")
    values = rows[1].split(",")
    assert len(header) == len(values)
    assert values[header.index("velocity_source")] == "odometry"
    assert float(values[header.index("joint_state_age_sec")]) == 0.02
    assert float(values[header.index("imu_accel_x")]) == 0.0
    assert float(values[header.index("previous_policy_action_11")]) == 44.0
    assert float(values[header.index(f"policy_action_{JOINT_NAMES[-1]}")]) == 111.0
