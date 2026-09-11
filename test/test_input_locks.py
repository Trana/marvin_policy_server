from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
from geometry_msgs.msg import Twist
from marvin_policy_interfaces.srv import SetPolicyInputLock
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, JointState
from std_srvs.srv import SetBool

from marvin_policy_server.marvin_policy_server import MarvinPolicyServer
from marvin_policy_server.env_config_loader import EnvConfigLoader
from marvin_policy_server.observation import ObservationBuilder, ObservationConfig, ObservationState


def _server_without_node() -> MarvinPolicyServer:
    server = object.__new__(MarvinPolicyServer)
    server.joint_names = ["joint_a", "joint_b"]
    server._cmd_vel = Twist()
    server._latest_imu = None
    server._latest_joint_state = None
    server._locked_command_input = None
    server._locked_imu_input = None
    server._locked_joint_position_relative = None
    server._locked_joint_velocity = None
    server._locked_joint_command = None
    server._locked_estimator_velocity = None
    server._locked_estimator_fk = None
    server._locked_estimator_foot_height = None
    server._locked_estimator_contact = None
    server._command_input_locked = False
    server._imu_input_locked = False
    server._joint_position_input_locked = False
    server._velocity_input_locked = False
    server._joint_command_output_locked = False
    server._estimator_velocity_override_mode = "live"
    server._latest_policy_joint_command = None
    server._last_published_joint_command = None
    server._obs_state = SimpleNamespace(
        lin_vel_b=np.zeros(3),
        joint_pos_history=np.empty((0, 2)),
    )
    server.default_pos = np.zeros(2)
    server.action = np.zeros(2)
    server.policy_path = "/tmp/policy.pt"
    server._env_path = Path("/tmp/env.yaml")
    server._policy_observation_schema = []
    server._base_lin_vel_required = False
    server._base_lin_vel_source = "imu_integration"
    server._base_lin_vel_timeout_sec = 0.25
    server._imu_topic = "/imu"
    server._odometry_topic = "/isaac_odometry"
    server._odometry_child_frame_id = "trunk"
    server._latest_odometry_velocity = None
    server._latest_odometry_received_monotonic = None
    server._odometry_validation_error = None
    server._velocity_estimator = None
    server._velocity_estimator_feature_builder = None
    server._latest_estimated_velocity = None
    server._velocity_estimator_error = None
    server._concurrent_policy_enabled = False
    server._policy_runner = None
    server._joint_position_filter = None
    server._logger = SimpleNamespace(warn=lambda _message: None)
    server._safety_limits_enabled = False
    server._safety_limits_loaded = False
    return server


def test_input_lock_callbacks_snapshot_and_release_values() -> None:
    server = _server_without_node()

    server._cmd_vel.linear.x = 0.4
    assert server._set_policy_input_lock("command", True)[0] is True
    np.testing.assert_allclose(server._locked_command_input, [0.4, 0.0, 0.0])

    imu = Imu()
    imu.orientation.w = 1.0
    imu.angular_velocity.y = 0.25
    imu.linear_acceleration.z = 9.81
    server._latest_imu = imu
    assert server._set_policy_input_lock("imu", True)[0] is True
    np.testing.assert_allclose(server._locked_imu_input["base_ang_vel"], [0.0, 0.25, 0.0])
    np.testing.assert_allclose(server._locked_imu_input["projected_gravity"], [0.0, 0.0, -1.0])

    joint_state = JointState()
    joint_state.name = ["joint_b", "joint_a"]
    joint_state.position = [0.4, 0.3]
    joint_state.velocity = [2.0, 1.0]
    server._latest_joint_state = joint_state
    assert server._set_policy_input_lock("joint_position", True)[0] is True
    np.testing.assert_allclose(
        server._locked_joint_position_relative,
        [0.3, 0.4],
    )
    assert server._set_policy_input_lock("velocity", True)[0] is True
    np.testing.assert_allclose(server._locked_joint_velocity, [1.0, 2.0])

    server._set_policy_input_lock("command", False)
    server._set_policy_input_lock("imu", False)
    server._set_policy_input_lock("joint_position", False)
    server._set_policy_input_lock("velocity", False)
    assert server._locked_command_input is None
    assert server._locked_imu_input is None
    assert server._locked_joint_position_relative is None
    assert server._locked_joint_velocity is None


def test_joint_command_output_can_hold_latest_apply_values_and_release() -> None:
    server = _server_without_node()
    published = []
    server._publish_legacy_commands = True
    server._publish_traced_commands = False
    server._joint_publisher = SimpleNamespace(
        publish=lambda message: published.append(list(message.data))
    )
    server._joint_publisher_traced = None
    server._last_published_joint_command = np.array([0.1, -0.2])

    success, message = server._set_policy_input_lock("joint_command", True)

    assert success is True
    assert "held" in message
    np.testing.assert_allclose(server._locked_joint_command, [0.1, -0.2])
    server._publish_joint_command([0.8, 0.9], trace_id=1)
    np.testing.assert_allclose(published[-1], [0.1, -0.2])
    np.testing.assert_allclose(server._latest_policy_joint_command, [0.8, 0.9])

    assert server._set_policy_input_lock("joint_command", True, [0.3, -0.4])[0] is True
    server._publish_joint_command([0.7, 0.6], trace_id=2)
    np.testing.assert_allclose(published[-1], [0.3, -0.4])

    server._safety_limits_enabled = True
    server._safety_limits_loaded = True
    server._safety_min_pos = np.array([-0.5, -0.5])
    server._safety_max_pos = np.array([0.5, 0.5])
    success, message = server._set_policy_input_lock("joint_command", True, [0.6, 0.0])
    assert success is False
    assert "outside" in message
    np.testing.assert_allclose(server._locked_joint_command, [0.3, -0.4])

    assert server._set_policy_input_lock("joint_command", False)[0] is True
    server._publish_joint_command([0.5, -0.5], trace_id=3)
    np.testing.assert_allclose(published[-1], [0.5, -0.5])


def test_embedded_estimator_velocity_can_hold_zero_follow_odometry_and_release() -> None:
    server = _server_without_node()
    server._concurrent_policy_enabled = True
    server._policy_runner = SimpleNamespace(supports_velocity_override=True)
    server._latest_estimated_velocity = np.array([0.4, -0.2, 0.1])
    server._latest_odometry_velocity = np.array([0.01, 0.02, -0.03])
    server._latest_odometry_received_monotonic = time.monotonic()

    assert server._set_policy_input_lock("estimator_velocity", True)[0] is True
    assert server._estimator_velocity_override_mode == "hold"
    np.testing.assert_allclose(server._effective_estimator_velocity_override(), [0.4, -0.2, 0.1])

    assert server._set_policy_input_lock("estimator_velocity", True, [0.0, 0.0, 0.0])[0] is True
    assert server._estimator_velocity_override_mode == "zero"
    np.testing.assert_allclose(server._effective_estimator_velocity_override(), 0.0)

    assert server._set_policy_input_lock("estimator_velocity", True)[0] is True
    assert server._estimator_velocity_override_mode == "hold"
    np.testing.assert_allclose(server._effective_estimator_velocity_override(), [0.4, -0.2, 0.1])

    assert server._set_policy_input_lock("estimator_velocity_odometry", True)[0] is True
    assert server._estimator_velocity_override_mode == "odometry"
    np.testing.assert_allclose(server._effective_estimator_velocity_override(), [0.01, 0.02, -0.03])

    assert server._set_policy_input_lock("estimator_velocity", False)[0] is True
    assert server._estimator_velocity_override_mode == "live"
    assert server._effective_estimator_velocity_override() is None


def test_embedded_estimator_auxiliary_outputs_can_be_held() -> None:
    """Snapshot auxiliary actor inputs without replacing raw estimates."""
    server = _server_without_node()
    server._concurrent_policy_enabled = True
    server._policy_runner = SimpleNamespace(
        supports_velocity_override=True,
        latest_estimated_state=np.array([
            0.4, -0.2, 0.1,
            0.11, 0.12, 0.13, 0.14,
            0.8, 0.7, 0.6, 0.5,
        ]),
        current_foot_positions=lambda: np.arange(12, dtype=np.float64),
    )

    assert server._set_policy_input_lock("estimator_fk", True)[0] is True
    assert server._set_policy_input_lock(
        "estimator_foot_height", True
    )[0] is True
    assert server._set_policy_input_lock(
        "estimator_contact", True
    )[0] is True
    np.testing.assert_allclose(server._locked_estimator_fk, np.arange(12))
    np.testing.assert_allclose(
        server._locked_estimator_foot_height,
        [0.11, 0.12, 0.13, 0.14],
    )
    np.testing.assert_allclose(
        server._locked_estimator_contact,
        [0.8, 0.7, 0.6, 0.5],
    )

    assert server._set_policy_input_lock(
        "estimator_fk", False
    )[0] is True
    assert server._set_policy_input_lock(
        "estimator_foot_height", False
    )[0] is True
    assert server._set_policy_input_lock(
        "estimator_contact", False
    )[0] is True
    assert server._locked_estimator_fk is None
    assert server._locked_estimator_foot_height is None
    assert server._locked_estimator_contact is None


def test_observation_builder_uses_joint_velocity_override() -> None:
    cfg = ObservationConfig(
        base_lin_vel_history_len=0,
        base_ang_vel_history_len=1,
        projected_gravity_history_len=1,
        velocity_commands_history_len=1,
        joint_pos_history_len=1,
        joint_vel_history_len=1,
        actions_history_len=1,
    )
    builder = ObservationBuilder(["joint_a", "joint_b"], cfg)
    state = ObservationState(
        lin_vel_b=np.zeros(3),
        action_history=np.zeros((1, 2)),
        default_pos=np.zeros(2),
        base_lin_vel_history=np.empty((0, 3)),
        base_ang_vel_history=np.full((1, 3), np.nan),
        projected_gravity_history=np.full((1, 3), np.nan),
        velocity_commands_history=np.full((1, 3), np.nan),
        joint_pos_history=np.full((1, 2), np.nan),
        joint_vel_history=np.full((1, 2), np.nan),
    )
    joint_state = JointState()
    joint_state.name = ["joint_a", "joint_b"]
    joint_state.position = [0.1, 0.2]
    joint_state.velocity = [8.0, 9.0]
    imu = Imu()
    imu.orientation.w = 1.0

    builder.build(
        joint_state,
        imu,
        Twist(),
        0.02,
        state,
        joint_velocity_override=np.array([1.5, 2.5]),
    )

    np.testing.assert_allclose(state.joint_vel_history[-1], [1.5, 2.5])


def test_observation_builder_uses_relative_joint_position_override() -> None:
    """Hold actor joint position without changing estimator input."""
    cfg = ObservationConfig(
        base_lin_vel_history_len=0,
        base_ang_vel_history_len=1,
        projected_gravity_history_len=1,
        velocity_commands_history_len=1,
        joint_pos_history_len=2,
        joint_vel_history_len=1,
        actions_history_len=1,
    )
    builder = ObservationBuilder(["joint_a", "joint_b"], cfg)
    state = ObservationState(
        lin_vel_b=np.zeros(3),
        action_history=np.zeros((1, 2)),
        default_pos=np.array([0.5, -0.5]),
        base_lin_vel_history=np.empty((0, 3)),
        base_ang_vel_history=np.full((1, 3), np.nan),
        projected_gravity_history=np.full((1, 3), np.nan),
        velocity_commands_history=np.full((1, 3), np.nan),
        joint_pos_history=np.full((2, 2), np.nan),
        joint_vel_history=np.full((1, 2), np.nan),
    )
    joint_state = JointState()
    joint_state.name = ["joint_a", "joint_b"]
    joint_state.position = [9.0, 9.0]
    joint_state.velocity = [1.0, 2.0]
    imu = Imu()
    imu.orientation.w = 1.0

    builder.build(
        joint_state,
        imu,
        Twist(),
        0.02,
        state,
        joint_position_relative_override=np.array([0.1, -0.2]),
    )

    np.testing.assert_allclose(
        state.joint_pos_history,
        [[0.1, -0.2], [0.1, -0.2]],
    )


def test_observation_builder_uses_base_linear_velocity_override() -> None:
    cfg = ObservationConfig(
        base_lin_vel_history_len=1,
        base_ang_vel_history_len=1,
        projected_gravity_history_len=1,
        velocity_commands_history_len=1,
        joint_pos_history_len=1,
        joint_vel_history_len=1,
        actions_history_len=1,
    )
    builder = ObservationBuilder(["joint_a", "joint_b"], cfg)
    state = ObservationState(
        lin_vel_b=np.zeros(3),
        action_history=np.zeros((1, 2)),
        default_pos=np.zeros(2),
        base_lin_vel_history=np.full((1, 3), np.nan),
        base_ang_vel_history=np.full((1, 3), np.nan),
        projected_gravity_history=np.full((1, 3), np.nan),
        velocity_commands_history=np.full((1, 3), np.nan),
        joint_pos_history=np.full((1, 2), np.nan),
        joint_vel_history=np.full((1, 2), np.nan),
    )
    joint_state = JointState()
    joint_state.name = ["joint_a", "joint_b"]
    joint_state.position = [0.0, 0.0]
    joint_state.velocity = [0.0, 0.0]
    imu = Imu()
    imu.orientation.w = 1.0
    imu.linear_acceleration.x = 99.0

    builder.build(
        joint_state,
        imu,
        Twist(),
        0.02,
        state,
        base_lin_vel_override=np.array([1.0, -2.0, 0.5]),
    )

    np.testing.assert_allclose(state.base_lin_vel_history[-1], [1.0, -2.0, 0.5])
    np.testing.assert_allclose(state.lin_vel_b, [1.0, -2.0, 0.5])


def test_odometry_source_requires_fresh_correctly_framed_sample() -> None:
    server = _server_without_node()
    server._base_lin_vel_required = True
    server._base_lin_vel_source = "odometry"

    ready, message = server._base_lin_vel_readiness()
    assert ready is False
    assert "/isaac_odometry" in message

    odometry = Odometry()
    odometry.child_frame_id = "trunk"
    odometry.twist.twist.linear.x = 1.25
    odometry.twist.twist.linear.y = -0.5
    odometry.twist.twist.linear.z = 0.1
    server._odometry_cb(odometry)

    ready, _ = server._base_lin_vel_readiness(server._latest_odometry_received_monotonic)
    assert ready is True
    np.testing.assert_allclose(server._current_base_lin_vel(), [1.25, -0.5, 0.1])

    stale_time = server._latest_odometry_received_monotonic + server._base_lin_vel_timeout_sec + 0.01
    ready, message = server._base_lin_vel_readiness(stale_time)
    assert ready is False
    assert "stale" in message

    odometry.child_frame_id = "base_link"
    server._odometry_cb(odometry)
    ready, message = server._base_lin_vel_readiness()
    assert ready is False
    assert "child_frame_id" in message

    odometry.child_frame_id = "trunk"
    odometry.twist.twist.linear.x = float("nan")
    server._odometry_cb(odometry)
    ready, message = server._base_lin_vel_readiness()
    assert ready is False
    assert "non-finite" in message


def test_activation_rejects_required_missing_odometry() -> None:
    server = _server_without_node()
    server._base_lin_vel_required = True
    server._base_lin_vel_source = "odometry"
    response = server._set_active_cb(SetBool.Request(data=True), SetBool.Response())

    assert response.success is False
    assert "activation rejected" in response.message.lower()
    assert "/isaac_odometry" in response.message


def test_estimator_source_reports_loaded_warmup_and_supplies_prediction() -> None:
    server = _server_without_node()
    server._base_lin_vel_required = True
    server._base_lin_vel_source = "estimator"
    server._velocity_estimator = SimpleNamespace()
    server._velocity_estimator_feature_builder = SimpleNamespace()

    ready, message = server._base_lin_vel_readiness()
    assert ready is True
    assert "initialize" in message

    server._latest_estimated_velocity = np.array([0.8, -0.2, 0.05])
    ready, message = server._base_lin_vel_readiness()
    assert ready is True
    assert "learned velocity estimator" in message
    np.testing.assert_allclose(server._base_lin_vel_override(), [0.8, -0.2, 0.05])
    np.testing.assert_allclose(server._current_base_lin_vel(), [0.8, -0.2, 0.05])


def test_estimator_source_rejects_missing_or_invalid_estimator() -> None:
    server = _server_without_node()
    server._base_lin_vel_required = True
    server._base_lin_vel_source = "estimator"

    ready, message = server._base_lin_vel_readiness()
    assert ready is False
    assert "not loaded" in message

    server._velocity_estimator = SimpleNamespace()
    server._velocity_estimator_feature_builder = SimpleNamespace()
    server._velocity_estimator_error = "bad feature frame"
    ready, message = server._base_lin_vel_readiness()
    assert ready is False
    assert "bad feature frame" in message

    server._velocity_estimator_error = None
    server._latest_estimated_velocity = np.array([float("nan"), 0.0, 0.0])
    ready, message = server._base_lin_vel_readiness()
    assert ready is False
    assert "invalid" in message


def test_typed_input_lock_service_acknowledges_applied_values() -> None:
    server = _server_without_node()
    request = SetPolicyInputLock.Request()
    request.input_name = "command"
    request.locked = True
    request.use_latest = False
    request.values = [0.2, -0.1, 0.3]

    response = server._set_policy_input_lock_cb(request, SetPolicyInputLock.Response())

    assert response.success is True
    assert response.locked is True
    np.testing.assert_allclose(response.values, [0.2, -0.1, 0.3])
    np.testing.assert_allclose(response.live_values, [0.0, 0.0, 0.0])


def test_env_loader_returns_policy_observation_schema_in_file_order(tmp_path: Path) -> None:
    env_path = tmp_path / "env.yaml"
    env_path.write_text(
        "observations:\n"
        "  policy:\n"
        "    history_length: null\n"
        "    custom_term:\n"
        "      func: package.module:custom_observation\n"
        "      history_length: 4\n"
        "    velocity_commands:\n"
        "      func: isaaclab.envs.mdp.observations:generated_commands\n"
        "      history_length: 0\n",
        encoding="utf-8",
    )

    assert EnvConfigLoader(env_path).get_policy_observation_schema() == [
        {
            "key": "custom_term",
            "func": "package.module:custom_observation",
            "history_length": 4,
        },
        {
            "key": "velocity_commands",
            "func": "isaaclab.envs.mdp.observations:generated_commands",
            "history_length": 1,
        },
    ]
