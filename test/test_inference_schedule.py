import pytest

from marvin_policy_server.inference_schedule import InferenceSchedule


def test_normal_and_delayed_callbacks_preserve_deadlines():
    schedule = InferenceSchedule(20_000_000)
    assert schedule.due(0)
    schedule.complete(0, 6_000_000)
    assert not schedule.due(10_000_000)
    assert not schedule.due(15_000_000)
    assert schedule.due(25_000_000)  # Late, but next deadline stays 40 ms.
    assert schedule.observation_dt(25_000_000) == .025
    schedule.complete(25_000_000, 31_000_000)
    assert schedule.deadline_ns == 40_000_000
    assert schedule.due(40_000_000)


def test_long_work_skips_expired_deadlines_without_burst():
    schedule = InferenceSchedule(20_000_000)
    assert schedule.due(0)
    schedule.complete(0, 65_000_000)
    assert schedule.deadline_ns == 80_000_000
    assert not schedule.due(65_000_000)
    assert not schedule.due(70_000_000)
    assert schedule.due(80_000_000)
    schedule.complete(80_000_000, 80_000_000)
    assert not schedule.due(80_000_000)


def test_paused_sim_clock_reset_and_reactivation():
    schedule = InferenceSchedule(20_000_000)
    assert schedule.due(100_000_000)
    assert schedule.observation_dt(100_000_000) == .02
    schedule.complete(100_000_000, 100_000_000)
    assert not schedule.due(100_000_000)
    assert schedule.due(0)  # Backward sim-clock jump.
    assert schedule.observation_dt(0) == .02
    schedule.complete(0, 0)
    schedule.reset()  # Inactive/activation boundary.
    assert schedule.due(5_000_000)
    assert schedule.observation_dt(5_000_000) == .02


def test_forward_jump_and_clock_reset_during_inference():
    schedule = InferenceSchedule(20_000_000)
    assert schedule.due(0)
    schedule.complete(0, 1)
    assert schedule.due(10_000_000_000)
    schedule.complete(10_000_000_000, 10_001_000_000)
    assert schedule.deadline_ns == 10_020_000_000
    schedule.complete(10_020_000_000, 0)
    assert schedule.due(0)
    assert schedule.observation_dt(0) == .02


def test_invalid_period():
    for value in (0, -1, .02):
        with pytest.raises(ValueError):
            InferenceSchedule(value)


def test_real_ros_simulated_clock_pause_and_rewind(monkeypatch):
    rclpy = pytest.importorskip('rclpy')
    import time
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.parameter import Parameter
    from rosgraph_msgs.msg import Clock

    # Never publish a test clock onto the robot or workstation runtime domain.
    monkeypatch.setenv('ROS_DOMAIN_ID', '211')
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    context = Context()
    rclpy.init(context=context)
    node = rclpy.create_node(
        'inference_schedule_clock_test', context=context,
        parameter_overrides=[Parameter('use_sim_time', value=True)])
    publisher = node.create_publisher(Clock, '/clock', 10)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    schedule = InferenceSchedule(20_000_000)

    def set_time(ns):
        message = Clock()
        message.clock.sec, message.clock.nanosec = divmod(ns, 1_000_000_000)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            publisher.publish(message)
            executor.spin_once(timeout_sec=.02)
            if node.get_clock().now().nanoseconds == ns:
                return
        pytest.fail('ROS simulated clock did not update')

    try:
        set_time(1_000_000_000)
        assert schedule.due(node.get_clock().now().nanoseconds)
        schedule.complete(1_000_000_000, node.get_clock().now().nanoseconds)
        # Wall time advances while simulation is paused: no new inference due.
        for _ in range(5):
            executor.spin_once(timeout_sec=.02)
            assert not schedule.due(node.get_clock().now().nanoseconds)
        set_time(1_020_000_000)
        assert schedule.due(node.get_clock().now().nanoseconds)
        schedule.complete(1_020_000_000, node.get_clock().now().nanoseconds)
        set_time(500_000_000)
        assert schedule.due(node.get_clock().now().nanoseconds)
        assert schedule.observation_dt(500_000_000) == .02
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown(context=context)


def test_runner_explicit_decision_overrides_callback_counter():
    torch = pytest.importorskip('torch')
    import numpy as np
    from marvin_policy_server.policy_runner import PolicyRunner, PolicyRunnerConfig
    calls = []

    def policy(obs):
        calls.append(obs)
        return torch.ones((1, 2)) * len(calls)

    runner = PolicyRunner(policy, 2, PolicyRunnerConfig())
    assert np.all(runner.step(None, infer=False) == 0)  # Counter 0 would infer.
    assert np.all(runner.step(np.zeros(3), infer=True) == 1)
    assert np.all(runner.step(np.zeros(3), infer=True) == 2)
    assert np.all(runner.step(None, infer=False) == 2)
    assert len(calls) == 2


@pytest.mark.parametrize('permission', [True, False])
def test_node_tick_uses_deadlines_and_preserves_stale_input_guard(permission):
    pytest.importorskip('torch')
    pytest.importorskip('rclpy')
    import numpy as np
    from types import SimpleNamespace as NS
    from marvin_policy_server.marvin_policy_server import MarvinPolicyServer
    from marvin_policy_server.operator_permission import OperatorPermission
    from marvin_policy_server.policy_runner import PolicyRunner, PolicyRunnerConfig

    now = [1_000_000_000]
    observations, published = [], []
    server = object.__new__(MarvinPolicyServer)
    server._operator_permission = OperatorPermission(required=True, clock=lambda: 0.)
    server._operator_permission.update(permission)
    server._timing = None
    server._inference_schedule = InferenceSchedule(20_000_000)
    server.get_clock = lambda: NS(now=lambda: NS(nanoseconds=now[0]))
    sensor = NS(header=NS(stamp=NS(sec=1, nanosec=0)))
    server._latest_joint_state = server._latest_imu = sensor
    server._stale_warn_count = 0
    server._activation_mgr = NS(is_active=lambda: True, compute_activation_factor=lambda t: 1.)
    server._update_velocity_estimator = lambda *args: None
    server._base_lin_vel_readiness = lambda: (True, '')
    server._next_trace_id = lambda: 1
    server._publish_trace_event = lambda *args, **kwargs: None
    server._command_input_locked = True
    server._locked_command_input = np.array([.5, .1, .2])
    server._imu_input_locked = False
    server._joint_position_input_locked = server._velocity_input_locked = False
    server._joint_position_filter = None
    server._base_lin_vel_override = lambda: None
    server._cmd_vel = None
    server._obs_state = NS(action_history=np.empty((0, 2)))
    def build(*args, **kwargs):
        expected = server._locked_command_input if permission else np.zeros(3)
        np.testing.assert_array_equal(kwargs['command_override'], expected)
        observations.append(args[3])
        return np.zeros(3)

    server._obs_builder = NS(build=build)
    server._concurrent_policy_feature_builder = None
    server._policy_runner = PolicyRunner(lambda obs: obs[:, :2], 2, PolicyRunnerConfig())
    server._record_velocity_estimator_diagnostic = lambda action: None
    server._map_action_to_joint_positions = lambda action: action
    server._publish_joint_command = lambda action, **kwargs: published.append(action)
    server._logger = NS(warn=lambda message: None)
    for ms in (0, 10, 15, 25, 35, 40):
        now[0] = 1_000_000_000 + ms * 1_000_000
        server._timer_tick_impl()
    assert observations == pytest.approx([.02, .025, .015])
    assert len(published) == 6
    now[0] = 2_000_000_000  # Header now stale: neither infer nor publish.
    server._timer_tick_impl()
    assert len(observations) == 3 and len(published) == 6
    assert server._inference_schedule.deadline_ns is None
