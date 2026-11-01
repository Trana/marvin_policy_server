import pytest

from marvin_policy_server.timing import Timing


def test_capture_expires_into_metrics_without_resetting_on_snapshot():
    now = [0]
    timing = Timing('capture', 2.0, clock=lambda: now[0])
    assert timing.capture_enabled()
    now[0] = 1_000_000_000
    timing.snapshot()
    now[0] = 2_000_000_000
    assert not timing.capture_enabled()
    assert timing.mode == 'metrics'


def test_windowed_metrics_preserve_tick_gap_across_windows():
    now = [0]
    timing = Timing(clock=lambda: now[0])
    timing.tick(5_000_000)
    now[0] = 5_000_000
    timing.tick(5_000_000)
    first = timing.snapshot()
    assert first['counts']['ticks'] == 2
    assert first['durations_ms']['tick_interval']['mean'] == 5
    now[0] = 15_000_000
    timing.tick(5_000_000)
    second = timing.snapshot()
    assert second['counts']['ticks'] == 1
    assert second['counts']['late_tick_intervals'] == 1


@pytest.mark.parametrize('mode,seconds', [('invalid', 30), ('capture', 0),
                                         ('metrics', float('nan')), ('capture', 301)])
def test_invalid_configuration_rejected(mode, seconds):
    with pytest.raises(ValueError):
        Timing(mode, seconds)


def test_off_never_enables_capture():
    assert not Timing('off').capture_enabled()


def test_nested_checkpoint_preserves_parent_stage():
    now = [0]
    timing = Timing(clock=lambda: now[0])
    timing.tick(5_000_000)
    now[0] = 1_000_000
    assert timing.checkpoint('publish_safety_limits', 0) == 1_000_000
    now[0] = 3_000_000
    timing.stage('command_publish')
    result = timing.snapshot()['durations_ms']
    assert result['publish_safety_limits']['mean'] == 1
    assert result['command_publish']['mean'] == 3


@pytest.mark.parametrize('mode', ['off', 'metrics', 'capture'])
@pytest.mark.parametrize('legacy,locked', [(False, False), (True, True)])
def test_publish_profiling_preserves_commands(mode, legacy, locked):
    from types import SimpleNamespace
    import numpy as np
    from builtin_interfaces.msg import Time
    from marvin_policy_server.marvin_policy_server import MarvinPolicyServer
    timing = Timing(mode)
    if mode == 'off':
        timing.clock = lambda: pytest.fail('off publication read timing clock')
        assert timing.checkpoint('unused', 42) == 42
    traced, old = [], []
    node = SimpleNamespace(
        _timing=timing, _apply_safety_position_limits=lambda values: [0.5, -0.5],
        _joint_command_output_locked=locked, _locked_joint_command=np.array([0.1, 0.2]),
        _publish_legacy_commands=legacy, _joint_publisher=SimpleNamespace(publish=old.append),
        _publish_traced_commands=True, _joint_publisher_traced=SimpleNamespace(publish=traced.append),
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: Time(sec=7))),
        _publish_trace_event=lambda *args, **kwargs: None,
    )
    assert MarvinPolicyServer._publish_joint_command(node, [1., -1.], trace_id=42,
                                                     emit_trace_event=True) == 42
    expected = [0.1, 0.2] if locked else [0.5, -0.5]
    assert list(traced[0].data) == expected
    assert traced[0].trace_id == 42 and traced[0].stamp.sec == 7
    assert len(old) == int(legacy)
    if legacy:
        assert list(old[0].data) == expected
    np.testing.assert_array_equal(node._latest_policy_joint_command, [0.5, -0.5])
    np.testing.assert_array_equal(node._last_published_joint_command, expected)
    if mode == 'off':
        assert timing.durations == {} and timing.counters == {}
    else:
        assert timing.counters['commands'] == 1
        assert {'publish_safety_limits', 'publish_array_copies', 'publish_traced_build',
                'publish_traced_ros', 'publish_trace_event'} <= timing.durations.keys()
        assert ('publish_legacy_ros' in timing.durations) == legacy
    # Snapshots, caller input and successive messages must never alias.
    first_snapshot = node._latest_policy_joint_command
    first_message = traced[0]
    first_message.data[0] = 99.
    np.testing.assert_array_equal(node._last_published_joint_command, expected)
    if legacy:
        assert list(old[0].data) == expected
    MarvinPolicyServer._publish_joint_command(node, [2., -2.], trace_id=43)
    assert traced[1] is not first_message
    assert list(traced[1].data) == expected
    node._latest_policy_joint_command[0] = 7.
    np.testing.assert_array_equal(first_snapshot, [0.5, -0.5])


def test_typed_command_array_preserves_wire_representation():
    from array import array
    from rclpy.serialization import serialize_message
    from marvin_trace_msgs.msg import TracedFloat64MultiArray
    from std_msgs.msg import Float64MultiArray
    for message_type in (TracedFloat64MultiArray, Float64MultiArray):
        for values in ([0., -0., 1.25, -3.5], [float('inf'), float('-inf'), float('nan')]):
            original, prepared = message_type(), message_type()
            original.data = values
            prepared.data = array('d', values)
            assert serialize_message(original) == serialize_message(prepared)


def test_callback_stages_budget_and_between_callbacks():
    now = [0]
    timing = Timing(clock=lambda: now[0])
    started = timing.tick(5_000_000)
    now[0] = 2_000_000
    timing.stage('observation_build')
    now[0] = 6_000_000
    timing.stage('inference_stage')
    now[0] = 7_000_000
    timing.finish_callback(started, 5_000_000)
    result = timing.snapshot()
    assert result['counts']['callback_over_budget'] == 1
    durations = result['durations_ms']
    assert durations['callback']['mean'] == 7
    assert durations['observation_build']['mean'] == 2
    assert durations['inference_stage']['mean'] == 4
    assert durations['callback_tail']['mean'] == 1
    # End-to-next-start is not claimed to be scheduler latency: other callbacks
    # and deliberate timer waiting also occupy this interval.
    now[0] = 10_000_000
    started = timing.tick(5_000_000)
    now[0] = 11_000_000
    timing.finish_callback(started, 5_000_000)
    result = timing.snapshot()
    assert result['durations_ms']['between_callbacks']['mean'] == 3
    assert result['durations_ms']['callback_tail']['mean'] == 1
    assert 'callback_over_budget' not in result['counts']


def test_off_stage_does_not_read_clock_or_allocate_metrics():
    timing = Timing('off')
    timing.clock = lambda: pytest.fail('off stage read the clock')
    timing.stage('observation_build')
    assert timing.durations == {}


@pytest.mark.parametrize('raises', [False, True])
def test_callback_wrapper_records_returns_and_exceptions(raises):
    from types import SimpleNamespace
    from marvin_policy_server.marvin_policy_server import MarvinPolicyServer
    now = [0]
    timing = Timing(clock=lambda: now[0])

    def callback():
        now[0] = 6_000_000
        if raises:
            raise RuntimeError('test failure')
        return 'early return'

    node = SimpleNamespace(_timing=timing, _expected_tick_ns=5_000_000,
                           _timer_tick_impl=callback)
    if raises:
        with pytest.raises(RuntimeError, match='test failure'):
            MarvinPolicyServer._timer_tick(node)
    else:
        assert MarvinPolicyServer._timer_tick(node) == 'early return'
    result = timing.snapshot()
    assert result['durations_ms']['callback']['mean'] == 6
    assert result['durations_ms']['callback_tail']['mean'] == 6
    assert result['counts']['callback_over_budget'] == 1


def test_callback_wrapper_off_bypasses_timing():
    from types import SimpleNamespace
    from marvin_policy_server.marvin_policy_server import MarvinPolicyServer
    timing = Timing('off')
    timing.clock = lambda: pytest.fail('off callback read the timing clock')
    node = SimpleNamespace(_timing=timing, _timer_tick_impl=lambda: 42)
    assert MarvinPolicyServer._timer_tick(node) == 42
    assert timing.counters == {}


@pytest.mark.parametrize('mode,expected', [('off', 0), ('metrics', 0), ('capture', 1)])
def test_trace_event_gate(mode, expected):
    from types import SimpleNamespace
    from marvin_policy_server.marvin_policy_server import MarvinPolicyServer
    messages = []
    node = SimpleNamespace(
        _timing=Timing(mode), _publish_trace_events=True,
        _trace_event_publisher=SimpleNamespace(publish=messages.append),
    )
    MarvinPolicyServer._publish_trace_event(node, 42, 'infer_start', stamp_ns=123)
    assert len(messages) == expected
