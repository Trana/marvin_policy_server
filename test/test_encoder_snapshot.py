from types import SimpleNamespace as NS

import pytest

from marvin_policy_server.encoder_snapshot import decode_snapshot
from marvin_policy_server.input_timing import InputTiming
from marvin_policy_server.timing import Timing


def snapshot():
    return NS(header=object(), joint_names=['b', 'a'], interface_values=[
        NS(interface_names=['velocity', 'encoder_receipt_monotonic_s', 'position'],
           values=[2.0, 0.98, 1.0]),
        NS(interface_names=['position', 'velocity', 'encoder_receipt_monotonic_s'],
           values=[3.0, 4.0, 0.99]),
    ])


def test_values_and_receipts_share_snapshot_and_policy_order():
    msg = snapshot()
    state, receipts = decode_snapshot(msg, ['a', 'b'], NS)
    assert state.header is msg.header
    assert state.name == ['a', 'b']
    assert state.position == [3, 1]
    assert state.velocity == [4, 2]
    assert receipts == {'a': 0.99, 'b': 0.98}


@pytest.mark.parametrize('fault', ['duplicate', 'missing', 'length', 'nan', 'interfaces'])
def test_malformed_control_values_are_rejected(fault):
    msg = snapshot()
    if fault == 'duplicate':
        msg.joint_names = ['a', 'a']
    elif fault == 'missing':
        msg.joint_names = ['c', 'a']
    elif fault == 'length':
        msg.interface_values.pop()
    elif fault == 'nan':
        msg.interface_values[0].values[0] = float('nan')
    else:
        msg.interface_values[0].interface_names[0] = 'position'
    with pytest.raises(ValueError):
        decode_snapshot(msg, ['a', 'b'], NS)


def test_age_reuse_and_identity_check():
    timing = Timing(clock=lambda: 0)
    inputs = InputTiming(timing, clock=lambda: 1_000_000_000)
    state, receipts = decode_snapshot(snapshot(), ['a', 'b'], NS)
    inputs.received_encoders(state, receipts)
    inputs.encoder_inference(state)
    inputs.encoder_inference(state)
    inputs.encoder_inference(NS())
    result = timing.snapshot()
    assert result['distributions_ms']['encoder_oldest_age_at_inference']['mean'] == pytest.approx(20)
    assert result['distributions_ms']['encoder_receipt_spread_at_inference']['mean'] == pytest.approx(10)
    assert result['counts']['encoder_a_sample_reused'] == 1
    assert result['counts']['encoder_snapshot_mismatch'] == 1
    inputs.reset_interval()
    inputs.encoder_inference(state)
    assert 'encoder_a_sample_reused' not in timing.snapshot()['counts']


def test_missing_future_regression_and_off():
    timing = Timing(clock=lambda: 0)
    inputs = InputTiming(timing, clock=lambda: 1_000_000_000)
    state = NS()
    inputs.received_encoders(state, {'a': .99})
    inputs.encoder_inference(state)
    inputs.received_encoders(state, {'a': .98, 'b': float('nan'), 'c': 2.0})
    inputs.encoder_inference(state)
    counts = timing.snapshot()['counts']
    assert counts['encoder_a_receipt_regressed'] == 1
    assert counts['encoder_b_receipt_invalid'] == 1
    assert counts['encoder_c_receipt_invalid'] == 1
    assert counts['encoder_inference_incomplete'] == 1
    timing.mode = 'off'
    inputs.clock = lambda: pytest.fail('off accessed clock')
    inputs.encoder_inference(None)


def test_missing_timestamp_does_not_discard_control_values():
    msg = snapshot()
    msg.interface_values[1].interface_names.pop()
    msg.interface_values[1].values.pop()
    state, receipts = decode_snapshot(msg, ['a', 'b'], NS)
    assert state.position == [3, 1]
    assert receipts['a'] != receipts['a']  # NaN


def test_ros_wire_snapshot_reaches_policy_callback_without_hardware():
    pytest.importorskip('torch')
    pytest.importorskip('rclpy')
    from control_msgs.msg import DynamicJointState, InterfaceValue
    from rclpy.serialization import serialize_message, deserialize_message
    from marvin_policy_server.marvin_policy_server import MarvinPolicyServer

    message = DynamicJointState()
    message.header.stamp.sec = 1
    message.joint_names = ['b', 'a']
    message.interface_values = [
        InterfaceValue(interface_names=['position', 'velocity', 'encoder_receipt_monotonic_s'],
                       values=[1.0, 2.0, 0.98]),
        InterfaceValue(interface_names=['position', 'velocity', 'encoder_receipt_monotonic_s'],
                       values=[3.0, 4.0, 0.99]),
    ]
    server = object.__new__(MarvinPolicyServer)
    server.joint_names = ['a', 'b']
    server._timing = Timing(clock=lambda: 0)
    server._input_timing = InputTiming(server._timing, clock=lambda: 1_000_000_000)
    server._latest_joint_state = None
    decoded = deserialize_message(serialize_message(message), DynamicJointState)
    server._dynamic_joint_state_cb(decoded)
    state = server._latest_joint_state
    assert list(state.position) == [3.0, 1.0]
    server._input_timing.encoder_inference(state)
    result = server._timing.snapshot()
    assert result['distributions_ms']['encoder_oldest_age_at_inference']['mean'] == pytest.approx(20)
    # No fallback to the earlier valid state after a malformed snapshot.
    decoded.interface_values.pop()
    server._dynamic_joint_state_cb(decoded)
    assert server._latest_joint_state is None
    assert server._timing.snapshot()['counts']['encoder_snapshot_invalid'] == 1
