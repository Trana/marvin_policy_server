from types import SimpleNamespace

from marvin_policy_server.input_timing import InputTiming
from marvin_policy_server.timing import AGE_BOUNDS_MS, Timing


def message(ns):
    return SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(
        sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)))


def test_distributions_are_bounded_and_reset_with_window():
    timing = Timing(clock=lambda: 0)
    for _ in range(10000):
        timing.observe_ms('age', 6)
    timing.observe_ms('overflow', 1001)
    for value in (-1, float('nan'), float('inf')):
        timing.observe_ms('age', value)
    result = timing.snapshot()
    age = result['distributions_ms']['age']
    assert age['count'] == 10000
    assert age['mean'] == age['max'] == 6
    assert age['p50_upper_ms'] == age['p95_upper_ms'] == age['p99_upper_ms'] == 7.5
    assert len(age['bucket_counts']) == len(AGE_BOUNDS_MS) + 1
    assert sum(age['bucket_counts']) == 10000
    assert result['distributions_ms']['overflow']['p99_upper_ms'] is None
    assert result['counts']['age_invalid'] == 3
    assert timing.snapshot()['distributions_ms'] == {}


def test_header_age_and_callback_age_use_separate_clocks():
    now = [0]
    timing = Timing(clock=lambda: now[0])
    inputs = InputTiming(timing, clock=lambda: now[0])
    inputs.received('joint')
    now[0] = 5_000_000
    inputs.inference(1_020_000_000, {'joint': message(1_000_000_000)})
    now[0] = 25_000_000
    inputs.inference(1_040_000_000, {'joint': message(1_000_000_000)})
    result = timing.snapshot()
    assert result['distributions_ms']['joint_header_age']['mean'] == 30
    assert result['distributions_ms']['joint_callback_age']['mean'] == 15
    assert result['distributions_ms']['inference_interval']['mean'] == 20
    assert result['counts']['joint_callback_sample_reused'] == 1
    inputs.received('joint')
    inputs.reset_interval()
    inputs.inference(1_040_000_000, {'joint': message(1_000_000_000)})
    result = timing.snapshot()
    assert 'inference_interval' not in result['distributions_ms']
    assert 'joint_callback_sample_reused' not in result['counts']


def test_missing_future_and_overridden_inputs_are_explicit():
    timing = Timing(clock=lambda: 0)
    inputs = InputTiming(timing, clock=lambda: 0)
    inputs.inference(100, {'joint': message(0), 'imu': message(200)})
    inputs.inference(100, {'joint': None}, overridden=['joint'])
    result = timing.snapshot()
    assert result['counts'] == {
        'joint_header_age_missing': 1, 'joint_receipt_missing': 1,
        'imu_header_age_invalid': 1, 'imu_receipt_missing': 1,
        'joint_input_overridden': 1,
    }


def test_off_skips_clocks_and_message_access():
    timing = Timing('off', clock=lambda: 0)

    def forbidden():
        raise AssertionError('clock called with instrumentation off')

    inputs = InputTiming(timing, clock=forbidden)
    inputs.received('joint')
    inputs.inference(None, {'joint': None})
    timing.observe_ms('age', None)
    assert not inputs.receipts
    assert not timing.histograms
