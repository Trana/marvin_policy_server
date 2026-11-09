import importlib.util
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location(
    'guard', Path(__file__).parents[1] / 'scripts/joint_test_capture_guard.py'
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sample(seq, now):
    return SimpleNamespace(
        controller_instance_id='instance',
        activation_generation=1,
        joint_names=['joint'],
        positions=[0.2],
        update_sequence=seq,
        sampled_at_monotonic_s=now,
    )


def primed():
    guard = module.CaptureGuard('instance', 1, ['joint'])
    for seq, now in [(1, 1.0), (2, 1.01)]:
        assert guard.target(sample(seq, now), now)
        assert guard.feedback(['joint'], [0.2], now)
    return guard


def test_requires_persisted_fresh_data():
    guard = primed()
    assert not guard.ready(1.01)
    guard.committed(1.01)
    assert guard.ready(1.02)
    assert not guard.ready(1.2)


def test_live_freshness_is_independent_of_disk_commit():
    guard = primed()
    assert guard.live_ready(1.02)
    assert not guard.ready(1.02)  # Legacy combined check still needs a commit.
    guard.committed(1.20)
    assert not guard.live_ready(1.20)  # A new commit cannot revive old live data.


def test_live_freshness_uses_target_source_time_not_only_arrival():
    guard = primed()
    guard.target(sample(3, 1.02), 1.16)
    guard.feedback(['joint'], [0.2], 1.16)
    assert guard.live_ready(1.16)
    assert not guard.live_ready(1.18)


def test_sequence_gap_latches_and_counts_missing():
    guard = primed()
    assert not guard.target(sample(5, 1.02), 1.02)
    assert guard.missing == 2
    guard.committed(1.02)
    assert not guard.ready(1.02)


def test_wrong_identity_and_invalid_feedback():
    guard = primed()
    changed = sample(3, 1.02)
    changed.activation_generation = 2
    assert not guard.target(changed, 1.02)
    assert not primed().feedback(['joint'], [float('nan')], 1.02)


def test_stale_and_regressing_source_time():
    assert not primed().target(sample(3, 0.8), 1.02)
    assert not primed().target(sample(3, 1.0), 1.02)
