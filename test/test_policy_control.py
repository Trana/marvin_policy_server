from copy import deepcopy
from uuid import uuid4

from marvin_policy_server.policy_control import PolicyControl
from marvin_policy_server.policy_control import PolicyControlError
import pytest


def fixture(limit=1024):
    lock = {
        'locked': False,
        'labels': ['x', 'y', 'yaw'],
        'values': [1.0, 2.0, 3.0],
        'live_values': [1.0, 2.0, 3.0],
    }
    calls = []

    def observe():
        return {
            'policy_path': 'policy.pt',
            'env_path': 'env.yaml',
            'observations': [{'key': 'velocity_commands'}],
            'locks': {'command': deepcopy(lock)},
        }

    def apply(control, locked, values):
        calls.append((control, locked, values))
        lock['locked'] = locked
        if values is not None:
            lock['values'] = values
        return True, 'Applied'

    owner = PolicyControl(observe, apply, limit=limit)
    request = {
        'command_id': str(uuid4()),
        'session_id': owner.session_id,
        'expected_revision': 0,
        'control': 'command',
        'mode': 'hold',
        'values': [],
    }
    return owner, request, calls, lock


def test_hold_retry_is_exactly_once_and_returns_original_applied_values():
    owner, request, calls, lock = fixture()
    result = owner.submit(request)
    assert result['status'] == 'applied'
    assert result['applied_revision'] == 1
    lock['values'] = [4.0, 5.0, 6.0]
    assert owner.submit(request) == result
    assert owner.get(owner.session_id, request['command_id']) == result
    assert len(calls) == 1
    with pytest.raises(PolicyControlError, match='different arguments'):
        owner.submit({**request, 'mode': 'live'})


def test_new_session_does_not_adopt_or_replay_old_commands():
    owner, request, calls, _ = fixture()
    owner.submit(request)
    replacement, _, _, _ = fixture()
    assert replacement.session_id != owner.session_id
    assert replacement.state()['controls']['command']['mode'] == 'live'
    with pytest.raises(PolicyControlError, match='session changed'):
        replacement.submit(request)


def test_loaded_artifacts_are_optional_and_copied_with_session():
    owner, _, _, _ = fixture()
    assert owner.state()['loaded_artifacts'] is None
    receipt = {'policy.pt': {'path': '/policy.pt', 'sha256': 'a' * 64, 'size': 10}}
    owner = PolicyControl(
        lambda: {
            'policy_path': '/policy.pt',
            'env_path': '/env.yaml',
            'loaded_artifacts': receipt,
            'observations': [],
            'locks': {},
        },
        lambda *args: None,
    )
    state = owner.state()
    assert state['session_id'] == owner.session_id
    assert state['loaded_artifacts'] == receipt
    state['loaded_artifacts']['policy.pt']['size'] = 20
    assert owner.state()['loaded_artifacts']['policy.pt']['size'] == 10


@pytest.mark.parametrize(
    'changes',
    [
        {'expected_revision': 5},
        {'mode': 'unsupported'},
        {'control': 'missing'},
        {'mode': 'fixed', 'values': [1.0]},
        {'mode': 'fixed', 'values': [1.0, float('nan'), 2.0]},
        {'mode': 'hold', 'values': [1.0, 2.0, 3.0]},
        {'mode': 'odometry'},
    ],
)
def test_invalid_or_stale_requests_do_not_mutate(changes):
    owner, request, calls, _ = fixture()
    assert owner.submit({**request, **changes})['status'] == 'rejected'
    assert calls == []
    assert owner.revision == 0


def test_legacy_writes_invalidate_revision_and_fixed_mode_is_explicit():
    owner, request, calls, _ = fixture()
    owner.changed('command', True, [1, 2, 3])
    assert owner.state()['controls']['command']['mode'] == 'fixed'
    assert owner.submit(request)['status'] == 'rejected'
    assert calls == []


def test_journal_is_bounded_without_eviction_and_reads_do_not_release():
    owner, request, calls, _ = fixture(limit=1)
    result = owner.submit(request)
    for _ in range(10):
        assert owner.state()['controls']['command']['mode'] == 'hold'
    with pytest.raises(PolicyControlError, match='journal full'):
        owner.submit({**request, 'command_id': str(uuid4()), 'expected_revision': 1})
    assert owner.submit(request) == result
    assert len(calls) == 1


def test_ambiguous_mutation_is_never_replayed():
    owner, request, calls, _ = fixture()

    def fail(*args):
        calls.append(args)
        raise RuntimeError('partial apply')

    owner.apply = fail
    assert owner.submit(request)['status'] == 'unknown'
    assert owner.submit(request)['status'] == 'unknown'
    assert owner.state()['writes_blocked'] is True
    assert (
        owner.submit({**request, 'command_id': str(uuid4()), 'expected_revision': 1})['status']
        == 'rejected'
    )
    assert len(calls) == 1


def test_timing_is_forwarded_without_sharing_mutable_state():
    timing = {'schema': 'marvin.policy-timing.v1', 'cadence': {'pattern_ticks': [4, 5]}}
    owner = PolicyControl(
        lambda: {
            'policy_path': '/policy.pt',
            'env_path': '/env.yaml',
            'timing': timing,
            'observations': [],
            'locks': {},
        },
        lambda *args: None,
    )
    assert owner.state()['timing'] == timing
    owner.state()['timing']['cadence']['pattern_ticks'].append(9)
    assert timing['cadence']['pattern_ticks'] == [4, 5]
