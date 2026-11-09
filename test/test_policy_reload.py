from concurrent.futures import Future
from threading import Event
from types import SimpleNamespace as NS
from uuid import uuid4

from marvin_policy_server.policy_reload import PolicyReload
import pytest


class Worker:
    def __init__(self):
        self.future = Future()
        self.calls = []

    def submit(self, fn, request):
        self.calls.append(request)
        return self.future

    def shutdown(self, **kwargs):
        pass


@pytest.fixture
def rig():
    installed, closed, released = [], [], [True]
    manager = PolicyReload(
        lambda _: None, installed.append, lambda: released[0], lambda: 'session'
    )
    manager.worker.shutdown()
    manager.worker = Worker()
    context = NS(close=lambda: closed.append(True))
    request = {
        'action': 'start',
        'request_id': str(uuid4()),
        'bundle_id': 'a' * 64,
        'copy_revision': str(uuid4()),
        'expected_incarnation': str(uuid4()),
        'expected_session_id': 'session',
    }
    return manager, request, installed, closed, released, context


def test_loading_and_commit_are_separate_and_exact_id_deduplicates(rig):
    m, req, installed, closed, _, context = rig
    assert m.handle(req)['state'] == 'loading' and m.blocked
    assert m.handle(req)['state'] == 'loading' and len(m.worker.calls) == 1
    m.worker.future.set_result(context)
    assert m.handle({'action': 'get', 'request_id': req['request_id']})['state'] == 'ready'
    assert not installed
    assert m.handle({'action': 'commit', 'request_id': req['request_id']})['state'] == 'succeeded'
    assert installed == [context] and not closed and not m.blocked
    m.handle({'action': 'commit', 'request_id': req['request_id']})
    assert installed == [context]


def test_cancel_does_not_release_reference_while_loader_is_running(rig):
    m, req, installed, closed, _, context = rig
    m.handle(req)
    m.handle({'action': 'cancel', 'request_id': req['request_id']})
    assert m.blocked and not closed
    m.worker.future.set_result(context)
    m.poll()
    assert closed == [True] and not installed and m.blocked


def test_failed_load_blocks_activation_and_conflicting_retry(rig):
    m, req, *_ = rig
    m.handle(req)
    with pytest.raises(ValueError, match='different inputs'):
        m.handle({**req, 'bundle_id': 'b' * 64})
    m.worker.future.set_exception(ValueError('invalid policy'))
    assert m.handle({'action': 'get', 'request_id': req['request_id']})['state'] == 'failed'
    assert m.blocked


def test_release_is_rechecked_at_commit(rig):
    m, req, installed, _, released, context = rig
    m.handle(req)
    m.worker.future.set_result(context)
    m.poll()
    released[0] = False
    with pytest.raises(ValueError, match='no longer confirmed'):
        m.handle({'action': 'commit', 'request_id': req['request_id']})
    assert not installed


def test_actual_worker_never_installs_from_background_thread():
    started, finish = Event(), Event()
    installed = []

    def load(_):
        started.set()
        assert finish.wait(2)
        return NS(close=lambda: None)

    m = PolicyReload(load, installed.append, lambda: True, lambda: 'session')
    req = {
        'action': 'start',
        'request_id': str(uuid4()),
        'bundle_id': 'a' * 64,
        'copy_revision': str(uuid4()),
        'expected_incarnation': str(uuid4()),
        'expected_session_id': 'session',
    }
    try:
        assert m.handle(req)['state'] == 'loading'
        assert started.wait(1) and not installed
        # A concurrent executor callback can return while model loading waits.
        assert m.handle({'action': 'get', 'request_id': req['request_id']})['state'] == 'loading'
    finally:
        finish.set()
        m.close()
    assert not installed
