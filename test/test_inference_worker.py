from threading import Event
import time

from marvin_policy_server.inference_worker import InferenceWorker


def wait_for(predicate):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.001)
    raise AssertionError('Worker did not complete')


def test_no_queue_and_reset_discards_result_before_reactivation():
    entered, release = Event(), Event()
    resets = []

    def compute(value):
        entered.set()
        assert release.wait(2)
        return value * 2

    worker = InferenceWorker(compute, lambda: resets.append(1))
    try:
        assert worker.submit(1, 'old')
        assert entered.wait(2)
        assert not worker.submit(2, 'queued')
        worker.invalidate()
        assert not worker.ready  # No second computation until native work finishes.
        release.set()
        wait_for(lambda: worker.ready)
        assert worker.poll() is None
        assert worker.submit(3, 'new')
        result = wait_for(worker.poll)
        assert result.value == 6 and result.context == 'new'
        assert result.error is None
        assert len(resets) == 2
    finally:
        release.set()
        assert worker.close()


def test_exceptions_are_results_and_shutdown_rejects_submissions():
    def compute(value):
        raise ValueError('invalid output')

    worker = InferenceWorker(compute, lambda: None)
    assert worker.submit(None, 1)
    result = wait_for(worker.poll)
    assert isinstance(result.error, ValueError)
    assert result.finished_ns >= result.submitted_ns
    assert worker.close()
    assert not worker.submit(None, 2)


def test_invalidation_discards_already_completed_result():
    done = Event()
    worker = InferenceWorker(lambda x: done.set(), lambda: None)
    try:
        assert worker.submit(None, None)
        assert done.wait(2)
        worker.invalidate()
        assert worker.poll() is None
    finally:
        assert worker.close()
