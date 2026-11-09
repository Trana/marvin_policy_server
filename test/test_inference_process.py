import multiprocessing
import time

from marvin_policy_server.inference_process import InferenceProcess
import pytest


class Task:
    def __init__(self, spec):
        self.spec = spec
        self.count = 0

    def reset(self):
        self.count = 0

    def __call__(self, value):
        if self.spec:
            self.spec[0].set()
            assert self.spec[1].wait(3)
        if value == 'fail':
            raise ValueError('invalid model output')
        self.count += 1
        return value, self.count


def factory(spec):
    return Task(spec)


def failing_factory(spec):
    raise ValueError('invalid artifact')


def receive(worker):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        value = worker.poll()
        if value is not None:
            return value
        time.sleep(0.001)
    pytest.fail('No inference result')


def test_process_one_job_generation_reset_and_errors():
    context = multiprocessing.get_context('spawn')
    entered, release = context.Event(), context.Event()
    worker = InferenceProcess(factory, (entered, release))
    try:
        assert worker.submit(1, 'old')
        assert entered.wait(3)
        assert not worker.submit(2, 'queued')
        worker.invalidate()
        release.set()
        deadline = time.monotonic() + 3
        while not worker.ready and time.monotonic() < deadline:
            assert worker.poll() is None
            time.sleep(0.001)
        assert worker.ready
        assert worker.submit(3, 'new')
        result = receive(worker)
        assert result.value == (3, 1) and result.context == 'new'
        assert worker.submit('fail', 'error')
        assert 'invalid model output' in str(receive(worker).error)
    finally:
        release.set()
        assert worker.close()
    assert not worker.submit(4, 'closed')


def test_startup_failure_is_explicit():
    with pytest.raises(RuntimeError, match='invalid artifact'):
        InferenceProcess(failing_factory, None)


def test_unexpected_child_exit_is_explicit():
    worker = InferenceProcess(factory, None)
    worker._process.terminate()
    worker._process.join(2)
    try:
        with pytest.raises((RuntimeError, EOFError)):
            worker.poll()
    finally:
        assert worker.close()
