import importlib.util
from pathlib import Path
from threading import Event
import time

import pytest

spec = importlib.util.spec_from_file_location(
    'writer', Path(__file__).parents[1] / 'scripts/joint_capture_writer.py'
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def progress(now):
    return {'last_target': now, 'last_feedback': now, 'targets': 2, 'feedback': 2}


def test_blocked_sync_does_not_block_submit_or_claim_durability(tmp_path):
    entered, release = Event(), Event()

    def sync(_):
        entered.set()
        assert release.wait(2)

    writer = module.CaptureWriter(tmp_path / 'samples', interval=0.001, sync=sync)
    try:
        writer.submit('first\n', progress(time.monotonic()))
        assert entered.wait(1)
        writer.submit('second\n', progress(time.monotonic()))
        assert writer.queue.qsize() == 1
        assert writer.progress() is None
        assert not module.durable_ready(writer.progress(), time.monotonic(), 0.15)
    finally:
        release.set()
        writer.close()
    assert (tmp_path / 'samples').read_text() == 'first\nsecond\n'
    assert writer.progress()['targets'] == 2


def test_bounded_queue_overflow_is_explicit(tmp_path):
    entered, release = Event(), Event()

    def sync(_):
        entered.set()
        assert release.wait(2)

    writer = module.CaptureWriter(tmp_path / 'samples', capacity=1, interval=0.001, sync=sync)
    try:
        writer.submit('a\n', progress(time.monotonic()))
        assert entered.wait(1)
        writer.submit('b\n', progress(time.monotonic()))
        with pytest.raises(RuntimeError, match='overflow'):
            writer.submit('c\n', progress(time.monotonic()))
    finally:
        release.set()
        writer.close()


def test_sync_failure_never_reports_committed_data(tmp_path):
    def sync(_):
        raise OSError('disk failed')

    writer = module.CaptureWriter(tmp_path / 'samples', sync=sync)
    writer.submit('a\n', progress(time.monotonic()))
    with pytest.raises(RuntimeError, match='disk failed'):
        writer.close()
    assert writer.progress() is None


def test_recent_sync_cannot_make_old_samples_fresh():
    assert not module.durable_ready(dict(progress(1.0), committed_at=2.0), 2.0, 0.15)
    assert module.durable_ready(dict(progress(2.0), committed_at=2.01), 2.02, 0.15)


def test_storage_allowance_does_not_require_150ms_commit():
    saved = dict(progress(1.0), committed_at=1.24)
    assert not module.durable_ready(saved, 1.25, 0.15)
    assert module.durable_ready(saved, 1.25, 0.5)
    assert not module.durable_ready(saved, 1.501, 0.5)
    # A fresh sync of old data cannot hide a stalled queue.
    saved['committed_at'] = 1.6
    assert not module.durable_ready(saved, 1.6, 0.5)
