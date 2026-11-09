"""Spawned inference process: no ROS entities, one request, no backlog."""

import multiprocessing
import time

from .inference_worker import InferenceResult


def _serve(connection, factory, spec):
    try:
        task = factory(spec)
        connection.send(('ready', None))
        generation = None
        while True:
            job = connection.recv()
            if job is None:
                break
            current, payload, context, submitted = job
            value = error = None
            try:
                if current != generation:
                    task.reset()
                    generation = current
                value = task(payload)
            except Exception as exc:
                error = RuntimeError(f'{type(exc).__name__}: {exc}')
            connection.send((
                'result',
                InferenceResult(current, context, value, error, submitted, time.monotonic_ns()),
            ))
    except (EOFError, BrokenPipeError, KeyboardInterrupt):
        pass
    except Exception as exc:
        connection.send(('error', f'{type(exc).__name__}: {exc}'))
    finally:
        connection.close()


class InferenceProcess:
    """Same lifecycle as InferenceWorker, without the shared Python GIL."""

    def __init__(self, factory, spec, *, startup_timeout=30.0):
        context = multiprocessing.get_context('spawn')
        self._connection, child = context.Pipe()
        self._process = context.Process(
            target=_serve, args=(child, factory, spec), name='policy-inference', daemon=True
        )
        self._generation = 0
        self._busy = self._closed = False
        self._process.start()
        child.close()
        try:
            if not self._connection.poll(startup_timeout):
                raise RuntimeError('Inference process startup timed out')
            kind, detail = self._connection.recv()
            if kind != 'ready':
                raise RuntimeError(f'Inference process startup failed: {detail}')
        except Exception:
            self.close()
            raise

    @property
    def ready(self):
        return not self._closed and not self._busy and self._process.is_alive()

    def invalidate(self):
        self._generation += 1

    def submit(self, payload, context):
        if not self.ready:
            return False
        self._connection.send((self._generation, payload, context, time.monotonic_ns()))
        self._busy = True
        return True

    def poll(self):
        if self._closed:
            return None
        if not self._connection.poll():
            if not self._process.is_alive():
                raise RuntimeError('Inference process exited unexpectedly')
            return None
        kind, result = self._connection.recv()
        self._busy = False
        if kind != 'result':
            raise RuntimeError(f'Inference process failed: {result}')
        return result if result.generation == self._generation else None

    def close(self, timeout=1.0):
        if self._closed:
            return not self._process.is_alive()
        self._closed = True
        self._generation += 1
        if self._process.is_alive() and not self._busy:
            try:
                self._connection.send(None)
            except (BrokenPipeError, EOFError):
                pass
        self._process.join(timeout)
        if self._process.is_alive():
            # Only our private compute child. It has no ROS or motor publishers.
            self._process.terminate()
            self._process.join(timeout)
        self._connection.close()
        return not self._process.is_alive()
