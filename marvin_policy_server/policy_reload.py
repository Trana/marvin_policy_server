"""Inactive-policy replacement. Loading never grants command authority."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import time
from uuid import UUID


class PolicyReload:
    def __init__(self, load, install, released, session, *, clock=time.monotonic):
        self.load, self.install, self.released, self.session = load, install, released, session
        self.clock = clock
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='policy-load')
        self.operations = {}
        self.pending = None
        self.unavailable = False

    @property
    def blocked(self):
        return self.pending is not None or self.unavailable

    def handle(self, request):
        if not isinstance(request, dict):
            raise ValueError('Reload request must be an object')
        action = request.get('action')
        operation_id = str(UUID(request['request_id']))
        self.poll()
        old = self.operations.get(operation_id)
        if action == 'start':
            required = {
                'action',
                'request_id',
                'bundle_id',
                'copy_revision',
                'expected_incarnation',
                'expected_session_id',
            }
            if set(request) != required:
                raise ValueError('Invalid policy reload fields')
            if old:
                if old['request'] != request:
                    raise ValueError('Reload ID has different inputs')
                return self.view(old)
            if len(self.operations) >= 128:
                raise ValueError('Reload journal full; restart runtime to reconcile')
            if self.pending or not self.released():
                raise ValueError('Policy output must be confirmed released')
            if request['expected_session_id'] != self.session():
                raise ValueError('Policy session changed; refresh before loading')
            old = {
                'request_id': operation_id,
                'request': deepcopy(request),
                'state': 'loading',
                'message': 'Loading policy',
                'started': self.clock(),
            }
            self.operations[operation_id] = old
            self.pending = old
            self.unavailable = True
            # No ROS or live policy state access is permitted in this worker.
            old['future'] = self.worker.submit(self.load, deepcopy(request))
        elif action in {'get', 'commit', 'cancel'}:
            if set(request) != {'action', 'request_id'} or old is None:
                raise ValueError('Unknown policy reload operation')
            if action == 'cancel' and old['state'] in {'loading', 'ready'}:
                old['state'], old['message'] = (
                    'cancelled',
                    'Reload cancelled; policy remains inactive',
                )
                if old.get('context') is not None:
                    self._discard(old)
            if action == 'commit' and old['state'] == 'ready':
                if not self.released():
                    raise ValueError('Policy output release is no longer confirmed')
                try:
                    old['loaded'] = self.install(old['context'])
                    old['state'], old['message'] = (
                        'succeeded',
                        'Policy loaded; switch to Policy explicitly',
                    )
                    old.pop('context')
                    self.pending = None
                    self.unavailable = False
                except Exception as error:
                    self.unavailable = True
                    old['state'], old['message'] = 'failed', str(error)
                    self._discard(old)
        else:
            raise ValueError('Unsupported policy reload action')
        return self.view(old)

    def _discard(self, old):
        context = old.pop('context', None)
        if context is not None:
            context.close()
        if old.get('future') is None:
            self.pending = None

    def poll(self):
        old = self.pending
        if not old:
            return
        future = old.get('future')
        if future is not None and future.done():
            old.pop('future')
            try:
                old['context'] = future.result()
                if old['state'] == 'cancelled':
                    self._discard(old)
                else:
                    old['state'], old['message'] = 'ready', 'Policy ready for inactive replacement'
            except Exception as error:
                old['state'], old['message'] = 'failed', str(error)
                self.unavailable = True
                self.pending = None
        if self.pending is old and self.clock() - old['started'] > 90:
            # Never abandon a running loader or allow a second load after timeout.
            old['state'], old['message'] = (
                'cancelled',
                'Loading deadline exceeded; waiting for loader cleanup',
            )
            self.unavailable = True
            if old.get('future') is None:
                self._discard(old)

    @staticmethod
    def view(old):
        result = {key: deepcopy(old[key]) for key in ('request_id', 'state', 'message')}
        if old.get('loaded') is not None:
            result['loaded'] = deepcopy(old['loaded'])
        return result

    def close(self):
        if self.pending:
            self.pending['state'] = 'cancelled'
            if self.pending.get('context') is not None:
                self._discard(self.pending)
        self.worker.shutdown(wait=True, cancel_futures=True)
        self.poll()
