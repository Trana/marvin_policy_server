"""Session-scoped overrides; no ROS, Torch, transport or automatic retry here."""

from copy import deepcopy
from datetime import datetime
from datetime import timezone
import math
from uuid import UUID
from uuid import uuid4


class PolicyControlError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class PolicyControl:
    def __init__(self, observe, apply, *, limit=1024):
        self.session_id = str(uuid4())
        self.revision = 0
        self.observe, self.apply = observe, apply
        self.limit = limit
        self._operations = {}
        self._modes = {}
        self._uncertain = False

    def changed(self, control, locked, values):
        """Legacy service calls share revision/mode ownership with the new API."""
        mode = 'live' if not locked else 'hold' if values is None else 'fixed'
        if control == 'estimator_velocity_odometry':
            control, mode = 'estimator_velocity', 'odometry' if locked else 'live'
        self._modes[control] = mode
        self.revision += 1

    def state(self):
        observation = self.observe()
        keys = {term['key'] for term in observation['observations']}
        relevant = {
            'command': 'velocity_commands' in keys,
            'imu': bool(keys & {'base_lin_vel', 'base_ang_vel', 'projected_gravity'}),
            'joint_position': 'joint_pos' in keys,
            'velocity': 'joint_vel' in keys,
        }
        controls = {}
        for name, lock in observation['locks'].items():
            available = bool(lock.get('available', relevant.get(name, True)))
            modes = ['live', 'hold', 'fixed'] if available else []
            if name == 'estimator_velocity' and available:
                modes.append('odometry')
            mode = self._modes.get(name, 'hold' if lock['locked'] else 'live')
            controls[name] = {
                'available': available,
                'modes': modes,
                'size': len(lock['values']),
                'reference_available': bool(lock.get('reference_available', False)),
                'mode': mode,
                'labels': lock['labels'],
                'values': lock['values'],
                'live_values': lock['live_values'],
                'reference_values': lock.get('reference_values', []),
            }
        return {
            'schema_version': 'policy-control.v1',
            'session_id': self.session_id,
            'revision': self.revision,
            'observed_at': datetime.now(timezone.utc).isoformat(),
            'writes_blocked': self._uncertain,
            'policy_path': observation['policy_path'],
            'env_path': observation['env_path'],
            'loaded_artifacts': deepcopy(observation.get('loaded_artifacts')),
            'timing': deepcopy(observation.get('timing')),
            'observations': observation['observations'],
            'controls': controls,
        }

    def get(self, session_id, command_id):
        self._session(session_id)
        if command_id not in self._operations:
            raise PolicyControlError(
                'POLICY_NOT_FOUND', 'Command ID is not recorded in this session'
            )
        return deepcopy(self._operations[command_id])

    def _session(self, session_id):
        if session_id != self.session_id:
            raise PolicyControlError(
                'POLICY_CONFLICT', 'Policy session changed; refresh state, do not replay'
            )

    def submit(self, request):
        self._session(request['session_id'])
        try:
            if str(UUID(request['command_id'])) != request['command_id']:
                raise ValueError('Noncanonical UUID')
        except (ValueError, TypeError, AttributeError) as exc:
            raise PolicyControlError(
                'POLICY_INVALID_REQUEST', 'command_id must be a canonical UUID'
            ) from exc
        command_id = request['command_id']
        if command_id in self._operations:
            old = self._operations[command_id]
            if old['requested'] != request:
                raise PolicyControlError(
                    'POLICY_CONFLICT', 'Command ID already has different arguments'
                )
            return deepcopy(old)
        if len(self._operations) >= self.limit:
            raise PolicyControlError(
                'POLICY_CONFLICT', 'Policy command journal full; no IDs evicted'
            )
        outcome = {
            'command_id': command_id,
            'session_id': self.session_id,
            'requested': deepcopy(request),
            'status': 'rejected',
            'message': '',
            'applied_revision': None,
            'applied': None,
        }
        # Reserve before mutation. The node's mutually exclusive callbacks serialize
        # this operation with inference and both legacy/new control callbacks.
        self._operations[command_id] = outcome
        control, mode, values = request['control'], request['mode'], request['values']
        spec = self.state()['controls'].get(control)
        if self._uncertain:
            outcome['message'] = (
                'An earlier application failed ambiguously; restart/reconcile policy session'
            )
        elif request['expected_revision'] != self.revision:
            outcome['message'] = 'Override revision changed; refresh state before a new command'
        elif not spec or not spec['available'] or mode not in spec['modes']:
            outcome['message'] = 'Control or mode is not supported by the loaded policy'
        elif mode == 'odometry' and not spec['reference_available']:
            outcome['message'] = 'Fresh body-frame odometry is unavailable'
        elif mode == 'fixed' and (
            len(values) != spec['size'] or not all(math.isfinite(value) for value in values)
        ):
            outcome['message'] = 'Fixed values must match control size and be finite'
        elif mode != 'fixed' and values:
            outcome['message'] = 'Only fixed mode accepts values'
        else:
            try:
                success, message = self.apply(
                    'estimator_velocity_odometry' if mode == 'odometry' else control,
                    mode != 'live',
                    values if mode == 'fixed' else None,
                )
                outcome['message'] = message
                if success:
                    self._modes[control] = mode
                    self.revision += 1
                    outcome.update(
                        status='applied',
                        applied_revision=self.revision,
                        applied=self.state()['controls'][control],
                    )
            except Exception as exc:  # retain ambiguity; never replay a partial mutation
                self._uncertain = True
                self.revision += 1
                outcome.update(status='unknown', message=f'Application outcome unknown: {exc}')
        return deepcopy(outcome)
