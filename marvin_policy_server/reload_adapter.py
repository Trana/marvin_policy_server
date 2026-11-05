"""ROS adapter for supervisor-authorized inactive policy replacement."""
from copy import deepcopy
import json
import numpy as np
from runtime_interfaces.srv import PositionDiagnostic
from .policy_control import PolicyControl
from .execution_preparation import PolicyPreparation
from .policy_reload import PolicyReload
from .reload_context import build_context, manifest_contract


class ReloadAdapter:
    def __init__(self, node):
        self.node = node
        self.loaded_context = None
        self.manager = PolicyReload(self.load, self.install, self.released,
                                    lambda: node._policy_control.session_id)
        node.create_service(PositionDiagnostic, 'reload_policy', self.handle)
        # Management requests advance loading through handle(). Do not add a
        # periodic Python callback to the policy executor while reload is idle.
        # Capture the startup contract, not a mutable environment file at reload time.
        self.spec = {
            'env_sha256': node._loaded_artifacts['env.yaml']['sha256'],
            'metadata': json.loads(node._concurrent_policy_manifest.path.read_text()) if node._concurrent_policy_manifest else None,
            'manifest_contract': manifest_contract(node._concurrent_policy_manifest),
            'compact': bool(node.get_parameter('compact_estimator').value),
            'action_count': len(node.joint_names), 'observation_count': node._expected_obs_dim,
            'estimator_config': node._velocity_estimator.config if node._velocity_estimator else None,
            'estimator_indices': node._velocity_estimator.feature_indices if node._velocity_estimator else None,
        }
        self.observation_template = deepcopy(node._obs_state)

    def released(self):
        n = self.node
        return (n._native_io and n._execution_gateway and n._execution_state()['output_released']
                and not n._activation_mgr.is_active())

    def load(self, request):
        from runtime_manager.policy_store import configured
        store = configured()
        if store is None:
            raise ValueError('Target policy store is not configured')
        return build_context(request, self.spec, store)

    def install(self, context):
        n = self.node
        if not self.released():
            raise ValueError('Policy output must remain released')
        # All callbacks and installation share the node executor. Heavy model work
        # stays in the loader. No publication or native reconfiguration occurs here.
        old_context = self.loaded_context
        old_identity = deepcopy(n._loaded_artifacts)
        old_session = n._policy_control.session_id
        n.policy, n._policy_runner = context.policy, context.runner
        n._concurrent_policy_manifest = context.manifest
        n._velocity_estimator = context.estimator
        n.policy_path, n._env_path = str(context.path), context.path.parent / 'env.yaml'
        n._loaded_artifacts = context.artifacts
        n._obs_state = deepcopy(self.observation_template)
        n._inference_schedule.reset()
        n.action = np.zeros(len(n.joint_names))
        n._latest_estimated_velocity = None
        n._velocity_estimator_error = None
        n._pending_velocity_estimator_diagnostic = None
        n._latest_policy_joint_command = n._last_published_joint_command = None
        n._pending_result, n._history_requested = None, False
        n._execution_preparation = PolicyPreparation(n.joint_names)
        if n._joint_position_filter is not None:
            n._joint_position_filter.reset()
        for name in ('command', 'imu', 'joint_position', 'velocity'):
            setattr(n, f'_{name}_input_locked', False)
        n._joint_command_output_locked = False
        n._estimator_velocity_override_mode = 'live'
        for name in ('command_input', 'imu_input', 'joint_position_relative', 'joint_velocity',
                     'joint_command', 'estimator_velocity', 'estimator_fk', 'estimator_foot_height', 'estimator_contact'):
            setattr(n, '_locked_' + name, None)
        n._policy_control = PolicyControl(n._observation_status_payload, n._apply_policy_input_lock)
        self.loaded_context = context
        if old_context:
            old_context.close()
        return {'old_artifacts': old_identity, 'new_artifacts': deepcopy(context.artifacts),
                'old_session_id': old_session, 'new_session_id': n._policy_control.session_id}

    def handle(self, request, response):
        payload = {}
        try:
            if len(request.request_json) > 8192:
                raise ValueError('Reload request exceeds 8192 bytes')
            payload = json.loads(request.request_json)
            result = self.manager.handle(payload)
        except (ValueError, TypeError, KeyError) as error:
            result = {'state': 'failed', 'message': str(error),
                      'request_id': payload.get('request_id', '') if isinstance(payload, dict) else ''}
        response.response_json = json.dumps(result, allow_nan=False)
        return response
