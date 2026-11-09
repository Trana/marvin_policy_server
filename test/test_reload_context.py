"""Real CPU TorchScript loading from the installed-copy authority, no ROS/CAN."""

import hashlib
import json
from uuid import uuid4

from marvin_policy_server.reload_context import build_context
import numpy as np
import pytest
from runtime_manager.inference_owner import transact
from runtime_manager.policy_bundle import install
from runtime_manager.policy_bundle import prepare
from runtime_manager.policy_store import PolicyStore
import torch


@pytest.fixture
def rig(tmp_path):
    root, authority = tmp_path / 'policies', tmp_path / 'authority'
    root.mkdir()
    authority.mkdir()
    (authority / 'owner.json').write_text(json.dumps({'owner': None, 'token': None}))
    store = PolicyStore(root, authority)
    bundles = []
    for i in range(2):
        source = tmp_path / f'source{i}'
        source.mkdir()
        model = torch.nn.Linear(7, 12)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.fill_(float(i))
        torch.jit.trace(model, torch.zeros(1, 7)).save(str(source / 'policy.pt'))
        (source / 'env.yaml').write_text('contract: same\n')
        prepared = prepare(
            {'policy.pt': source / 'policy.pt', 'env.yaml': source / 'env.yaml'},
            tmp_path / f'prepared{i}',
            run_id=f'run{i}',
            checkpoint='model_1.pt',
        )
        install(tmp_path / f'prepared{i}', root)
        bundles.append(prepared['bundle_id'])
    store.migrate({
        'revision': 0,
        'bundle_id': None,
        'deployment_id': None,
        'run_id': None,
        'checkpoint': None,
    })
    transact(authority, 'acquire', 'robot', str(uuid4()))
    spec = {
        'env_sha256': hashlib.sha256(b'contract: same\n').hexdigest(),
        'manifest_contract': None,
        'metadata': None,
        'compact': False,
        'action_count': 12,
        'observation_count': 7,
        'estimator_config': None,
        'estimator_indices': None,
    }
    return store, bundles, spec


def request(store, bundle):
    inv = store.inventory()
    copy = next(c for c in inv['copies'] if c['bundle_id'] == bundle)
    return {
        'request_id': str(uuid4()),
        'bundle_id': bundle,
        'copy_revision': copy['copy_revision'],
        'expected_incarnation': inv['incarnation'],
    }


def test_a_b_a_loads_exact_models_and_resets_warmup_state(rig):
    store, bundles, spec = rig
    for index in (0, 1, 0):
        ctx = build_context(request(store, bundles[index]), spec, store)
        try:
            assert ctx.runner.counter == 0
            assert np.array_equal(ctx.runner.current_action, np.zeros(12))
            np.testing.assert_allclose(
                ctx.runner.step(np.zeros(7), infer=True), np.full(12, index)
            )
            with store.transaction() as state:
                assert state['references'][ctx.reference]['copy_revision']
            assert store.owner() == 'robot'
        finally:
            ctx.close()
    with store.transaction() as state:
        assert not state['references']


def test_incompatible_environment_and_bad_output_cleanup_references(rig):
    store, bundles, spec = rig
    with pytest.raises(ValueError, match='environment file'):
        build_context(request(store, bundles[0]), {**spec, 'env_sha256': '0' * 64}, store)
    with pytest.raises(ValueError, match='invalid joint actions'):
        build_context(request(store, bundles[0]), {**spec, 'action_count': 13}, store)
    with store.transaction() as state:
        assert not state['references']


def test_stale_copy_cannot_be_loaded(rig):
    store, bundles, spec = rig
    req = request(store, bundles[0])
    req['copy_revision'] = str(uuid4())
    with pytest.raises(ValueError, match='copy changed'):
        build_context(req, spec, store)
    with store.transaction() as state:
        assert not state['references']


def test_combined_model_discards_warmup_history_and_rejects_metadata_drift(rig, tmp_path):
    from marvin_policy_server.concurrent_policy.manifest import load_concurrent_policy_manifest
    from marvin_policy_server.reload_context import manifest_contract

    class Combined(torch.nn.Module):
        def __init__(self, offset):
            super().__init__()
            self.register_buffer('offset', torch.tensor(float(offset)))

        def forward(self, observation, history):
            return observation[:, :1].repeat(1, 12) + self.offset, history[:, -1, :]

    store, _, spec = rig
    source = tmp_path / 'combined'
    source.mkdir()
    (source / 'env.yaml').write_text('contract: same\n')
    metadata = {
        'schema': 'marvin.concurrent_policy_export.v1',
        'artifacts': {'combined_policy': 'policy.pt'},
        'dimensions': {
            'base_policy_observation': 7,
            'actor_observation': 27,
            'raw_estimator_feature': 3,
            'estimator_history': 2,
            'estimated_state': 3,
            'joint_action': 12,
        },
    }
    bundles = []
    for index in range(2):
        torch.jit.trace(Combined(index), (torch.zeros(1, 7), torch.zeros(1, 2, 3))).save(
            str(source / 'policy.pt')
        )
        identity = {
            **metadata,
            'policy_sha256': hashlib.sha256((source / 'policy.pt').read_bytes()).hexdigest(),
            'created_at': str(index),
            'checkpoint_path': f'/training/model_{index}.pt',
            'task': 'training-task' if index == 0 else None,
        }
        (source / 'metadata.json').write_text(json.dumps(identity))
        staging = tmp_path / f'prepared-combined-{index}'
        prepared = prepare(
            {name: source / name for name in ('policy.pt', 'env.yaml', 'metadata.json')},
            staging,
            run_id='combined',
            checkpoint=f'model_{index}.pt',
        )
        install(staging, store.root)
        bundles.append(prepared['bundle_id'])
    path = store.root / bundles[0] / 'policy.pt'
    manifest = load_concurrent_policy_manifest(path)
    spec = {
        **spec,
        'metadata': json.loads(manifest.path.read_text()),
        'manifest_contract': manifest_contract(manifest),
    }
    for index in (0, 1, 0):
        context = build_context(request(store, bundles[index]), spec, store)
        try:
            assert context.runner.history.count == 0
            assert context.runner.latest_estimated_state is None
            np.testing.assert_allclose(
                context.runner.step(np.ones(7), np.ones(3), infer=True), np.full(12, 1 + index)
            )
            assert context.artifacts['metadata.json']['sha256']
        finally:
            context.close()
    with pytest.raises(ValueError, match='history contract differs'):
        build_context(
            request(store, bundles[0]),
            {**spec, 'metadata': {**spec['metadata'], 'combined_outputs': ['wrong']}},
            store,
        )
    with store.transaction() as state:
        assert not state['references']
