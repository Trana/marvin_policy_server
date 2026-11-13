"""Build compatible model replacements without touching the live node."""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch

from .artifact_identity import artifact_identity
from .concurrent_policy.manifest import load_concurrent_policy_manifest
from .concurrent_policy.optimized import optimize_policy
from .concurrent_policy.runner import ConcurrentPolicyRunner
from .inference_engine import DEFAULT_ENGINE, prepare_policy
from .policy_runner import PolicyRunner
from .policy_runner import PolicyRunnerConfig
from .velocity_estimator import VelocityEstimator


def manifest_contract(manifest):
    if manifest is None:
        return None
    return tuple(
        getattr(manifest, name)
        for name in (
            'base_observation_count',
            'actor_observation_count',
            'raw_feature_count',
            'history_length',
            'estimated_state_count',
            'action_count',
        )
    )


def metadata_contract(metadata):
    """
    Compare semantics while allowing a newly exported model identity.

    Unknown metadata remains part of the contract. Only known provenance fields
    are excluded. The installed bundle separately validates all artifact hashes.
    """
    if metadata is None:
        return None
    result = deepcopy(metadata)
    for key in (
        'created_at',
        'task',
        'policy_path',
        'policy_sha256',
        'artifact_sha256',
        'checkpoint_name',
        'checkpoint_path',
        'checkpoint_sha256',
    ):
        result.pop(key, None)
    # The store normalizes this reference when installing an exported bundle.
    result['artifacts']['combined_policy'] = 'policy.pt'
    return result


@dataclass
class ReloadContext:
    policy: object
    runner: object
    manifest: object
    estimator: object
    artifacts: dict
    path: Path
    reference: str
    store: object

    def close(self):
        self.store.release(self.reference)


def build_context(request, spec, store):
    """Pin the exact copy, validate its contract, then load and warm on CPU."""
    ref = request['request_id']
    store.acquire_exact(
        ref,
        request['bundle_id'],
        request['copy_revision'],
        request['expected_incarnation'],
        'Policy reload',
    )
    try:
        root = store.root / request['bundle_id']
        env_content = (root / 'env.yaml').read_bytes()
        if hashlib.sha256(env_content).hexdigest() != spec['env_sha256']:
            raise ValueError('Runtime restart required: environment file differs')
        path = root / 'policy.pt'
        content = path.read_bytes()
        manifest = load_concurrent_policy_manifest(path)
        metadata = json.loads(manifest.path.read_text()) if manifest else None
        if manifest_contract(manifest) != spec['manifest_contract'] or metadata_contract(
            metadata
        ) != metadata_contract(spec['metadata']):
            raise ValueError('Runtime restart required: policy family or history contract differs')
        # Exact loader bytes must agree with the validated immutable bundle.
        from runtime_manager.policy_bundle import validate

        validate(root, request['bundle_id'])
        policy = torch.jit.load(io.BytesIO(content), map_location='cpu').eval()
        if spec['compact']:
            policy = optimize_policy(policy, manifest)
        policy = prepare_policy(
            policy, manifest=manifest, observation_count=spec['observation_count'],
            action_count=spec['action_count'], engine=spec.get('inference_engine', DEFAULT_ENGINE),
        )
        runner = (
            ConcurrentPolicyRunner(policy, manifest, decimation=4)
            if manifest
            else PolicyRunner(policy, spec['action_count'], PolicyRunnerConfig())
        )
        args = (np.zeros(spec['observation_count'], dtype=np.float32),)
        if manifest:
            args += (np.zeros(manifest.raw_feature_count, dtype=np.float32),)
        for _ in range(20):
            result = runner.step(*args, infer=True)
            if result.shape != (spec['action_count'],) or not np.isfinite(result).all():
                raise ValueError('Policy warmup produced invalid joint actions')
        runner.reset()
        estimator = None
        artifacts = {
            'policy.pt': artifact_identity(path, content),
            'env.yaml': artifact_identity(root / 'env.yaml', env_content),
        }
        if manifest:
            artifacts['metadata.json'] = manifest.loaded_artifact
        if spec['estimator_config'] is not None:
            estimator = VelocityEstimator(root / 'velocity_estimator.pt', engine=spec.get('inference_engine', DEFAULT_ENGINE))
            if (
                estimator.config != spec['estimator_config']
                or estimator.feature_indices != spec['estimator_indices']
            ):
                raise ValueError('Runtime restart required: estimator feature contract differs')
            for _ in range(20):
                result = estimator.predict(np.zeros(estimator.raw_feature_count))
                if not np.isfinite(result).all():
                    raise ValueError('Estimator warmup produced invalid velocity')
            estimator.reset()
            artifacts['velocity_estimator.pt'] = estimator.loaded_artifact
        # A bundle cannot change between validation and construction.
        manifest_json = json.loads((root / 'manifest.json').read_text())
        for name, artifact in artifacts.items():
            entry = manifest_json['files'].get(name)
            if entry is None or entry['sha256'] != artifact['sha256']:
                raise ValueError('Loaded bytes differ from installed bundle')
        return ReloadContext(
            policy, runner, manifest, estimator, deepcopy(artifacts), path, ref, store
        )
    except BaseException:
        store.release(ref)
        raise
