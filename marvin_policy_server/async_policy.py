"""Worker-owned runners and copied results. No node state or ROS calls here."""

from copy import deepcopy


class RunnerTask:
    def __init__(self, runner):
        self.runner = runner

    def reset(self):
        self.runner.reset()

    def __call__(self, payload):
        args, kwargs = payload
        action = self.runner.step(*args, **kwargs).copy()
        state = {}
        for name in (
            'latest_estimated_state',
            'latest_effective_velocity',
            'latest_foot_positions',
            'latest_result_monotonic',
            'history',
        ):
            if hasattr(self.runner, name):
                state[name] = deepcopy(getattr(self.runner, name))
        return action, state


def create_runner_task(spec):
    """Load and warm a CPU-only model in the spawned process, never a ROS node."""
    import hashlib
    import io
    from pathlib import Path

    import numpy as np
    import torch

    from .concurrent_policy.manifest import load_concurrent_policy_manifest
    from .concurrent_policy.optimized import optimize_policy
    from .concurrent_policy.runner import ConcurrentPolicyRunner
    from .inference_engine import DEFAULT_ENGINE, prepare_policy
    from .policy_runner import PolicyRunner
    from .policy_runner import PolicyRunnerConfig

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    content = Path(spec['policy_path']).read_bytes()
    if hashlib.sha256(content).hexdigest() != spec['policy_sha256']:
        raise ValueError('Policy artifact changed before inference process startup')
    policy = torch.jit.load(io.BytesIO(content), map_location='cpu').eval()
    manifest = load_concurrent_policy_manifest(Path(spec['policy_path']))
    if manifest is not None and manifest.loaded_artifact['sha256'] != spec['metadata_sha256']:
        raise ValueError('Policy metadata changed before inference process startup')
    if spec['compact_estimator']:
        if manifest is None:
            raise ValueError('Compact estimator requires a combined artifact')
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
    task = RunnerTask(runner)
    args = (np.zeros(spec['observation_count'], dtype=np.float32),)
    if manifest:
        args += (np.zeros(manifest.raw_feature_count, dtype=np.float32),)
    for _ in range(20):
        task((args, {'infer': True}))
    task.reset()
    return task
