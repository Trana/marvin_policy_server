"""Read and validate metadata beside a concurrent estimator-policy export."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


SCHEMA = 'marvin.concurrent_policy_export.v1'


@dataclass(frozen=True)
class ConcurrentPolicyManifest:
    """Validated dimensions associated with one combined policy artifact."""

    base_observation_count: int
    actor_observation_count: int
    raw_feature_count: int
    history_length: int
    estimated_state_count: int
    action_count: int
    path: Path


def load_concurrent_policy_manifest(
    policy_path: Path,
) -> ConcurrentPolicyManifest | None:
    """Return metadata only for the declared combined model."""
    policy_path = Path(policy_path).expanduser().resolve()
    metadata_path = policy_path.parent / 'metadata.json'
    if not metadata_path.is_file():
        metadata_path = policy_path.parent / 'manifest.json'
    if not metadata_path.is_file():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError(
            f'Cannot read concurrent policy metadata {metadata_path}: {exc}'
        ) from exc
    if not isinstance(payload, dict) or payload.get('schema') != SCHEMA:
        return None
    artifacts = payload.get('artifacts', {})
    if (
        not isinstance(artifacts, dict)
        or policy_path.name != artifacts.get('combined_policy')
    ):
        return None
    dimensions = payload.get('dimensions', {})
    if not isinstance(dimensions, dict):
        raise ValueError(
            f'Concurrent policy dimensions are missing in {metadata_path}'
        )
    try:
        manifest = ConcurrentPolicyManifest(
            base_observation_count=int(dimensions['base_policy_observation']),
            actor_observation_count=int(dimensions['actor_observation']),
            raw_feature_count=int(dimensions['raw_estimator_feature']),
            history_length=int(dimensions['estimator_history']),
            estimated_state_count=int(dimensions['estimated_state']),
            action_count=int(dimensions['joint_action']),
            path=metadata_path,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f'Invalid concurrent policy dimensions in {metadata_path}: {exc}'
        ) from exc
    values = (
        manifest.base_observation_count,
        manifest.actor_observation_count,
        manifest.raw_feature_count,
        manifest.history_length,
        manifest.estimated_state_count,
        manifest.action_count,
    )
    if min(values) <= 0:
        raise ValueError(
            f'Concurrent policy dimensions must be positive in {metadata_path}'
        )
    if (
        manifest.actor_observation_count
        != manifest.base_observation_count + 20
    ):
        raise ValueError(
            'Concurrent actor observation must add FK(12), height(4), and '
            'contact(4)'
        )
    return manifest


__all__ = ['ConcurrentPolicyManifest', 'load_concurrent_policy_manifest']
