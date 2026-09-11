"""Inference/decimation adapter for a two-input concurrent policy artifact."""

from __future__ import annotations

import numpy as np

import torch

from .history import EstimatorFeatureHistory
from .manifest import ConcurrentPolicyManifest


class ConcurrentPolicyRunner:
    """Run a combined artifact at the configured control decimation."""

    def __init__(
        self, policy, manifest: ConcurrentPolicyManifest, *, decimation: int
    ):
        """Initialize action state and estimator history."""
        if decimation <= 0:
            raise ValueError('decimation must be positive')
        self.policy = policy
        self.manifest = manifest
        self.decimation = int(decimation)
        self.counter = 0
        self.current_action = np.zeros(
            manifest.action_count, dtype=np.float64
        )
        self.latest_estimated_state: np.ndarray | None = None
        self.latest_effective_velocity: np.ndarray | None = None
        self.latest_foot_positions: np.ndarray | None = None
        self.history = EstimatorFeatureHistory(
            manifest.history_length, manifest.raw_feature_count
        )

    @property
    def supports_velocity_override(self) -> bool:
        """Return whether the combined artifact exposes its estimator and actor."""
        return hasattr(self.policy, 'estimator') and hasattr(self.policy, 'actor')

    def should_infer(self) -> bool:
        """Return true on control ticks that require model inference."""
        return self.counter % self.decimation == 0

    def current_foot_positions(self) -> np.ndarray | None:
        """Evaluate current history once to expose FK for a hold snapshot."""
        if not self.supports_velocity_override or self.history.count == 0:
            return None
        history = torch.from_numpy(
            self.history.values.copy()
        ).unsqueeze(0)
        with torch.inference_mode():
            _, _, _, foot_positions = self.policy.estimator.actor_estimates(
                history
            )
        values = foot_positions.detach().reshape(-1).cpu().numpy()
        if values.shape != (12,) or not np.isfinite(values).all():
            return None
        self.latest_foot_positions = values.astype(
            np.float64, copy=True
        )
        return self.latest_foot_positions.copy()

    def reset(self) -> None:
        """Reset action and estimator history on policy activation."""
        self.counter = 0
        self.current_action.fill(0.0)
        self.latest_estimated_state = None
        self.latest_effective_velocity = None
        self.latest_foot_positions = None
        self.history.reset()

    @staticmethod
    def _override_tensor(
        values: np.ndarray | None,
        live: torch.Tensor,
        width: int,
        name: str,
    ) -> torch.Tensor:
        """Replace one estimator output after validating its shape."""
        if values is None:
            return live
        override = np.asarray(values, dtype=np.float32).reshape(-1)
        if override.shape != (width,):
            raise ValueError(
                f'{name} override has shape {override.shape}, expected '
                f'{(width,)}'
            )
        if not np.isfinite(override).all():
            raise ValueError(f'{name} override contains non-finite values')
        return torch.as_tensor(
            override,
            dtype=live.dtype,
            device=live.device,
        ).reshape_as(live)

    def _infer_with_estimate_overrides(
        self,
        observation: torch.Tensor,
        history: torch.Tensor,
        velocity_override: np.ndarray | None,
        foot_positions_override: np.ndarray | None,
        foot_height_override: np.ndarray | None,
        contact_override: np.ndarray | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.supports_velocity_override:
            raise ValueError(
                'Concurrent artifact does not expose estimator/actor modules '
                'required for estimator-output overrides'
            )
        velocity, foot_height, contact, foot_positions = (
            self.policy.estimator.actor_estimates(history)
        )
        self.latest_foot_positions = foot_positions.detach().reshape(
            -1
        ).cpu().numpy().astype(np.float64, copy=True)
        effective_velocity = self._override_tensor(
            velocity_override, velocity, 3, 'velocity'
        )
        effective_foot_positions = self._override_tensor(
            foot_positions_override,
            foot_positions,
            12,
            'foot position',
        )
        effective_foot_height = self._override_tensor(
            foot_height_override, foot_height, 4, 'foot height'
        )
        effective_contact = self._override_tensor(
            contact_override, contact, 4, 'contact'
        )
        actor_observation = observation.clone()
        actor_observation[..., :3] = effective_velocity
        actor_observation = torch.cat(
            (
                actor_observation,
                effective_foot_positions.flatten(start_dim=-2),
                effective_foot_height,
                effective_contact,
            ),
            dim=-1,
        )
        action = self.policy.actor(actor_observation)
        estimated_state = torch.cat((velocity, foot_height, contact), dim=-1)
        return action, estimated_state

    def step(
        self,
        observation: np.ndarray | None,
        raw_features: np.ndarray | None = None,
        estimated_velocity_override: np.ndarray | None = None,
        estimated_foot_positions_override: np.ndarray | None = None,
        estimated_foot_height_override: np.ndarray | None = None,
        estimated_contact_override: np.ndarray | None = None,
    ) -> np.ndarray:
        """Advance one control tick and hold actions between inferences."""
        if self.should_infer():
            if observation is None or raw_features is None:
                raise ValueError(
                    'Concurrent observation and estimator features are '
                    'required on inference ticks'
                )
            observation = np.asarray(observation, dtype=np.float32).reshape(-1)
            if observation.shape != (self.manifest.base_observation_count,):
                raise ValueError(
                    'Concurrent base observation has the wrong width: '
                    f'{observation.shape} != '
                    f'{(self.manifest.base_observation_count,)}'
                )
            history = self.history.push(raw_features)
            with torch.inference_mode():
                observation_tensor = torch.from_numpy(observation).unsqueeze(0)
                history_tensor = torch.from_numpy(history.copy()).unsqueeze(0)
                has_override = any(
                    item is not None
                    for item in (
                        estimated_velocity_override,
                        estimated_foot_positions_override,
                        estimated_foot_height_override,
                        estimated_contact_override,
                    )
                )
                output = (
                    self._infer_with_estimate_overrides(
                        observation_tensor,
                        history_tensor,
                        estimated_velocity_override,
                        estimated_foot_positions_override,
                        estimated_foot_height_override,
                        estimated_contact_override,
                    )
                    if has_override
                    else self.policy(observation_tensor, history_tensor)
                )
                if not isinstance(output, tuple) or len(output) != 2:
                    raise ValueError(
                        'Concurrent policy must return '
                        '(joint_action, estimated_state)'
                    )
                action, estimated_state = output
            action_values = action.detach().reshape(-1).cpu().numpy()
            state_values = estimated_state.detach().reshape(-1).cpu().numpy()
            if action_values.shape != (self.manifest.action_count,):
                raise ValueError(
                    'Concurrent policy returned an invalid joint-action width'
                )
            if state_values.shape != (self.manifest.estimated_state_count,):
                raise ValueError(
                    'Concurrent policy returned an invalid estimated-state '
                    'width'
                )
            if (
                not np.isfinite(action_values).all()
                or not np.isfinite(state_values).all()
            ):
                raise ValueError(
                    'Concurrent policy returned non-finite output'
                )
            self.current_action = action_values.astype(
                np.float64, copy=True
            )
            self.latest_estimated_state = state_values.astype(
                np.float64, copy=True
            )
            self.latest_effective_velocity = (
                state_values[:3].astype(np.float64, copy=True)
                if estimated_velocity_override is None
                else np.asarray(
                    estimated_velocity_override, dtype=np.float64
                ).reshape(3).copy()
            )
        self.counter += 1
        return self.current_action


__all__ = ['ConcurrentPolicyRunner']
