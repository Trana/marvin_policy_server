"""Policy inference and decimation management for Marvin policy server.

This module wraps the logic for running the TorchScript policy at a reduced
rate (decimation) while tracking the previous action.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class PolicyRunnerConfig:
    decimation: int = 4


class PolicyRunner:
    def __init__(self, policy, action_dim: int, cfg: PolicyRunnerConfig):
        self.policy = policy
        self.cfg = cfg
        self.counter = 0
        self.previous_action = np.zeros(action_dim)
        self.current_action = np.zeros(action_dim)

    def should_infer(self) -> bool:
        return self.counter % self.cfg.decimation == 0

    def step(self, obs: np.ndarray | None):
        """Advance one control tick, running policy only on decimated inference ticks."""
        if self.should_infer():
            if obs is None:
                raise ValueError("Policy observation is required on inference ticks")
            with torch.no_grad():
                t_obs = torch.from_numpy(obs).view(1, -1).float()
                act = self.policy(t_obs).detach().view(-1).cpu().numpy()
            self.current_action = act
            self.previous_action = act.copy()
        self.counter += 1
        return self.current_action

__all__ = [
    'PolicyRunner',
    'PolicyRunnerConfig'
]
