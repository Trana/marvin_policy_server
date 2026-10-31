"""Policy inference and held actions for Marvin policy server.

This module wraps the logic for running the TorchScript policy at a reduced
rate while tracking the previous action. The node supplies deadline decisions;
callers omitting ``infer`` retain callback-count decimation.
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

    def reset(self):
        self.counter = 0
        self.previous_action.fill(0.)
        self.current_action.fill(0.)

    def step(self, obs: np.ndarray | None, *, infer: bool | None = None):
        """Infer or hold; use legacy decimation only when no decision is supplied."""
        should_infer = self.should_infer() if infer is None else infer
        if should_infer:
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
