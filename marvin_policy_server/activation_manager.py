"""Activation and deactivation ramp management for Marvin policy server.

This module encapsulates the logic governing active/inactive state transitions,
including ramp-in (activation) and ramp-out (deactivation) blending factors and
release semantics. It mirrors the original inline implementation but provides
an explicit API so the main node code is cleaner.

All state variables are kept here; the node supplies current time via a callback.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np


@dataclass
class ActivationConfig:
    activation_ramp_duration: float = 1.0
    deactivation_ramp_duration: float = 1.0
    release_after_deactivate: bool = True


@dataclass
class ActivationState:
    active: bool = False
    released: bool = False
    activation_start_time: Optional[float] = None
    deactivation_start_time: Optional[float] = None
    deactivation_start_action: Optional[np.ndarray] = None


class ActivationManager:
    """Encapsulates activation/deactivation logic with ramps and release flag."""
    def __init__(self, cfg: ActivationConfig):
        self.cfg = cfg
        self.state = ActivationState()

    # --- Public API -----------------------------------------------------
    def request_activate(self, now: float) -> Tuple[bool, str]:
        if self.state.active:
            return True, 'Already active'
        self.state.active = True
        self.state.released = False
        self.state.activation_start_time = now
        self.state.deactivation_start_time = None
        self.state.deactivation_start_action = None
        return True, 'Policy activation ramp started'

    def request_deactivate(self, now: float, current_action: np.ndarray) -> Tuple[bool, str]:
        if not self.state.active:
            return True, 'Already inactive'
        self.state.active = False
        self.state.activation_start_time = None
        if self.cfg.deactivation_ramp_duration <= 0.0:
            # Immediate
            self.state.deactivation_start_time = None
            self.state.deactivation_start_action = None
            if self.cfg.release_after_deactivate:
                self.state.released = True
            return True, 'Policy deactivated immediately'
        # Start ramp
        self.state.deactivation_start_time = now
        self.state.deactivation_start_action = current_action.copy()
        return True, 'Policy deactivation ramp started'

    def compute_activation_factor(self, now: float) -> float:
        if not self.state.active:
            return 0.0
        if self.state.activation_start_time is None:
            return 1.0
        dt = now - self.state.activation_start_time
        if dt >= self.cfg.activation_ramp_duration:
            self.state.activation_start_time = None
            return 1.0
        if self.cfg.activation_ramp_duration <= 0.0:
            return 1.0
        return max(0.0, min(1.0, dt / self.cfg.activation_ramp_duration))

    def compute_deactivation_blend(self, now: float) -> Optional[np.ndarray]:
        # Returns blended action (scaled) or None if ramp finished or not in ramp.
        if self.state.active:
            return None
        if self.state.deactivation_start_time is None:
            return None
        if self.state.deactivation_start_action is None:
            return None
        dt = now - self.state.deactivation_start_time
        dur = self.cfg.deactivation_ramp_duration
        if dur <= 0.0:
            # Should have been handled in request_deactivate
            self.state.deactivation_start_time = None
            return None
        if dt >= dur:
            # Ramp complete
            self.state.deactivation_start_time = None
            self.state.deactivation_start_action = None
            if self.cfg.release_after_deactivate:
                self.state.released = True
            return None
        t = max(0.0, min(1.0, dt / dur))
        return (1.0 - t) * self.state.deactivation_start_action

    def is_active(self) -> bool:
        return self.state.active

    def is_released(self) -> bool:
        return self.state.released

    def clear_after_publish_default(self):
        # Helper to reset action history after final default stance publish.
        if not self.state.active and self.state.deactivation_start_time is None:
            # Ready state after deactivation
            pass

__all__ = [
    'ActivationConfig',
    'ActivationState',
    'ActivationManager'
]
