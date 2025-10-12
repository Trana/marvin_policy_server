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
    def __init__(self, cfg: ActivationConfig, logger: Optional[object] = None):
        self.cfg = cfg
        self.state = ActivationState()
        # Use provided ROS 2 logger if given; otherwise fall back to Python logging
        self._log = logger

    # Internal: format-and-log helper to handle ROS logger (single-arg) and Python logger uniformly
    def _logf(self, level: str, msg: str, *args) -> None:
        text = msg % args if args else msg
        try:
            method = getattr(self._log, level)
        except Exception:
            method = None
        if callable(method):
            try:
                method(text)
                return
            except Exception:
                pass
        

    # --- Public API -----------------------------------------------------
    def request_activate(self, now: float) -> Tuple[bool, str]:
        if self.state.active:
            self._logf(
                'debug',
                "Activation requested but already active | released=%s, ramp_started=%s",
                self.state.released,
                self.state.activation_start_time is not None,
            )
            return True, 'Already active'
        self._logf(
            'info',
            "Policy activation requested | t=%.3f, ramp_duration=%.3fs, was_released=%s",
            now,
            max(0.0, self.cfg.activation_ramp_duration),
            self.state.released,
        )
        self.state.active = True
        self.state.released = False
        self.state.activation_start_time = now
        self.state.deactivation_start_time = None
        self.state.deactivation_start_action = None
        if self.cfg.activation_ramp_duration <= 0.0:
            self._logf(
                'info',
                "Activation ramp duration <= 0, treating as immediate activation"
            )
        return True, 'Policy activation ramp started'

    def request_deactivate(self, now: float, current_action: np.ndarray) -> Tuple[bool, str]:
        if not self.state.active:
            self._logf(
                'debug',
                "Deactivation requested but already inactive | released=%s",
                self.state.released,
            )
            return True, 'Already inactive'
        self.state.active = False
        self.state.activation_start_time = None
        if self.cfg.deactivation_ramp_duration <= 0.0:
            # Immediate
            self.state.deactivation_start_time = None
            self.state.deactivation_start_action = None
            if self.cfg.release_after_deactivate:
                self.state.released = True
            self._logf(
                'info',
                "Policy deactivated immediately | t=%.3f, release=%s",
                now,
                self.state.released,
            )
            return True, 'Policy deactivated immediately'
        # Start ramp
        self.state.deactivation_start_time = now
        try:
            self.state.deactivation_start_action = current_action.copy()
        except Exception:  # noqa: BLE001 - being defensive on numpy input
            self._logf(
                'warning',
                "Current action not copyable, using as-is | type=%s",
                type(current_action),
            )
            self.state.deactivation_start_action = current_action
        act = self.state.deactivation_start_action
        self._logf(
            'info',
            "Policy deactivation ramp started | t=%.3f, ramp_duration=%.3fs, action_shape=%s, action_norm=%.6f, will_release=%s",
            now,
            max(0.0, self.cfg.deactivation_ramp_duration),
            getattr(act, 'shape', None),
            float(np.linalg.norm(act)) if isinstance(act, np.ndarray) else float('nan'),
            self.cfg.release_after_deactivate,
        )
        return True, 'Policy deactivation ramp started'

    def compute_activation_factor(self, now: float) -> float:
        if not self.state.active:
            self._logf('debug', "Activation factor queried while inactive -> 0.0")
            return 0.0
        if self.state.activation_start_time is None:
            # Already fully active
            self._logf('debug', "Activation ramp already complete -> 1.0")
            return 1.0
        dt = now - self.state.activation_start_time
        dur = max(0.0, self.cfg.activation_ramp_duration)
        if dur <= 0.0:
            self._logf('debug', "Activation duration <= 0 -> immediate active | dt=%.3f", dt)
            self.state.activation_start_time = None
            return 1.0
        if dt >= dur:
            self.state.activation_start_time = None
            self._logf('info', "Activation ramp complete | dt=%.3f/%.3fs -> factor=1.0", dt, dur)
            return 1.0
        factor = max(0.0, min(1.0, dt / dur))
        self._logf('debug', "Activation ramp progress | dt=%.3f/%.3fs -> factor=%.3f", dt, dur, factor)
        return factor

    def compute_deactivation_blend(self, now: float) -> Optional[np.ndarray]:
        # Returns blended action (scaled) or None if ramp finished or not in ramp.
        if self.state.active:
            self._logf('debug', "Deactivation blend queried while active -> None (no ramp)")
            return None
        if self.state.deactivation_start_time is None:
            self._logf('debug', "No deactivation ramp in progress -> None")
            return None
        if self.state.deactivation_start_action is None:
            self._logf('debug', "Missing start action for deactivation ramp -> None")
            return None
        dt = now - self.state.deactivation_start_time
        dur = max(0.0, self.cfg.deactivation_ramp_duration)
        if dur <= 0.0:
            # Should have been handled in request_deactivate
            self.state.deactivation_start_time = None
            self._logf('warning', "Deactivation duration <= 0 encountered in blend | treating as complete")
            return None
        if dt >= dur:
            # Ramp complete
            self.state.deactivation_start_time = None
            self.state.deactivation_start_action = None
            if self.cfg.release_after_deactivate:
                self.state.released = True
            self._logf('info', "Deactivation ramp complete | dt=%.3f/%.3fs, released=%s", dt, dur, self.state.released)
            return None
        t = max(0.0, min(1.0, dt / dur))
        act = self.state.deactivation_start_action
        blended = (1.0 - t) * act
        try:
            act_norm = float(np.linalg.norm(act))
            blended_norm = float(np.linalg.norm(blended))
        except Exception:  # noqa: BLE001
            act_norm = float('nan')
            blended_norm = float('nan')
        self._logf('debug', "Deactivation ramp progress | dt=%.3f/%.3fs, t=%.3f, start_norm=%.6f, blended_norm=%.6f", dt, dur, t, act_norm, blended_norm)
        return blended

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
