"""Select the action representation fed back into actor history."""

import numpy as np


VALID_ACTION_HISTORY_FEEDBACK_MODES = frozenset(
    {'raw', 'zero', 'applied'}
)


def action_history_feedback(
    mode: str,
    raw_action: np.ndarray,
    policy_joint_target: np.ndarray,
    default_position: np.ndarray,
    action_scale: float,
) -> np.ndarray:
    """Return one actor-history frame for the selected diagnostic mode."""
    raw = np.asarray(raw_action, dtype=np.float64)
    target = np.asarray(policy_joint_target, dtype=np.float64)
    default = np.asarray(default_position, dtype=np.float64)
    if raw.shape != target.shape or raw.shape != default.shape:
        raise ValueError(
            'raw action, policy target, and default position must have '
            'matching shapes'
        )
    if mode not in VALID_ACTION_HISTORY_FEEDBACK_MODES:
        choices = ', '.join(sorted(VALID_ACTION_HISTORY_FEEDBACK_MODES))
        raise ValueError(
            f'action history feedback mode must be one of: {choices}; '
            f'got {mode!r}'
        )
    if mode == 'raw':
        return raw.copy()
    if mode == 'zero':
        return np.zeros_like(raw)
    if not np.isfinite(action_scale) or action_scale <= 0.0:
        raise ValueError('action scale must be finite and greater than zero')
    return (target - default) / action_scale
