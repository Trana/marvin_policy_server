"""One-entry cache for deterministic joint position clipping, not publication."""

import numpy as np


class CommandClipCache:
    """Compare array contents so in-place changes cannot reuse an old result."""

    def __init__(self):
        self._key = None
        self._values = ()
        self._clipped_indices = ()

    def apply(self, command, lower, upper):
        key = tuple((value.shape, value.dtype, value.tobytes())
                    for value in (command, lower, upper))
        if key != self._key:
            clipped = np.clip(command, lower, upper)
            indices = tuple(np.flatnonzero(np.abs(clipped - command) > 1e-9))
            # Store immutable results. Callers must not mutate cached values.
            self._values = tuple(clipped.tolist())
            self._clipped_indices = indices
            self._key = key
        return list(self._values), self._clipped_indices
