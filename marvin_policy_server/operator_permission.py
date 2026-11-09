"""Fail-closed permission for requested velocity, never for balance inference."""

import time


class OperatorPermission:
    def __init__(self, required=False, clock=time.monotonic, timeout=0.3):
        self.required, self.clock, self.timeout = required, clock, timeout
        self.last_at = None
        self.permitted = False

    def update(self, permitted):
        self.permitted = bool(permitted)
        self.last_at = self.clock()

    def allowed(self):
        return not self.required or (
            self.permitted
            and self.last_at is not None
            and self.clock() - self.last_at < self.timeout
        )
