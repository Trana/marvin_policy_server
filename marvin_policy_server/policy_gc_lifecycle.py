"""Bound cyclic garbage collection to inactive native policy windows."""

from enum import Enum, auto
import gc


class _WindowState(Enum):
    IDLE = auto()
    PROTECTED = auto()
    RESTORE_PENDING = auto()


class PolicyGcLifecycle:
    """Suppress automatic cyclic GC while native policy output may be active.

    Python reference counting remains active. The prior cyclic-GC state is
    restored without collecting only after native output is known to have
    stopped.
    """

    def __init__(self, gc_api=None):
        self._gc = gc if gc_api is None else gc_api
        self._state = _WindowState.IDLE
        self._restore_enabled = False

    def enter_output_window(self):
        """Protect the native output window without pausing activation."""
        if self._state is _WindowState.PROTECTED:
            return
        if self._state is _WindowState.RESTORE_PENDING:
            self._state = _WindowState.PROTECTED
            return

        self._restore_enabled = self._gc.isenabled()
        if self._restore_enabled:
            self._gc.disable()
        self._state = _WindowState.PROTECTED

    def request_restore(self):
        """Defer restoration until native output release is confirmed."""
        if self._state is _WindowState.PROTECTED:
            self._state = _WindowState.RESTORE_PENDING

    def observe_native_state(self, *, active, output_released):
        """Restore after a native status proves that output has stopped."""
        if (self._state is _WindowState.RESTORE_PENDING and
                not active and output_released):
            self.confirm_output_stopped()

    def confirm_output_stopped(self):
        """Restore the prior GC state without running a collection."""
        if self._state is _WindowState.IDLE:
            return
        if self._restore_enabled:
            self._gc.enable()
        self._restore_enabled = False
        self._state = _WindowState.IDLE
