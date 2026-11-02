"""Cyclic-GC ownership across native policy output windows."""

from marvin_policy_server.policy_gc_lifecycle import PolicyGcLifecycle


class FakeGc:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.calls = []

    def isenabled(self):
        self.calls.append('isenabled')
        return self.enabled

    def collect(self):
        self.calls.append('collect')

    def disable(self):
        self.calls.append('disable')
        self.enabled = False

    def enable(self):
        self.calls.append('enable')
        self.enabled = True


def test_enabled_gc_is_disabled_until_confirmed_release():
    gc = FakeGc()
    lifecycle = PolicyGcLifecycle(gc)

    lifecycle.enter_output_window()
    lifecycle.request_restore()
    lifecycle.observe_native_state(active=False, output_released=False)

    assert gc.calls == ['isenabled', 'disable']
    assert not gc.enabled

    lifecycle.observe_native_state(active=False, output_released=True)
    assert gc.calls == ['isenabled', 'disable', 'enable']
    assert gc.enabled


def test_initially_disabled_gc_is_neither_collected_nor_enabled():
    gc = FakeGc(enabled=False)
    lifecycle = PolicyGcLifecycle(gc)

    lifecycle.enter_output_window()
    lifecycle.request_restore()
    lifecycle.confirm_output_stopped()

    assert gc.calls == ['isenabled']
    assert not gc.enabled


def test_reactivation_cancels_pending_restore_without_collecting():
    gc = FakeGc()
    lifecycle = PolicyGcLifecycle(gc)

    lifecycle.enter_output_window()
    lifecycle.request_restore()
    lifecycle.enter_output_window()
    lifecycle.observe_native_state(active=False, output_released=True)

    assert gc.calls == ['isenabled', 'disable']
    assert not gc.enabled

    lifecycle.request_restore()
    lifecycle.observe_native_state(active=False, output_released=True)
    assert gc.calls[-1] == 'enable'
    assert gc.enabled


def test_contradictory_active_released_status_does_not_restore():
    gc = FakeGc()
    lifecycle = PolicyGcLifecycle(gc)

    lifecycle.enter_output_window()
    lifecycle.request_restore()
    lifecycle.observe_native_state(active=True, output_released=True)

    assert gc.calls == ['isenabled', 'disable']
    assert not gc.enabled


def test_repeated_lifecycle_events_are_idempotent():
    gc = FakeGc()
    lifecycle = PolicyGcLifecycle(gc)

    lifecycle.enter_output_window()
    lifecycle.enter_output_window()
    lifecycle.request_restore()
    lifecycle.request_restore()
    lifecycle.confirm_output_stopped()
    lifecycle.confirm_output_stopped()

    assert gc.calls == ['isenabled', 'disable', 'enable']
