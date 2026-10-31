from marvin_policy_server.operator_permission import OperatorPermission


def test_missing_expired_and_negative_permission_zeroes_motion_gate():
    now = [0.]
    gate = OperatorPermission(required=True, clock=lambda: now[0])
    assert not gate.allowed()
    gate.update(True)
    assert gate.allowed()
    now[0] = .31
    assert not gate.allowed()
    gate.update(False)
    assert not gate.allowed()
    gate.update(True)
    assert gate.allowed()


def test_explicit_legacy_profile_does_not_require_permission():
    assert OperatorPermission(required=False).allowed()
