"""Optional same-host encoder diagnostics transported with their joint values."""
import math


def decode_snapshot(message, joint_names, joint_state_factory):
    """Reject ambiguous/incomplete control values; missing timing remains diagnostic."""
    if (len(message.joint_names) != len(message.interface_values)
            or len(set(message.joint_names)) != len(message.joint_names)):
        raise ValueError('Invalid dynamic joint names')
    rows = dict(zip(message.joint_names, message.interface_values))
    state = joint_state_factory()
    state.header = message.header
    state.name = list(joint_names)
    positions, velocities, receipts = [], [], {}
    for name in joint_names:
        row = rows.get(name)
        if (row is None or len(row.interface_names) != len(row.values)
                or len(set(row.interface_names)) != len(row.interface_names)):
            raise ValueError('Invalid dynamic joint interfaces')
        values = dict(zip(row.interface_names, row.values))
        position, velocity = values.get('position'), values.get('velocity')
        if (position is None or velocity is None
                or not math.isfinite(position) or not math.isfinite(velocity)):
            raise ValueError('Missing or non-finite joint position/velocity')
        positions.append(position)
        velocities.append(velocity)
        receipts[name] = values.get('encoder_receipt_monotonic_s', float('nan'))
    state.position = positions
    state.velocity = velocities
    return state, receipts
