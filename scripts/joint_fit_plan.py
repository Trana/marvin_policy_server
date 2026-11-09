"""No-ROS, bounded excitation planning from an inherited command pose."""

import math

PROFILE_ID = 'marvin-stand-all-joints-fit-v1'


def plan_joints(profile, positions, measured=None):
    names = profile['joints']
    if profile['joint_test_profile.id'] != PROFILE_ID or len(positions) != len(names):
        raise ValueError('Unexpected fitting profile or pose shape')
    if measured is None or len(measured) != len(names):
        raise ValueError('Fresh measured pose is required')
    start = profile['joint_test_profile.chirp_start_hz']
    end = profile['joint_test_profile.chirp_end_hz']
    if (start, end) != (0.05, 0.15):
        raise ValueError('Unapproved fitting sweep')
    duration = 20.0
    envelope = math.pi / duration
    omega = 2 * math.pi * end
    sweep = 2 * math.pi * (end - start) / duration
    result = []
    for i, (name, q) in enumerate(zip(names, positions)):

        def limit(key):
            return profile['joint_test_profile.' + key][i]

        values = [
            q,
            limit('lower_rad'),
            limit('upper_rad'),
            limit('speed_rad_s'),
            limit('acceleration_rad_s2'),
            limit('max_amplitude_rad'),
            measured[i],
            profile['max_handover_error'][i],
        ]
        if not all(math.isfinite(v) for v in values) or not values[1] <= q <= values[2]:
            raise ValueError('Invalid or out-of-bounds baseline: ' + name)
        clearance = min(q - values[1], values[2] - q) - math.radians(0.25)
        # Leave room even if the joint initially does not follow the target.
        tracking_room = (
            profile['max_handover_error'][i] - abs(q - measured[i]) - math.radians(0.25)
        )
        amplitude = min(
            math.radians(5),
            limit('max_amplitude_rad'),
            clearance,
            tracking_room,
            0.95 * limit('speed_rad_s') / (omega + envelope),
            0.95
            * limit('acceleration_rad_s2')
            / (omega**2 + 2 * omega * envelope + 2 * envelope**2 + sweep),
        )
        result.append({
            'joint': name,
            'baseline_rad': q,
            'amplitude_rad': max(0.0, amplitude),
            'transition_s': 1.0,
            'excitation_s': duration,
            'start_hz': start,
            'end_hz': end,
            'status': 'ready' if amplitude >= math.radians(0.25) else 'insufficient_clearance',
        })
    return result
