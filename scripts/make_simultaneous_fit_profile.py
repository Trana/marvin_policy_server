#!/usr/bin/env python3
"""Generate an opt-in test profile. Does not connect to ROS or activate motors."""
import argparse
import copy
import math
from pathlib import Path
import yaml

PROFILE_ID = 'marvin-stand-simultaneous-fit-v1'
SHORT_PROFILE_ID = 'marvin-stand-simultaneous-report-only-8s-v1'
WIDE_SHORT_PROFILE_ID = 'marvin-stand-simultaneous-7deg-3hz-report-only-8s-v1'
TEN_SHORT_PROFILE_ID = 'marvin-stand-simultaneous-10deg-3hz-report-only-8s-v1'
FAST_SHORT_PROFILE_ID = 'marvin-stand-simultaneous-10deg-4hz-report-only-8s-v1'
FIVE_HZ_PROFILE_ID = 'marvin-stand-simultaneous-10deg-5hz-report-only-8s-v1'
LONG_PROFILE_ID = 'marvin-stand-simultaneous-10deg-5hz-report-only-60s-v1'


def build_profile(template, limits, standing_pose_envelope=False, short_report_only=False,
                  short_7deg_3hz=False, short_10deg_3hz=False, short_10deg_4hz=False,
                  short_10deg_5hz=False, long_10deg_5hz=False):
    if long_10deg_5hz:
        if any((short_report_only, short_7deg_3hz, short_10deg_3hz,
                short_10deg_4hz, short_10deg_5hz)):
            raise ValueError('Select only one duration/frequency profile')
        result = build_profile(template, limits, standing_pose_envelope, short_10deg_5hz=True)
        p = result['/joint_test_controller']['ros__parameters']
        p['joint_test_profile.id'] = LONG_PROFILE_ID
        p['joint_test_profile.max_total_seconds'] = 61.
        return result
    if sum((short_10deg_3hz,short_10deg_4hz,short_10deg_5hz))>1:
        raise ValueError('Select only one short frequency profile')
    if short_10deg_4hz or short_10deg_5hz:
        if short_10deg_3hz:
            raise ValueError('Select only one short frequency profile')
        short_10deg_3hz = True
    if short_10deg_3hz and (not standing_pose_envelope or short_report_only or short_7deg_3hz):
        raise ValueError('10-degree short test requires standing envelope and excludes other short modes')
    if short_7deg_3hz and (not standing_pose_envelope or short_report_only):
        raise ValueError('7-degree short test requires standing envelope and excludes other short mode')
    result = copy.deepcopy(template)
    p = result['/joint_test_controller']['ros__parameters']
    names = p['joints']
    if len(names) != 12 or len(set(names)) != 12:
        raise ValueError('Expected twelve distinct Marvin joints')
    p['joint_test_profile.id'] = PROFILE_ID
    for field, source in [('lower_rad', 'min_position'), ('upper_rad', 'max_position')]:
        p['joint_test_profile.'+field] = [limits['limits'][name][source] for name in names]
    if standing_pose_envelope:
        # Operator-approved test envelope, not measured mechanical stops.
        # Extend only the sides needed by the recorded IK stance plus ±5 degrees.
        for name, field, angle in (
                ('FL_thigh_joint','upper_rad',44.),
                ('FR_thigh_joint','lower_rad',-44.),
                ('RL_thigh_joint','upper_rad',58.),
                ('RR_thigh_joint','lower_rad',-58.)):
            p['joint_test_profile.'+field][names.index(name)] = math.radians(angle)
    p['joint_test_profile.speed_rad_s'] = [1.2]*12
    p['joint_test_profile.acceleration_rad_s2'] = [15.]*12
    p['joint_test_profile.max_amplitude_rad'] = [math.radians(5)]*12
    p['max_handover_error'] = [math.radians(5)]*12
    p['joint_test_profile.chirp_start_hz'] = .1
    p['joint_test_profile.chirp_end_hz'] = 2.
    p['joint_test_profile.max_total_seconds'] = 21.
    p['joint_test_profile.max_cycles'] = 1
    if short_report_only:
        p['joint_test_profile.id'] = SHORT_PROFILE_ID
        p['joint_test_profile.max_total_seconds'] = 9.
    if short_7deg_3hz or short_10deg_3hz:
        p['joint_test_profile.id'] = TEN_SHORT_PROFILE_ID if short_10deg_3hz else WIDE_SHORT_PROFILE_ID
        p['joint_test_profile.max_total_seconds'] = 9.
        p['joint_test_profile.chirp_end_hz'] = 3.
        p['joint_test_profile.max_amplitude_rad'] = [math.radians(10 if short_10deg_3hz else 7)]*12
        p['joint_test_profile.speed_rad_s'] = [3.5 if short_10deg_3hz else 2.5]*12
        p['joint_test_profile.acceleration_rad_s2'] = [70. if short_10deg_3hz else 50.]*12
        front, rear = (49.,63.) if short_10deg_3hz else (46.,60.)
        for name, field, angle in (
                ('FL_thigh_joint','upper_rad',front), ('FR_thigh_joint','lower_rad',-front),
                ('RL_thigh_joint','upper_rad',rear), ('RR_thigh_joint','lower_rad',-rear)):
            p['joint_test_profile.'+field][names.index(name)] = math.radians(angle)
    p['joint_test_profile.simultaneous_directions'] = [
        (-1. if name.startswith(('FR_', 'RR_')) else 1.) *
        (-1. if '_calf_' in name else 1.) for name in names]
    if short_10deg_4hz:
        p['joint_test_profile.id'] = FAST_SHORT_PROFILE_ID
        p['joint_test_profile.chirp_end_hz'] = 4.
        p['joint_test_profile.speed_rad_s'] = [4.6]*12
        p['joint_test_profile.acceleration_rad_s2'] = [120.]*12
    if short_10deg_5hz:
        p['joint_test_profile.id'] = FIVE_HZ_PROFILE_ID
        p['joint_test_profile.chirp_end_hz'] = 5.
        p['joint_test_profile.speed_rad_s'] = [5.7]*12
        p['joint_test_profile.acceleration_rad_s2'] = [180.]*12
    return result


def simultaneous_trial(profile, positions, measured):
    names = profile['joints']
    fast = profile['joint_test_profile.id'] == FAST_SHORT_PROFILE_ID
    long = profile['joint_test_profile.id'] == LONG_PROFILE_ID
    five = long or profile['joint_test_profile.id'] == FIVE_HZ_PROFILE_ID
    ten = five or profile['joint_test_profile.id'] in (TEN_SHORT_PROFILE_ID,FAST_SHORT_PROFILE_ID)
    wide = ten or profile['joint_test_profile.id'] == WIDE_SHORT_PROFILE_ID
    amplitude = math.radians(10 if ten else 7 if wide else 5)
    short = wide or profile['joint_test_profile.id'] == SHORT_PROFILE_ID
    if (profile['joint_test_profile.id'] not in (PROFILE_ID, SHORT_PROFILE_ID, WIDE_SHORT_PROFILE_ID,TEN_SHORT_PROFILE_ID,FAST_SHORT_PROFILE_ID,FIVE_HZ_PROFILE_ID,LONG_PROFILE_ID) or len(names) != 12 or
            len(positions) != 12 or len(measured) != 12):
        raise ValueError('Invalid simultaneous profile or pose')
    end_hz = 5. if five else 4. if fast else 3. if wide else 2.
    if (profile['joint_test_profile.chirp_start_hz'], profile['joint_test_profile.chirp_end_hz']) != (.1, end_hz):
        raise ValueError('Unexpected sweep frequencies')
    for i, name in enumerate(names):
        q, actual = positions[i], measured[i]
        lower = profile['joint_test_profile.lower_rad'][i]
        upper = profile['joint_test_profile.upper_rad'][i]
        error = profile['max_handover_error'][i]
        if not all(math.isfinite(x) for x in (q, actual, lower, upper, error)):
            raise ValueError('Nonfinite pose or bound: '+name)
        if not (lower <= q-amplitude and q+amplitude <= upper):
            raise ValueError('Starting pose lacks requested amplitude clearance: '+name)
        if not lower <= actual <= upper or abs(q-actual) > error:
            raise ValueError('Initial position or tracking error exceeds bound: '+name)
    # Reserved report-only profiles still check initial tracking.
    return dict(joint='__all__', joints=list(names), baseline_rad=list(positions),
        amplitude_rad=amplitude, transition_s=1., excitation_s=60. if long else 8. if short else 20.,
        start_hz=.1, end_hz=end_hz, status='ready')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--limits', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--standing-pose-envelope', action='store_true',
                        help='Explicit temporary thigh extensions for the approved IK standing test')
    parser.add_argument('--short-report-only', action='store_true',
                        help='Eight-second test without dynamic tracking abort; initial tracking still checked')
    parser.add_argument('--short-7deg-3hz', action='store_true',
                        help='Approved eight-second 7-degree 0.1–3 Hz test with expanded temporary bounds')
    parser.add_argument('--short-10deg-3hz', action='store_true',
                        help='Approved eight-second 10-degree 0.1–3 Hz test with expanded temporary bounds')
    parser.add_argument('--short-10deg-4hz', action='store_true',
                        help='Approved eight-second 10-degree 0.1–4 Hz test; same position bounds')
    parser.add_argument('--short-10deg-5hz', action='store_true',
                        help='Approved eight-second 10-degree 0.1–5 Hz test; same position bounds')
    parser.add_argument('--long-10deg-5hz', action='store_true',
                        help='Sixty-second 10-degree 0.1–5 Hz test; same position bounds')
    args = parser.parse_args()
    profile = build_profile(yaml.safe_load(args.template.read_text()), yaml.safe_load(args.limits.read_text()),
                            args.standing_pose_envelope, args.short_report_only, args.short_7deg_3hz,
                            args.short_10deg_3hz, args.short_10deg_4hz, args.short_10deg_5hz,
                            args.long_10deg_5hz)
    with args.output.open('x') as stream:
        yaml.safe_dump(profile, stream, sort_keys=False)


if __name__ == '__main__':
    main()
