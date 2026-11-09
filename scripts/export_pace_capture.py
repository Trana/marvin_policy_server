#!/usr/bin/env python3
"""Offline, explicit-clock conversion of a finalized reference bag for PACE."""

import argparse
from bisect import bisect_right
import hashlib
import json
import math
from pathlib import Path


def verify_capture(root):
    receipt = json.loads((root / 'complete.json').read_text())
    if receipt.get('schema') != 'reference-complete.v1' or not receipt.get('files'):
        raise ValueError('Missing finalized capture receipt')
    actual = {
        str(p.relative_to(root))
        for p in root.rglob('*')
        if p.is_file() and p != root / 'complete.json'
    }
    if actual != set(receipt['files']):
        raise ValueError('Capture file inventory differs from completion receipt')
    for name, identity in receipt['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError('Invalid capture file: ' + name)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != identity['sha256'] or path.stat().st_size != identity['size']:
            raise ValueError('Capture checksum mismatch: ' + name)
    return receipt


def resample(commands, feedback, names, start, end, dt_ns, max_gap_ns):
    """Same-clock source stamps only. Hold commands; interpolate feedback."""
    if not names or len(names) != len(set(names)):
        raise ValueError('Joint order must be explicit and unique')
    if not 0 < start < end or dt_ns <= 0 or max_gap_ns <= 0:
        raise ValueError('Invalid time window, sample interval or gap limit')
    if (end - start) // dt_ns > 1_000_000:
        raise ValueError('Export exceeds one million samples')
    normalized = []
    for kind, rows in [('commands', commands), ('feedback', feedback)]:
        if len(rows) < 2:
            raise ValueError('Missing ' + kind)
        stamps, values = [], []
        for row in rows:
            stamp = row[0]
            if stamp <= 0 or (stamps and stamp <= stamps[-1]):
                raise ValueError('Duplicate or regressing source timestamps in ' + kind)
            if kind == 'feedback':
                _, observed, positions = row
                if (
                    len(observed) != len(set(observed))
                    or len(observed) != len(positions)
                    or not set(names) <= set(observed)
                ):
                    raise ValueError('Invalid feedback joint mapping')
                mapping = dict(zip(observed, positions))
                value = [mapping[n] for n in names]
            else:
                value = row[1]
            if len(value) != len(names) or not all(math.isfinite(v) for v in value):
                raise ValueError('Invalid shape or nonfinite joint data')
            stamps.append(stamp)
            values.append(value)
        if start < stamps[0] or end > stamps[-1]:
            raise ValueError('Requested window is not covered by ' + kind)
        if any(b - a > max_gap_ns for a, b in zip(stamps, stamps[1:]) if b >= start and a <= end):
            raise ValueError('Recording gap exceeds limit in ' + kind)
        normalized.append((stamps, values))
    (ct, cv), (ft, fv) = normalized
    result = {'time': [], 'des_dof_pos': [], 'dof_pos': []}
    for t in range(start, end, dt_ns):
        c = bisect_right(ct, t) - 1
        f = bisect_right(ft, t) - 1
        if t - ct[c] > max_gap_ns:
            raise ValueError('Stale command')
        alpha = (t - ft[f]) / (ft[f + 1] - ft[f])
        result['time'].append((t - start) / 1e9)
        result['des_dof_pos'].append(list(cv[c]))
        result['dof_pos'].append([a + (b - a) * alpha for a, b in zip(fv[f], fv[f + 1])])
    if len(result['time']) < 2:
        raise ValueError('Too few output samples')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture', type=Path)
    p.add_argument('--output', type=Path, required=True, help='New output directory')
    p.add_argument(
        '--joint-names',
        nargs='+',
        required=True,
        help='Exact traced command array order in radians',
    )
    p.add_argument(
        '--source-id', type=int, required=True, help='Explicit traced command producer, 0..255'
    )
    p.add_argument('--command-topic', default='/marvin_joint_controller/commands_traced')
    p.add_argument('--feedback-topic', default='/joint_states')
    p.add_argument('--start-ros-ns', type=int, required=True)
    p.add_argument('--end-ros-ns', type=int, required=True)
    p.add_argument(
        '--dt-ns', type=int, required=True, help='Must equal the PACE environment step interval'
    )
    p.add_argument('--max-gap-ms', type=float, required=True)
    p.add_argument(
        '--confirm-common-clock',
        action='store_true',
        required=True,
        help='Confirm command and feedback source stamps use one clock',
    )
    args = p.parse_args()
    if not 0 <= args.source_id <= 255:
        p.error('source-id must be between 0 and 255')
    if not math.isfinite(args.max_gap_ms) or args.max_gap_ms <= 0:
        p.error('Gap limit must be finite and positive')
    receipt = verify_capture(args.capture)
    metadata = json.loads((args.capture / 'metadata.json').read_text())
    session = metadata.get('session') or {}
    if (
        session.get('joint_names') != args.joint_names
        or session.get('position_units') != 'rad'
        or not session.get('support')
        or not session.get('control_mode')
    ):
        p.error(
            'Session must specify matching joint_names, position_units=rad, support and'
            ' control_mode'
        )
    from analyze_joint_tracking import read_bag

    selection = {}
    commands, feedback = read_bag(
        args.capture / 'topics', args.command_topic, args.feedback_topic, args.source_id, selection
    )
    data = resample(
        commands,
        feedback,
        args.joint_names,
        args.start_ros_ns,
        args.end_ros_ns,
        args.dt_ns,
        int(args.max_gap_ms * 1e6),
    )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'data.json').write_text(json.dumps(data, allow_nan=False))
    report = {
        'schema': 'pace-export.v1',
        'joint_order': args.joint_names,
        'start_ros_ns': args.start_ros_ns,
        'end_ros_ns': args.end_ros_ns,
        'dt_ns': args.dt_ns,
        'max_gap_ms': args.max_gap_ms,
        'selection': selection,
        'command_topic': args.command_topic,
        'feedback_topic': args.feedback_topic,
        'input_receipt': receipt,
        'session': session,
        'interpolation': 'command: zero-order hold; feedback: linear',
        'limits': (
            'Requested targets, not verified hardware writes. Source stamps may be publication'
            ' times. Resampling adds no sensor information. This is data-format validation, not'
            ' approval for fitting. Review fixed-base support, excitation, gains, filters and'
            ' model configuration separately.'
        ),
    }
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    print(
        f'Exported {len(data["time"])} samples. Convert data.json to tensors in the PACE'
        ' environment.'
    )


if __name__ == '__main__':
    main()
