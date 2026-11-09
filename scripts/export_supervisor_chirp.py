#!/usr/bin/env python3
"""Export one completed supervisor chirp from an integrity-checked recorder bag."""

import argparse
from bisect import bisect_right
import hashlib
import json
from pathlib import Path
import sqlite3

from export_pace_capture import resample

JOINTS = [
    f'{leg}_{joint}_joint'
    for joint in ('hip', 'thigh', 'calf')
    for leg in ('FL', 'RL', 'FR', 'RR')
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_decisions(events, request_id):
    requested = [
        e
        for e in events
        if e.get('request_id') == request_id and e['kind'] == 'position_diagnostic_requested'
    ]
    completed = [
        e
        for e in events
        if e.get('request_id') == request_id and e['kind'] == 'position_diagnostic_completed'
    ]
    require(len(requested) == len(completed) == 1, 'Need one requested and one completed chirp')
    start, end = requested[0], completed[0]
    require(start['trajectory_kind'] == end['trajectory_kind'] == 'chirp', 'Not a chirp')
    require(start['supervisor_instance'] == end['supervisor_instance'], 'Supervisor changed')
    require(start['sequence'] < end['sequence'], 'Completion precedes request')
    require(
        not any(
            e.get('request_id') == request_id and e['kind'] == 'position_diagnostic_cancelled'
            for e in events
        ),
        'Cancelled request',
    )
    window = [
        e for e in events if start['monotonic_ns'] <= e['monotonic_ns'] <= end['monotonic_ns']
    ]
    require(
        all(
            e['supervisor_instance'] == start['supervisor_instance']
            and e['telemetry_dropped'] == 0
            for e in window
        ),
        'Mixed supervisor or dropped telemetry',
    )
    require(
        all(b['sequence'] == a['sequence'] + 1 for a, b in zip(window, window[1:])),
        'Telemetry sequence gap',
    )
    decisions = [
        e for e in window if e['kind'] == 'command_decision' and e['owner'] == 'diagnostic'
    ]
    require(
        len(decisions) > 50
        and all(e['accepted'] and e['owner'] == 'diagnostic' for e in decisions),
        'Rejected or mixed-owner commands',
    )
    require(
        not any(
            e['kind'] == 'command_decision'
            and e['owner'] != 'diagnostic'
            and decisions[0]['monotonic_ns'] <= e['monotonic_ns'] <= decisions[-1]['monotonic_ns']
            for e in window
        ),
        'Mixed owners during chirp',
    )
    require(len({e['generation'] for e in decisions}) == 1, 'Diagnostic generation changed')
    require(
        all(
            b['command_sequence'] == a['command_sequence'] + 1
            for a, b in zip(decisions, decisions[1:])
        ),
        'Command sequence gap',
    )
    require(
        all(type(e.get('source_stamp_ns')) is int and e['source_stamp_ns'] > 0 for e in decisions),
        'Missing exact publication timestamps; old captures cannot be inferred',
    )
    require(
        all(b['source_stamp_ns'] > a['source_stamp_ns'] for a, b in zip(decisions, decisions[1:])),
        'Command clock regressed or duplicated',
    )
    require(
        max(abs(a - b) for a, b in zip(decisions[0]['targets'], decisions[-1]['targets'])) < 1e-4,
        'Chirp did not return to baseline',
    )
    return start, end, decisions


def check_context(rows, decisions, instance):
    for service in ('supervisor', 'configuration'):
        observations = [r for r in rows if r['service'] == service]
        stamps = [r['monotonic_ns'] for r in observations]
        require(all(b > a for a, b in zip(stamps, stamps[1:])), 'Context timestamps regressed')
        for decision in decisions:
            index = bisect_right(stamps, decision['monotonic_ns']) - 1
            require(index >= 0, 'Context missing at chirp start')
            row = observations[index]
            require(
                decision['monotonic_ns'] - row['monotonic_ns'] <= 3_000_000_000
                and 'error' not in row,
                'Stale or failed context in chirp',
            )
            require(
                row['payload'].get('supervisor_instance') == instance,
                'Context supervisor mismatch',
            )


def export(capture, request_id, session_path, output, max_gap_ms=100.0):
    from marvin_trace_msgs.msg import TracedFloat64MultiArray
    from rclpy.serialization import deserialize_message
    from recording_manager.export import verified_files
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String

    capture = capture.resolve()
    files = {name: path for path, name in verified_files(capture.parent, capture.name)}
    manifest = json.loads(files['manifest.json'].read_text())
    require(
        manifest['schema_version'] == 'robot-recording.v1' and manifest['state'] == 'complete',
        'Complete recorder manifest required',
    )
    receipt = json.loads(files['worker-result.json'].read_text())
    require(
        receipt['clean']
        and receipt['dropped'] == 0
        and not receipt['missing_runtime_streams']
        and receipt['health_violation_samples'] == 0,
        'Unhealthy recorder capture',
    )
    session = json.loads(session_path.read_text())
    require(
        session.get('schema') == 'marvin-identification-session.v1',
        'Identification session required',
    )
    require(
        session.get('robot_id') == manifest['robot_id']
        and session.get('recording_id') == capture.name,
        'Session identity mismatch',
    )
    require(
        session.get('support') == 'fixed_base_no_contacts',
        'Fixed base and no contacts must be documented',
    )
    require(
        session.get('kind') == 'simulation'
        and session.get('robot_id') == 'marvin-sim'
        and session.get('clock') == 'ros_sim_time',
        'This verified adapter currently supports simulation; physical calibration provenance is'
        ' not yet implemented',
    )
    require(len(session.get('artifacts', [])) >= 3, 'Scene and model identities required')
    for artifact in session['artifacts']:
        require(
            digest(artifact['path']) == artifact['sha256'],
            'Simulation source changed since capture',
        )
    events, commands, feedback = [], [], []
    for name, path in files.items():
        if name.startswith('topics/') and name.endswith('.db3'):
            with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
                topics = dict(db.execute('select id,name from topics'))
                for topic, data in db.execute(
                    'select topic_id,data from messages order by timestamp,id'
                ):
                    name = topics[topic]
                    if name == '/sim/robot_control/supervisor/telemetry':
                        events.append(json.loads(deserialize_message(data, String).data))
                    elif name == '/sim/marvin_joint_controller/commands_traced':
                        m = deserialize_message(data, TracedFloat64MultiArray)
                        if m.trace_id >> 56 == 5:
                            commands.append((
                                m.stamp.sec * 1_000_000_000 + m.stamp.nanosec,
                                list(m.data),
                                m.trace_id,
                            ))
                    elif name == '/isaac_joint_states':
                        m = deserialize_message(data, JointState)
                        feedback.append((
                            m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec,
                            list(m.name),
                            list(m.position),
                        ))
    start, end, decisions = selected_decisions(events, request_id)
    instances = [
        json.loads(path.read_text())
        for name, path in files.items()
        if name.startswith('motion-configuration-')
    ]
    configs = [c for c in instances if c['supervisor_instance'] == start['supervisor_instance']]
    require(
        configs and all(c['joint_names'] == JOINTS for c in configs),
        'Missing configuration/joint order',
    )
    for c in configs:
        require(
            c['topics']['targets'] == '/sim/marvin_joint_controller/commands_traced'
            and c['topics']['joints'] == '/isaac_joint_states',
            'Configuration topics differ',
        )
        for key in ('pose', 'limits'):
            require(
                hashlib.sha256(c[key]['content'].encode()).hexdigest() == c[key]['sha256'],
                'Embedded configuration hash mismatch',
            )
    rows = [json.loads(line) for line in files['runtime-context.jsonl'].read_text().splitlines()]
    require(
        not any(
            'error' in r
            and decisions[0]['monotonic_ns'] <= r['monotonic_ns'] <= decisions[-1]['monotonic_ns']
            for r in rows
        ),
        'Context error during selected chirp',
    )
    check_context(rows, decisions, start['supervisor_instance'])
    provenance = [
        r for r in session.get('requests', []) if r['request']['request_id'] == request_id
    ]
    require(
        len(provenance) == 1
        and provenance[0]['request']['chirp'] == start['trajectory']
        and provenance[0]['request']['duration_s'] == start['duration_s'],
        'Session trajectory mismatch',
    )
    lo, hi = decisions[0]['source_stamp_ns'], decisions[-1]['source_stamp_ns']
    selected = [c for c in commands if lo <= c[0] <= hi]
    require(
        len(selected) == len(decisions), 'Published command count differs from accepted decisions'
    )
    for c, d in zip(selected, decisions):
        require(
            c[0] == d['source_stamp_ns'] and c[1] == d['targets'] and c[2] == d['trace_id'],
            'Published command/decision mismatch',
        )
    # Keep feedback in receive order, including one bracketing sample on each side.
    before = [i for i, f in enumerate(feedback) if f[0] <= lo]
    after = [i for i, f in enumerate(feedback) if f[0] >= hi]
    require(before and after, 'Feedback does not cover chirp')
    feedback = feedback[max(before): min(after) + 1]
    data = resample(
        [(c[0], c[1]) for c in selected],
        feedback,
        JOINTS,
        lo,
        hi,
        5_000_000,
        int(max_gap_ms * 1e6),
    )
    report = {
        'schema': 'pace-supervisor-export.v1',
        'joint_order': JOINTS,
        'dt_ns': 5_000_000,
        'start_ros_ns': lo,
        'end_ros_ns': hi,
        'robot_id': manifest['robot_id'],
        'recording_id': capture.name,
        'request_id': request_id,
        'supervisor_instance': start['supervisor_instance'],
        'generation': decisions[0]['generation'],
        'manifest_sha256': digest(files['manifest.json']),
        'input_artifacts': manifest['artifacts'],
        'implementation_sha256': {
            str(p): digest(p)
            for p in (Path(__file__), Path(__file__).with_name('export_pace_capture.py'))
        },
        'session': session,
        'session_sha256': digest(session_path),
        'trajectory': start['trajectory'],
        'capture_quality': manifest['quality_status'],
        'excluded_context_missing_samples': receipt['context_missing_samples'],
        'excluded_context_errors': receipt['context_errors'],
        'selected_context_verified': True,
        'commanded_duration_wall_s': start['duration_s'],
        'recorded_duration_ros_s': (hi - lo) / 1e9,
        'trajectory_clock': 'executor monotonic; replay uses recorded ROS publication stamps',
        'max_gap_ms': max_gap_ms,
        'source_command_samples': len(selected),
        'source_feedback_samples': len(feedback),
        'interpolation': 'Commands held; feedback linearly interpolated. No timestamp shifting.',
        'limits': (
            'Simulation only. Publication timestamps are not motor application times. Export does'
            ' not establish fit quality.'
        ),
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / 'data.json').write_text(json.dumps(data, allow_nan=False))
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture', type=Path)
    p.add_argument('--request-id', required=True)
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-gap-ms', type=float, default=100.0)
    a = p.parse_args()
    report = export(a.capture, a.request_id, a.session, a.output, a.max_gap_ms)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    'recording_id',
                    'request_id',
                    'source_command_samples',
                    'source_feedback_samples',
                    'capture_quality',
                )
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
