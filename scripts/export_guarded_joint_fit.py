#!/usr/bin/env python3
"""Offline PACE input from a finalized, fault-free all-joint JSONL capture."""
import argparse
import hashlib
import json
from pathlib import Path
from export_pace_capture import resample


def convert(root, dt_ns, max_gap_ns):
    metadata=json.loads((root/'metadata.json').read_text())
    health=json.loads((root/'health.json').read_text())
    plan=json.loads((root/'excitation-plan.json').read_text())
    names=metadata['joints']
    if len(plan)==1 and plan[0].get('joint')=='__all__' and plan[0].get('joints')==names:
        plan=[dict(plan[0],joint=name) for name in names]
    if (health.get('running') or not health.get('finalized') or health.get('fault') or
            health.get('missing') or health.get('writer_queue') or
            metadata['instance']!=health['instance'] or metadata['activation']!=health['activation'] or
            metadata['capture_id']!=health['capture_id']):
        raise ValueError('Capture is not finalized and fault-free')
    if len(plan)!=len(names) or {t['joint'] for t in plan}!=set(names) or any(t['status']!='completed' for t in plan):
        raise ValueError('Not all joints completed; do not fit this as a complete dataset')
    commands=[];feedback=[];sequence=None
    def stamp(s):return s['sec']*1_000_000_000+s['nanosec']
    for line in (root/'samples.jsonl').open():
        row=json.loads(line);s=row['sample']
        if row['kind']=='target':
            if (s['joint_names']!=names or s['controller_instance_id']!=metadata['instance'] or
                    s['activation_generation']!=metadata['activation'] or
                    s['update_sequence'] <= 0 or
                    (sequence is not None and s['update_sequence']!=sequence+1)):
                raise ValueError('Target identity/order/sequence mismatch')
            sequence=s['update_sequence']
            commands.append((s['stamp_sec']*1_000_000_000+s['stamp_nanosec'],s['positions']))
        elif row['kind']=='feedback':
            feedback.append((stamp(s['header']['stamp']),s['name'],s['position']))
        else:raise ValueError('Unknown sample kind')
    for kind,rows in [('targets',commands),('feedback',feedback)]:
        if len(rows)<2 or health[kind]!=len(rows) or health['durable'][kind]!=len(rows):
            raise ValueError('Durable record count mismatch: '+kind)
    start=max(commands[0][0],feedback[0][0]);end=min(commands[-1][0],feedback[-1][0])
    data=resample(commands,feedback,names,start,end,dt_ns,max_gap_ns)
    # Descriptive coverage only. Output-side motion and identifiability are not proven.
    ranges=[max(row[j] for row in data['dof_pos'])-min(row[j] for row in data['dof_pos']) for j in range(len(names))]
    return data,dict(schema='guarded-pace-export.v1',joint_names=names,joint_order=names,position_units='rad',
        dt_ns=dt_ns,start_ros_ns=start,end_ros_ns=end,max_gap_ns=max_gap_ns,
        feedback_position_ranges_rad=ranges,excitation_plan=plan,
        clock='operator-confirmed common ROS source clock; commands held, feedback interpolated',
        caveat='Joint encoder positions are not independent output-side backlash measurements')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture',type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--dt-ns',required=True,type=int,help='Must match the fitting simulation step')
    p.add_argument('--max-gap-ms',required=True,type=float)
    p.add_argument('--confirm-common-clock',required=True,action='store_true')
    a=p.parse_args()
    data,info=convert(a.capture,a.dt_ns,int(a.max_gap_ms*1_000_000))
    info['source_sha256']={name:hashlib.sha256((a.capture/name).read_bytes()).hexdigest()
        for name in ['samples.jsonl','metadata.json','health.json','profile.yaml','excitation-plan.json','service-timings.json']}
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'data.json').write_text(json.dumps(data,allow_nan=False))
    (a.output/'metadata.json').write_text(json.dumps(info,indent=2,allow_nan=False))
    (a.output/'report.json').write_text(json.dumps(info,indent=2,allow_nan=False))


if __name__=='__main__':main()
