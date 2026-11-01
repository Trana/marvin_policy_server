import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from export_guarded_joint_fit import convert


def capture(root):
    meta=dict(joints=['joint'],instance='id',activation=1,capture_id='capture')
    health=dict(instance='id',activation=1,capture_id='capture',running=False,finalized=True,
        fault=None,missing=0,writer_queue=0,targets=3,feedback=3,durable=dict(targets=3,feedback=3))
    (root/'metadata.json').write_text(json.dumps(meta))
    (root/'health.json').write_text(json.dumps(health))
    (root/'excitation-plan.json').write_text(json.dumps([dict(joint='joint',status='completed')]))
    rows=[]
    for i in range(3):
        rows.append(dict(kind='target',sample=dict(joint_names=['joint'],controller_instance_id='id',
            activation_generation=1,update_sequence=i+1,stamp_sec=1,stamp_nanosec=i*10_000_000,positions=[i*.1])))
        rows.append(dict(kind='feedback',sample=dict(header=dict(stamp=dict(sec=1,nanosec=i*10_000_000)),name=['joint'],position=[i*.05])))
    (root/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return health


def test_export_uses_source_clock_and_pace_arrays(tmp_path):
    capture(tmp_path)
    data,meta=convert(tmp_path,5_000_000,20_000_000)
    assert data['time']==[0.,.005,.01,.015]
    assert data['des_dof_pos']==[[0.],[0.],[.1],[.1]]
    assert data['dof_pos'][1]==pytest.approx([.025])
    assert meta['joint_names']==['joint']
    assert meta['joint_order']==['joint']


@pytest.mark.parametrize('field,value',[('running',True),('finalized',False),('fault','TARGET_SEQUENCE_GAP'),('missing',1),('targets',2)])
def test_rejects_incomplete_or_faulted_capture(tmp_path,field,value):
    health=capture(tmp_path);health[field]=value
    (tmp_path/'health.json').write_text(json.dumps(health))
    with pytest.raises(ValueError):convert(tmp_path,5_000_000,20_000_000)


def test_rejects_skipped_joints(tmp_path):
    capture(tmp_path)
    (tmp_path/'excitation-plan.json').write_text(json.dumps([dict(joint='joint',status='insufficient_clearance')]))
    with pytest.raises(ValueError):convert(tmp_path,5_000_000,20_000_000)


def test_simultaneous_requires_complete_matching_joint_order(tmp_path):
    capture(tmp_path)
    p=tmp_path/'excitation-plan.json'
    p.write_text(json.dumps([dict(joint='__all__',joints=['joint'],status='completed')]))
    _,meta=convert(tmp_path,5_000_000,20_000_000)
    assert meta['excitation_plan'][0]['joint']=='joint'
    p.write_text(json.dumps([dict(joint='__all__',joints=['wrong'],status='completed')]))
    with pytest.raises(ValueError):convert(tmp_path,5_000_000,20_000_000)
