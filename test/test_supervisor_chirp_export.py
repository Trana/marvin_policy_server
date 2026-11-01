from pathlib import Path
import sys
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
from export_supervisor_chirp import selected_decisions, check_context


def events():
    start = dict(kind='position_diagnostic_requested',request_id='r',trajectory_kind='chirp',
                 supervisor_instance='s',sequence=1,monotonic_ns=1,telemetry_dropped=0)
    commands = [dict(kind='command_decision',accepted=True,owner='diagnostic',generation='g',
                     command_sequence=i,source_stamp_ns=i*20_000_000,targets=[0.]*12,
                     supervisor_instance='s',sequence=i+1,monotonic_ns=i+1,telemetry_dropped=0)
                for i in range(1,61)]
    end = dict(start,kind='position_diagnostic_completed',sequence=62,monotonic_ns=62)
    return [start,*commands,end]


def test_completed_window_retains_exact_source_stamps():
    _,_,rows=selected_decisions(events(),'r')
    assert len(rows)==60 and rows[0]['source_stamp_ns']==20_000_000


@pytest.mark.parametrize('fault',['missing_stamp','duplicate_stamp','gap','generation','cancel','rejected','baseline'])
def test_ineligible_windows_are_rejected(fault):
    rows=events()
    if fault=='missing_stamp': rows[10].pop('source_stamp_ns')
    if fault=='duplicate_stamp': rows[10]['source_stamp_ns']=rows[9]['source_stamp_ns']
    if fault=='gap': rows.pop(10)
    if fault=='generation': rows[10]['generation']='other'
    if fault=='cancel': rows.append(dict(rows[-1],kind='position_diagnostic_cancelled'))
    if fault=='rejected': rows[10]['accepted']=False
    if fault=='baseline': rows[-2]['targets'][0]=.1
    with pytest.raises(ValueError): selected_decisions(rows,'r')


def test_context_requires_both_fresh_same_instance_sources():
    rows=[dict(service=s,monotonic_ns=1,payload={'supervisor_instance':'s'}) for s in ('supervisor','configuration')]
    decisions=[{'monotonic_ns':2}]
    check_context(rows,decisions,'s')
    with pytest.raises(ValueError): check_context(rows[:1],decisions,'s')
    with pytest.raises(ValueError): check_context(rows,[{'monotonic_ns':4_000_000_000}],'s')
    with pytest.raises(ValueError): check_context(rows,decisions,'other')
