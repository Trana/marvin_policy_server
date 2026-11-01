"""Full client sequence with fake time/services. No ROS graph or motors."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import pytest
pytest.importorskip('rclpy')
import rclpy

ROOT=Path(__file__).parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('batch_chirp',ROOT/'scripts/run_robot_joint_chirp.py')
client=importlib.util.module_from_spec(spec);spec.loader.exec_module(client)


@pytest.mark.parametrize('fail_at,renewal_race,simultaneous', [(None,False,False),(3,False,False),(None,True,False),(None,False,True),(None,'tracking',True)])
def test_full_sequence_and_abort(monkeypatch,tmp_path,fail_at,renewal_race,simultaneous):
    p=ROOT.parent/'marvin/bringup/config/robot_joint_test_fit_all.yaml'
    if simultaneous:
        from make_simultaneous_fit_profile import build_profile
        limits=client.yaml.safe_load((ROOT.parent/'marvin/description/models/limits/marvin_joint_limits.yaml').read_text())
        generated=build_profile(client.yaml.safe_load(p.read_text()),limits)
        p=tmp_path/'simultaneous.yaml';p.write_text(client.yaml.safe_dump(generated))
    profile=client.yaml.safe_load(p.read_text())['/joint_test_controller']['ros__parameters']
    clock=[100.]; switched=[False]; hardware=['active']; trial=[0]; phase=['holding']; started=[0.]; seq=[0]
    events=[]; child=NS(returncode=None)
    def stopped(_):
        events.append('recorder_stop'); child.returncode=1 if renewal_race=='tracking' else 0
    child.poll=lambda:child.returncode
    child.send_signal=stopped; child.wait=lambda **_:0
    output=tmp_path/'capture'
    def spawn(_):
        output.mkdir();return child
    def state():
        current=phase[0]
        reason=''
        if current=='executing' and clock[0]-started[0]>=21:
            current='execution_hold';reason='TEST_COMPLETED'
        return NS(controller_instance_id='instance',activation_generation=1,session_id=trial[0],
            hold_applied=True,feedback_receipt_verified=True,tracking_fault_latched=current=='tracking_fault',
            feedback_fault_latched=False,phase=current,reason_code=reason,
            permit_challenge='challenge',applied_sequence=seq[0])
    def respond(endpoint,request):
        clock[0]+=.003
        if endpoint.endswith('list_hardware_components'):
            return NS(component=[NS(name='robot',plugin_name='odrive_ros2_control_plugin/ODriveHardwareInterface',state=NS(label=hardware[0]))])
        if endpoint.endswith('list_controllers'):
            return NS(controller=[NS(name='marvin_joint_controller',state='inactive' if switched[0] else 'active',claimed_interfaces=[] if switched[0] else ['joint/position']),
                NS(name='joint_test_controller',state='active' if switched[0] else 'inactive',claimed_interfaces=[j+'/position' for j in profile['joints']] if switched[0] else [])])
        if endpoint.endswith('switch_controller'): switched[0]=True;return NS(ok=True)
        if endpoint.endswith('joint_test/state'): return NS(state=state())
        if endpoint.endswith('joint_test/prepare'):
            if simultaneous:assert request.joint_name=='__all__'
            assert phase[0] in ('holding','cancelled')
            trial[0]+=1;seq[0]=0;phase[0]='prepared';events.append('prepare')
            return NS(accepted=True,state=state())
        if endpoint.endswith('joint_test/control'):
            seq[0]=request.sequence
            if request.action==request.START:
                phase[0]='executing';started[0]=clock[0];events.append('start')
                if trial[0]==fail_at: return NS(accepted=False,error_code='INJECTED_FAILURE')
            elif request.action==request.CANCEL: phase[0]='cancelled'
            elif request.action==request.RENEW and renewal_race=='tracking':
                phase[0]='tracking_fault'
                return NS(accepted=False,error_code='NOT_READY')
            elif request.action==request.RENEW and renewal_race and clock[0]-started[0]>20.8:
                clock[0]=started[0]+21.01
                return NS(accepted=False,error_code='NOT_READY')
            return NS(accepted=True)
        if endpoint.endswith('set_hardware_component_state'):
            assert request.target_state.label=='inactive';hardware[0]='inactive';events.append('motor_stop');return NS(ok=True)
        raise AssertionError(endpoint)
    class Service:
        def __init__(self,path):self.path=path
        def wait_for_service(self,**_):return True
        def call_async(self,req):
            answer=respond(self.path,req);return NS(done=lambda:True,result=lambda:answer)
    node=NS(create_client=lambda _,path:Service(path),destroy_node=lambda:None)
    for name,value in [('init',lambda **_:None),('create_node',lambda _:node),('spin_until_future_complete',lambda *a,**k:None),('try_shutdown',lambda:None)]:
        monkeypatch.setattr(rclpy,name,value)
    monkeypatch.setattr(client.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(client.time,'sleep',lambda s:clock.__setitem__(0,clock[0]+s))
    monkeypatch.setattr(client.signal,'signal',lambda *a:None)
    monkeypatch.setattr(client.subprocess,'Popen',spawn)
    original=Path.read_text
    def read(path,*a,**kw):
        if path.name=='health.json':return json.dumps(dict(ready=True,instance='instance',activation=1,sampled_at_monotonic_s=clock[0],target_positions=[0.]*12,feedback_positions=[0.]*12))
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,'read_text',read)
    monkeypatch.setenv('ROS_DOMAIN_ID','0');monkeypatch.setenv('ROS_LOCALHOST_ONLY','1')
    monkeypatch.setattr(sys,'argv',['chirp','--operator-ready','--simultaneous' if simultaneous else '--all-joints','--profile',str(p),'--output',str(output)])
    if renewal_race=='tracking':
        with pytest.raises(RuntimeError,match='NOT_READY:.*tracking_fault=True'):client.main()
    elif fail_at:
        with pytest.raises(RuntimeError,match='INJECTED_FAILURE'):client.main()
    else:client.main()
    assert hardware[0]=='inactive'
    count=1 if simultaneous else 12
    assert events.count('start')==(fail_at or count)
    plan=json.loads((output/'excitation-plan.json').read_text())
    assert sum(t['status']=='completed' for t in plan)==(0 if renewal_race=='tracking' else fail_at-1 if fail_at else count)
    if renewal_race=='tracking':
        assert plan[0]['status']=='failed'
        assert 'tracking_fault=True' in plan[0]['error']
    assert events.index('recorder_stop')<events.index('motor_stop')
    assert (output/'service-timings.json').exists()
