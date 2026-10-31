import math
from pathlib import Path
import sys
import pytest
import yaml

ROOT=Path(__file__).parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from make_simultaneous_fit_profile import build_profile, simultaneous_trial


def profile():
    root=ROOT.parent/'marvin'
    template=yaml.safe_load((root/'bringup/config/robot_joint_test_fit_all.yaml').read_text())
    limits=yaml.safe_load((root/'description/models/limits/marvin_joint_limits.yaml').read_text())
    p=build_profile(template,limits)['/joint_test_controller']['ros__parameters']
    for i,name in enumerate(p['joints']):
        assert p['joint_test_profile.lower_rad'][i]==limits['limits'][name]['min_position']
        assert p['joint_test_profile.upper_rad'][i]==limits['limits'][name]['max_position']
    return p


def test_all_joints_and_existing_offset_are_supported():
    p=profile();q=[0.]*12;measured=[0.]*11+[math.radians(2.6)]
    trial=simultaneous_trial(p,q,measured)
    assert trial['joint']=='__all__' and len(trial['joints'])==12
    assert trial['amplitude_rad']==math.radians(5)


def test_long_profile_changes_only_identity_and_duration():
    root=ROOT.parent/'marvin'
    template=yaml.safe_load((root/'bringup/config/robot_joint_test_fit_all.yaml').read_text())
    limits=yaml.safe_load((root/'description/models/limits/marvin_joint_limits.yaml').read_text())
    short=build_profile(template,limits,True,short_10deg_5hz=True)['/joint_test_controller']['ros__parameters']
    long=build_profile(template,limits,True,long_10deg_5hz=True)['/joint_test_controller']['ros__parameters']
    assert {k for k in short if short[k]!=long[k]}=={'joint_test_profile.id','joint_test_profile.max_total_seconds'}
    assert long['joint_test_profile.max_total_seconds']==61.
    trial=simultaneous_trial(long,[0.]*12,[0.]*12)
    assert trial['excitation_s']==60. and trial['transition_s']==1.
    assert trial['end_hz']==5. and trial['amplitude_rad']==math.radians(10)
    with pytest.raises(ValueError,match='tracking'):
        simultaneous_trial(long,[0.]*12,[math.radians(5.1)]*12)
    with pytest.raises(ValueError):
        build_profile(template,limits,long_10deg_5hz=True)
    with pytest.raises(ValueError):
        build_profile(template,limits,True,long_10deg_5hz=True,short_10deg_5hz=True)


@pytest.mark.parametrize('index,angle',[(4,37.65),(5,52.),(7,-52.),(4,-37.65)])
def test_unsafe_starting_clearance_is_rejected(index,angle):
    p=profile();q=[0.]*12;q[index]=math.radians(angle)
    with pytest.raises(ValueError,match='clearance'):
        simultaneous_trial(p,q,q)


def test_tracking_fault_is_not_ignored():
    with pytest.raises(ValueError,match='tracking'):
        simultaneous_trial(profile(),[0.]*12,[0.]*11+[math.radians(5.1)])


def test_five_degree_tracking_bound():
    p=profile()
    assert p['max_handover_error']==[math.radians(5)]*12
    assert simultaneous_trial(p,[0.]*12,[math.radians(4.9)]*12)['status']=='ready'


def test_short_report_only_profile_still_checks_initial_tracking():
    root=ROOT.parent/'marvin'
    template=yaml.safe_load((root/'bringup/config/robot_joint_test_fit_all.yaml').read_text())
    limits=yaml.safe_load((root/'description/models/limits/marvin_joint_limits.yaml').read_text())
    p=build_profile(template,limits,True,True)['/joint_test_controller']['ros__parameters']
    assert p['joint_test_profile.max_total_seconds']==9.
    assert simultaneous_trial(p,[0.]*12,[0.]*12)['excitation_s']==8.
    with pytest.raises(ValueError,match='tracking'):
        simultaneous_trial(p,[0.]*12,[math.radians(5.1)]*12)


@pytest.mark.parametrize('ten',[False,True])
def test_seven_degree_three_hz_short_profile(ten):
    root=ROOT.parent/'marvin'
    template=yaml.safe_load((root/'bringup/config/robot_joint_test_fit_all.yaml').read_text())
    limits=yaml.safe_load((root/'description/models/limits/marvin_joint_limits.yaml').read_text())
    p=build_profile(template,limits,True,short_7deg_3hz=not ten,short_10deg_3hz=ten)['/joint_test_controller']['ros__parameters']
    pose=[math.radians(v) for v in (-5.81,2.16,5.81,-2.16,37.85,52.15,-37.85,-52.15,76.58,70.21,-76.58,-70.21)]
    trial=simultaneous_trial(p,pose,pose)
    assert trial['excitation_s']==8. and trial['end_hz']==3.
    assert trial['amplitude_rad']==math.radians(10 if ten else 7)
    assert p['joint_test_profile.max_total_seconds']==9.
    assert p['joint_test_profile.speed_rad_s']==[3.5 if ten else 2.5]*12
    assert p['joint_test_profile.acceleration_rad_s2']==[70. if ten else 50.]*12
    assert p['max_handover_error']==[math.radians(5)]*12
    bad=pose.copy();bad[4]=math.radians(40)
    with pytest.raises(ValueError,match='clearance'):simultaneous_trial(p,bad,bad)
    with pytest.raises(ValueError):build_profile(template,limits,short_7deg_3hz=True)


def test_explicit_standing_envelope_changes_only_four_boundaries():
    root=ROOT.parent/'marvin'
    template=yaml.safe_load((root/'bringup/config/robot_joint_test_fit_all.yaml').read_text())
    limits=yaml.safe_load((root/'description/models/limits/marvin_joint_limits.yaml').read_text())
    p=build_profile(template,limits,True)['/joint_test_controller']['ros__parameters']
    standard=profile()
    changed=[]
    for field in ('lower_rad','upper_rad'):
        for i,name in enumerate(p['joints']):
            if p['joint_test_profile.'+field][i] != standard['joint_test_profile.'+field][i]:
                changed.append((name,field))
    assert set(changed)=={('FL_thigh_joint','upper_rad'),('FR_thigh_joint','lower_rad'),
                          ('RL_thigh_joint','upper_rad'),('RR_thigh_joint','lower_rad')}
    pose=[math.radians(v) for v in (-5.81,2.16,5.81,-2.16,37.65,52.05,-37.65,-52.05,76.54,70.32,-76.54,-70.32)]
    measured=pose.copy();measured[-1]+=math.radians(2.6)
    assert simultaneous_trial(p,pose,measured)['status']=='ready'
    assert p['max_handover_error']==standard['max_handover_error']
    assert p['joint_test_lease_s']==standard['joint_test_lease_s']


@pytest.mark.parametrize('hz',[4,5])
def test_four_hz_changes_no_position_or_stop_bounds(hz):
    root=ROOT.parent/'marvin'
    template=yaml.safe_load((root/'bringup/config/robot_joint_test_fit_all.yaml').read_text())
    limits=yaml.safe_load((root/'description/models/limits/marvin_joint_limits.yaml').read_text())
    old=build_profile(template,limits,True,short_10deg_3hz=True)['/joint_test_controller']['ros__parameters']
    new=build_profile(template,limits,True,**{f'short_10deg_{hz}hz':True})['/joint_test_controller']['ros__parameters']
    assert {k for k in new if new[k]!=old[k]}=={
        'joint_test_profile.id','joint_test_profile.chirp_end_hz',
        'joint_test_profile.speed_rad_s','joint_test_profile.acceleration_rad_s2'}
    trial=simultaneous_trial(new,[0.]*12,[0.]*12)
    assert trial['end_hz']==hz and trial['excitation_s']==8.
    assert trial['amplitude_rad']==math.radians(10)
    with pytest.raises(ValueError):build_profile(template,limits,True,short_10deg_3hz=True,short_10deg_4hz=True)
