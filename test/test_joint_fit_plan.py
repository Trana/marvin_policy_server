import math
from pathlib import Path
import sys
import pytest
import yaml

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from joint_fit_plan import plan_joints


def profile():
    return yaml.safe_load((ROOT.parent/'marvin/bringup/config/robot_joint_test_fit_all.yaml').read_text())['/joint_test_controller']['ros__parameters']


def test_all_joints_stay_inside_existing_limits_and_dynamic_bounds():
    p = profile()
    pose = [math.radians(v) for v in [-5.78,2.12,5.83,-2.2,37.61,52.,-37.63,-52.05,76.52,70.31,-76.55,-70.37]]
    trials = plan_joints(p, pose, pose)
    assert len(trials) == 12 and all(t['status']=='ready' for t in trials)
    for i,t in enumerate(trials):
        a=t['amplitude_rad']; q=t['baseline_rad']
        assert a <= math.radians(2.75)+1e-12
        assert q-a >= p['joint_test_profile.lower_rad'][i]+math.radians(.25)-1e-12
        assert q+a <= p['joint_test_profile.upper_rad'][i]-math.radians(.25)+1e-12
        w=2*math.pi*.15; e=math.pi/20; s=2*math.pi*.1/20
        assert a*(w+e)<.1 and a*(w*w+2*w*e+2*e*e+s)<.2
    assert trials[4]['amplitude_rad'] < math.radians(1.2)


def test_no_clamping_or_baseline_motion_at_limits():
    p=profile(); pose=[0.]*12
    pose[0]=p['joint_test_profile.upper_rad'][0]
    assert plan_joints(p,pose,pose)[0]['status']=='insufficient_clearance'
    pose[0]+=.001
    with pytest.raises(ValueError): plan_joints(p,pose,pose)
    pose[0]=float('nan')
    with pytest.raises(ValueError): plan_joints(p,pose,pose)
