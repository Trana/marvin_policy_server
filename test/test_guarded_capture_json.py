import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('guarded_recorder', SCRIPTS/'record_joint_test_guarded.py')
recorder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recorder)


def test_unavailable_optional_fields_are_null_not_zero():
    raw = dict(position=[1.], velocity=[float('inf')], effort=[float('nan')])
    result = recorder.optional_feedback_fields(raw)
    assert result == dict(position=[1.], velocity=[None], effort=[None])
    json.dumps(result, allow_nan=False)


def test_invalid_required_position_still_fails_serialization():
    result = recorder.optional_feedback_fields(dict(position=[float('nan')], effort=[]))
    with pytest.raises(ValueError):
        json.dumps(result, allow_nan=False)


def test_loop_timing_is_bounded_and_snapshot_is_independent():
    timing = recorder.LoopTiming()
    for i in range(100):
        timing.observe('spin_once', float(i), i+.05)
    saved = timing.snapshot()
    assert len(saved['slow']) == 32
    assert saved['maximum_ms']['spin_once'] == pytest.approx(50)
    timing.observe('health_write', 100., 100.001)
    assert 'health_write' not in saved['maximum_ms']


def test_live_health_uses_ram_and_final_report_replaces_link(tmp_path):
    output=tmp_path/'capture';output.mkdir()
    ram=tmp_path/'ram';ram.mkdir()
    health=recorder.LiveHealthFile(output, ram)
    live_dir=health.live.parent
    try:
        health.write(dict(running=True,finalized=False))
        assert (output/'health.json').is_symlink()
        assert (output/'health.json').resolve()==health.live
        assert json.loads((output/'health.json').read_text())['running']
        health.write(dict(running=False,finalized=True,fault=None))
        assert not (output/'health.json').is_symlink()
    finally:
        health.close()
    assert not live_dir.exists()
    assert json.loads((output/'health.json').read_text())['finalized']


def test_fixed_schema_matches_ros_conversion():
    pytest.importorskip('rclpy')
    from sensor_msgs.msg import JointState
    from runtime_interfaces.msg import JointTestTargetSample
    from rosidl_runtime_py.convert import message_to_ordereddict
    target=JointTestTargetSample(controller_instance_id='test', activation_generation=1,
        update_sequence=2, sampled_at_monotonic_s=3., stamp_sec=4, stamp_nanosec=5,
        joint_names=['a'], positions=[.25])
    feedback=JointState(name=['a'],position=[.5],velocity=[.1],effort=[float('nan')])
    for kind,msg in [('target',target),('feedback',feedback)]:
        original=message_to_ordereddict(msg)
        if kind=='feedback':original=recorder.optional_feedback_fields(original)
        assert recorder.sample_fields(kind,msg)==original


def test_storage_allowance_does_not_relax_live_freshness():
    assert recorder.MAX_DISK_LAG_S == 2.
    guard=recorder.CaptureGuard('instance',1,['joint'])
    assert guard.max_age == .15
    saved=dict(last_target=10.,last_feedback=10.,committed_at=10.7,targets=2,feedback=2)
    assert recorder.durable_ready(saved,10.8,recorder.MAX_DISK_LAG_S)
    assert not recorder.durable_ready(saved,12.01,recorder.MAX_DISK_LAG_S)
