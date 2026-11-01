"""Private-domain recorder smoke test with synthetic publishers and read-only state."""
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time

import pytest
pytest.importorskip('rclpy')
import rclpy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState


@pytest.mark.skipif(os.environ.get('ROS_DOMAIN_ID') != '231' or os.environ.get('ROS_LOCALHOST_ONLY') != '1',
                   reason='requires isolated localhost domain 231')
def test_joint_capture_preserves_topics_timestamps_and_typed_state(tmp_path):
    from runtime_interfaces.srv import GetJointTestState
    artifact = tmp_path / 'synthetic-profile.yaml'
    artifact.write_text('# synthetic recorder fixture, not a robot profile\n')
    output = tmp_path / 'capture'
    script = Path(__file__).parents[1] / 'scripts/capture_reference_run.py'
    rclpy.init()
    node = rclpy.create_node('joint_capture_fixture')
    process = None
    try:
        def state_callback(_, response):
            response.state.phase = 'holding'
            response.state.revision = 12
            response.state.applied_sequence = 4
            response.state.permit_challenge = 'do-not-record'
            return response
        node.create_service(GetJointTestState, '/joint_test/joint_test_controller/joint_test/state', state_callback)
        publishers = [node.create_publisher(JointState, '/joint_test/isaac_joint_' + suffix, 10)
                      for suffix in ('commands', 'states')]
        def publish():
            for index, publisher in enumerate(publishers):
                msg = JointState()
                msg.header.stamp.sec = 40 + index
                msg.header.stamp.nanosec = 123
                msg.name = ['probe_joint']
                msg.position = [.2 + index*.01]
                publisher.publish(msg)
        node.create_timer(.02, publish)
        process = subprocess.Popen([sys.executable, str(script), '--target', 'sim_joint_test',
            '--output', str(output), '--seconds', '2.5', '--artifact', str(artifact)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        deadline = time.monotonic() + 20
        while process.poll() is None:
            assert time.monotonic() < deadline, 'Recorder did not finish'
            rclpy.spin_once(node, timeout_sec=.02)
        log, _ = process.communicate()
        assert process.returncode == 0, log
        receipt = json.loads((output / 'complete.json').read_text())
        assert receipt['schema'] == 'reference-complete.v1'
        for name, identity in receipt['files'].items():
            assert hashlib.sha256((output / name).read_bytes()).hexdigest() == identity['sha256']
        assert (output / 'artifacts/000-synthetic-profile.yaml').read_bytes() == artifact.read_bytes()
        metadata = json.loads((output / 'metadata.json').read_text())
        assert metadata['target'] == 'sim_joint_test' and metadata['ros_domain_id'] == '231'
        assert metadata['status_type'] == 'runtime_interfaces/srv/GetJointTestState'
        assert metadata['status_period_s'] == .2
        assert metadata['artifacts'][0]['sha256'] == hashlib.sha256(artifact.read_bytes()).hexdigest()
        rows = [json.loads(line) for line in (output / 'status.jsonl').read_text().splitlines()]
        typed = [row for row in rows if 'state' in row]
        assert len(typed) >= 2, rows
        assert all(row['state']['phase'] == 'holding' and row['state']['applied_sequence'] == 4 for row in typed)
        assert all(row['received_monotonic_ns'] >= row['request_monotonic_ns'] for row in typed)
        assert 'do-not-record' not in (output / 'status.jsonl').read_text()
        databases = list((output / 'topics').glob('*.db3'))
        assert len(databases) == 1
        with sqlite3.connect(databases[0]) as database:
            for index, suffix in enumerate(('commands', 'states')):
                records = database.execute('SELECT m.data FROM messages m JOIN topics t ON m.topic_id=t.id '
                    'WHERE t.name=?', ('/joint_test/isaac_joint_' + suffix,)).fetchall()
                assert len(records) >= 10, (suffix, len(records))
                for (raw,) in records:
                    msg = deserialize_message(raw, JointState)
                    assert msg.header.stamp.sec == 40 + index and msg.header.stamp.nanosec == 123
                    assert msg.name == ['probe_joint'] and list(msg.position) == [.2 + index*.01]
    finally:
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=25)
        node.destroy_node()
        rclpy.shutdown()


@pytest.mark.skipif(os.environ.get('ROS_DOMAIN_ID') != '231' or os.environ.get('ROS_LOCALHOST_ONLY') != '1',
                   reason='requires isolated localhost domain 231')
@pytest.mark.parametrize('manual_stop', [False, True])
def test_topics_only_record_and_pace_export(tmp_path, manual_stop):
    from marvin_trace_msgs.msg import TracedFloat64MultiArray
    scripts = Path(__file__).parents[1] / 'scripts'
    sys.path.insert(0, str(scripts))
    from analyze_joint_tracking import read_bag
    from export_pace_capture import verify_capture
    from capture_control import control
    output = tmp_path / 'capture'
    context = tmp_path / 'session.json'
    context.write_text(json.dumps(dict(joint_names=['probe_joint'], position_units='rad',
                                      support='synthetic', control_mode='fixture')))
    # Test-only domain substitution. Production robot admission still requires domain 0.
    harness = ('import capture_reference_run as c; '
               'c.TARGET_DOMAINS["robot"]="231"; c.main()')
    env = dict(os.environ, PYTHONPATH=str(scripts) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    rclpy.init()
    node = rclpy.create_node('pace_recorder_fixture')
    process = None
    try:
        cmd = node.create_publisher(TracedFloat64MultiArray, '/marvin_joint_controller/commands_traced', 10)
        feedback = node.create_publisher(JointState, '/joint_states', 10)
        def publish():
            stamp = node.get_clock().now().to_msg()
            target = TracedFloat64MultiArray()
            target.stamp = stamp
            target.trace_id = 1 << 56
            target.data = [.2]
            state = JointState()
            state.header.stamp = stamp
            state.name, state.position = ['probe_joint'], [.19]
            cmd.publish(target)
            feedback.publish(state)
        node.create_timer(.02, publish)
        process = subprocess.Popen([sys.executable, '-c', harness, '--target', 'robot',
            '--output', str(output), '--seconds', '30' if manual_stop else '3', '--topics-only', '--session-config', str(context)],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        deadline = time.monotonic() + 25
        started = time.monotonic()
        requested = False
        while process.poll() is None:
            assert time.monotonic() < deadline
            rclpy.spin_once(node, timeout_sec=.01)
            if (output / 'progress.json').exists():
                progress = control(output)
                assert progress['state'] in ('recording', 'finalized')
                if manual_stop and not requested and time.monotonic() - started > 4:
                    assert control(output, stop=True)['stop_requested']
                    requested = True
        log, _ = process.communicate()
        assert process.returncode == 0, log
        receipt = verify_capture(output)
        assert receipt['stop_reason'] == ('operator_request' if manual_stop else 'duration')
        assert control(output)['state'] == 'finalized'
        assert not (output / 'status.jsonl').read_text()
        commands, states = read_bag(output / 'topics', '/marvin_joint_controller/commands_traced', '/joint_states', 1)
        assert len(commands) > 20 and len(states) > 20
        start = max(commands[0][0], states[0][0])
        end = min(commands[-1][0], states[-1][0])
        result = subprocess.run([sys.executable, str(scripts / 'export_pace_capture.py'), str(output),
            '--output', str(tmp_path / 'export'), '--joint-names', 'probe_joint', '--source-id', '1',
            '--start-ros-ns', str(start), '--end-ros-ns', str(end), '--dt-ns', '20000000',
            '--max-gap-ms', '150', '--confirm-common-clock'], env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        data = json.loads((tmp_path / 'export/data.json').read_text())
        assert len(data['time']) > 20
        assert all(abs(row[0] - .2) < 1e-8 for row in data['des_dof_pos'])
        assert all(abs(row[0] - .19) < 1e-8 for row in data['dof_pos'])
    finally:
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=25)
        node.destroy_node()
        rclpy.shutdown()
