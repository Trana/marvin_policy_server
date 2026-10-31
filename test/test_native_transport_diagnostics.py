"""Fault injection against an explicitly supplied, private-domain native helper."""

import json
import os
from pathlib import Path
import socket
import subprocess
import time

import pytest

from marvin_policy_server.native_protocol import Flags, Kind, Packet, receive, send


@pytest.mark.parametrize('activate', [False, True])
def test_heartbeat_protects_active_output_but_allows_idle_startup(activate, monkeypatch):
    binary = os.environ.get('MARVIN_TEST_NATIVE_BINARY')
    if not binary:
        pytest.skip('Set MARVIN_TEST_NATIVE_BINARY to a built native helper')
    assert Path(binary).is_absolute() and Path(binary).is_file()
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    parent.settimeout(5.)
    process = None
    node = context = None
    if activate:
        import rclpy
        from sensor_msgs.msg import JointState, Imu
        from rclpy.qos import qos_profile_sensor_data
        monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
        context = rclpy.context.Context()
        rclpy.init(context=context, domain_id=184)
        node = rclpy.create_node('native_watchdog_fixture', context=context)
        joints = node.create_publisher(JointState, '/policy_probe/joints', qos_profile_sensor_data)
        imu = node.create_publisher(Imu, '/policy_probe/imu', qos_profile_sensor_data)
    try:
        process = subprocess.Popen(
            [binary, str(child.fileno()), '--ros-args', '-p',
             'joint_names:=[' + ','.join(f'probe_joint_{i}' for i in range(12)) + ']'],
            pass_fds=(child.fileno(),), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=dict(os.environ, ROS_DOMAIN_ID='184', ROS_LOCALHOST_ONLY='1',
                                MARVIN_POLICY_NATIVE_PROFILE='test'))
        child.close()
        config = (0.,) * 12 + (-1.,) * 12 + (1.,) * 12 + (.25, 1., 1., 0., 1000. if activate else 100., 5., 30.)
        send(parent, Packet(Kind.CONFIGURE, sequence=1, flags=2, values=config))
        deadline = time.monotonic() + 5.
        configured = False
        status_count = 0
        activation_sent = faulted = False
        while time.monotonic() < deadline:
            if node is not None:
                stamp = node.get_clock().now().to_msg()
                joint = JointState()
                joint.header.stamp = stamp
                joint.name = [f'probe_joint_{i}' for i in range(12)]
                joint.position = joint.velocity = [0.] * 12
                sensor = Imu()
                sensor.header.stamp = stamp
                sensor.orientation.w = 1.
                joints.publish(joint)
                imu.publish(sensor)
            packet = receive(parent)
            if packet.kind == Kind.ACK:
                assert packet.flags & Flags.SUCCESS
                if not configured:
                    deadline = time.monotonic() + (5. if activate else 1.)
                configured = True
            elif packet.kind == Kind.STATUS:
                assert not packet.flags & Flags.OUTPUT_VALID
                status_count += 1
                if activate and packet.flags & Flags.SENSORS_VALID and not activation_sent:
                    send(parent, Packet(Kind.ACTIVATE, sequence=3))
                    activation_sent = True
                if packet.flags & Flags.FAULTED:
                    assert activate and packet.trace_id == 3
                    assert packet.flags & Flags.OUTPUT_RELEASED
                    faulted = True
                    break
                if not activate:
                    assert not packet.flags & Flags.ACTIVE
                    assert packet.flags & Flags.OUTPUT_RELEASED
            # Deliberately withhold heartbeat and inference replies.
        assert configured and status_count >= 5
        assert faulted == activate
        send(parent, Packet(Kind.SHUTDOWN, sequence=2))
        output, _ = process.communicate(timeout=5.)
        assert process.returncode == 0, output
        reports = [json.loads(line.split('NATIVE_TRANSPORT_FAULT ', 1)[1])
                   for line in output.splitlines() if 'NATIVE_TRANSPORT_FAULT ' in line]
        if activate:
            assert len(reports) == 1, output
            assert reports[0]['peer_age_ms'] > 300
        else:
            assert reports == [], output
    finally:
        if node is not None:
            node.destroy_node()
            context.shutdown()
        parent.close()
        child.close()
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.)
