"""Explicit, bounded Isaac restart check. Never use a physical ROS domain."""

import argparse
import json
import math
import os
import time
import urllib.request


def http(path, data=None):
    request = urllib.request.Request(
        path,
        data=None if data is None else json.dumps(data).encode(),
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(request, timeout=30.0) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--restart-isaac', action='store_true', required=True)
    parser.parse_args()
    if os.environ.get('ROS_DOMAIN_ID') != '10' or os.environ.get('ROS_LOCALHOST_ONLY') != '1':
        raise SystemExit('Isaac check requires localhost-only domain 10')
    assert http('http://127.0.0.1:8212/api/v1/operator/status')['target'] is None
    api = 'http://127.0.0.1:8091/api/v1/sim/runtime/isaac/'
    initial = http(api + 'status')
    assert initial['running'] and initial['ros_domain_id'] == 10
    assert (
        initial['scene_path']
        == '/home/trana/Development/isaac/sim/Environments/marvin_ros2_default_backlash.usda'
    )
    from controller_manager_msgs.srv import ListHardwareComponents
    from marvin_trace_msgs.msg import TracedFloat64MultiArray
    from nav_msgs.msg import Odometry
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from rosgraph_msgs.msg import Clock
    from runtime_interfaces.srv import SetBehavior
    from sensor_msgs.msg import Joy
    from std_srvs.srv import Trigger

    rclpy.init(args=[])
    node = rclpy.create_node('native_sim_restart_check')
    joy_pub = node.create_publisher(Joy, '/joy', 1)
    state = {'clock': 0.0, 'clock_updates': 0, 'button': False, 'odom': None, 'commands': 0}

    def joy():
        buttons = [0] * 11
        buttons[0] = int(state['button'])
        joy_pub.publish(Joy(axes=[0.0] * 8, buttons=buttons))

    def clock(message):
        state['clock'] = message.clock.sec + message.clock.nanosec / 1e9
        state['clock_updates'] += 1

    def command(message):
        if message.trace_id >> 56 == 1:
            assert len(message.data) == 12 and all(math.isfinite(v) for v in message.data)
            state['commands'] += 1

    node.create_timer(0.02, joy)
    node.create_subscription(Clock, '/clock', clock, qos_profile_sensor_data)
    node.create_subscription(
        Odometry, '/isaac_odometry', lambda msg: state.update(odom=msg), qos_profile_sensor_data
    )
    node.create_subscription(
        TracedFloat64MultiArray,
        '/sim/marvin_joint_controller/commands_traced',
        command,
        qos_profile_sensor_data,
    )
    behavior_client = node.create_client(SetBehavior, '/sim/robot_control/behavior/set')
    observation_client = node.create_client(Trigger, '/sim/get_policy_observations')
    hardware_client = node.create_client(
        ListHardwareComponents, '/sim/controller_manager/list_hardware_components'
    )

    def spin(seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.01)

    def call(client, request):
        assert client.wait_for_service(timeout_sec=8.0), 'Simulation service missing'
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)
        assert future.done(), 'Simulation service timeout'
        reply = future.result()
        assert getattr(reply, 'success', True), str(reply)
        return reply

    def status():
        return json.loads(call(observation_client, Trigger.Request()).message)['execution']

    def behavior(value):
        reply = call(behavior_client, SetBehavior.Request(behavior=value))
        print(json.dumps({'behavior': value, 'hardware': reply.hardware_state}), flush=True)
        return reply

    def authorize():
        state['button'] = True
        spin(0.12)
        state['button'] = False
        spin(0.8)

    def policy_phase(label):
        behavior('ai_locomotion')
        before = state['commands']
        spin(4.0)
        assert state['commands'] > before + 100
        execution = status()
        assert execution['native_active'] and not execution['fault'], execution
        q = state['odom'].pose.pose.orientation
        upright = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        assert upright > 0.75, 'Simulation robot is not upright'
        print(
            json.dumps({'phase': label, 'upright_z': upright, 'execution': execution}), flush=True
        )
        return execution['generation']

    validated = False
    try:
        spin(2.0)
        hardware = call(hardware_client, ListHardwareComponents.Request()).component
        assert hardware and all(
            h.plugin_name == 'topic_based_ros2_control/TopicBasedSystem' for h in hardware
        )
        validated = True
        assert status()['io_backend'] == 'native'
        authorize()
        behavior('stand')
        spin(2.0)
        generation = policy_phase('before_restart')
        old_clock = state['clock']
        http(api + 'stop', {})
        spin(1.0)
        before = state['commands']
        spin(0.5)
        assert state['commands'] == before, 'Policy continued publishing with stopped simulation'
        execution = status()
        assert execution['paused'] or not execution['sensors_valid'], execution
        print(json.dumps({'phase': 'isaac_stopped', 'execution': execution}), flush=True)
        http(
            api + 'start',
            {
                'scene_path': initial['scene_path'],
                'play_sim_on_start': True,
                'use_imu': True,
                'headless': 'native',
                'sim_ros_domain_id': 10,
                'custom_args': '--/app/omni.graph.scriptnode/opt_in=true',
            },
        )
        clock_updates = state['clock_updates']
        deadline = time.monotonic() + 60.0
        while state['clock_updates'] < clock_updates + 10:
            assert time.monotonic() < deadline, 'Restarted Isaac has no clock'
            spin(0.1)
        assert state['clock'] < old_clock, 'Test did not observe a simulator clock reset'
        spin(2.0)
        assert status()['generation'] > generation
        authorize()
        behavior('stand')
        spin(2.0)
        policy_phase('after_restart')
        behavior('stand')
        spin(2.0)
        assert behavior('rest').hardware_state == 'inactive'
        print('NATIVE_SIM_RESET_CHECK_PASS', flush=True)
    finally:
        if validated:
            try:
                behavior('rest')
            except Exception as error:
                print(f'Simulation Rest cleanup failed: {error}', flush=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
