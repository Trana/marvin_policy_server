"""
Private-domain status callback timing and short-lived monitoring clients.

This is a test helper, not a production entry point. It never enables hardware.
"""

import json
import os
import sys
import time


def main():
    if os.environ.get('ROS_DOMAIN_ID') != '184' or os.environ.get('ROS_LOCALHOST_ONLY') != '1':
        raise SystemExit('Monitoring probe requires localhost-only domain 184')
    mode = sys.argv.pop(1)
    if mode == 'server':
        from marvin_policy_server import native_server
        from marvin_policy_server.marvin_policy_server import MarvinPolicyServer

        original = MarvinPolicyServer._get_policy_observations_cb
        samples = []

        def measured(self, request, response):
            wall, cpu = time.monotonic_ns(), time.thread_time_ns()
            try:
                return original(self, request, response)
            finally:
                if len(samples) < 1000:
                    samples.append({
                        'monotonic_ns': wall,
                        'elapsed_ms': (time.monotonic_ns() - wall) / 1e6,
                        'cpu_ms': (time.thread_time_ns() - cpu) / 1e6,
                    })

        MarvinPolicyServer._get_policy_observations_cb = measured
        try:
            native_server.main()
        finally:
            print('MONITOR_CALLBACKS ' + json.dumps(samples), flush=True)
        return
    if mode not in {'fresh', 'discovery'}:
        raise SystemExit('Expected server, fresh or discovery')
    started = time.monotonic()
    import rclpy
    from std_srvs.srv import Trigger

    rclpy.init(args=[])
    node = rclpy.create_node('private_short_lived_monitor')
    try:
        client = node.create_client(Trigger, '/get_policy_observations')
        if not client.wait_for_service(timeout_sec=3.0):
            raise RuntimeError('Private policy service not discovered')
        ready = time.monotonic()
        if mode == 'fresh':
            future = client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(node, future, timeout_sec=2.0)
            if not future.done() or not future.result().success:
                raise RuntimeError('Private status query failed')
        print(
            'MONITOR_CLIENT '
            + json.dumps({
                'mode': mode,
                'startup_discovery_ms': (ready - started) * 1000,
                'query_ms': (time.monotonic() - ready) * 1000,
            }),
            flush=True,
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
