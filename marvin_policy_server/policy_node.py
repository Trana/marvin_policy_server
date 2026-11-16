"""Optional lower-overhead ROS endpoints. Parameter services remain available."""

from rclpy.event_handler import PublisherEventCallbacks
from rclpy.event_handler import SubscriptionEventCallbacks
from rclpy.node import Node


class PolicyNode(Node):
    def __init__(self, name):
        # Node construction retains the normal parameter-event publisher.
        self._lean_qos_events = False
        super().__init__(name)
        self.declare_parameter('lean_qos_events', False)
        self._lean_qos_events = bool(self.get_parameter('lean_qos_events').value)
        if self._lean_qos_events:
            self.get_logger().warn(
                'Lean QoS events enabled: default per-endpoint incompatibility warnings are'
                ' disabled. Sensor freshness checks and parameter services remain enabled.'
            )

    def create_publisher(self, *args, **kwargs):
        if self._lean_qos_events and 'event_callbacks' not in kwargs:
            kwargs['event_callbacks'] = PublisherEventCallbacks(use_default_callbacks=False)
        return super().create_publisher(*args, **kwargs)

    def create_subscription(self, *args, **kwargs):
        if self._lean_qos_events and 'event_callbacks' not in kwargs:
            kwargs['event_callbacks'] = SubscriptionEventCallbacks(use_default_callbacks=False)
        return super().create_subscription(*args, **kwargs)
