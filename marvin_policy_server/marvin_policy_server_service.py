#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
from std_msgs.msg import String


class ToggleNode(Node):
    def __init__(self):
        super().__init__('toggle_node')
        # Publisher (only active when running)
        self.pub = self.create_publisher(String, 'toggle_topic', 10)
        # Timer (disabled initially)
        self.timer = None
        self.active = False

        # Start/stop service
        self.srv = self.create_service(SetBool, 'toggle_node/set_active', self.on_set_active)
        self.get_logger().info("ToggleNode ready. Call /toggle_node/set_active {data: true|false}")

    def on_set_active(self, request, response):
        if request.data and not self.active:
            # Start
            self.timer = self.create_timer(1.0, self.on_timer)
            self.active = True
            response.success = True
            response.message = "Node started"
        elif not request.data and self.active:
            # Stop
            self.timer.cancel()
            self.timer = None
            self.active = False
            response.success = True
            response.message = "Node stopped"
        else:
            # Already in requested state
            response.success = False
            response.message = f"Node already {'active' if self.active else 'inactive'}"
        self.get_logger().info(response.message)
        
        return response

    def on_timer(self):
        msg = String()
        msg.data = "Hello from ToggleNode!"
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ToggleNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
