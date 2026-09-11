"""ROS2 node that records raw Joy samples and derived policy commands."""

from __future__ import annotations

from pathlib import Path
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Joy

from .joy_command_recording import JoyCommandSessionWriter


class JoyCommandRecorderNode(Node):
    """Record one Joy session for the lifetime of this ROS node."""

    def __init__(self) -> None:
        super().__init__('joy_command_recorder')
        self.declare_parameter('joy_topic', '/joy')
        self.declare_parameter('output_dir', '')
        self.declare_parameter('flush_every_samples', 50)

        joy_topic = str(self.get_parameter('joy_topic').value).strip() or '/joy'
        output_dir = str(self.get_parameter('output_dir').value).strip()
        output_root = (
            Path(output_dir).expanduser()
            if output_dir
            else Path.home() / '.ros' / 'marvin_joy_recordings'
        )
        flush_every_samples = int(
            self.get_parameter('flush_every_samples').value
        )
        self._writer = JoyCommandSessionWriter(
            output_root,
            joy_topic=joy_topic,
            flush_every_samples=flush_every_samples,
        )
        self._subscription = self.create_subscription(
            Joy,
            joy_topic,
            self._joy_callback,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            f'Recording {joy_topic} to {self._writer.session_dir}'
        )

    def _joy_callback(self, message: Joy) -> None:
        ros_time_sec = self.get_clock().now().nanoseconds * 1.0e-9
        source_stamp_sec = (
            float(message.header.stamp.sec)
            + float(message.header.stamp.nanosec) * 1.0e-9
        )
        self._writer.record(
            axes=message.axes,
            buttons=message.buttons,
            ros_time_sec=ros_time_sec,
            source_stamp_sec=source_stamp_sec,
            monotonic_time_sec=time.monotonic(),
        )

    def close(self) -> None:
        """Finalize the active recording."""
        self._writer.close()
        if rclpy.ok():
            self.get_logger().info(
                f'Saved {self._writer.sample_count} Joy samples to '
                f'{self._writer.session_dir}'
            )


def main(args=None) -> None:
    """Run the Joy command recorder until ROS shutdown."""
    rclpy.init(args=args)
    node = None
    try:
        node = JoyCommandRecorderNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
