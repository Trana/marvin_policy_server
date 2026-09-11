"""Replay one chopped Joy-command episode through the ROS policy-server path."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Joy
from std_srvs.srv import SetBool


COMMAND_FIELDS = (
    "command_linear_x",
    "command_linear_y",
    "command_angular_z",
)


def load_commands(path: Path) -> tuple[tuple[float, float, float], ...]:
    commands: list[tuple[float, float, float]] = []
    with path.expanduser().resolve().open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(COMMAND_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Replay CSV is missing fields: {sorted(missing)}")
        for row in reader:
            commands.append(tuple(float(row[field]) for field in COMMAND_FIELDS))
    if not commands:
        raise ValueError("Replay CSV contains no commands")
    return tuple(commands)


def command_to_axes(command: tuple[float, float, float]) -> list[float]:
    """Invert Marvin's deployed Joy mapping for an already-mapped command."""

    linear_x, linear_y, angular_z = command
    forward_axis = linear_x / 1.5 if linear_x >= 0.0 else linear_x
    return [angular_z, forward_axis, 1.0, linear_y, 0.0, 1.0, 0.0, 0.0]


class JoyCommandReplayNode(Node):
    def __init__(self, *, joy_topic: str, activation_service: str):
        super().__init__(
            "marvin_joy_command_replay",
            parameter_overrides=[Parameter("use_sim_time", value=True)],
        )
        self.publisher = self.create_publisher(Joy, joy_topic, 10)
        self.activation_client = self.create_client(SetBool, activation_service)

    def publish_command(self, command: tuple[float, float, float]) -> None:
        message = Joy()
        message.header.stamp = self.get_clock().now().to_msg()
        message.axes = command_to_axes(command)
        message.buttons = [0] * 11
        self.publisher.publish(message)

    def set_active(self, active: bool, timeout_sec: float) -> None:
        if not self.activation_client.wait_for_service(timeout_sec=timeout_sec):
            raise TimeoutError("Policy activation service did not become ready")
        future = self.activation_client.call_async(SetBool.Request(data=active))
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        response = future.result()
        if response is None or not response.success:
            detail = response.message if response is not None else "no response"
            raise RuntimeError(f"Policy set_active={active} failed: {detail}")


def _publish_sequence(
    node: JoyCommandReplayNode,
    commands: tuple[tuple[float, float, float], ...],
    *,
    sample_rate_hz: float,
    wall_timeout_sec: float,
) -> None:
    period_ns = round(1.0e9 / sample_rate_hz)
    start_ns = node.get_clock().now().nanoseconds
    wall_deadline = time.monotonic() + wall_timeout_sec
    for index, command in enumerate(commands):
        target_ns = start_ns + index * period_ns
        while node.get_clock().now().nanoseconds < target_ns:
            if time.monotonic() > wall_deadline:
                raise TimeoutError("Simulation clock stopped during Joy replay")
            rclpy.spin_once(node, timeout_sec=0.01)
        node.publish_command(command)
        rclpy.spin_once(node, timeout_sec=0.0)


def main(args=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--joy-topic", default="/joy")
    parser.add_argument("--activation-service", default="/sim/set_active")
    parser.add_argument("--sample-rate-hz", type=float, default=50.0)
    parser.add_argument("--pre-activation-seconds", type=float, default=1.0)
    parser.add_argument("--active-warmup-seconds", type=float, default=2.0)
    parser.add_argument("--post-replay-seconds", type=float, default=1.0)
    parser.add_argument("--service-timeout-seconds", type=float, default=30.0)
    parsed, ros_args = parser.parse_known_args(args)
    if parsed.sample_rate_hz <= 0.0:
        raise ValueError("--sample-rate-hz must be positive")
    commands = load_commands(parsed.episode)
    neutral = (0.0, 0.0, 0.0)

    rclpy.init(args=ros_args)
    node = JoyCommandReplayNode(
        joy_topic=parsed.joy_topic,
        activation_service=parsed.activation_service,
    )
    try:
        while node.get_clock().now().nanoseconds == 0:
            rclpy.spin_once(node, timeout_sec=0.1)
        pre_count = round(parsed.pre_activation_seconds * parsed.sample_rate_hz)
        warmup_count = round(parsed.active_warmup_seconds * parsed.sample_rate_hz)
        post_count = round(parsed.post_replay_seconds * parsed.sample_rate_hz)
        wall_timeout = max(30.0, (pre_count + warmup_count + len(commands) + post_count) / parsed.sample_rate_hz * 5.0)

        _publish_sequence(
            node,
            (neutral,) * pre_count,
            sample_rate_hz=parsed.sample_rate_hz,
            wall_timeout_sec=wall_timeout,
        )
        node.set_active(True, parsed.service_timeout_seconds)
        _publish_sequence(
            node,
            (neutral,) * warmup_count + commands + (neutral,) * post_count,
            sample_rate_hz=parsed.sample_rate_hz,
            wall_timeout_sec=wall_timeout,
        )
        node.set_active(False, parsed.service_timeout_seconds)
        _publish_sequence(
            node,
            (neutral,) * round(parsed.sample_rate_hz * 1.2),
            sample_rate_hz=parsed.sample_rate_hz,
            wall_timeout_sec=wall_timeout,
        )
        print(
            f"Replay complete: episode={parsed.episode.resolve()} samples={len(commands)} "
            f"rate_hz={parsed.sample_rate_hz:g}"
        )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
