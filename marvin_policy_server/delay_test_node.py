#!/usr/bin/env python3
"""Measure delay from a commanded step to first observed motion."""

import csv
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


@dataclass
class StepResult:
    step_idx: int
    joint_name: str
    cmd_value: float
    t_cmd: float
    t_move: float
    delay_s: float
    peak_abs_vel: float
    method: str


class DelayTestNode(Node):
    """Measures first response after a command change.

    This assumes commands and joint_states are on the SAME machine. If commands
    are coming from another computer, you must embed timestamps in the command
    or use ROS headers.    

    Ros2 command example:
    ros2 run marvin_policy_server marvin_delay_test --ros-args -p joint_names_csv:="FL_hip_joint,RL_hip_joint,FR_hip_joint,RR_hip_joint,FL_thigh_joint,RL_thigh_joint,FR_thigh_joint,RR_thigh_joint,FL_calf_joint,RL_calf_joint,FR_calf_joint,RR_calf_joint"   -p target_joints_csv:="FL_thigh_joint,RL_thigh_joint,FR_thigh_joint,RR_thigh_joint,FL_calf_joint,RL_calf_joint,FR_calf_joint,RR_calf_joint"
    """

    def __init__(self) -> None:
        super().__init__("marvin_delay_test")

        self.declare_parameter("command_topic", "marvin_joint_controller/commands")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("joint_names", [""])
        self.declare_parameter("joint_names_csv", "")
        self.declare_parameter("target_joint", "")
        self.declare_parameter("target_joints", [""])
        self.declare_parameter("target_joints_csv", "")
        self.declare_parameter("period_s", 1.0)
        self.declare_parameter("value_a", 0.0)
        self.declare_parameter("value_b", 0.3)
        self.declare_parameter("num_steps", 60)
        self.declare_parameter("use_velocity", True)
        self.declare_parameter("vel_threshold", 0.02)
        self.declare_parameter("pos_threshold", 0.002)
        self.declare_parameter("consecutive_samples", 2)
        self.declare_parameter("publish_rate_hz", 200.0)
        self.declare_parameter("motion_timeout_s", 2.0)
        self.declare_parameter("warmup_samples", 10)
        self.declare_parameter("csv_path", "")

        self.command_topic = self.get_parameter("command_topic").value
        self.joint_states_topic = self.get_parameter("joint_states_topic").value
        raw_joint_names = list(self.get_parameter("joint_names").value)
        joint_names_csv = self.get_parameter("joint_names_csv").value.strip()
        if joint_names_csv:
            self.joint_names = [name.strip() for name in joint_names_csv.split(",") if name.strip()]
        else:
            self.joint_names = [name for name in raw_joint_names if name]
        self.target_joint = self.get_parameter("target_joint").value.strip()
        raw_target_joints = list(self.get_parameter("target_joints").value)
        target_joints_csv = self.get_parameter("target_joints_csv").value.strip()
        self.period_s = float(self.get_parameter("period_s").value)
        self.value_a = float(self.get_parameter("value_a").value)
        self.value_b = float(self.get_parameter("value_b").value)
        self.num_steps = int(self.get_parameter("num_steps").value)
        self.use_velocity = bool(self.get_parameter("use_velocity").value)
        self.vel_threshold = float(self.get_parameter("vel_threshold").value)
        self.pos_threshold = float(self.get_parameter("pos_threshold").value)
        self.consecutive_samples = int(self.get_parameter("consecutive_samples").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.motion_timeout_s = float(self.get_parameter("motion_timeout_s").value)
        self.warmup_samples = int(self.get_parameter("warmup_samples").value)
        self.csv_path = self.get_parameter("csv_path").value.strip()

        if not self.joint_names:
            raise RuntimeError(
                "Set joint_names (string array) or joint_names_csv (comma-separated list)."
            )
        if target_joints_csv:
            self.target_joints = [
                name.strip() for name in target_joints_csv.split(",") if name.strip()
            ]
        else:
            self.target_joints = [name for name in raw_target_joints if name]
        if not self.target_joints and self.target_joint:
            self.target_joints = [self.target_joint]
        if not self.target_joints:
            raise RuntimeError(
                "Set target_joint or target_joints/target_joints_csv."
            )

        self.target_indices = {}
        for name in self.target_joints:
            try:
                self.target_indices[name] = self.joint_names.index(name)
            except ValueError as exc:
                raise RuntimeError(
                    f"target_joint '{name}' not found in joint_names"
                ) from exc

        self.cmd_pub = self.create_publisher(Float64MultiArray, self.command_topic, 10)
        self.js_sub = self.create_subscription(JointState, self.joint_states_topic, self.on_joint_state, 50)

        self.current_cmd = self.value_a
        self.last_toggle_time = time.monotonic()
        self.step_idx = 0

        self.pending = False
        self.t_cmd = 0.0
        self.cmd_value_for_step = 0.0
        self.q_at_cmd = {}
        self.peak_abs_vel = {}
        self.hit_count = {}
        self.done = {}
        self.results: List[StepResult] = []

        publish_dt = 1.0 / max(1.0, self.publish_rate_hz)
        self.pub_timer = self.create_timer(publish_dt, self.publish_command_stream)

        self.get_logger().info(
            f"Delay test starting (command_topic={self.command_topic}, "
            f"joint_states_topic={self.joint_states_topic}, "
            f"target_joints={self.target_joints})"
        )

    def publish_command_stream(self) -> None:
        now = time.monotonic()
        can_toggle = not self.pending and (now - self.last_toggle_time) >= self.period_s
        if self.step_idx < self.num_steps and can_toggle:
            self.last_toggle_time = now
            self.current_cmd = self.value_b if self.current_cmd == self.value_a else self.value_a

            self.pending = True
            self.t_cmd = time.monotonic()
            self.cmd_value_for_step = self.current_cmd
            self.q_at_cmd = {name: None for name in self.target_joints}
            self.peak_abs_vel = {name: 0.0 for name in self.target_joints}
            self.hit_count = {name: 0 for name in self.target_joints}
            self.done = {name: False for name in self.target_joints}

            self.step_idx += 1
            self.get_logger().info(
                f"[step {self.step_idx}/{self.num_steps}] cmd -> {self.current_cmd:.4f} rad"
            )

        msg = Float64MultiArray()
        msg.data = [0.0] * len(self.joint_names)
        for name, idx in self.target_indices.items():
            msg.data[idx] = float(self.current_cmd)
        self.cmd_pub.publish(msg)

        if self.pending and (now - self.t_cmd) >= self.motion_timeout_s:
            for name in self.target_joints:
                if self.done.get(name):
                    continue
                res = StepResult(
                    step_idx=self.step_idx,
                    joint_name=name,
                    cmd_value=self.cmd_value_for_step,
                    t_cmd=self.t_cmd,
                    t_move=float("nan"),
                    delay_s=float("nan"),
                    peak_abs_vel=self.peak_abs_vel.get(name, 0.0),
                    method="timeout",
                )
                self.results.append(res)
                self.done[name] = True
                self.get_logger().warn(f"  -> {name}: timeout after {self.motion_timeout_s:.2f}s")
            self.pending = False

        if self.step_idx >= self.num_steps and not self.pending:
            self.summarize_and_exit()

    def on_joint_state(self, msg: JointState) -> None:
        if not self.pending:
            return

        t_sample = time.monotonic()
        name_to_index = {name: msg.name.index(name) for name in self.target_joints if name in msg.name}
        for name in self.target_joints:
            if self.done.get(name):
                continue
            idx = name_to_index.get(name)
            if idx is None or idx >= len(msg.position):
                continue

            q = float(msg.position[idx])
            dq = None
            if self.use_velocity and idx < len(msg.velocity):
                dq = float(msg.velocity[idx])

            if self.q_at_cmd.get(name) is None:
                self.q_at_cmd[name] = q

            if dq is not None:
                self.peak_abs_vel[name] = max(self.peak_abs_vel[name], abs(dq))

            moved = False
            method = "vel"
            if self.use_velocity and dq is not None:
                if abs(dq) >= self.vel_threshold:
                    moved = True
            else:
                method = "pos"
                if abs(q - (self.q_at_cmd.get(name) or 0.0)) >= self.pos_threshold:
                    moved = True

            if moved:
                self.hit_count[name] += 1
            else:
                self.hit_count[name] = 0

            if self.hit_count[name] >= self.consecutive_samples:
                t_move = t_sample
                delay_s = t_move - self.t_cmd

                res = StepResult(
                    step_idx=self.step_idx,
                    joint_name=name,
                    cmd_value=self.cmd_value_for_step,
                    t_cmd=self.t_cmd,
                    t_move=t_move,
                    delay_s=delay_s,
                    peak_abs_vel=self.peak_abs_vel[name],
                    method=method,
                )
                self.results.append(res)
                self.done[name] = True

                self.get_logger().info(
                    f"  -> {name}: delay={delay_s * 1000.0:.1f} ms "
                    f"(peak|dq|={self.peak_abs_vel[name]:.3f}, method={method})"
                )

        if self.pending and all(self.done.get(name, False) for name in self.target_joints):
            self.pending = False

    def summarize_and_exit(self) -> None:
        if not self.results:
            self.get_logger().error(
                "No results collected. Check target_joint, topics, thresholds, and that the joint moves."
            )
            rclpy.shutdown()
            return

        usable = [r for r in self.results if r.step_idx > self.warmup_samples]
        if not usable:
            self.get_logger().error(
                f"Not enough samples after warmup steps ({self.warmup_samples})."
            )
            rclpy.shutdown()
            return

        self.get_logger().info("===== Delay summary (command -> motion) =====")
        self.get_logger().info(
            f"samples: {len(usable)} (warmup steps skipped: {self.warmup_samples})"
        )
        for name in self.target_joints:
            joint_samples = [r for r in usable if r.joint_name == name and np.isfinite(r.delay_s)]
            if not joint_samples:
                self.get_logger().info(f"{name}: no samples")
                continue
            delays_ms = np.array([r.delay_s * 1000.0 for r in joint_samples], dtype=float)
            self.get_logger().info(
                f"{name}: n={len(delays_ms)} min={float(np.min(delays_ms)):.1f} "
                f"mean={float(np.mean(delays_ms)):.1f} p50={float(np.percentile(delays_ms, 50)):.1f} "
                f"p90={float(np.percentile(delays_ms, 90)):.1f} p95={float(np.percentile(delays_ms, 95)):.1f} "
                f"max={float(np.max(delays_ms)):.1f}"
            )
        self.get_logger().info("============================================")

        if self.csv_path:
            self.write_csv(self.csv_path)
            self.get_logger().info(f"Wrote CSV: {self.csv_path}")

        rclpy.shutdown()

    def write_csv(self, path: str) -> None:
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "step_idx",
                    "joint_name",
                    "cmd_value",
                    "t_cmd_monotonic_s",
                    "t_move_monotonic_s",
                    "delay_s",
                    "delay_ms",
                    "peak_abs_vel",
                    "method",
                ]
            )
            for r in self.results:
                writer.writerow(
                    [
                        r.step_idx,
                        r.joint_name,
                        r.cmd_value,
                        r.t_cmd,
                        r.t_move,
                        r.delay_s,
                        r.delay_s * 1000.0,
                        r.peak_abs_vel,
                        r.method,
                    ]
                )


def main() -> None:
    rclpy.init()
    node = DelayTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
