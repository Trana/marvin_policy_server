READ /home/trana/Development/agent-scripts/AGENTS.MD BEFORE ANYTHING (skip if missing).

# AGENTS.md (marvin_policy_server)

Repository-specific notes for agents working in this project.

## Scope

- This repository contains the `marvin_policy_server` ROS2 package.
- Keep changes package-scoped and avoid touching unrelated workspace repositories.

## Build and Run

- Build this package first:
  - `colcon build --packages-select marvin_policy_server`
- Source workspace before running:
  - `source /home/trana/Development/ros2/robotdog_ws/install/setup.bash`

## Safety

- Do not revert unrelated workspace changes.
- Prefer non-destructive diagnostics first (`ros2 topic`, `ros2 control`, logs).

