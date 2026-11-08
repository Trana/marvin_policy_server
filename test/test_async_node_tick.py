"""No ROS graph or hardware: exercise the real timer and activation methods."""
from threading import Event
from types import SimpleNamespace as NS
import time

import numpy as np
import pytest

pytest.importorskip('rclpy')
import torch
from marvin_policy_server.activation_manager import ActivationManager, ActivationConfig
from marvin_policy_server.async_policy import RunnerTask
from marvin_policy_server.inference_schedule import InferenceSchedule
from marvin_policy_server.inference_worker import InferenceWorker
from marvin_policy_server.marvin_policy_server import MarvinPolicyServer
from marvin_policy_server.operator_permission import OperatorPermission
from marvin_policy_server.policy_runner import PolicyRunner, PolicyRunnerConfig


def test_pending_inference_does_not_block_ticks_and_cannot_survive_stop():
    entered, release = Event(), Event()
    def policy(obs):
        entered.set()
        assert release.wait(2)
        return torch.ones((1,2))
    node = object.__new__(MarvinPolicyServer)
    now=[1_000_000_000]
    node.get_clock=lambda: NS(now=lambda: NS(nanoseconds=now[0]))
    sensor=NS(header=NS(stamp=NS(sec=1,nanosec=0)))
    node._latest_joint_state=node._latest_imu=sensor
    node._timing=None
    node._inference_schedule=InferenceSchedule(20_000_000)
    node._activation_mgr=ActivationManager(ActivationConfig(0.,0.,True))
    node._operator_permission=OperatorPermission()
    node._async_pending_ns=node._async_fault=node._async_last_clock_ns=node._async_permission=None
    node._async_has_action=False
    node._inference_max_age_ns=100_000_000
    node._policy_runner=PolicyRunner(policy,2,PolicyRunnerConfig())
    task=RunnerTask(PolicyRunner(policy,2,PolicyRunnerConfig()))
    node._async_worker=InferenceWorker(task,task.reset)
    node._velocity_estimator=None
    node._concurrent_policy_enabled=False
    node._concurrent_policy_feature_builder=None
    node._joint_position_filter=None
    node._update_velocity_estimator=lambda *args: None
    node._base_lin_vel_readiness=lambda: (True,'')
    node._base_lin_vel_override=lambda: None
    node._next_trace_id=lambda: 1
    node._publish_trace_event=lambda *args,**kw: None
    node._command_input_locked=node._imu_input_locked=False
    node._joint_position_input_locked=node._velocity_input_locked=False
    node._cmd_vel=None
    node._obs_state=NS(action_history=np.zeros((2,2)))
    node._obs_builder=NS(build=lambda *args,**kw: np.zeros(3))
    node._actor_action_history_mode='raw'
    node._action_scale=.25
    node.default_pos=np.zeros(2)
    node.action=np.zeros(2)
    node._record_velocity_estimator_diagnostic=lambda action: None
    node._map_action_to_joint_positions=lambda action: action
    published=[]
    def publish(action,**kw):
        published.append(action)
        node._latest_policy_joint_command=np.asarray(action)
    node._publish_joint_command=publish
    node._logger=NS(warn=lambda msg: None,error=lambda msg: None)
    try:
        node._set_active_cb(NS(data=True),NS())
        node._timer_tick_impl()
        assert entered.wait(2)
        now[0]+=5_000_000
        node._timer_tick_impl()  # Returns while inference is still blocked.
        assert not published
        node._set_active_cb(NS(data=False),NS())
        release.set()
        deadline=time.monotonic()+2
        while not node._async_worker.ready and time.monotonic()<deadline:
            time.sleep(.001)
        assert node._async_worker.ready
        node._timer_tick_impl()
        assert not published
        node._set_active_cb(NS(data=True),NS())
        node._timer_tick_impl()
        deadline=time.monotonic()+1
        while not published and time.monotonic()<deadline:
            time.sleep(.001)
            now[0]+=1_000_000
            node._timer_tick_impl()
        assert published == [[1.,1.]]
        np.testing.assert_array_equal(node._obs_state.action_history,[[0.,0.],[1.,1.]])
        assert node._async_fault is None
    finally:
        release.set()
        assert node._async_worker.close()
