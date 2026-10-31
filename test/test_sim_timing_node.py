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


from marvin_policy_server.sim_timing import SimActionRelease


def make_node():
    def policy(obs):
        return torch.ones((1,2))
    node = object.__new__(MarvinPolicyServer)
    now=[1_000_000_000]
    node.get_clock=lambda: NS(now=lambda: NS(nanoseconds=now[0]))
    sensor=NS(header=NS(stamp=NS(sec=1,nanosec=0)))
    node._latest_joint_state=node._latest_imu=sensor
    node._timing=None
    node._inference_schedule=InferenceSchedule()
    node._activation_mgr=ActivationManager(ActivationConfig(0.,0.,True))
    node._operator_permission=OperatorPermission()
    node._async_pending_ns=node._async_fault=node._async_last_clock_ns=node._async_permission=None
    node._async_has_action=False
    node._inference_max_age_ns=100_000_000
    node._policy_runner=PolicyRunner(policy,2,PolicyRunnerConfig())
    task=RunnerTask(PolicyRunner(policy,2,PolicyRunnerConfig()))
    node._async_worker=None
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
    node.joint_names=['a','b']
    node.default_pos=np.zeros(2)
    node.action=np.zeros(2)
    node._record_velocity_estimator_diagnostic=lambda action: None
    node._map_action_to_joint_positions=lambda action: action
    published=[]
    def publish(action,**kw):
        published.append(action)
        node._latest_policy_joint_command=np.asarray(action)
        node._last_published_joint_command=np.asarray(action)
    node._publish_joint_command=publish
    node._logger=NS(warn=lambda msg: None,error=lambda msg: None)

    node._sim_release=SimActionRelease(dict(revision='marvin-timing-v1',physics_dt=.005,
        inference_delay_ms=[15,15]),np.zeros(2))
    released=[]
    node._sim_release_publisher=NS(publish=lambda msg:released.append(msg.data))
    return node, now, published, released


def test_sync_simulation_release_and_history_are_delayed_once():
    node, now, published, released = make_node()
    node._set_active_cb(NS(data=True),NS())
    node._timer_tick_impl()
    assert published==[[0.,0.]]
    np.testing.assert_array_equal(node.action,[0,0])
    np.testing.assert_array_equal(node._obs_state.action_history,[[0,0],[0,0]])
    now[0]+=5_000_000
    node._timer_tick_impl()
    assert published[-1]==[0.,0.]
    now[0]+=10_000_000
    node._timer_tick_impl()
    assert published[-1]==[1.,1.]
    assert len(released)==1
    np.testing.assert_array_equal(node._obs_state.action_history,[[0,0],[1,1]])
    # A new result cannot survive explicit deactivation.
    now[0]+=5_000_000
    node._timer_tick_impl()
    assert node._sim_release.pending is not None
    node._set_active_cb(NS(data=False),NS())
    assert node._sim_release.pending is None
    now[0]+=30_000_000
    node._timer_tick_impl()
    assert len(released)==1


def test_async_keeps_original_observation_deadline():
    node, now, published, released = make_node()
    task=RunnerTask(PolicyRunner(lambda obs:torch.ones((1,2)),2,PolicyRunnerConfig()))
    node._async_worker=InferenceWorker(task,task.reset)
    try:
        node._set_active_cb(NS(data=True),NS())
        node._timer_tick_impl()
        now[0]+=5_000_000
        deadline=time.monotonic()+.08
        while node._sim_release.pending is None and time.monotonic()<deadline:
            time.sleep(.001)
            node._timer_tick_impl()
        assert node._sim_release.pending is not None
        assert node._sim_release.pending[2]==1_000_000_000
        assert not released
        now[0]+=10_000_000
        node._timer_tick_impl()
        assert len(released)==1
        np.testing.assert_array_equal(node._obs_state.action_history,[[0,0],[1,1]])
    finally:
        assert node._async_worker.close()


def test_scene_reset_cancels_pending_and_requires_reactivation():
    import json
    node, now, published, released = make_node()
    node._set_active_cb(NS(data=True),NS())
    node._timer_tick_impl()
    assert node._sim_release.pending is not None
    node._sim_identity(NS(data=json.dumps(dict(profile=json.loads(node._sim_release.profile),
        session_id='new-scene',sim_time_ns=0,imu_acceleration_convention='specific_force'))))
    assert node._sim_release.pending is None
    assert not node._activation_mgr.is_active()
    np.testing.assert_array_equal(node._obs_state.action_history,[[0,0],[0,0]])
