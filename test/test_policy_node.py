import pytest

rclpy = pytest.importorskip('rclpy')
from std_msgs.msg import String
from marvin_policy_server.policy_node import PolicyNode


@pytest.mark.parametrize('lean', [False, True])
def test_lean_endpoints_keep_parameter_services(monkeypatch,lean):
    monkeypatch.setenv('ROS_DOMAIN_ID','212')
    monkeypatch.setenv('ROS_LOCALHOST_ONLY','1')
    rclpy.init(args=['--ros-args','-p',f'lean_qos_events:={str(lean).lower()}'])
    node=PolicyNode('policy_endpoint_test')
    try:
        before=len(list(node.waitables))
        node.create_publisher(String,'diagnostic_only',1)
        node.create_subscription(String,'diagnostic_only',lambda msg: None,1)
        after=len(list(node.waitables))
        assert after==before if lean else after>before
        services={service.srv_name.rsplit('/',1)[-1] for service in node.services}
        assert {'get_parameters','set_parameters','list_parameters'} <= services
    finally:
        node.destroy_node()
        rclpy.shutdown()
