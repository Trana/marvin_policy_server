"""Build one runner request for both native and Python ROS frontends.

No scheduling, ROS calls, output ownership or command publication belongs here.
The caller owns observation histories and supplies a consistent sensor snapshot.
"""


def runner_inputs(observation, *, feature_builder=None, joint_state=None, imu=None,
                  previous_action=None, filtered_joint_position=None,
                  estimated_velocity_override=None, estimated_foot_positions_override=None,
                  estimated_foot_height_override=None, estimated_contact_override=None):
    """Keep actor and embedded-estimator input semantics identical across runtimes."""
    if feature_builder is None:
        return (observation,), {'infer': True}
    features = feature_builder.build(
        joint_state, imu, previous_action,
        joint_position_relative_override=filtered_joint_position)
    return (observation, features), {
        'estimated_velocity_override': estimated_velocity_override,
        'estimated_foot_positions_override': estimated_foot_positions_override,
        'estimated_foot_height_override': estimated_foot_height_override,
        'estimated_contact_override': estimated_contact_override,
        'infer': True,
    }
