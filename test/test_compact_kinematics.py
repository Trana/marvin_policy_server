from types import SimpleNamespace

import pytest
import torch

from marvin_policy_server.concurrent_policy.optimized import CompactForwardKinematics


def geometry():
    generator = torch.Generator().manual_seed(4)
    shape = (5, 4)
    axes = torch.randn(*shape, 3, generator=generator)
    axes /= axes.norm(dim=-1, keepdim=True)
    rotations = torch.linalg.qr(torch.randn(*shape, 3, 3, generator=generator))[0]
    return SimpleNamespace(
        motor_indices=torch.tensor([[-1]*4, [0,3,6,9], [1,4,7,10], [-1]*4, [2,5,8,11]]),
        valid=torch.ones(*shape, dtype=torch.bool), axes=axes,
        parent_positions=torch.randn(*shape, 3, generator=generator)*.1,
        parent_rotations=rotations,
        child_inverse_positions=torch.randn(*shape, 3, generator=generator)*.1,
        child_inverse_rotations=rotations.transpose(-1,-2))


def reference(source, joints):
    batch = joints.shape[0]
    position = torch.zeros(batch,4,3)
    rotation = torch.eye(3).expand(batch,4,3,3).clone()
    for step in range(source.motor_indices.shape[0]):
        axis = source.axes[step]
        x,y,z = axis.unbind(-1)
        zero = torch.zeros_like(x)
        skew = torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),-1).reshape(4,3,3)
        indices = source.motor_indices[step]
        angle = joints[:,indices.clamp_min(0)]
        angle = torch.where(indices >= 0, angle, torch.zeros_like(angle))
        c,s = angle.cos()[...,None,None],angle.sin()[...,None,None]
        joint_rotation = c*torch.eye(3)+(1-c)*axis.unsqueeze(-1)*axis.unsqueeze(-2)+s*skew
        candidate_p = position + (rotation @ source.parent_positions[step].unsqueeze(-1)).squeeze(-1)
        candidate_r = rotation @ source.parent_rotations[step] @ joint_rotation
        candidate_p += (candidate_r @ source.child_inverse_positions[step].unsqueeze(-1)).squeeze(-1)
        candidate_r = candidate_r @ source.child_inverse_rotations[step]
        valid = source.valid[step]
        position = torch.where(valid[:,None],candidate_p,position)
        rotation = torch.where(valid[:,None,None],candidate_r,rotation)
    return position


@pytest.mark.parametrize('batch', [1, 10, 128])
def test_compact_fk_matches_generic_matrix_chain_and_scripts(batch):
    source = geometry()
    # Missing fixed steps in some chains must not change motor indexing.
    source.valid[0,1] = False
    source.valid[3,2] = False
    joints = torch.linspace(-6.28,6.28,batch*12).reshape(batch,12)
    optimized = torch.jit.script(CompactForwardKinematics(source))
    torch.testing.assert_close(optimized(joints), reference(source,joints), rtol=1e-5, atol=2e-6)


def test_unequal_motor_chains_are_rejected():
    source = geometry()
    source.valid[2,1] = False
    with pytest.raises(ValueError, match='equally sized'):
        CompactForwardKinematics(source)
