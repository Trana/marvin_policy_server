"""CPU deployment optimization. Original artifact bytes and weights stay intact."""

import torch
from torch import nn


class CompactForwardKinematics(nn.Module):
    """Fold fixed transforms, then rotate position vectors instead of matrices."""

    def __init__(self, source):
        super().__init__()
        chains, tails = [], []
        for foot in range(source.motor_indices.shape[1]):
            position, rotation = torch.zeros(3), torch.eye(3)
            chain = []
            for step in range(source.motor_indices.shape[0]):
                if not bool(source.valid[step, foot]):
                    continue
                position = position + rotation @ source.parent_positions[step, foot]
                rotation = rotation @ source.parent_rotations[step, foot]
                index = int(source.motor_indices[step, foot])
                if index >= 0:
                    chain.append((position, rotation, source.axes[step, foot], index))
                    position, rotation = torch.zeros(3), torch.eye(3)
                position = position + rotation @ source.child_inverse_positions[step, foot]
                rotation = rotation @ source.child_inverse_rotations[step, foot]
            chains.append(chain)
            tails.append(position)
        lengths = {len(chain) for chain in chains}
        if len(lengths) != 1 or not next(iter(lengths)):
            raise ValueError('Compact FK requires equally sized, nonempty motor chains')
        for name, field in [('positions', 0), ('rotations', 1), ('axes', 2)]:
            self.register_buffer(name, torch.stack([
                torch.stack([chain[step][field] for chain in chains])
                for step in range(len(chains[0]))]))
        self.register_buffer('indices', torch.tensor([
            [chain[step][3] for chain in chains] for step in range(len(chains[0]))]))
        self.register_buffer('tails', torch.stack(tails))

    def forward(self, joints: torch.Tensor) -> torch.Tensor:
        position = self.tails.unsqueeze(0).expand(joints.shape[0], -1, -1)
        for offset in range(self.indices.shape[0]):
            step = self.indices.shape[0] - offset - 1
            angle = joints[:, self.indices[step]].unsqueeze(-1)
            axis = self.axes[step].unsqueeze(0).expand_as(position)
            cosine, sine = torch.cos(angle), torch.sin(angle)
            rotated = (cosine * position + sine * torch.cross(axis, position, dim=-1)
                       + (1.0 - cosine) * axis * (axis * position).sum(-1, keepdim=True))
            position = self.positions[step] + (
                self.rotations[step] @ rotated.unsqueeze(-1)).squeeze(-1)
        return position


class CompactEstimator(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.forward_kinematics = CompactForwardKinematics(source.forward_kinematics)
        for name in ('encoder', 'velocity_head', 'foot_height_head', 'contact_head'):
            setattr(self, name, getattr(source, name))
        for name in ('default_joint_positions', 'augmented_feature_scale', 'velocity_scale'):
            self.register_buffer(name, getattr(source, name).detach().clone())
        self.foot_height_scale = float(source.foot_height_scale)
        self.max_abs_velocity = float(source.max_abs_velocity)
        self.min_foot_height = float(source.min_foot_height)
        self.max_foot_height = float(source.max_foot_height)

    @torch.jit.export
    def actor_estimates(self, history: torch.Tensor):
        joints = history[..., 9:21] + self.default_joint_positions
        feet = self.forward_kinematics(joints.reshape(-1, 12)).reshape(
            history.shape[0], history.shape[1], 4, 3)
        augmented = torch.cat((history, feet.flatten(start_dim=-2)), dim=-1)
        latent = self.encoder((augmented / self.augmented_feature_scale).flatten(start_dim=-2))
        velocity = torch.clamp(self.velocity_head(latent) * self.velocity_scale,
                               -self.max_abs_velocity, self.max_abs_velocity)
        height = torch.clamp(self.foot_height_head(latent) * self.foot_height_scale,
                             self.min_foot_height, self.max_foot_height)
        return velocity, height, torch.sigmoid(self.contact_head(latent)), feet[:, -1]


class CompactPolicy(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.estimator = CompactEstimator(source.estimator)
        self.actor = source.actor

    def forward(self, observation: torch.Tensor, history: torch.Tensor):
        action, state, _ = self.inference_snapshot(observation, history)
        return action, state

    @torch.jit.export
    def inference_snapshot(self, observation: torch.Tensor, history: torch.Tensor):
        velocity, height, contact, feet = self.estimator.actor_estimates(history)
        observation = observation.clone()
        observation[..., :3] = velocity
        action = self.actor(torch.cat((observation, feet.flatten(start_dim=-2), height, contact), -1))
        return action, torch.cat((velocity, height, contact), -1), feet


def optimize_policy(source, manifest):
    """Return a verified in-memory adapter or raise. Never overwrite the artifact."""
    if manifest.raw_feature_count != 45 or manifest.action_count != 12:
        raise ValueError('Unsupported compact estimator dimensions')
    candidate = torch.jit.script(CompactPolicy(source).eval())
    generator = torch.Generator().manual_seed(20261010)
    with torch.inference_mode():
        # Include full revolutions, not just the current pose. Do not change global RNG.
        joints = torch.rand(128, 12, generator=generator) * (4 * torch.pi) - 2 * torch.pi
        torch.testing.assert_close(candidate.estimator.forward_kinematics(joints),
                                   source.estimator.forward_kinematics(joints), rtol=1e-5, atol=2e-6)
        for scale in (0., .1, 1., 3.):
            observation = torch.randn(2, manifest.base_observation_count, generator=generator) * scale
            history = torch.randn(2, manifest.history_length, 45, generator=generator) * scale
            for actual, expected in zip(candidate(observation, history), source(observation, history)):
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        # Exported TorchScript methods have separate first-call compilation costs.
        # Warm the exact batch-one entry point before motion can be activated.
        observation = torch.zeros(1, manifest.base_observation_count)
        history = torch.zeros(1, manifest.history_length, 45)
        for _ in range(20):
            action, state, feet = candidate.inference_snapshot(observation, history)
        expected_action, expected_state = source(observation, history)
        torch.testing.assert_close(action, expected_action, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(state, expected_state, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(feet, source.estimator.actor_estimates(history)[3], rtol=2e-5, atol=2e-5)
    return candidate
