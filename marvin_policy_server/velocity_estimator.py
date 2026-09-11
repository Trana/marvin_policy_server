"""Small, dependency-free runtime wrapper for Marvin's velocity estimator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class VelocityEstimatorConfig:
    history_length: int
    feature_count: int
    hidden_sizes: tuple[int, ...]
    output_count: int


class _VelocityEstimatorMlp(nn.Module):
    """Architecture used by the clean fixed-history training baseline."""

    def __init__(self, config: VelocityEstimatorConfig):
        super().__init__()
        sizes = (
            config.history_length * config.feature_count,
            *config.hidden_sizes,
            config.output_count,
        )
        layers: list[nn.Module] = []
        for input_size, output_size in zip(sizes[:-2], sizes[1:-1], strict=True):
            layers.extend((nn.Linear(input_size, output_size), nn.ELU()))
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.network(history.flatten(start_dim=-2))


class VelocityEstimator:
    """Load a training checkpoint and maintain one reset-safe feature history."""

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        raw_config = checkpoint["model_config"]
        self.config = VelocityEstimatorConfig(
            history_length=int(raw_config["history_length"]),
            feature_count=int(raw_config["feature_count"]),
            hidden_sizes=tuple(int(value) for value in raw_config["hidden_sizes"]),
            output_count=int(raw_config["output_count"]),
        )
        if self.config.output_count != 3:
            raise ValueError(
                f"Velocity estimator must produce three values, got {self.config.output_count}"
            )

        self.model = _VelocityEstimatorMlp(self.config)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()

        normalization = checkpoint["normalization"]
        self.feature_mean = self._tensor(normalization["feature_mean"])
        self.feature_std = self._tensor(normalization["feature_std"])
        self.target_mean = self._tensor(normalization["target_mean"])
        self.target_std = self._tensor(normalization["target_std"])
        if self.feature_mean.shape != (self.config.feature_count,):
            raise ValueError(
                "Estimator normalization width does not match model feature count: "
                f"{tuple(self.feature_mean.shape)} != {(self.config.feature_count,)}"
            )
        if self.target_mean.shape != (3,) or self.target_std.shape != (3,):
            raise ValueError("Estimator target normalization must contain vx, vy, and vz")

        raw_indices = checkpoint.get("feature_set", {}).get(
            "indices", range(self.config.feature_count)
        )
        self.feature_indices = tuple(int(index) for index in raw_indices)
        if len(self.feature_indices) != self.config.feature_count:
            raise ValueError(
                "Estimator feature selection width does not match model feature count: "
                f"{len(self.feature_indices)} != {self.config.feature_count}"
            )
        self.raw_feature_count = max(self.feature_indices) + 1
        self._history = torch.zeros(
            (self.config.history_length, self.config.feature_count), dtype=torch.float32
        )
        self.history_count = 0

    @staticmethod
    def _tensor(value) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device="cpu").detach().clone()

    @property
    def history_ready(self) -> bool:
        return self.history_count >= self.config.history_length

    def reset(self) -> None:
        self._history.zero_()
        self.history_count = 0

    def predict(self, raw_features: np.ndarray) -> np.ndarray:
        """Append a frame and return body-frame velocity in metres per second.

        The first frame is repeated across the empty history, matching the clean
        closed-loop evaluator. ``history_ready`` distinguishes that warmup output
        from estimates backed by ten distinct 50 Hz samples.
        """

        values = np.asarray(raw_features, dtype=np.float32).reshape(-1)
        if values.shape != (self.raw_feature_count,):
            raise ValueError(
                f"Expected {self.raw_feature_count} raw estimator features, got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("Estimator input contains non-finite values")

        frame = torch.from_numpy(values[list(self.feature_indices)]).to(dtype=torch.float32)
        if self.history_count == 0:
            self._history[:] = frame
        else:
            self._history[:-1] = self._history[1:].clone()
            self._history[-1] = frame
        self.history_count = min(self.history_count + 1, self.config.history_length)

        normalized = (self._history - self.feature_mean) / self.feature_std
        with torch.inference_mode():
            prediction = self.model(normalized.unsqueeze(0)).squeeze(0)
            velocity = prediction * self.target_std + self.target_mean
        return velocity.numpy().astype(np.float64, copy=True)


__all__ = ["VelocityEstimator", "VelocityEstimatorConfig"]
