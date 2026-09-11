from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml


class EnvYamlLoader(yaml.SafeLoader):
    @staticmethod
    def _ignore_unknown(node: yaml.Node) -> None:
        return None

    @staticmethod
    def _construct_tuple(loader: "EnvYamlLoader", node: yaml.Node) -> tuple:
        return tuple(loader.construct_sequence(node))

    @staticmethod
    def _construct_slice(loader: "EnvYamlLoader", node: yaml.Node) -> slice:
        values = loader.construct_sequence(node)
        return slice(*values)


def _register_constructors(loader_cls: type[EnvYamlLoader]) -> None:
    loader_cls.add_constructor("tag:yaml.org,2002:python/tuple", loader_cls._construct_tuple)
    loader_cls.add_constructor("tag:yaml.org,2002:python/object/apply:builtins.slice", loader_cls._construct_slice)
    loader_cls.add_constructor(None, loader_cls._ignore_unknown)


_register_constructors(EnvYamlLoader)


class EnvConfigLoader:
    """Lightweight loader for env.yaml to support future config access."""

    def __init__(self, env_path: str | Path):
        self._env_path = Path(env_path)
        self._data: dict[str, Any] | None = None

    def _load(self) -> dict[str, Any]:
        if self._data is None:
            with self._env_path.open("r", encoding="utf-8") as handle:
                # env.yaml uses !!python/tuple and python/object/apply:builtins.slice.
                self._data = yaml.load(handle, Loader=EnvYamlLoader)
        return self._data

    def get_default_joint_positions(self) -> np.ndarray:
        data = self._load()
        joint_pos = data["scene"]["robot"]["init_state"]["joint_pos"]
        return np.array(list(joint_pos.values()), dtype=float)

    def get_joint_names(self) -> list[str]:
        data = self._load()
        joint_pos = data["scene"]["robot"]["init_state"]["joint_pos"]
        return list(joint_pos.keys())

    def get_action_history_length(self) -> int:
        data = self._load()
        history_len = (
            data.get("observations", {})
            .get("policy", {})
            .get("actions", {})
            .get("history_length")
        )
        try:
            history_len = int(history_len)
        except (TypeError, ValueError):
            history_len = 1
        return max(1, history_len)

    def get_policy_observation_terms(self) -> list[str]:
        """Return policy observation term names in env.yaml order."""
        data = self._load()
        policy_cfg = data.get("observations", {}).get("policy", {})
        return [
            name
            for name, term_cfg in policy_cfg.items()
            if isinstance(term_cfg, dict) and "func" in term_cfg
        ]

    def get_policy_observation_schema(self) -> list[dict[str, Any]]:
        """Return env-defined policy terms in order with effective history metadata."""
        term_names = self.get_policy_observation_terms()
        history_lengths = self.get_policy_term_history_lengths(term_names)
        policy_cfg = self._load().get("observations", {}).get("policy", {})
        return [
            {
                "key": name,
                "func": str(policy_cfg.get(name, {}).get("func") or ""),
                "history_length": history_lengths[name],
            }
            for name in term_names
        ]

    def get_policy_term_history_lengths(self, term_names: Sequence[str]) -> dict[str, int]:
        """Return effective history lengths for policy observation terms.

        Isaac Lab semantics:
        - ``history_length <= 0`` means no history buffer (single current sample).
        - ``history_length > 0`` means flattened history of exactly that many samples.
        - group-level ``observations.policy.history_length`` overrides term-level values.
        """
        data = self._load()
        policy_cfg = data.get("observations", {}).get("policy", {})
        group_history_raw = policy_cfg.get("history_length", None)
        group_history = self._coerce_int(group_history_raw, default=None)

        output: dict[str, int] = {}
        for term_name in term_names:
            if term_name not in policy_cfg:
                output[term_name] = 0
                continue
            if group_history is not None:
                term_history_raw = group_history
            else:
                term_history_raw = policy_cfg.get(term_name, {}).get("history_length", 0)
            term_history = self._coerce_int(term_history_raw, default=0)
            output[term_name] = max(1, term_history)
        return output

    @staticmethod
    def _coerce_int(value: Any, default: int | None) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
