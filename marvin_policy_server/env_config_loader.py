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
