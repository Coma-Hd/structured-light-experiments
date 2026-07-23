"""配置加载：读取 config.yaml 并提供带默认值的访问。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict

import yaml


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path: str | None = None) -> Dict[str, Any]:
    """加载 YAML 配置。path 为空时默认读取项目根目录 config.yaml。"""
    if path is None:
        path = os.path.join(_project_root(), "config.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到配置文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_path(cfg: Dict[str, Any], rel: str) -> str:
    """把配置里的相对路径解析成基于项目根目录的绝对路径。"""
    if os.path.isabs(rel):
        return rel
    return os.path.join(_project_root(), rel)


@dataclass
class CharucoConfig:
    squares_x: int
    squares_y: int
    square_length: float
    marker_length: float
    dictionary: str

    @staticmethod
    def from_cfg(cfg: Dict[str, Any]) -> "CharucoConfig":
        c = cfg["charuco"]
        return CharucoConfig(
            squares_x=int(c["squares_x"]),
            squares_y=int(c["squares_y"]),
            square_length=float(c["square_length"]),
            marker_length=float(c["marker_length"]),
            dictionary=str(c["dictionary"]),
        )
