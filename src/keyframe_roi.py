"""关键帧图像 ROI：按导轨行程插值。

JSON 格式示例::

    {
      "version": 1,
      "normalized": true,
      "keyframes": [
        {
          "image": "img_xxx.png",
          "distance_mm": 0.0,
          "roi": {"x_min": 0.58, "x_max": 0.78, "y_min": 0.48, "y_max": 0.59}
        }
      ]
    }
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


ROI_KEYS = ("x_min", "x_max", "y_min", "y_max")


def resolve_keyframe_roi_path(cfg: Dict, project_root: Optional[str] = None) -> Optional[str]:
    """从 config 解析关键帧 ROI JSON 路径；未启用则返回 None。"""
    lz = cfg.get("laser", {}) or {}
    kf = lz.get("keyframe_roi", {}) or {}
    if not bool(kf.get("enabled", False)):
        return None
    path = kf.get("path") or kf.get("json") or ""
    path = str(path).strip()
    if not path:
        return None
    if os.path.isabs(path):
        return path
    if project_root:
        return os.path.normpath(os.path.join(project_root, path))
    return os.path.normpath(path)


def load_keyframe_roi_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"keyframe ROI JSON must be an object: {path}")
    frames = data.get("keyframes") or []
    if len(frames) < 1:
        raise ValueError(f"keyframe ROI JSON has no keyframes: {path}")
    normalized = bool(data.get("normalized", True))
    cleaned: List[Dict[str, Any]] = []
    for i, fr in enumerate(frames):
        roi = fr.get("roi") or {}
        for k in ROI_KEYS:
            if k not in roi:
                raise ValueError(f"keyframe[{i}] missing roi.{k}")
        dist_mm = fr.get("distance_mm", None)
        if dist_mm is None and fr.get("distance_m") is not None:
            dist_mm = float(fr["distance_m"]) * 1000.0
        if dist_mm is None:
            raise ValueError(f"keyframe[{i}] missing distance_mm")
        cleaned.append({
            "image": str(fr.get("image", "")),
            "distance_mm": float(dist_mm),
            "roi": {k: float(roi[k]) for k in ROI_KEYS},
        })
    cleaned.sort(key=lambda x: x["distance_mm"])
    return {
        "version": int(data.get("version", 1)),
        "normalized": normalized,
        "path": path,
        "keyframes": cleaned,
    }


def load_keyframe_roi_from_cfg(
    cfg: Dict,
    project_root: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    path = resolve_keyframe_roi_path(cfg, project_root=project_root)
    if path is None:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"laser.keyframe_roi.enabled 但找不到文件: {path}"
        )
    return load_keyframe_roi_file(path)


def _lerp(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * t)


def interpolate_roi_box(
    keyframes: Sequence[Dict[str, Any]],
    distance_mm: float,
) -> Dict[str, float]:
    """按 distance_mm 在关键帧之间线性插值 ROI 盒。"""
    if not keyframes:
        raise ValueError("empty keyframes")
    xs = [float(k["distance_mm"]) for k in keyframes]
    if distance_mm <= xs[0]:
        return dict(keyframes[0]["roi"])
    if distance_mm >= xs[-1]:
        return dict(keyframes[-1]["roi"])

    i = int(np.searchsorted(xs, distance_mm, side="right") - 1)
    i = max(0, min(i, len(keyframes) - 2))
    d0, d1 = xs[i], xs[i + 1]
    t = 0.0 if d1 <= d0 else (distance_mm - d0) / (d1 - d0)
    r0, r1 = keyframes[i]["roi"], keyframes[i + 1]["roi"]
    out = {k: _lerp(r0[k], r1[k], t) for k in ROI_KEYS}
    # keep ordering
    if out["x_min"] > out["x_max"]:
        out["x_min"], out["x_max"] = out["x_max"], out["x_min"]
    if out["y_min"] > out["y_max"]:
        out["y_min"], out["y_max"] = out["y_max"], out["y_min"]
    return out


def roi_override_for_distance_m(
    kf_data: Dict[str, Any],
    distance_m: float,
) -> Dict[str, Any]:
    """返回可直接喂给 extract_laser_centers 的 image_roi 字典。"""
    box = interpolate_roi_box(kf_data["keyframes"], float(distance_m) * 1000.0)
    return {
        "enabled": True,
        "normalized": bool(kf_data.get("normalized", True)),
        **box,
    }


def pick_keyframe_indices(n_images: int, n_keys: int) -> List[int]:
    """在 [0, n-1] 上均匀取 n_keys 个索引（含首尾）。"""
    if n_images <= 0:
        return []
    n_keys = max(1, min(int(n_keys), n_images))
    if n_keys == 1:
        return [0]
    return [
        int(round(i * (n_images - 1) / (n_keys - 1)))
        for i in range(n_keys)
    ]


def save_keyframe_roi_file(
    path: str,
    keyframes: Sequence[Dict[str, Any]],
    normalized: bool = True,
) -> None:
    payload = {
        "version": 1,
        "normalized": bool(normalized),
        "keyframes": [
            {
                "image": str(fr["image"]),
                "distance_mm": float(fr["distance_mm"]),
                "roi": {k: float(fr["roi"][k]) for k in ROI_KEYS},
            }
            for fr in sorted(keyframes, key=lambda x: float(x["distance_mm"]))
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
