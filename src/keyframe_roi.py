"""关键帧图像 ROI：按导轨行程、转台角度或扫描帧序号插值。

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

机械臂/手持扫描无行程与转角时，使用 ``frame_index``（排序后的扫描序号）插值。
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
        angle_deg = fr.get("angle_deg", None)
        dist_mm = fr.get("distance_mm", None)
        frame_index = fr.get("frame_index", None)
        if dist_mm is None and fr.get("distance_m") is not None:
            dist_mm = float(fr["distance_m"]) * 1000.0
        if angle_deg is not None:
            parameter_key = "angle_deg"
            parameter_value = float(angle_deg)
        elif dist_mm is not None:
            parameter_key = "distance_mm"
            parameter_value = float(dist_mm)
        elif frame_index is not None:
            parameter_key = "frame_index"
            parameter_value = float(frame_index)
        else:
            raise ValueError(
                f"keyframe[{i}] missing distance_mm, angle_deg or frame_index")
        item = {
            "image": str(fr.get("image", "")),
            "parameter_key": parameter_key,
            "parameter_value": parameter_value,
            "roi": {k: float(roi[k]) for k in ROI_KEYS},
        }
        item[parameter_key] = parameter_value
        cleaned.append(item)
    parameter_keys = {item["parameter_key"] for item in cleaned}
    if len(parameter_keys) != 1:
        raise ValueError(
            "keyframe ROI cannot mix distance_mm / angle_deg / frame_index"
        )
    parameter_key = next(iter(parameter_keys))
    cleaned.sort(key=lambda x: x["parameter_value"])
    return {
        "version": int(data.get("version", 1)),
        "normalized": normalized,
        "path": path,
        "parameter_key": parameter_key,
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
    parameter_value: float,
) -> Dict[str, float]:
    """按行程、角度或帧序号在关键帧之间线性插值 ROI 盒。"""
    if not keyframes:
        raise ValueError("empty keyframes")
    xs = [
        float(k.get(
            "parameter_value",
            k.get("angle_deg", k.get("distance_mm", k.get("frame_index"))),
        ))
        for k in keyframes
    ]
    if parameter_value <= xs[0]:
        return dict(keyframes[0]["roi"])
    if parameter_value >= xs[-1]:
        return dict(keyframes[-1]["roi"])

    i = int(np.searchsorted(xs, parameter_value, side="right") - 1)
    i = max(0, min(i, len(keyframes) - 2))
    d0, d1 = xs[i], xs[i + 1]
    t = 0.0 if d1 <= d0 else (parameter_value - d0) / (d1 - d0)
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


def roi_override_for_angle_deg(
    kf_data: Dict[str, Any],
    angle_deg: float,
) -> Dict[str, Any]:
    """按转台角度返回可直接用于激光提取的 image_roi。"""
    if kf_data.get("parameter_key") != "angle_deg":
        raise ValueError("keyframe ROI file is not parameterized by angle_deg")
    box = interpolate_roi_box(kf_data["keyframes"], float(angle_deg))
    return {
        "enabled": True,
        "normalized": bool(kf_data.get("normalized", True)),
        **box,
    }


def roi_override_for_frame_index(
    kf_data: Dict[str, Any],
    frame_index: float,
) -> Dict[str, Any]:
    """按扫描帧序号返回可直接用于激光提取的 image_roi。

    ``frame_index`` 对应扫描目录排序后的图片下标（从 0 开始），
    用于无导轨行程/转台角度的机械臂或手持扫描。
    """
    if kf_data.get("parameter_key") != "frame_index":
        raise ValueError("keyframe ROI file is not parameterized by frame_index")
    box = interpolate_roi_box(kf_data["keyframes"], float(frame_index))
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
    if all("angle_deg" in frame for frame in keyframes):
        parameter_key = "angle_deg"
    elif all("frame_index" in frame for frame in keyframes):
        parameter_key = "frame_index"
    elif all("distance_mm" in frame for frame in keyframes):
        parameter_key = "distance_mm"
    else:
        raise ValueError(
            "keyframes must all contain the same parameter: "
            "distance_mm, angle_deg or frame_index"
        )
    sorted_frames = sorted(
        keyframes, key=lambda item: float(item[parameter_key]))
    payload = {
        "version": 1,
        "normalized": bool(normalized),
        "parameter_key": parameter_key,
        "keyframes": [
            {
                "image": str(fr["image"]),
                parameter_key: (
                    int(fr[parameter_key])
                    if parameter_key == "frame_index"
                    else float(fr[parameter_key])
                ),
                "roi": {k: float(fr["roi"][k]) for k in ROI_KEYS},
            }
            for fr in sorted_frames
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
