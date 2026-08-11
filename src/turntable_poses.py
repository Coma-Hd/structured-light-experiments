"""转台角度记录、转轴标定结果与点云反向旋转。"""
from __future__ import annotations

import csv
import os
from typing import Dict, Mapping, Optional

import cv2
import numpy as np
import yaml


_ANGLE_KEYS = (
    "angle_deg",
    "angle_degree",
    "angle",
    "theta_deg",
    "theta",
    "rotation_deg",
)


def detect_angle_column(fieldnames) -> Optional[str]:
    if not fieldnames:
        return None
    lower_map = {name.lower().strip(): name for name in fieldnames}
    for key in _ANGLE_KEYS:
        if key in lower_map:
            return lower_map[key]
    return None


def load_turntable_angles(csv_path: str) -> Dict[str, float]:
    """读取 angles.csv，返回 ``{image_basename: angle_deg}``。"""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"turntable angles file not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"empty angles csv: {csv_path}")
        fields = [column.strip() for column in reader.fieldnames]
        lower_map = {column.lower(): column for column in fields}
        image_column = next(
            (lower_map[key] for key in ("image", "file", "filename", "name", "img")
             if key in lower_map),
            None,
        )
        if image_column is None:
            raise ValueError(
                f"angles csv missing image column (image/file/filename): {csv_path}"
            )
        angle_column = detect_angle_column(fields)
        if angle_column is None:
            raise ValueError(
                f"angles csv missing angle column (angle_deg/angle/theta_deg): {csv_path}"
            )

        result: Dict[str, float] = {}
        for row in reader:
            name = (row.get(image_column) or "").strip()
            raw_angle = (row.get(angle_column) or "").strip()
            if not name or not raw_angle:
                continue
            basename = os.path.basename(name.replace("\\", "/"))
            result[basename] = float(raw_angle)

    if not result:
        raise ValueError(f"no valid rows in angles csv: {csv_path}")
    return result


def resolve_angles_path(image_dir: str, angles_file: str) -> str:
    """角度表可使用绝对路径、扫描目录相对路径或工程相对路径。"""
    if os.path.isabs(angles_file) and os.path.isfile(angles_file):
        return angles_file
    beside_images = os.path.join(image_dir, angles_file)
    if os.path.isfile(beside_images):
        return beside_images
    if os.path.isfile(angles_file):
        return angles_file
    raise FileNotFoundError(
        f"angles file not found: tried '{beside_images}' and '{angles_file}'"
    )


def lookup_angle(angles: Mapping[str, float], image_path: str) -> Optional[float]:
    return angles.get(os.path.basename(image_path))


def normalize_axis(axis) -> np.ndarray:
    vector = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("turntable axis_direction must be a non-zero 3-vector")
    return vector / norm


def rotation_matrix(axis, angle_deg: float) -> np.ndarray:
    """按右手定则生成绕单位轴旋转 ``angle_deg`` 的矩阵。"""
    unit_axis = normalize_axis(axis)
    matrix, _ = cv2.Rodrigues(unit_axis * np.deg2rad(float(angle_deg)))
    return matrix


def transform_cam_by_turntable_rotation(
    points_cam: np.ndarray,
    angle_deg: float,
    reference_angle_deg: float,
    axis_point_m,
    axis_direction,
) -> np.ndarray:
    """把当前转角下的相机系点反转到参考角度的相机坐标系。

    相机与激光固定，物体绕相机系中的固定轴旋转。当前物体点满足
    ``P_current = C + R(delta) (P_reference - C)``，因此融合时使用
    ``R(-delta)`` 消除物体转动。
    """
    points = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    center = np.asarray(axis_point_m, dtype=np.float64).reshape(3)
    delta_deg = float(angle_deg) - float(reference_angle_deg)
    undo_rotation = rotation_matrix(axis_direction, -delta_deg)
    return (points - center[None, :]) @ undo_rotation.T + center[None, :]


def load_turntable_axis(path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """读取转轴 YAML，返回轴上一点、单位方向和完整内容。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"turntable axis calibration not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if "axis_point_m" not in payload or "axis_direction" not in payload:
        raise ValueError(
            f"turntable axis yaml must contain axis_point_m and axis_direction: {path}"
        )
    point = np.asarray(payload["axis_point_m"], dtype=np.float64).reshape(3)
    direction = normalize_axis(payload["axis_direction"])
    return point, direction, payload
