"""内参、激光平面等标定结果的读写（YAML 格式）。"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def imread_color(path: str) -> Optional[np.ndarray]:
    """读取 BGR 图；兼容 Windows 下含中文的路径（cv2.imread 会失败）。"""
    # Prefer imdecode first when path is non-ASCII to avoid OpenCV warnings.
    try:
        path.encode("ascii")
        ascii_path = True
    except UnicodeEncodeError:
        ascii_path = False

    if ascii_path:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            return img

    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: str, image_bgr: np.ndarray) -> bool:
    """写入图像；兼容 Windows 下含中文的路径。"""
    ensure_dir(os.path.dirname(os.path.abspath(path)) or ".")
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, image_bgr)
    if not ok:
        return False
    try:
        buf.tofile(path)
        return True
    except OSError:
        return False


def save_intrinsic(path: str, K: np.ndarray, dist: np.ndarray,
                   image_size: Tuple[int, int], reproj_error: float) -> None:
    """保存相机内参。image_size = (width, height)。"""
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    dist = np.asarray(dist).reshape(-1)
    data = {
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "reproj_error_px": float(reproj_error),
        "K": {
            "fx": float(K[0, 0]), "fy": float(K[1, 1]),
            "cx": float(K[0, 2]), "cy": float(K[1, 2]),
        },
        "camera_matrix": K.astype(float).tolist(),
        "dist_coeffs": dist.astype(float).tolist(),
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def load_intrinsic(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (K 3x3, dist 1xN)。"""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    K = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"], dtype=np.float64).reshape(1, -1)
    return K, dist


def load_intrinsic_size(path: str) -> Tuple[int, int]:
    """返回内参标定时的图像尺寸 (width, height)。缺失则返回 (0, 0)。"""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return int(data.get("image_width", 0)), int(data.get("image_height", 0))


def save_laser_plane(path: str, plane: np.ndarray, rms: float,
                     num_points: int) -> None:
    """保存激光平面 [a,b,c,d]，满足 a*x+b*y+c*z+d=0（相机坐标系）。"""
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    plane = np.asarray(plane, dtype=float).reshape(-1)
    data = {
        "plane_abcd": plane.tolist(),
        "normal": plane[:3].tolist(),
        "d": float(plane[3]),
        "fit_rms_m": float(rms),
        "num_points": int(num_points),
        "frame": "camera",
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def load_laser_plane(path: str) -> np.ndarray:
    """返回激光平面 [a,b,c,d]。"""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return np.array(data["plane_abcd"], dtype=np.float64).reshape(-1)
