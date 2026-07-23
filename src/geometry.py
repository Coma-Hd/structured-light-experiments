"""几何核心：去畸变、反投影、射线-平面求交、平面拟合、刚体变换。

坐标约定：
- 相机光心在原点，光轴 +Z 朝前。
- 板 -> 相机 位姿 (R, t): P_cam = R @ P_board + t
- 平面表示 [a,b,c,d]，满足 a*x+b*y+c*z+d = 0。
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


# ---------------- 内参分辨率缩放 ----------------

def scale_intrinsic(K: np.ndarray, from_size, to_size) -> np.ndarray:
    """把内参 K 从 from_size=(w,h) 缩放到 to_size=(w,h)。

    仅当目标图像是标定图像的等比/等区域缩放时才严格成立；
    若是裁剪得到的分辨率，结果不准确。畸变系数无量纲不需缩放。
    """
    K = np.asarray(K, dtype=np.float64).copy()
    sx = float(to_size[0]) / float(from_size[0])
    sy = float(to_size[1]) / float(from_size[1])
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy
    return K


# ---------------- 像素 → 相机射线 ----------------

def pixels_to_rays(pixels: np.ndarray, K: np.ndarray,
                   dist: np.ndarray) -> np.ndarray:
    """把像素点去畸变并反投影成相机坐标系下的射线方向 (N,3)。

    射线从光心 (0,0,0) 出发，方向未归一化 (z=1)。
    """
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
    # 返回归一化坐标 (x=(u-cx)/fx 去畸变后)
    norm = cv2.undistortPoints(pixels, K, dist).reshape(-1, 2)
    rays = np.hstack([norm, np.ones((norm.shape[0], 1))])
    return rays


# ---------------- 射线 ∩ 平面 ----------------

def ray_plane_intersect(rays: np.ndarray, plane: np.ndarray,
                        origin: np.ndarray | None = None) -> np.ndarray:
    """射线与平面求交，返回交点 (N,3)。

    射线: P = origin + s * dir  (dir=rays[i], origin 默认光心)
    平面: n·P + d = 0
    背向/平行的点会被过滤 (s<=0 或分母≈0)。
    """
    rays = np.asarray(rays, dtype=np.float64).reshape(-1, 3)
    n = plane[:3].astype(np.float64)
    d = float(plane[3])
    if origin is None:
        origin = np.zeros(3)
    origin = np.asarray(origin, dtype=np.float64).reshape(3)

    denom = rays @ n                       # (N,)
    num = -(d + origin @ n)
    valid = np.abs(denom) > 1e-12
    s = np.zeros_like(denom)
    s[valid] = num / denom[valid]
    valid &= s > 0
    pts = origin[None, :] + s[:, None] * rays
    return pts[valid]


def ray_plane_intersect_masked(rays: np.ndarray, plane: np.ndarray
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """同上，但返回 (交点, 有效掩码)，保持与输入一一对应。"""
    rays = np.asarray(rays, dtype=np.float64).reshape(-1, 3)
    n = plane[:3].astype(np.float64)
    d = float(plane[3])
    denom = rays @ n
    valid = np.abs(denom) > 1e-12
    s = np.zeros_like(denom)
    s[valid] = -d / denom[valid]
    valid &= s > 0
    pts = s[:, None] * rays
    return pts, valid


# ---------------- 板平面（相机系） ----------------

def board_plane_in_camera(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """由板位姿得到板所在平面 (z_board=0) 在相机系的方程 [a,b,c,d]。"""
    R, _ = cv2.Rodrigues(rvec)
    n = R[:, 2]                       # 板 z 轴在相机系的方向 = 平面法向
    t = tvec.reshape(3)
    d = -float(n @ t)                # 平面过点 t
    return np.array([n[0], n[1], n[2], d], dtype=np.float64)


# ---------------- 刚体变换 ----------------

def transform_cam_to_board(points_cam: np.ndarray, rvec: np.ndarray,
                           tvec: np.ndarray) -> np.ndarray:
    """相机系点 -> 板坐标系点：P_board = R^T (P_cam - t)。"""
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    pc = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    return (pc - t[None, :]) @ R      # (R^T (p-t))^T = (p-t) R


def transform_board_to_cam(points_board: np.ndarray, rvec: np.ndarray,
                           tvec: np.ndarray) -> np.ndarray:
    """板坐标系点 -> 相机系点：P_cam = R P_board + t。"""
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    pb = np.asarray(points_board, dtype=np.float64).reshape(-1, 3)
    return pb @ R.T + t[None, :]


def transform_cam_by_rail_translation(
        points_cam: np.ndarray,
        s_m: float,
        axis: np.ndarray,
        s_ref_m: float = 0.0) -> np.ndarray:
    """相机系点 -> 导轨世界系点（纯平移、姿态不变）。

    约定：s=s_ref 时导轨系与相机系重合；
    P_world = P_cam + (s_m - s_ref_m) * axis_unit
    """
    pc = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    ax = np.asarray(axis, dtype=np.float64).reshape(3)
    nrm = float(np.linalg.norm(ax))
    if nrm < 1e-12:
        raise ValueError("rail axis must be a non-zero 3-vector")
    ax = ax / nrm
    delta = float(s_m) - float(s_ref_m)
    return pc + delta * ax[None, :]


# ---------------- 平面拟合 ----------------

def fit_plane_lstsq(points: np.ndarray) -> np.ndarray:
    """最小二乘拟合平面 [a,b,c,d]，法向单位化。"""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    centroid = pts.mean(axis=0)
    # full_matrices=False: for (N,3) points avoid allocating U as (N,N)
    # (N≈9e4 would otherwise request tens of GiB and crash).
    _, _, vh = np.linalg.svd(pts - centroid, full_matrices=False)
    normal = vh[-1]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    d = -float(normal @ centroid)
    return np.array([normal[0], normal[1], normal[2], d], dtype=np.float64)


def plane_point_distance(points: np.ndarray, plane: np.ndarray) -> np.ndarray:
    """点到平面距离 (N,)。法向应已单位化。"""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = plane[:3]
    return np.abs(pts @ n + plane[3]) / (np.linalg.norm(n) + 1e-12)


def fit_plane_ransac(points: np.ndarray, threshold: float = 1e-3,
                     iters: int = 1000, seed: int = 0
                     ) -> Tuple[np.ndarray, np.ndarray, float]:
    """RANSAC 平面拟合。返回 (plane, inlier_mask, rms)。"""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = pts.shape[0]
    if n < 3:
        raise ValueError("拟合平面至少需要 3 个点")
    rng = np.random.default_rng(seed)
    best_inliers = None
    best_count = -1
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        p = fit_plane_lstsq(pts[idx])
        if not np.isfinite(p).all():
            continue
        dist = plane_point_distance(pts, p)
        inliers = dist < threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers
    if best_inliers is None or best_count < 3:
        plane = fit_plane_lstsq(pts)
        rms = float(np.sqrt(np.mean(plane_point_distance(pts, plane) ** 2)))
        return plane, np.ones(n, dtype=bool), rms
    # 用全部内点重新拟合
    plane = fit_plane_lstsq(pts[best_inliers])
    rms = float(np.sqrt(np.mean(plane_point_distance(pts[best_inliers], plane) ** 2)))
    return plane, best_inliers, rms
