"""激光平面标定（阶段 4）。

对每张「同时含 ChArUco + 激光线」的图片：
  1. 检测板位姿 -> 得到板平面(相机系)
  2. 提取激光中心像素 -> 反投影成射线
  3. 射线 ∩ 板平面 -> 该帧激光 3D 点(相机系)
累积所有帧的 3D 点，用 RANSAC 拟合出激光平面。
"""
from __future__ import annotations

import os
from typing import Dict, List

import cv2
import numpy as np

from .calib_intrinsic import list_images
from .charuco import CharucoTarget
from .config import CharucoConfig
from .geometry import (board_plane_in_camera, fit_plane_lstsq,
                       fit_plane_ransac, pixels_to_rays,
                       ray_plane_intersect_masked)
from .io_utils import load_intrinsic, save_laser_plane
from .laser_center import extract_laser_centers


def calibrate_laser_plane(cfg: Dict, image_dir: str, intrinsic_path: str,
                          out_path: str, verbose: bool = True) -> float:
    """执行激光平面标定，返回拟合 RMS (m)。"""
    K, dist = load_intrinsic(intrinsic_path)
    target = CharucoTarget(CharucoConfig.from_cfg(cfg))
    files = list_images(image_dir)
    filename_prefix = str(
        (cfg.get("laser_calibration") or {}).get("filename_prefix", "")
    ).strip()
    if filename_prefix:
        files = [
            path for path in files
            if os.path.basename(path).startswith(filename_prefix)
        ]
    if not files:
        raise FileNotFoundError(f"目录中没有图片: {image_dir}")

    gating = cfg.get("gating", {})
    min_corners = int(gating.get("min_charuco_corners", 6))
    min_laser = int(gating.get("min_laser_points", 20))
    max_reproj = float(gating.get("max_reproj_error", 2.0))

    all_pts: List[np.ndarray] = []
    used = 0

    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        det = target.detect(img)
        if det is None or det.count < min_corners:
            if verbose:
                print(f"  跳过(角点不足): {os.path.basename(f)}")
            continue
        pose = target.estimate_pose(det, K, dist)
        if pose is None:
            continue
        rvec, tvec = pose
        reproj_error = target.reproj_error(det, rvec, tvec, K, dist)
        if reproj_error > max_reproj:
            if verbose:
                print(
                    f"  跳过(位姿误差 {reproj_error:.2f}px): "
                    f"{os.path.basename(f)}"
                )
            continue
        board_plane = board_plane_in_camera(rvec, tvec)

        centers = extract_laser_centers(img, cfg)
        # 只有落在实体标定板上的激光像素才能与 board_plane 求交。
        # 板外的地面、墙面和物体激光不能拿“无限延伸的板平面”反投影，
        # 否则会形成一个RMS看似很低、几何位置却完全错误的激光平面。
        hull = cv2.convexHull(
            np.asarray(det.corners, dtype=np.float32).reshape(-1, 1, 2)
        )
        inside_board = np.array([
            cv2.pointPolygonTest(
                hull, (float(point[0]), float(point[1])), False
            ) >= 0
            for point in centers
        ], dtype=bool)
        centers = centers[inside_board]
        if centers.shape[0] < min_laser:
            if verbose:
                print(
                    f"  跳过(板内激光点不足 {centers.shape[0]}): "
                    f"{os.path.basename(f)}"
                )
            continue

        rays = pixels_to_rays(centers, K, dist)
        pts_cam, valid = ray_plane_intersect_masked(rays, board_plane)
        pts_cam = pts_cam[valid]
        if pts_cam.shape[0] < min_laser:
            continue
        all_pts.append(pts_cam)
        used += 1
        if verbose:
            print(f"  使用({pts_cam.shape[0]} 激光点): {os.path.basename(f)}")

    if used < 2:
        raise RuntimeError(f"有效激光标定图片过少({used})，至少需要 2 张不同姿态。")

    pts = np.vstack(all_pts)
    # Reject infinite board-plane extensions far from each board pose centroid.
    # (Keeps laser points near the physical board; record-mode duplicates stay OK.)
    dists = np.linalg.norm(pts - pts.mean(axis=0)[None, :], axis=1)
    keep = dists < max(0.35, float(np.percentile(dists, 90)))
    pts = pts[keep]
    if pts.shape[0] < 100:
        raise RuntimeError("过滤离群点后激光 3D 点过少，请检查标定板姿态多样性与激光提取。")

    pf = cfg.get("plane_fit", {})
    if bool(pf.get("ransac", True)):
        plane, inliers, rms = fit_plane_ransac(
            pts,
            threshold=float(pf.get("ransac_threshold", 1e-3)),
            iters=int(pf.get("ransac_iters", 1000)),
        )
        n_in = int(inliers.sum())
    else:
        plane = fit_plane_lstsq(pts)
        from .geometry import plane_point_distance
        rms = float(np.sqrt(np.mean(plane_point_distance(pts, plane) ** 2)))
        n_in = pts.shape[0]

    inlier_ratio = float(n_in / max(pts.shape[0], 1))
    min_inlier_ratio = float(pf.get("min_inlier_ratio", 0.65))
    if inlier_ratio < min_inlier_ratio:
        raise RuntimeError(
            f"激光平面内点率仅 {inlier_ratio*100:.1f}% "
            f"({n_in}/{pts.shape[0]})，低于 {min_inlier_ratio*100:.1f}%。"
            "标定目录可能混入不同相机/激光安装状态、板外激光或旧批次图片；"
            "请清空旧图或用 --filename-prefix 只选择同一批数据。"
        )

    # Optical axis is +Z in camera frame. Laser plane must cut it at a healthy angle.
    n = plane[:3] / (np.linalg.norm(plane[:3]) + 1e-12)
    d = float(plane[3])
    dist_origin_mm = abs(d) * 1000.0
    # Angle between plane and optical axis: 0° = plane contains +Z (degenerate).
    # sin(phi) = |n·z| ; want |n_z| not tiny.
    nz = abs(float(n[2]))
    if nz > 1e-9:
        z_hit_m = -d / float(n[2])
    else:
        z_hit_m = float("inf")

    save_laser_plane(out_path, plane, rms, n_in)
    if verbose:
        print(f"\n激光平面标定完成: {used} 张图, {pts.shape[0]} 点, 内点 {n_in}")
        print(f"平面内点率 = {inlier_ratio*100:.1f}%")
        print(f"平面 [a,b,c,d] = {plane}")
        print(f"拟合 RMS = {rms * 1000:.4f} mm")
        print(f"|d| (相机光心到平面距离) = {dist_origin_mm:.3f} mm")
        print(f"|n_z| (与光轴夹角指标) = {nz:.4f}  (越大越好, <0.15 通常无法测距)")
        if np.isfinite(z_hit_m) and z_hit_m > 0:
            print(f"平面穿过光轴位置 Z ≈ {z_hit_m * 1000:.1f} mm")
        print(f"已保存: {out_path}")
        if rms > 0.5e-3:
            print("提示: RMS 偏大(>0.5mm)，检查激光提取质量与标定板刚性。")
        if nz < 0.15 or dist_origin_mm < 2.0:
            print(
                "\n*** 标定几何不合格（测距会塌缩到几毫米）***\n"
                "原因: 激光平面几乎平行于相机视线（竖直线激光却几乎与光轴共面）。\n"
                "处理: 把激光头向场景中心内偏（toe-in），使蓝线在工作距离处穿过视野，\n"
                "      然后重新采集 data/laser_plane 并再跑本脚本。\n"
                "合格粗标准: |n_z| > 0.2，且平面穿过光轴的 Z 落在工作距离（如 300~700mm）。"
            )
    if nz < 0.12:
        raise RuntimeError(
            f"激光平面与光轴几乎平行 (|n_z|={nz:.4f})，三角测距会失败。"
            "请将激光向场景内偏转后重新采集并标定。"
        )
    return rms
