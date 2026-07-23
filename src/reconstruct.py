"""逐帧重建与融合（阶段 8）。

支持两种位姿来源：
  - charuco: 每帧检测标定板，变到板坐标系（旧流程，扫描时需要板）
  - rail:    导轨纯平移，姿态固定；扫描无板，用 positions.csv 行程拼点云

rail 模式：
  1. 提取蓝激光中心线
  2. 反投影成射线，与激光平面求交 -> 相机系 3D 点 P_C
  3. P_world = P_C + distance_scale * (s - s_ref) * axis
  4. 累积所有帧 -> 输出点云
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import cv2
import numpy as np
import yaml

from .calib_intrinsic import list_images
from .charuco import CharucoTarget
from .charuco_tracking import fit_charuco_rail_tracking
from .config import CharucoConfig
from .geometry import (pixels_to_rays, ray_plane_intersect_masked,
                       scale_intrinsic, transform_cam_by_rail_translation,
                       transform_cam_to_board)
from .io_utils import imread_color, load_intrinsic, load_intrinsic_size, load_laser_plane
from .keyframe_roi import load_keyframe_roi_from_cfg, roi_override_for_distance_m
from .laser_center import extract_laser_centers
from .rail_poses import load_rail_positions, lookup_distance, resolve_positions_path


def _normalize_axis(axis) -> np.ndarray:
    ax = np.asarray(axis, dtype=np.float64).reshape(3)
    nrm = float(np.linalg.norm(ax))
    if nrm < 1e-12:
        raise ValueError("rail.axis must be a non-zero 3-vector")
    return ax / nrm


def _distance_to_meters(value: float, unit: str) -> float:
    u = (unit or "mm").strip().lower()
    if u in ("mm", "millimeter", "millimetre"):
        return float(value) * 1e-3
    if u in ("m", "meter", "metre"):
        return float(value)
    if u in ("cm", "centimeter", "centimetre"):
        return float(value) * 1e-2
    raise ValueError(f"unsupported rail.distance_unit: {unit}")


def reconstruct(cfg: Dict, image_dir: str, intrinsic_path: str,
                laser_plane_path: str, out_ply: str,
                verbose: bool = True,
                pose_source: Optional[str] = None,
                positions_file: Optional[str] = None) -> np.ndarray:
    """执行逐帧重建，返回融合点云 (N,3)，并写出 PLY。

    pose_source:
      - None: 读 cfg['pose_source']，再否则若 rail.enabled 则 rail，否则 charuco
      - 'charuco' | 'rail'
    positions_file: 覆盖 cfg['rail']['positions_file']（rail 或 ChArUco rail_fit）
    """
    rail_cfg = cfg.get("rail", {}) or {}
    if pose_source is None:
        pose_source = cfg.get("pose_source")
    if pose_source is None:
        pose_source = "rail" if bool(rail_cfg.get("enabled", False)) else "charuco"
    pose_source = str(pose_source).strip().lower()
    if pose_source not in ("charuco", "rail"):
        raise ValueError(f"unsupported pose_source: {pose_source}")

    K, dist = load_intrinsic(intrinsic_path)
    plane = load_laser_plane(laser_plane_path)
    files = list_images(image_dir)
    if not files:
        raise FileNotFoundError(f"目录中没有扫描图片: {image_dir}")

    # 安全兜底：内参标定分辨率必须与扫描图分辨率一致，否则射线全错。
    calib_size = load_intrinsic_size(intrinsic_path)
    probe = imread_color(files[0])
    if probe is not None and calib_size[0] > 0:
        scan_size = (probe.shape[1], probe.shape[0])
        if scan_size != tuple(calib_size):
            K = scale_intrinsic(K, calib_size, scan_size)
            if verbose:
                print(f"[警告] 内参标定分辨率 {calib_size[0]}x{calib_size[1]} "
                      f"≠ 扫描分辨率 {scan_size[0]}x{scan_size[1]}，已按比例缩放 K。")
                print("      这是近似修正！最准确的做法是用同一分辨率重新标定内参与激光平面。")

    gating = cfg.get("gating", {})
    min_laser = int(gating.get("min_laser_points", 20))

    tracking_cfg = cfg.get("charuco_tracking", {}) or {}
    charuco_tracking_mode = str(
        tracking_cfg.get("mode", "per_frame")
    ).strip().lower()
    if charuco_tracking_mode not in ("per_frame", "rail_fit"):
        raise ValueError(
            "charuco_tracking.mode must be 'per_frame' or 'rail_fit'"
        )

    positions = None
    axis = None
    distance_scale = 1.0
    s_ref_m = 0.0
    needs_positions = (
        positions_file is not None
        or pose_source == "rail"
        or (
            pose_source == "charuco"
            and charuco_tracking_mode == "rail_fit"
        )
    )
    if needs_positions:
        pos_name = positions_file or rail_cfg.get("positions_file", "positions.csv")
        pos_path = resolve_positions_path(image_dir, pos_name)
        unit = str(rail_cfg.get("distance_unit", "mm"))
        positions = load_rail_positions(pos_path, distance_unit=unit)
    if pose_source == "rail":
        axis = _normalize_axis(rail_cfg.get("axis", [1.0, 0.0, 0.0]))
        distance_scale = float(rail_cfg.get("distance_scale", 1.0))
        if not np.isfinite(distance_scale) or distance_scale <= 0.0:
            raise ValueError("rail.distance_scale must be a positive number")
        s_ref_raw = float(rail_cfg.get("s_ref", 0.0))
        s_ref_m = _distance_to_meters(s_ref_raw, unit)
        if verbose:
            print(f"[重建] pose_source=rail  positions={pos_path}  "
                  f"axis={axis.tolist()}  distance_scale={distance_scale:.9f}  "
                  f"frames_in_csv={len(positions)}")
    else:
        if verbose:
            print("[重建] pose_source=charuco")

    target = None
    fitted_charuco_poses = None
    charuco_tracking_report = None
    min_corners = int(gating.get("min_charuco_corners", 6))
    max_reproj = float(gating.get("max_reproj_error", 2.0))
    if pose_source == "charuco":
        target = CharucoTarget(CharucoConfig.from_cfg(cfg))
        if charuco_tracking_mode == "rail_fit":
            assert positions is not None
            tracking_result = fit_charuco_rail_tracking(
                files,
                target,
                K,
                dist,
                positions,
                cfg,
                verbose=verbose,
            )
            fitted_charuco_poses = tracking_result["poses"]
            charuco_tracking_report = tracking_result["report"]

    # Optional: per-frame image ROI interpolated from keyframe boxes.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kf_roi = None
    try:
        kf_roi = load_keyframe_roi_from_cfg(cfg, project_root=project_root)
    except FileNotFoundError:
        kf_cfg = (cfg.get("laser") or {}).get("keyframe_roi") or {}
        if bool(kf_cfg.get("enabled", False)):
            raise
    if verbose and kf_roi is not None:
        print(
            f"[重建] keyframe_roi: {kf_roi['path']}  "
            f"n_keys={len(kf_roi['keyframes'])}  "
            f"(static laser.image_roi ignored when override is used)"
        )

    clouds: List[np.ndarray] = []
    used = 0
    dropped = 0

    for f in files:
        img = imread_color(f)
        if img is None:
            dropped += 1
            continue

        rvec = tvec = None
        s_m = lookup_distance(positions, f) if positions is not None else None
        if pose_source == "charuco":
            if fitted_charuco_poses is not None:
                pose = fitted_charuco_poses.get(os.path.basename(f))
                if pose is None:
                    dropped += 1
                    continue
                rvec, tvec = pose
            else:
                det = target.detect(img)
                if det is None or det.count < min_corners:
                    dropped += 1
                    continue
                pose = target.estimate_pose(det, K, dist)
                if pose is None:
                    dropped += 1
                    continue
                rvec, tvec = pose
                err = target.reproj_error(det, rvec, tvec, K, dist)
                if err > max_reproj:
                    if verbose:
                        print(
                            f"  跳过(位姿误差 {err:.2f}px): "
                            f"{os.path.basename(f)}"
                        )
                    dropped += 1
                    continue
        else:
            if s_m is None:
                if verbose:
                    print(f"  跳过(无导轨行程): {os.path.basename(f)}")
                dropped += 1
                continue

        roi_override = None
        if kf_roi is not None and s_m is not None:
            roi_override = roi_override_for_distance_m(kf_roi, float(s_m))
        centers = extract_laser_centers(img, cfg, image_roi_override=roi_override)
        if centers.shape[0] < min_laser:
            dropped += 1
            continue

        rays = pixels_to_rays(centers, K, dist)
        pts_cam, valid = ray_plane_intersect_masked(rays, plane)
        pts_cam = pts_cam[valid]
        if pts_cam.shape[0] < min_laser:
            dropped += 1
            continue

        if pose_source == "charuco":
            pts_out = transform_cam_to_board(pts_cam, rvec, tvec)
        else:
            pts_out = transform_cam_by_rail_translation(
                pts_cam,
                s_m=float(s_m) * distance_scale,
                axis=axis,
                s_ref_m=s_ref_m * distance_scale,
            )

        clouds.append(pts_out)
        used += 1
        if verbose:
            extra = f"s={s_m:.6f}m" if pose_source == "rail" else "charuco"
            print(f"  帧 {used}: {pts_out.shape[0]} 点  ({os.path.basename(f)}, {extra})")

    if not clouds:
        raise RuntimeError("没有任何有效帧，检查采集质量、positions.csv 与配置。")

    cloud = np.vstack(clouds)
    write_ply(out_ply, cloud)
    if charuco_tracking_report is not None:
        report_path = os.path.splitext(out_ply)[0] + "_charuco_tracking.yaml"
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                charuco_tracking_report,
                handle,
                allow_unicode=True,
                sort_keys=False,
            )
        if verbose:
            print(f"ChArUco 轨迹报告: {report_path}")
    if verbose:
        print(f"\n重建完成: 有效 {used} 帧, 丢弃 {dropped} 帧, 共 {cloud.shape[0]} 点")
        print(f"已保存点云: {out_ply}")
    return cloud


def read_ply_xyz(path: str) -> np.ndarray:
    """纯 numpy 读取 ASCII/二进制 PLY 的 xyz 点（不依赖 open3d）。"""
    with open(path, "rb") as f:
        # 解析头
        line = f.readline().strip()
        if line != b"ply":
            raise ValueError(f"不是合法 PLY 文件: {path}")
        fmt = None
        n_vert = 0
        props: List[str] = []
        in_vertex = False
        while True:
            line = f.readline()
            if not line:
                raise ValueError("PLY 头未结束")
            parts = line.split()
            if not parts:
                continue
            key = parts[0]
            if key == b"format":
                fmt = b" ".join(parts[1:]).decode()
            elif key == b"element":
                in_vertex = parts[1] == b"vertex"
                if in_vertex:
                    n_vert = int(parts[2])
            elif key == b"property" and in_vertex:
                props.append(parts[-1].decode())
            elif key == b"end_header":
                break

        xi, yi, zi = props.index("x"), props.index("y"), props.index("z")
        if fmt == "ascii 1.0":
            data = np.loadtxt(f, max_rows=n_vert)
            data = np.atleast_2d(data)
            return data[:, [xi, yi, zi]].astype(np.float64)
        # 二进制（仅支持 float32/float64 顺序属性，简单场景）
        raise NotImplementedError(
            "read_ply_xyz 仅支持 ASCII PLY；本项目 write_ply 输出即为 ASCII。")


def write_ply(path: str, points: np.ndarray) -> None:
    """写出 ASCII PLY 点云（不依赖 open3d，便于快速查看）。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    with open(path, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in pts:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
