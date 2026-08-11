"""激光平面标定（阶段 4）。

对每张「同时含 ChArUco + 激光线」的图片：
  1. 检测板位姿 -> 得到板平面(相机系)
  2. 提取激光中心像素 -> 反投影成射线
  3. 射线 ∩ 板平面 -> 该帧激光 3D 点(相机系)
累积所有帧的 3D 点，用 RANSAC 拟合出激光平面。
"""
from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from .calib_intrinsic import list_images
from .charuco import CharucoDetection, CharucoTarget
from .config import CharucoConfig
from .geometry import (board_plane_in_camera, fit_plane_lstsq,
                       fit_plane_ransac, pixels_to_rays,
                       ray_plane_intersect_masked)
from .io_utils import imread_color, load_intrinsic, save_laser_plane
from .laser_center import blue_laser_score, extract_laser_centers


def _filter_stripe_centers(
        centers: np.ndarray,
        score_img: np.ndarray,
        min_score: float,
        max_line_dist_px: float,
) -> np.ndarray:
    """标定专用二维强度和细直线过滤（兼容外部成功版本的核心算法）。"""
    pts = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return pts

    if min_score > 0:
        h, w = score_img.shape[:2]
        uv = np.rint(pts).astype(np.int64)
        uv[:, 0] = np.clip(uv[:, 0], 0, w - 1)
        uv[:, 1] = np.clip(uv[:, 1], 0, h - 1)
        keep = score_img[uv[:, 1], uv[:, 0]] >= min_score
        pts = pts[keep]

    # 先用 RANSAC 找主条带，再用 PCA 精修；相比直接 PCA，不会被少量散点
    # 拉偏整条线。平面标定板上的激光交线在去畸变前也应近似直线。
    if max_line_dist_px > 0 and len(pts) >= 10:
        rng = np.random.default_rng(0)
        best = np.zeros(len(pts), dtype=bool)
        for _ in range(300):
            i, j = rng.choice(len(pts), 2, replace=False)
            direction = pts[j] - pts[i]
            norm = float(np.linalg.norm(direction))
            if norm < 1e-6:
                continue
            direction /= norm
            normal = np.array([-direction[1], direction[0]])
            candidate = (
                np.abs((pts - pts[i]) @ normal) <= max_line_dist_px
            )
            if int(candidate.sum()) > int(best.sum()):
                best = candidate
        if int(best.sum()) >= 3:
            for _ in range(2):
                selected = pts[best]
                centered = selected - selected.mean(axis=0)
                _, _, vh = np.linalg.svd(
                    centered, full_matrices=False)
                normal = vh[-1]
                distance = np.abs(
                    (pts - selected.mean(axis=0)) @ normal)
                refined = distance <= max_line_dist_px
                if np.array_equal(refined, best):
                    break
                best = refined
            pts = pts[best]
    return pts


def _exclude_laser_occluded_corners(
        detection: CharucoDetection,
        centers: np.ndarray,
        exclusion_px: float,
        min_corners: int,
) -> CharucoDetection:
    """排除激光附近的板角点，避免蓝线破坏角点后把位姿整体拉偏。"""
    if exclusion_px <= 0 or len(centers) == 0:
        return detection
    delta = (
        detection.corners[:, None, :]
        - np.asarray(centers, dtype=np.float64)[None, :, :]
    )
    min_distance = np.sqrt(np.min(np.sum(delta * delta, axis=2), axis=1))
    keep = min_distance >= exclusion_px
    if int(keep.sum()) < max(4, min_corners):
        return detection
    return CharucoDetection(
        corners=detection.corners[keep],
        ids=detection.ids[keep],
        obj_points=detection.obj_points[keep],
    )


def _fit_frame_laser_line(
        points: np.ndarray,
        threshold: float,
        iters: int = 300,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """RANSAC 拟合单帧三维激光交线，并把内点投影回该直线。"""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 2:
        return pts, np.zeros(len(pts), dtype=bool), float("inf")
    rng = np.random.default_rng(0)
    best = np.zeros(len(pts), dtype=bool)
    for _ in range(iters):
        i, j = rng.choice(len(pts), 2, replace=False)
        direction = pts[j] - pts[i]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            continue
        direction /= norm
        residual = np.linalg.norm(
            np.cross(pts - pts[i], direction), axis=1)
        candidate = residual < threshold
        if int(candidate.sum()) > int(best.sum()):
            best = candidate
    if int(best.sum()) < 2:
        return pts, best, float("inf")

    for _ in range(3):
        selected = pts[best]
        origin = selected.mean(axis=0)
        _, _, vh = np.linalg.svd(
            selected - origin, full_matrices=False)
        direction = vh[0]
        residual = np.linalg.norm(
            np.cross(pts - origin, direction), axis=1)
        refined = residual < threshold
        if np.array_equal(refined, best):
            break
        best = refined

    selected = pts[best]
    origin = selected.mean(axis=0)
    _, _, vh = np.linalg.svd(selected - origin, full_matrices=False)
    direction = vh[0]
    signed = (selected - origin) @ direction
    projected = origin + signed[:, None] * direction[None, :]
    residual = np.linalg.norm(selected - projected, axis=1)
    rms = float(np.sqrt(np.mean(residual * residual)))
    return projected, best, rms


def _frame_consensus_plane(
        frames: Sequence[np.ndarray],
        threshold: float,
        iters: int,
        min_point_ratio: float = 0.50,
        median_factor: float = 1.50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """按“整帧激光交线”找共识，避免坏位姿贡献的整条线污染点级 RANSAC。"""
    from .geometry import plane_point_distance

    if len(frames) < 2:
        raise ValueError("帧级平面拟合至少需要 2 帧")

    def classify(plane: np.ndarray) -> Tuple[np.ndarray, int, float]:
        good = []
        point_count = 0
        medians = []
        for points in frames:
            residual = plane_point_distance(points, plane)
            ratio = float(np.mean(residual < threshold))
            median = float(np.median(residual))
            accepted = (
                ratio >= min_point_ratio
                and median <= threshold * median_factor
            )
            good.append(accepted)
            if accepted:
                point_count += int(np.sum(residual < threshold))
                medians.append(median)
        tie_break = float(np.median(medians)) if medians else float("inf")
        return np.asarray(good, dtype=bool), point_count, tie_break

    # 激光平面由至少两个不同板姿态上的激光交线约束。逐帧配对生成候选，
    # 比随机抽三个点更不容易抽到同一条近共线交线。
    candidates = []
    for i in range(len(frames) - 1):
        for j in range(i + 1, len(frames)):
            candidates.append(fit_plane_lstsq(
                np.vstack((frames[i], frames[j]))))
    point_plane, _, _ = fit_plane_ransac(
        np.vstack(frames), threshold=threshold, iters=iters)
    candidates.append(point_plane)

    best_plane = candidates[0]
    best_good = np.zeros(len(frames), dtype=bool)
    best_score = (-1, -1, float("-inf"))
    for candidate in candidates:
        if not np.isfinite(candidate).all():
            continue
        good, point_count, median = classify(candidate)
        score = (int(good.sum()), point_count, -median)
        if score > best_score:
            best_plane, best_good, best_score = candidate, good, score

    # 重新拟合并重新分类，最多三轮，使边界帧的判定不依赖初始帧对。
    for _ in range(3):
        if int(best_good.sum()) < 2:
            break
        selected = np.vstack([
            points for points, keep in zip(frames, best_good) if keep
        ])
        refined, _, _ = fit_plane_ransac(
            selected, threshold=threshold, iters=iters)
        good, _, _ = classify(refined)
        best_plane = refined
        if np.array_equal(good, best_good):
            break
        best_good = good

    selected = np.vstack([
        points for points, keep in zip(frames, best_good) if keep
    ])
    plane, inliers, rms = fit_plane_ransac(
        selected, threshold=threshold, iters=iters)
    return plane, inliers, best_good, rms


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
    calibration_cfg = cfg.get("laser_calibration") or {}
    corner_exclusion_px = float(
        calibration_cfg.get("charuco_laser_exclusion_px", 10.0))
    line_threshold = float(
        calibration_cfg.get("line_ransac_threshold", 0.001))
    min_line_inlier_ratio = float(
        calibration_cfg.get("min_line_inlier_ratio", 0.60))
    detection_channel = str(
        calibration_cfg.get(
            "charuco_detection_channel", "green")).lower()
    stripe_min_score = float(
        calibration_cfg.get("laser_min_score", 40.0))
    stripe_max_dist_px = float(
        calibration_cfg.get("stripe_max_dist_px", 2.5))

    all_pts: List[np.ndarray] = []
    used_files: List[str] = []
    used = 0

    for f in files:
        img = imread_color(f)
        if img is None:
            continue
        # 蓝激光会污染灰度图中的黑白边界；优先用绿色通道检测板，
        # 激光中心仍然从完整 BGR 图提取。
        if detection_channel == "green":
            board_image = img[:, :, 1]
        elif detection_channel == "red":
            board_image = img[:, :, 2]
        else:
            board_image = img
        det = target.detect(board_image)
        if det is None or det.count < min_corners:
            if verbose:
                print(f"  跳过(角点不足): {os.path.basename(f)}")
            continue
        # 只有落在实体标定板上的激光像素才能与 board_plane 求交。
        # 板外的地面、墙面和物体激光不能拿“无限延伸的板平面”反投影，
        # 否则会形成一个RMS看似很低、几何位置却完全错误的激光平面。
        hull = cv2.convexHull(
            np.asarray(det.corners, dtype=np.float32).reshape(-1, 1, 2)
        )
        # Mask before row/column argmax. Otherwise a brighter background laser
        # can win the row, then be discarded after extraction while the valid
        # board laser in the same row is never considered.
        board_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(
            board_mask,
            np.rint(hull.reshape(-1, 2)).astype(np.int32),
            1,
        )
        centers = extract_laser_centers(
            img, cfg, image_mask_override=board_mask.astype(bool))
        inside_board = np.array([
            cv2.pointPolygonTest(
                hull, (float(point[0]), float(point[1])), False
            ) >= 0
            for point in centers
        ], dtype=bool)
        centers = centers[inside_board]
        laser_cfg = cfg.get("laser") or {}
        score_img = blue_laser_score(
            img,
            mode=str(laser_cfg.get(
                "score_mode", "blue_minus_max")),
            blue_gate_threshold=float(
                laser_cfg.get("blue_gate_threshold", 20.0)),
            blue_gate_expand_px=int(
                laser_cfg.get("blue_gate_expand_px", 0)),
        )
        centers = _filter_stripe_centers(
            centers,
            score_img,
            min_score=stripe_min_score,
            max_line_dist_px=stripe_max_dist_px,
        )
        if centers.shape[0] < min_laser:
            if verbose:
                print(
                    f"  跳过(板内激光点不足 {centers.shape[0]}): "
                    f"{os.path.basename(f)}"
                )
            continue
        detected_count = int(centers.shape[0])
        pose_detection = _exclude_laser_occluded_corners(
            det, centers, corner_exclusion_px, min_corners)
        pose = target.estimate_pose(pose_detection, K, dist)
        if pose is None:
            continue
        rvec, tvec = pose
        reproj_error = target.reproj_error(
            pose_detection, rvec, tvec, K, dist)
        if reproj_error > max_reproj:
            if verbose:
                print(
                    f"  跳过(位姿误差 {reproj_error:.2f}px): "
                    f"{os.path.basename(f)}"
                )
            continue
        board_plane = board_plane_in_camera(rvec, tvec)

        rays = pixels_to_rays(centers, K, dist)
        pts_cam, valid = ray_plane_intersect_masked(rays, board_plane)
        pts_cam = pts_cam[valid]
        if pts_cam.shape[0] < min_laser:
            continue
        line_points, line_inliers, line_rms = _fit_frame_laser_line(
            pts_cam, threshold=line_threshold)
        line_ratio = float(line_inliers.sum() / max(len(pts_cam), 1))
        if (
            len(line_points) < min_laser
            or line_ratio < min_line_inlier_ratio
        ):
            if verbose:
                print(
                    f"  跳过(单帧激光线内点率 {line_ratio*100:.1f}%, "
                    f"RMS {line_rms*1000:.3f}mm): "
                    f"{os.path.basename(f)}"
                )
            continue

        max_points_per_frame = int(
            calibration_cfg.get("max_points_per_frame", 0)
        )
        if (
            max_points_per_frame > 0
            and len(line_points) > max_points_per_frame
        ):
            sample_indices = np.linspace(
                0, len(line_points) - 1,
                max_points_per_frame,
                dtype=int,
            )
            line_points = line_points[sample_indices]
        all_pts.append(line_points)
        used_files.append(os.path.basename(f))
        used += 1
        if verbose:
            print(
                f"  使用({line_points.shape[0]}/{detected_count} 激光点, "
                f"线RMS={line_rms*1000:.3f}mm, "
                f"板角点={pose_detection.count}/{det.count}): "
                f"{os.path.basename(f)}"
            )

    if used < 2:
        raise RuntimeError(f"有效激光标定图片过少({used})，至少需要 2 张不同姿态。")

    pts = np.vstack(all_pts)
    if pts.shape[0] < 100:
        raise RuntimeError("过滤离群点后激光 3D 点过少，请检查标定板姿态多样性与激光提取。")

    pf = cfg.get("plane_fit", {})
    if bool(pf.get("ransac", True)):
        threshold = float(pf.get("ransac_threshold", 1e-3))
        iters = int(pf.get("ransac_iters", 1000))
        frame_filter = pf.get("frame_consensus") or {}
        use_frame_consensus = bool(frame_filter.get("enabled", True))
        if use_frame_consensus:
            plane, inliers, good_frames, rms = _frame_consensus_plane(
                all_pts,
                threshold=threshold,
                iters=iters,
                min_point_ratio=float(
                    frame_filter.get("min_point_inlier_ratio", 0.50)),
                median_factor=float(
                    frame_filter.get("max_median_factor", 1.50)),
            )
            min_good_frames = int(frame_filter.get("min_good_frames", 8))
            min_good_frame_ratio = float(
                frame_filter.get("min_good_frame_ratio", 0.50))
            good_count = int(good_frames.sum())
            frame_ratio = good_count / max(len(all_pts), 1)
            if verbose:
                rejected = [
                    name for name, keep in zip(used_files, good_frames)
                    if not keep
                ]
                print(
                    f"\n帧级共识: {good_count}/{len(all_pts)} 张 "
                    f"({frame_ratio*100:.1f}%)")
                if rejected:
                    print("  自动拒绝的非共面帧:")
                    for name in rejected:
                        print(f"    {name}")
            if (
                good_count < min_good_frames
                or frame_ratio < min_good_frame_ratio
            ):
                raise RuntimeError(
                    f"只有 {good_count}/{len(all_pts)} 张图属于同一激光平面，"
                    f"要求至少 {min_good_frames} 张且占比不低于 "
                    f"{min_good_frame_ratio*100:.1f}%。这不是点数过多；"
                    "常见原因是手持标定板弯曲/拍摄时仍在运动、Gain过高导致"
                    "激光饱和，或板姿态变化不足。请固定刚性平板并用与扫描"
                    "一致的曝光增益，在多个距离和双轴倾角下静止后逐张拍摄。"
                )
            pts = np.vstack([
                points for points, keep in zip(all_pts, good_frames) if keep
            ])
        else:
            plane, inliers, rms = fit_plane_ransac(
                pts, threshold=threshold, iters=iters)
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
