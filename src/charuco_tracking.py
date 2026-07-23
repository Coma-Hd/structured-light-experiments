"""用固定 ChArUco 板拟合连续导轨扫描的平滑相机轨迹。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Sequence

import cv2
import numpy as np

from .charuco import CharucoTarget
from .io_utils import imread_color
from .rail_poses import lookup_distance


@dataclass
class TrackingSample:
    path: str
    distance_m: float
    rotation_board_to_camera: np.ndarray
    center_board: np.ndarray
    corner_count: int
    reprojection_error_px: float
    corner_area_ratio: float


def corner_area_ratio(corners: np.ndarray, image_shape: Sequence[int]) -> float:
    """检测角点凸包面积占整幅图像面积的比例。"""
    points = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    if len(points) < 3:
        return 0.0
    height, width = int(image_shape[0]), int(image_shape[1])
    image_area = float(max(width * height, 1))
    hull = cv2.convexHull(points.reshape(-1, 1, 2))
    return float(abs(cv2.contourArea(hull)) / image_area)


def _mean_rotation(rotations: Sequence[np.ndarray]) -> np.ndarray:
    matrix = np.sum(np.asarray(rotations, dtype=np.float64), axis=0)
    u, _, vt = np.linalg.svd(matrix)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    return u @ correction @ vt


def _rotation_errors_deg(
    rotations: Sequence[np.ndarray],
    reference: np.ndarray,
) -> np.ndarray:
    errors = []
    for rotation in rotations:
        relative = np.asarray(rotation) @ reference.T
        cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
        errors.append(np.degrees(np.arccos(cosine)))
    return np.asarray(errors, dtype=np.float64)


def _fit_centers(
    distances_m: np.ndarray,
    centers: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected_s = distances_m[mask]
    reference_s = float(np.median(selected_s))
    design = np.column_stack([
        np.ones(len(selected_s), dtype=np.float64),
        selected_s - reference_s,
    ])
    coefficients, _, _, _ = np.linalg.lstsq(
        design, centers[mask], rcond=None
    )
    all_design = np.column_stack([
        np.ones(len(distances_m), dtype=np.float64),
        distances_m - reference_s,
    ])
    predicted = all_design @ coefficients
    return coefficients, predicted, np.array([reference_s])


def fit_charuco_rail_tracking(
    files: Sequence[str],
    target: CharucoTarget,
    K: np.ndarray,
    dist: np.ndarray,
    positions: Dict[str, float],
    cfg: Dict,
    verbose: bool = True,
) -> Dict:
    """检测部分可见标定板，并拟合固定姿态、直线平移的相机轨迹。

    positions 中的距离只作为时间/行程参数。相机中心和实际位移尺度来自
    已知尺寸 ChArUco 板的 PnP 位姿。
    """
    tracking_cfg = cfg.get("charuco_tracking", {}) or {}
    gating = cfg.get("gating", {}) or {}
    min_corners = int(gating.get("min_charuco_corners", 6))
    max_reproj = float(gating.get("max_reproj_error", 2.0))
    min_area_ratio = float(
        tracking_cfg.get("min_corner_area_ratio", 0.002)
    )
    min_pose_frames = int(tracking_cfg.get("min_pose_frames", 8))
    min_span_m = float(tracking_cfg.get("min_pose_span_m", 0.030))
    max_center_residual_m = float(
        tracking_cfg.get("max_center_residual_m", 0.004)
    )
    max_rotation_deviation_deg = float(
        tracking_cfg.get("max_rotation_deviation_deg", 1.5)
    )

    samples: list[TrackingSample] = []
    rejection_counts = {
        "no_position": 0,
        "no_detection": 0,
        "too_few_corners": 0,
        "corners_too_clustered": 0,
        "pose_failed": 0,
        "reprojection_error": 0,
    }
    for path in files:
        distance_m = lookup_distance(positions, path)
        if distance_m is None:
            rejection_counts["no_position"] += 1
            continue
        image = imread_color(path)
        if image is None:
            rejection_counts["no_detection"] += 1
            continue
        detection = target.detect(image)
        if detection is None:
            rejection_counts["no_detection"] += 1
            continue
        if detection.count < min_corners:
            rejection_counts["too_few_corners"] += 1
            continue
        area_ratio = corner_area_ratio(detection.corners, image.shape)
        if area_ratio < min_area_ratio:
            rejection_counts["corners_too_clustered"] += 1
            continue
        pose = target.estimate_pose(detection, K, dist)
        if pose is None:
            rejection_counts["pose_failed"] += 1
            continue
        rvec, tvec = pose
        reprojection_error = target.reproj_error(
            detection, rvec, tvec, K, dist
        )
        if reprojection_error > max_reproj:
            rejection_counts["reprojection_error"] += 1
            continue
        rotation, _ = cv2.Rodrigues(rvec)
        center_board = -rotation.T @ np.asarray(tvec).reshape(3)
        samples.append(TrackingSample(
            path=path,
            distance_m=float(distance_m),
            rotation_board_to_camera=rotation,
            center_board=center_board,
            corner_count=detection.count,
            reprojection_error_px=reprojection_error,
            corner_area_ratio=area_ratio,
        ))

    if len(samples) < min_pose_frames:
        raise RuntimeError(
            f"ChArUco 有效位姿仅 {len(samples)} 帧，至少需要 "
            f"{min_pose_frames} 帧；拒绝统计={rejection_counts}"
        )

    distances = np.asarray(
        [sample.distance_m for sample in samples], dtype=np.float64
    )
    centers = np.vstack([sample.center_board for sample in samples])
    rotations = [sample.rotation_board_to_camera for sample in samples]
    mask = np.ones(len(samples), dtype=bool)

    for _ in range(4):
        if int(mask.sum()) < min_pose_frames:
            break
        coefficients, predicted, reference_array = _fit_centers(
            distances, centers, mask
        )
        center_residuals = np.linalg.norm(centers - predicted, axis=1)
        mean_rotation = _mean_rotation([
            rotation for rotation, keep in zip(rotations, mask) if keep
        ])
        rotation_errors = _rotation_errors_deg(rotations, mean_rotation)
        selected_residuals = center_residuals[mask]
        median = float(np.median(selected_residuals))
        mad = float(np.median(np.abs(selected_residuals - median)))
        robust_limit = max(
            max_center_residual_m,
            median + 3.5 * max(1.4826 * mad, 1e-6),
        )
        new_mask = (
            (center_residuals <= robust_limit)
            & (rotation_errors <= max_rotation_deviation_deg)
        )
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    if int(mask.sum()) < min_pose_frames:
        raise RuntimeError(
            f"ChArUco 轨迹剔除异常值后仅剩 {int(mask.sum())} 帧，"
            f"至少需要 {min_pose_frames} 帧"
        )

    coefficients, predicted, reference_array = _fit_centers(
        distances, centers, mask
    )
    reference_s = float(reference_array[0])
    center_at_reference = coefficients[0]
    motion_per_nominal_distance = coefficients[1]
    motion_scale = float(np.linalg.norm(motion_per_nominal_distance))
    if motion_scale < 1e-9:
        raise RuntimeError("ChArUco 拟合出的导轨运动向量接近零")
    axis_board = motion_per_nominal_distance / motion_scale
    mean_rotation = _mean_rotation([
        rotation for rotation, keep in zip(rotations, mask) if keep
    ])
    center_residuals = np.linalg.norm(centers - predicted, axis=1)
    rotation_errors = _rotation_errors_deg(rotations, mean_rotation)
    inlier_distances = distances[mask]
    pose_span_m = float(inlier_distances.max() - inlier_distances.min())
    if pose_span_m < min_span_m:
        raise RuntimeError(
            f"ChArUco 有效位姿仅覆盖 {pose_span_m*1000:.1f} mm，"
            f"至少需要 {min_span_m*1000:.1f} mm"
        )

    rvec_mean, _ = cv2.Rodrigues(mean_rotation)
    fitted_poses: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for path in files:
        distance_m = lookup_distance(positions, path)
        if distance_m is None:
            continue
        center = (
            center_at_reference
            + (float(distance_m) - reference_s)
            * motion_per_nominal_distance
        )
        tvec = (-mean_rotation @ center).reshape(3, 1)
        fitted_poses[os.path.basename(path)] = (
            rvec_mean.reshape(3, 1).copy(),
            tvec,
        )

    center_rms_m = float(np.sqrt(np.mean(center_residuals[mask] ** 2)))
    max_center_residual = float(center_residuals[mask].max())
    max_rotation_deviation = float(rotation_errors[mask].max())
    mean_reprojection_error = float(np.mean([
        sample.reprojection_error_px
        for sample, keep in zip(samples, mask) if keep
    ]))
    accepted = (
        center_rms_m <= max_center_residual_m
        and max_center_residual <= 2.0 * max_center_residual_m
        and max_rotation_deviation <= max_rotation_deviation_deg
        and mean_reprojection_error <= max_reproj
        and 0.5 <= motion_scale <= 1.5
    )
    report = {
        "mode": "fixed_board_rail_fit",
        "valid_detection_frames": int(len(samples)),
        "inlier_pose_frames": int(mask.sum()),
        "total_frames": int(len(files)),
        "pose_span_mm": pose_span_m * 1000.0,
        "axis_board": axis_board.tolist(),
        "motion_per_nominal_distance": (
            motion_per_nominal_distance.tolist()
        ),
        "actual_over_nominal_distance": motion_scale,
        "center_fit_rms_mm": center_rms_m * 1000.0,
        "max_center_residual_mm": max_center_residual * 1000.0,
        "max_rotation_deviation_deg": max_rotation_deviation,
        "mean_corner_count": float(np.mean([
            sample.corner_count for sample, keep in zip(samples, mask) if keep
        ])),
        "mean_reprojection_error_px": mean_reprojection_error,
        "min_corner_area_ratio": float(min(
            sample.corner_area_ratio
            for sample, keep in zip(samples, mask) if keep
        )),
        "rejections": rejection_counts,
        "accepted": accepted,
    }
    if verbose:
        print(
            "[ChArUco轨迹] "
            f"有效/内点/总帧={len(samples)}/{int(mask.sum())}/{len(files)}, "
            f"行程={pose_span_m*1000:.1f} mm"
        )
        print(
            "[ChArUco轨迹] "
            f"axis_board={axis_board.tolist()}, "
            f"实际/名义位移={motion_scale:.6f}"
        )
        print(
            "[ChArUco轨迹] "
            f"中心拟合RMS={center_rms_m*1000:.3f} mm, "
            f"最大姿态偏差={max_rotation_deviation:.3f}°, "
            f"accepted={accepted}"
        )
    if not accepted:
        raise RuntimeError(
            "ChArUco 轨迹未通过质量门限："
            f"center_rms={center_rms_m*1000:.3f} mm, "
            f"max_center={max_center_residual*1000:.3f} mm, "
            f"rotation={max_rotation_deviation:.3f}°, "
            f"scale={motion_scale:.6f}"
        )
    return {
        "poses": fitted_poses,
        "report": report,
    }
