"""Read-only diagnostics for four board-frame side scans."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calib_intrinsic import list_images
from src.charuco import CharucoTarget
from src.charuco_tracking import corner_area_ratio
from src.config import CharucoConfig
from src.four_face_alignment import (COLORS, FACES, PAIRS, corner_distances,
                                     fit_four_planes,
                                     fit_sampled_rail_trajectory, load_cloud,
                                     plane_angle_deg, proximity_coverage,
                                     write_json, xy_corners)
from src.geometry import (pixels_to_rays, ray_plane_intersect_masked,
                          scale_intrinsic, transform_board_to_cam,
                          transform_cam_to_board)
from src.io_utils import (imread_color, load_intrinsic, load_intrinsic_size,
                          load_laser_plane)
from src.keyframe_roi import (load_keyframe_roi_from_cfg,
                              pick_keyframe_indices,
                              roi_override_for_distance_m)
from src.laser_center import extract_laser_centers
from src.rail_poses import (load_rail_positions, lookup_distance,
                            resolve_positions_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="四面板坐标点云只读诊断")
    parser.add_argument("--input-root",
                        default="ceshi/rail/two_faces/input")
    parser.add_argument("--tracking-root",
                        default="ceshi/rail/two_faces/work")
    parser.add_argument("--laser-plane", default="output/laser_plane.yaml")
    parser.add_argument("--out-dir",
                        default="ceshi/rail/two_faces/output/auto_diagnostics")
    parser.add_argument("--plane-threshold-mm", type=float, default=1.5)
    parser.add_argument("--ransac-iterations", type=int, default=1500)
    parser.add_argument("--pose-compare-samples", type=int, default=9)
    parser.add_argument("--no-pose-mode-comparison", action="store_true")
    parser.add_argument("--write-colored-ply", action="store_true")
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False, "path": str(path)}
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    return {"available": True, "path": str(path), **value}


def tracking_summary(value: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "available", "path", "accepted", "total_frames",
        "valid_detection_frames", "inlier_pose_frames", "pose_span_mm",
        "center_fit_rms_mm", "max_center_residual_mm",
        "max_rotation_deviation_deg", "mean_corner_count",
        "mean_reprojection_error_px", "min_corner_area_ratio", "rejections",
    )
    return {key: value.get(key) for key in keys if key in value}


def transform_roundtrip() -> dict[str, float]:
    points = np.array([[0.10, -0.03, 0.70], [0.22, 0.08, 0.92],
                       [-0.04, 0.12, 0.55]], dtype=np.float64)
    rvec = np.array([0.17, -0.08, 0.04])
    tvec = np.array([0.02, -0.01, 0.61])
    board = transform_cam_to_board(points, rvec, tvec)
    recovered = transform_board_to_cam(board, rvec, tvec)
    error = np.linalg.norm(recovered - points, axis=1)
    return {
        "max_abs_m": float(np.max(np.abs(recovered - points))),
        "max_point_error_m": float(error.max()),
        "rms_point_error_m": float(np.sqrt(np.mean(error ** 2))),
    }


def compare_face_pose_modes(
    face: str,
    config_path: Path,
    laser_plane_path: Path,
    requested_samples: int,
) -> dict[str, Any]:
    """Compare raw PnP and sampled rail-fit poses on identical laser points."""
    base = {
        "available": False,
        "face": face,
        "config": str(config_path),
        "requested_samples": int(requested_samples),
        "interpretation": (
            "仅量化逐帧 PnP 与固定姿态直线轨迹平滑之间的点位差异；"
            "不单独证明坐标转换错误。"
        ),
    }
    try:
        if requested_samples < 3:
            raise ValueError("pose comparison requires at least 3 samples")
        if not config_path.is_file():
            raise FileNotFoundError(f"缺少扫描配置: {config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
        scan_dir = resolve(str((cfg.get("paths") or {}).get(
            "scan_images", "")))
        files = list_images(str(scan_dir))
        if not files:
            raise FileNotFoundError(f"扫描图目录为空或缺失: {scan_dir}")
        rail_cfg = cfg.get("rail", {}) or {}
        positions_name = str(rail_cfg.get("positions_file", "positions.csv"))
        positions_path = Path(resolve_positions_path(
            str(scan_dir), positions_name))
        positions = load_rail_positions(
            str(positions_path),
            distance_unit=str(rail_cfg.get("distance_unit", "mm")),
        )
        eligible = [
            path for path in files if lookup_distance(positions, path) is not None
        ]
        indices = pick_keyframe_indices(len(eligible), requested_samples)
        selected = [eligible[index] for index in indices]
        if len(selected) < 3:
            raise RuntimeError("有行程记录的扫描帧不足 3 帧")

        intrinsic_path = resolve(str((cfg.get("paths") or {}).get(
            "camera_intrinsic", "output/camera_intrinsic.yaml")))
        K_base, dist = load_intrinsic(str(intrinsic_path))
        calibration_size = load_intrinsic_size(str(intrinsic_path))
        laser_plane = load_laser_plane(str(laser_plane_path))
        target = CharucoTarget(CharucoConfig.from_cfg(cfg))
        kf_roi = load_keyframe_roi_from_cfg(
            cfg, project_root=str(PROJECT_ROOT))
        gating = cfg.get("gating", {}) or {}
        tracking_cfg = cfg.get("charuco_tracking", {}) or {}
        min_corners = int(gating.get("min_charuco_corners", 6))
        max_reprojection = float(gating.get("max_reproj_error", 2.0))
        min_area = float(tracking_cfg.get("min_corner_area_ratio", 0.002))
        min_laser = int(gating.get("min_laser_points", 12))
        rejections = {
            "image_read": 0, "detection": 0, "corners": 0, "area": 0,
            "pose": 0, "reprojection": 0, "laser": 0,
        }
        samples = []
        for path in selected:
            image = imread_color(path)
            if image is None:
                rejections["image_read"] += 1
                continue
            detection = target.detect(image)
            if detection is None:
                rejections["detection"] += 1
                continue
            if detection.count < min_corners:
                rejections["corners"] += 1
                continue
            if corner_area_ratio(detection.corners, image.shape) < min_area:
                rejections["area"] += 1
                continue
            K = K_base
            image_size = (image.shape[1], image.shape[0])
            if calibration_size[0] > 0 and image_size != calibration_size:
                K = scale_intrinsic(K_base, calibration_size, image_size)
            pose = target.estimate_pose(detection, K, dist)
            if pose is None:
                rejections["pose"] += 1
                continue
            rvec, tvec = pose
            reprojection = target.reproj_error(
                detection, rvec, tvec, K, dist)
            if reprojection > max_reprojection:
                rejections["reprojection"] += 1
                continue
            distance_m = float(lookup_distance(positions, path))
            roi_override = (
                roi_override_for_distance_m(kf_roi, distance_m)
                if kf_roi is not None else None
            )
            centers = extract_laser_centers(
                image, cfg, image_roi_override=roi_override)
            points_cam = np.empty((0, 3), dtype=np.float64)
            if len(centers) >= min_laser:
                rays = pixels_to_rays(centers, K, dist)
                triangulated, valid = ray_plane_intersect_masked(
                    rays, laser_plane)
                points_cam = triangulated[valid]
            if len(points_cam) < min_laser:
                rejections["laser"] += 1
            rotation, _ = cv2.Rodrigues(rvec)
            samples.append({
                "path": path,
                "distance_m": distance_m,
                "rotation": rotation,
                "center": -rotation.T @ np.asarray(tvec).reshape(3),
                "rvec": np.asarray(rvec).reshape(3, 1),
                "tvec": np.asarray(tvec).reshape(3, 1),
                "points_cam": points_cam,
            })

        configured_min = int(tracking_cfg.get("min_pose_frames", 8))
        min_samples = max(3, min(configured_min, requested_samples))
        fit = fit_sampled_rail_trajectory(
            np.asarray([sample["distance_m"] for sample in samples]),
            np.asarray([sample["rotation"] for sample in samples]),
            np.asarray([sample["center"] for sample in samples]),
            min_samples=min_samples,
            max_center_residual_m=float(
                tracking_cfg.get("max_center_residual_m", 0.004)),
            max_rotation_deviation_deg=float(
                tracking_cfg.get("max_rotation_deviation_deg", 1.5)),
        )
        mean_rvec, _ = cv2.Rodrigues(fit["mean_rotation"])
        differences = []
        comparison_frames = 0
        comparison_points = 0
        for index, sample in enumerate(samples):
            points_cam = sample["points_cam"]
            if not fit["inlier_mask"][index] or len(points_cam) < min_laser:
                continue
            fitted_tvec = (
                -fit["mean_rotation"] @ fit["predicted_centers"][index]
            ).reshape(3, 1)
            per_frame_points = transform_cam_to_board(
                points_cam, sample["rvec"], sample["tvec"])
            rail_fit_points = transform_cam_to_board(
                points_cam, mean_rvec, fitted_tvec)
            differences.append(np.linalg.norm(
                per_frame_points - rail_fit_points, axis=1))
            comparison_frames += 1
            comparison_points += len(points_cam)
        if not differences:
            raise RuntimeError("有效 PnP 样本没有可比较的激光三角测量点")
        values_mm = np.concatenate(differences) * 1000.0
        return {
            **base,
            "available": True,
            "scan_dir": str(scan_dir),
            "positions": str(positions_path),
            "selected_frames": int(len(selected)),
            "valid_pose_samples": int(len(samples)),
            "inlier_pose_samples": int(fit["inlier_mask"].sum()),
            "comparison_frames": comparison_frames,
            "comparison_points": comparison_points,
            "rejections": rejections,
            "sample_fit": {
                key: fit[key] for key in (
                    "center_fit_rms_mm", "max_center_residual_mm",
                    "max_rotation_deviation_deg",
                    "motion_per_nominal_distance",
                )
            },
            "per_frame_vs_sampled_rail_fit_point_difference_mm": {
                "rms": float(np.sqrt(np.mean(values_mm ** 2))),
                "median": float(np.median(values_mm)),
                "p95": float(np.percentile(values_mm, 95)),
                "max": float(values_mm.max()),
            },
        }
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as error:
        return {**base, "reason": str(error)}


def main() -> int:
    args = parse_args()
    input_root = resolve(args.input_root)
    tracking_root = resolve(args.tracking_root)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cloud_objects = {
        face: load_cloud(input_root / face / "cloud_clean.ply")
        for face in FACES
    }
    clouds = {
        face: np.asarray(cloud_objects[face].points).copy() for face in FACES
    }
    fitted = fit_four_planes(
        clouds,
        threshold=args.plane_threshold_mm * 1e-3,
        iterations=args.ransac_iterations,
    )
    z_reference = float(np.median(np.concatenate(
        [points[:, 2] for points in clouds.values()])))
    planes = {face: fitted[face]["plane"] for face in FACES}
    corners = xy_corners(planes, z_reference)

    face_report = {}
    for index, face in enumerate(FACES):
        previous = FACES[(index - 1) % 4]
        following = FACES[(index + 1) % 4]
        distance_metrics = corner_distances(
            clouds[face],
            corners[f"{previous}_{face}"],
            corners[f"{face}_{following}"],
        )
        entry = fitted[face]
        face_report[face] = {
            "point_count": int(len(clouds[face])),
            "plane": entry["plane"].tolist(),
            "inlier_count": int(entry["mask"].sum()),
            "inlier_ratio": float(entry["mask"].mean()),
            "plane_rms_mm": float(entry["rms_m"] * 1000.0),
            "inlier_center_m": entry["center"].tolist(),
            "distance_to_theoretical_adjacent_corners": distance_metrics,
        }

    pair_coverage = {}
    adjacent_angles = {}
    for first, second in PAIRS:
        key = f"{first}_{second}"
        adjacent_angles[key] = plane_angle_deg(planes[first], planes[second])
        pair_coverage[key] = proximity_coverage(clouds[first], clouds[second])
    opposite_angles = {
        "face1_face3": plane_angle_deg(planes["face1"], planes["face3"]),
        "face2_face4": plane_angle_deg(planes["face2"], planes["face4"]),
    }
    tracking = {}
    for face in FACES:
        path = tracking_root / face / "output" / "cloud_charuco_tracking.yaml"
        tracking[face] = tracking_summary(load_yaml(path))
    if args.no_pose_mode_comparison:
        pose_mode_comparison = {
            "enabled": False,
            "interpretation": "已通过 CLI 禁用；不影响其余诊断。",
        }
    else:
        pose_mode_comparison = {
            "enabled": True,
            "requested_samples_per_face": int(args.pose_compare_samples),
            "faces": {
                face: compare_face_pose_modes(
                    face,
                    resolve(
                        f"ceshi/rail/two_faces/{face}_scan.yaml"),
                    resolve(args.laser_plane),
                    args.pose_compare_samples,
                )
                for face in FACES
            },
            "interpretation": (
                "该比较只提供轨迹平滑误差证据，不参与坐标转换错误归因。"
            ),
        }

    coverage_25 = [
        pair_coverage[key]["25"]["symmetric_min"] for key in pair_coverage
    ]
    edge_p05 = [
        face_report[face]["distance_to_theoretical_adjacent_corners"]["p05_mm"]
        for face in FACES
    ]
    causes = []
    if min(coverage_25) < 0.20:
        causes.append({
            "code": "insufficient_overlap",
            "supported": True,
            "evidence": "至少一组相邻点云在 25 mm 内双向覆盖率低于 20%",
        })
    if max(edge_p05) > 10.0:
        causes.append({
            "code": "edge_coverage_missing",
            "supported": True,
            "evidence": "至少一面最靠近理论角点的 5% 点仍超过 10 mm",
        })
    causes.append({
        "code": "coordinate_transform_error",
        "supported": False,
        "evidence": (
            "数值往返测试仅验证实现自洽；当前几何证据不足以把问题归因于"
            "坐标转换。优先处理重叠与边缘覆盖。"
        ),
    })
    report = {
        "method": "read_only_four_face_board_frame_diagnostics",
        "units": {"coordinates": "m", "reported_distances": "mm"},
        "inputs": {
            face: str(input_root / face / "cloud_clean.ply") for face in FACES
        },
        "faces": face_report,
        "plane_angles_deg": {
            "adjacent": adjacent_angles,
            "opposite": opposite_angles,
        },
        "z_reference_m": z_reference,
        "adjacent_plane_xy_corners_m": {
            key: value.tolist() for key, value in corners.items()
        },
        "adjacent_cloud_proximity_coverage": pair_coverage,
        "tracking": tracking,
        "per_frame_vs_sampled_rail_fit": pose_mode_comparison,
        "laser_plane": load_yaml(resolve(args.laser_plane)),
        "transform_cam_to_board_roundtrip": transform_roundtrip(),
        "cause_classification": causes,
    }
    report_path = out_dir / "four_face_diagnostics.json"
    write_json(report_path, report)

    if args.write_colored_ply:
        colored = o3d.geometry.PointCloud()
        for face in FACES:
            copy = o3d.geometry.PointCloud(cloud_objects[face])
            copy.paint_uniform_color(COLORS[face])
            colored += copy
        o3d.io.write_point_cloud(str(out_dir / "four_faces_colored.ply"), colored)

    print(f"诊断报告: {report_path}")
    print("原因分类:", ", ".join(
        item["code"] for item in causes if item["supported"]))
    for key, metrics in pair_coverage.items():
        print(f"  {key}: 25mm symmetric coverage="
              f"{metrics['25']['symmetric_min']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
