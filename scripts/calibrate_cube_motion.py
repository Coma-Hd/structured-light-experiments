"""用已知直角正方体扫描反标定导轨运动向量。

该脚本不把最终点云直接投影到理想平面，而是从原始扫描图片重新提取
每帧相机系激光点，优化：

    P_world = P_camera + s_nominal * motion_per_distance

motion_per_distance 的方向是 rail.axis，模长是实际位移/名义位移比例。
目标是让每份扫描中的两个真实平面保持平整，并满足 90° 夹角。

仅适用于确定为标准直角件的标定扫描，不得用于未知夹角或缺陷测量数据。
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import open3d as o3d
import yaml
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calib_intrinsic import list_images  # noqa: E402
from src.config import load_config, resolve_path  # noqa: E402
from src.geometry import (pixels_to_rays, ray_plane_intersect_masked,  # noqa: E402
                          scale_intrinsic)
from src.io_utils import (imread_color, load_intrinsic,  # noqa: E402
                          load_intrinsic_size, load_laser_plane)
from src.keyframe_roi import (load_keyframe_roi_from_cfg,  # noqa: E402
                              roi_override_for_distance_m)
from src.laser_center import extract_laser_centers  # noqa: E402
from src.rail_poses import (load_rail_positions, lookup_distance,  # noqa: E402
                            resolve_positions_path)
from src.reconstruct import write_ply  # noqa: E402


@dataclass
class RawScan:
    name: str
    config_path: str
    points_camera: np.ndarray
    nominal_distance_m: np.ndarray


@dataclass
class PlaneSelection:
    point_indices: np.ndarray
    normal: np.ndarray
    inlier_count: int


@dataclass
class ScanPlanes:
    scan: RawScan
    first: PlaneSelection
    second: PlaneSelection


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用已知90°正方体联合自标定导轨轴向量和位移比例"
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "ceshi/rail/two_faces/face1_scan.yaml",
            "ceshi/rail/two_faces/face2_scan.yaml",
        ],
        help="参与联合优化的扫描配置，可只提供一份完整双平面扫描",
    )
    parser.add_argument(
        "--out",
        default="output/cube_motion_calibration.yaml",
        help="标定报告 YAML",
    )
    parser.add_argument(
        "--preview-dir",
        default="ceshi/rail/two_faces/work/cube_motion_calibration",
        help="优化前后原始点云预览目录",
    )
    parser.add_argument("--nominal-velocity-mm-s", type=float, default=1.0)
    parser.add_argument("--plane-threshold-mm", type=float, default=2.0)
    parser.add_argument("--plane-min-points", type=int, default=500)
    parser.add_argument("--min-plane-angle-deg", type=float, default=25.0)
    parser.add_argument("--max-plane-points", type=int, default=5000)
    parser.add_argument("--iterations", type=int, default=3,
                        help="重新分面和优化的交替次数")
    parser.add_argument("--max-evaluations", type=int, default=240)
    parser.add_argument("--plane-rms-target-mm", type=float, default=1.5)
    parser.add_argument("--angle-tolerance-deg", type=float, default=2.0)
    parser.add_argument("--speed-prior-fraction", type=float, default=0.30)
    parser.add_argument("--axis-prior-deg", type=float, default=35.0)
    parser.add_argument("--max-angle-error-deg", type=float, default=3.0)
    parser.add_argument("--max-plane-rms-mm", type=float, default=3.0)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="验收通过后把 axis 和 distance_scale 写入参与优化的配置",
    )
    return parser.parse_args()


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        raise ValueError("零向量不能归一化")
    return value / norm


def _acute_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    dot = float(np.clip(abs(np.dot(_unit(first), _unit(second))), 0.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def _fit_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    center = pts.mean(axis=0)
    covariance = (pts - center).T @ (pts - center) / max(len(pts), 1)
    values, vectors = np.linalg.eigh(covariance)
    normal = _unit(vectors[:, int(np.argmin(values))])
    residual = (pts - center) @ normal
    rms = float(np.sqrt(np.mean(residual ** 2)))
    return normal, rms


def _resolve_cfg_path(cfg: dict, value: str) -> str:
    return value if os.path.isabs(value) else resolve_path(cfg, value)


def _extract_raw_scan(config_path: str) -> RawScan:
    cfg = load_config(config_path)
    paths = cfg.get("paths") or {}
    image_dir = _resolve_cfg_path(cfg, str(paths["scan_images"]))
    intrinsic_path = _resolve_cfg_path(cfg, str(paths["camera_intrinsic"]))
    laser_path = _resolve_cfg_path(cfg, str(paths["laser_plane"]))
    files = list_images(image_dir)
    if not files:
        raise RuntimeError(f"{config_path}: 扫描目录没有图片")

    K, dist = load_intrinsic(intrinsic_path)
    calibration_size = load_intrinsic_size(intrinsic_path)
    probe = imread_color(files[0])
    if probe is None:
        raise RuntimeError(f"{config_path}: 第一张图无法读取")
    scan_size = (probe.shape[1], probe.shape[0])
    if calibration_size[0] > 0 and tuple(calibration_size) != scan_size:
        K = scale_intrinsic(K, calibration_size, scan_size)

    laser_plane = load_laser_plane(laser_path)
    rail_cfg = cfg.get("rail") or {}
    positions_path = resolve_positions_path(
        image_dir, str(rail_cfg.get("positions_file", "positions.csv"))
    )
    positions = load_rail_positions(
        positions_path, distance_unit=str(rail_cfg.get("distance_unit", "mm"))
    )
    min_laser = int((cfg.get("gating") or {}).get("min_laser_points", 20))
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    keyframe_roi = load_keyframe_roi_from_cfg(cfg, project_root=project_root)

    point_groups: list[np.ndarray] = []
    distance_groups: list[np.ndarray] = []
    used = 0
    for path in files:
        nominal_s = lookup_distance(positions, path)
        if nominal_s is None:
            continue
        image = imread_color(path)
        if image is None:
            continue
        roi_override = None
        if keyframe_roi is not None:
            roi_override = roi_override_for_distance_m(
                keyframe_roi, float(nominal_s)
            )
        centers = extract_laser_centers(
            image, cfg, image_roi_override=roi_override
        )
        if len(centers) < min_laser:
            continue
        rays = pixels_to_rays(centers, K, dist)
        points_camera, valid = ray_plane_intersect_masked(rays, laser_plane)
        points_camera = points_camera[valid]
        if len(points_camera) < min_laser:
            continue
        point_groups.append(points_camera)
        distance_groups.append(np.full(len(points_camera), float(nominal_s)))
        used += 1
    if not point_groups:
        raise RuntimeError(f"{config_path}: 没有提取到有效相机系激光点")
    name = Path(config_path).stem.replace("_scan", "")
    points = np.vstack(point_groups)
    distances = np.concatenate(distance_groups)
    print(
        f"[提取] {name}: {used}/{len(files)} 帧, "
        f"{len(points)} 点, 名义行程 "
        f"{(distances.max()-distances.min())*1000:.1f} mm"
    )
    return RawScan(
        name=name,
        config_path=str(Path(config_path).resolve()),
        points_camera=points,
        nominal_distance_m=distances,
    )


def _transform(scan: RawScan, motion: np.ndarray) -> np.ndarray:
    return (
        scan.points_camera
        + scan.nominal_distance_m[:, None] * np.asarray(motion)[None, :]
    )


def _segment_distinct_planes(
    scan: RawScan,
    motion: np.ndarray,
    threshold_m: float,
    min_points: int,
    min_angle_deg: float,
) -> ScanPlanes:
    points = _transform(scan, motion)
    remaining_points = points
    remaining_indices = np.arange(len(points), dtype=np.int64)
    first: PlaneSelection | None = None
    second: PlaneSelection | None = None

    if hasattr(o3d.utility, "random"):
        o3d.utility.random.seed(42)
    for _ in range(12):
        if len(remaining_points) < min_points:
            break
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(remaining_points)
        model, local_indices = cloud.segment_plane(
            distance_threshold=float(threshold_m),
            ransac_n=3,
            num_iterations=2500,
        )
        if len(local_indices) < min_points:
            break
        local_indices = np.asarray(local_indices, dtype=np.int64)
        original_indices = remaining_indices[local_indices]
        normal = _unit(np.asarray(model[:3], dtype=np.float64))
        selection = PlaneSelection(
            point_indices=original_indices,
            normal=normal,
            inlier_count=len(original_indices),
        )
        if first is None:
            first = selection
        elif _acute_angle_deg(first.normal, normal) >= min_angle_deg:
            second = selection
            break
        keep = np.ones(len(remaining_points), dtype=bool)
        keep[local_indices] = False
        remaining_points = remaining_points[keep]
        remaining_indices = remaining_indices[keep]

    if first is None or second is None:
        raise RuntimeError(
            f"{scan.name}: 无法找到两个夹角至少 {min_angle_deg:.1f}° "
            f"且各有 {min_points} 点的平面；请增加邻面条带"
        )
    return ScanPlanes(scan=scan, first=first, second=second)


def _sample_indices(indices: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    values = np.asarray(indices, dtype=np.int64)
    if len(values) <= maximum:
        return values
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(values, size=maximum, replace=False))


def _objective(
    motion: np.ndarray,
    selections: Sequence[ScanPlanes],
    initial_motion: np.ndarray,
    plane_rms_target_m: float,
    angle_tolerance_rad: float,
    speed_prior_fraction: float,
    axis_prior_rad: float,
    max_plane_points: int,
) -> float:
    vector = np.asarray(motion, dtype=np.float64)
    magnitude = float(np.linalg.norm(vector))
    if magnitude < 0.25 or magnitude > 2.0:
        return 1e6 + (magnitude - 1.0) ** 2 * 1e6

    total = 0.0
    for scan_index, item in enumerate(selections):
        transformed = _transform(item.scan, vector)
        first_idx = _sample_indices(
            item.first.point_indices, max_plane_points, 100 + scan_index * 2
        )
        second_idx = _sample_indices(
            item.second.point_indices, max_plane_points, 101 + scan_index * 2
        )
        first_normal, first_rms = _fit_plane(transformed[first_idx])
        second_normal, second_rms = _fit_plane(transformed[second_idx])
        orthogonality = abs(float(np.dot(first_normal, second_normal)))
        total += (first_rms / plane_rms_target_m) ** 2
        total += (second_rms / plane_rms_target_m) ** 2
        total += (orthogonality / max(np.sin(angle_tolerance_rad), 1e-4)) ** 2

    initial_magnitude = float(np.linalg.norm(initial_motion))
    speed_sigma = max(initial_magnitude * speed_prior_fraction, 1e-3)
    total += ((magnitude - initial_magnitude) / speed_sigma) ** 2
    axis_angle = np.arccos(np.clip(
        float(np.dot(_unit(vector), _unit(initial_motion))), -1.0, 1.0
    ))
    total += (axis_angle / max(axis_prior_rad, 1e-3)) ** 2
    return float(total)


def _plane_metrics(
    item: ScanPlanes,
    motion: np.ndarray,
) -> dict:
    transformed = _transform(item.scan, motion)
    first_normal, first_rms = _fit_plane(
        transformed[item.first.point_indices]
    )
    second_normal, second_rms = _fit_plane(
        transformed[item.second.point_indices]
    )
    return {
        "angle_deg": _acute_angle_deg(first_normal, second_normal),
        "plane_rms_mm": [first_rms * 1000.0, second_rms * 1000.0],
        "plane_points": [
            int(len(item.first.point_indices)),
            int(len(item.second.point_indices)),
        ],
        "normals": [first_normal.tolist(), second_normal.tolist()],
    }


def _update_config_motion(
    config_path: str,
    axis: np.ndarray,
    distance_scale: float,
) -> None:
    path = Path(config_path)
    text = path.read_text(encoding="utf-8")
    axis_text = ", ".join(f"{value:.9f}" for value in axis)
    axis_pattern = re.compile(r"(?m)^(\s*)axis:\s*\[[^\]]+\]\s*(?:#.*)?$")
    match = axis_pattern.search(text)
    if match is None:
        raise RuntimeError(f"{config_path}: 找不到 rail.axis")
    indent = match.group(1)
    replacement = f"{indent}axis: [{axis_text}]"
    text = axis_pattern.sub(replacement, text, count=1)

    scale_pattern = re.compile(r"(?m)^(\s*)distance_scale:\s*[^\r\n#]+(?:#.*)?$")
    if scale_pattern.search(text):
        text = scale_pattern.sub(
            f"{indent}distance_scale: {distance_scale:.9f}", text, count=1
        )
    else:
        axis_line = re.compile(
            rf"(?m)^{re.escape(indent)}axis:\s*\[[^\]]+\]\s*$"
        )
        text = axis_line.sub(
            lambda row: (
                row.group(0)
                + f"\n{indent}distance_scale: {distance_scale:.9f}"
            ),
            text,
            count=1,
        )
    path.write_text(text, encoding="utf-8")


def _write_report(
    path: str,
    initial_motion: np.ndarray,
    optimized_motion: np.ndarray,
    before: dict[str, dict],
    after: dict[str, dict],
    result,
    configs: Sequence[str],
    nominal_velocity_mm_s: float,
    accepted: bool,
) -> None:
    scale = float(np.linalg.norm(optimized_motion))
    report = {
        "method": "known_90deg_cube_motion_self_calibration",
        "warning": (
            "仅适用于确定为标准90°的标定件；不得用于未知夹角或缺陷测量。"
            "两个平面无法独立观测沿公共棱方向的运动分量，该分量由初始轴先验约束；"
            "完整三维独立标定仍需第三平面或ChArUco位姿。"
        ),
        "configs": [str(Path(value).resolve()) for value in configs],
        "initial": {
            "motion_per_nominal_distance": initial_motion.tolist(),
            "axis": _unit(initial_motion).tolist(),
            "distance_scale": float(np.linalg.norm(initial_motion)),
            "scans": before,
        },
        "optimized": {
            "motion_per_nominal_distance": optimized_motion.tolist(),
            "axis": _unit(optimized_motion).tolist(),
            "distance_scale": scale,
            "actual_velocity_estimate_mm_s": float(
                nominal_velocity_mm_s * scale
            ),
            "scans": after,
        },
        "optimizer": {
            "success": bool(result.success),
            "message": str(result.message),
            "evaluations": int(result.nfev),
            "objective": float(result.fun),
        },
        "accepted": bool(accepted),
        "usage": {
            "rail_axis": "使用 optimized.axis",
            "rail_distance_scale": "使用 optimized.distance_scale",
            "capture_velocity": (
                "若 positions.csv 仍按名义速度生成，则保留名义速度并使用 distance_scale；"
                "不要同时把采集速度改成实际速度，否则会重复缩放。"
            ),
        },
    }
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(report, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def main() -> int:
    args = _parse_args()
    if args.nominal_velocity_mm_s <= 0:
        print("[错误] --nominal-velocity-mm-s 必须大于 0")
        return 2
    try:
        scans = [_extract_raw_scan(path) for path in args.configs]
        first_cfg = load_config(args.configs[0])
        rail_cfg = first_cfg.get("rail") or {}
        initial_axis = _unit(rail_cfg.get("axis", [-1.0, 0.0, 0.0]))
        initial_scale = float(rail_cfg.get("distance_scale", 1.0))
        initial_motion = initial_axis * initial_scale
        motion = initial_motion.copy()
        threshold_m = float(args.plane_threshold_mm) * 1e-3
        rms_target_m = float(args.plane_rms_target_mm) * 1e-3
        angle_tolerance_rad = np.radians(float(args.angle_tolerance_deg))
        axis_prior_rad = np.radians(float(args.axis_prior_deg))

        preview_dir = Path(args.preview_dir).resolve()
        preview_dir.mkdir(parents=True, exist_ok=True)
        for scan in scans:
            write_ply(
                str(preview_dir / f"{scan.name}_before.ply"),
                _transform(scan, initial_motion),
            )

        initial_selections: list[ScanPlanes] = []
        for scan in scans:
            try:
                selected = _segment_distinct_planes(
                    scan,
                    motion,
                    threshold_m,
                    int(args.plane_min_points),
                    float(args.min_plane_angle_deg),
                )
                initial_selections.append(selected)
                print(
                    f"[分面] {scan.name}: "
                    f"{selected.first.inlier_count} + "
                    f"{selected.second.inlier_count} 点, "
                    f"初始夹角 "
                    f"{_acute_angle_deg(selected.first.normal, selected.second.normal):.2f}°"
                )
            except RuntimeError as error:
                print(f"[跳过] {error}")
        if not initial_selections:
            raise RuntimeError("没有任何扫描能可靠提取两个不同平面")

        before = {
            item.scan.name: _plane_metrics(item, initial_motion)
            for item in initial_selections
        }
        selections = initial_selections
        result = None
        for iteration in range(max(1, int(args.iterations))):
            result = minimize(
                _objective,
                motion,
                args=(
                    selections,
                    initial_motion,
                    rms_target_m,
                    angle_tolerance_rad,
                    float(args.speed_prior_fraction),
                    axis_prior_rad,
                    int(args.max_plane_points),
                ),
                method="Powell",
                bounds=[(-1.75, 1.75)] * 3,
                options={
                    "maxfev": int(args.max_evaluations),
                    "xtol": 1e-5,
                    "ftol": 1e-5,
                },
            )
            motion = np.asarray(result.x, dtype=np.float64)
            print(
                f"[优化 {iteration+1}] objective={result.fun:.4f}, "
                f"motion={motion.tolist()}, scale={np.linalg.norm(motion):.6f}"
            )
            refreshed: list[ScanPlanes] = []
            for item in selections:
                refreshed.append(_segment_distinct_planes(
                    item.scan,
                    motion,
                    threshold_m,
                    int(args.plane_min_points),
                    float(args.min_plane_angle_deg),
                ))
            selections = refreshed

        assert result is not None
        after = {
            item.scan.name: _plane_metrics(item, motion)
            for item in selections
        }
        accepted = all(
            abs(float(metrics["angle_deg"]) - 90.0)
            <= float(args.max_angle_error_deg)
            and max(float(value) for value in metrics["plane_rms_mm"])
            <= float(args.max_plane_rms_mm)
            for metrics in after.values()
        )
        for item in selections:
            write_ply(
                str(preview_dir / f"{item.scan.name}_after.ply"),
                _transform(item.scan, motion),
            )
        _write_report(
            args.out,
            initial_motion,
            motion,
            before,
            after,
            result,
            args.configs,
            float(args.nominal_velocity_mm_s),
            accepted,
        )

        axis = _unit(motion)
        scale = float(np.linalg.norm(motion))
        print("")
        print("[正方体运动自标定完成]")
        print(
            "  rail.axis = "
            f"[{axis[0]:.9f}, {axis[1]:.9f}, {axis[2]:.9f}]"
        )
        print(f"  rail.distance_scale = {scale:.9f}")
        print(
            f"  名义 {args.nominal_velocity_mm_s:.6f} mm/s 对应估计实际 "
            f"{args.nominal_velocity_mm_s*scale:.6f} mm/s"
        )
        for name, metrics in after.items():
            print(
                f"  {name}: angle={metrics['angle_deg']:.3f}°, "
                f"plane RMS={metrics['plane_rms_mm'][0]:.3f}/"
                f"{metrics['plane_rms_mm'][1]:.3f} mm"
            )
        print(f"  accepted = {accepted}")
        print(f"  报告：{Path(args.out).resolve()}")
        print(f"  预览：{preview_dir}")

        if args.apply:
            if not accepted:
                raise RuntimeError(
                    "结果未通过夹角/平面 RMS 验收，未修改配置；"
                    "请检查邻面条带、ROI、激光中心和标定参数"
                )
            for config_path in args.configs:
                _update_config_motion(config_path, axis, scale)
                print(f"  已更新配置：{Path(config_path).resolve()}")
            print("  下一步：重新运行两份 3_rebuild_face.ps1。")
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as error:
        print(f"[错误] {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
