"""固定的半球重建精度与重复性验证。

单次：
  python scripts/validate_hemisphere.py --inputs output/cloud_clean.ply \
      --diameter-mm 98.9 --out-md report.md

多次：
  python scripts/validate_hemisphere.py --inputs run1/cloud_clean.ply \
      run2/cloud_clean.ply run3/cloud_clean.ply --diameter-mm 98.9 \
      --out-md repeatability.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.postprocess import fit_sphere_ransac  # noqa: E402


def _load_cloud(path: str) -> np.ndarray:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("半球验证需要 open3d") from exc
    cloud = o3d.io.read_point_cloud(path)
    points = np.asarray(cloud.points, dtype=np.float64)
    if len(points) < 30:
        raise RuntimeError(f"点云点数过少({len(points)}): {path}")
    if not np.isfinite(points).all():
        points = points[np.isfinite(points).all(axis=1)]
    return points


def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return data if isinstance(data, dict) else {}


def _tracking_path_for_cloud(path: str) -> Optional[str]:
    candidate = os.path.join(
        os.path.dirname(os.path.abspath(path)),
        "cloud_charuco_tracking.yaml",
    )
    return candidate if os.path.isfile(candidate) else None


def _sample_points(
    points: np.ndarray,
    maximum: int,
    seed: int = 0,
) -> np.ndarray:
    if maximum <= 0 or len(points) <= maximum:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), maximum, replace=False)]


def fit_hemisphere(
    points: np.ndarray,
    inlier_threshold_m: float,
    maximum_fit_points: int = 50000,
) -> Dict[str, Any]:
    """RANSAC 初始化 + soft-L1 非线性球面拟合，返回固定口径指标。"""
    fit_points = _sample_points(points, maximum_fit_points)
    center0, radius0, _ = fit_sphere_ransac(
        fit_points, iters=500, inlier_k=2.5)

    def residual(parameters: np.ndarray, data: np.ndarray) -> np.ndarray:
        return np.linalg.norm(data - parameters[:3], axis=1) - parameters[3]

    solution = least_squares(
        residual,
        np.r_[center0, radius0],
        args=(fit_points,),
        loss="soft_l1",
        f_scale=0.0008,
        max_nfev=400,
    )

    center = solution.x[:3]
    radius = float(solution.x[3])
    all_residual = residual(solution.x, points)
    inlier_mask = np.abs(all_residual) <= inlier_threshold_m
    inlier_residual = all_residual[inlier_mask]
    if len(inlier_residual) < 30:
        raise RuntimeError("球面拟合内点过少，点云可能不是半球或包含大量背景。")

    abs_residual = np.abs(inlier_residual)
    spans = np.ptp(points, axis=0)
    return {
        "center_m": center.tolist(),
        "radius_m": radius,
        "diameter_mm": radius * 2000.0,
        "point_count": int(len(points)),
        "fit_point_count": int(len(fit_points)),
        "inlier_count": int(inlier_mask.sum()),
        "inlier_ratio": float(inlier_mask.mean()),
        "inlier_threshold_mm": inlier_threshold_m * 1000.0,
        "residual_mean_mm": float(np.mean(inlier_residual) * 1000.0),
        "residual_median_abs_mm": float(np.median(abs_residual) * 1000.0),
        "residual_rms_mm": float(
            np.sqrt(np.mean(inlier_residual ** 2)) * 1000.0),
        "residual_p95_abs_mm": float(
            np.percentile(abs_residual, 95) * 1000.0),
        "fraction_abs_lt_0_5mm": float(
            np.mean(abs_residual < 0.0005)),
        "fraction_abs_lt_1_0mm": float(
            np.mean(abs_residual < 0.0010)),
        "bbox_span_mm": (spans * 1000.0).tolist(),
    }


def _scan_name(path: str, index: int) -> str:
    parent = Path(path).resolve().parent.name
    stem = Path(path).stem
    if parent and parent.lower() not in {"output", "validation"}:
        return parent
    return f"{stem}_{index + 1}" if index else stem


def analyze_scan(
    path: str,
    name: str,
    true_diameter_mm: float,
    inlier_threshold_m: float,
) -> tuple[Dict[str, Any], np.ndarray]:
    points = _load_cloud(path)
    sphere = fit_hemisphere(points, inlier_threshold_m)
    diameter = float(sphere["diameter_mm"])
    error = diameter - true_diameter_mm
    tracking_path = _tracking_path_for_cloud(path)
    tracking = _load_yaml(tracking_path)
    result = {
        "name": name,
        "cloud_path": os.path.abspath(path),
        "tracking_path": tracking_path,
        "true_diameter_mm": true_diameter_mm,
        "diameter_error_mm": error,
        "diameter_error_percent": (
            error / true_diameter_mm * 100.0
            if true_diameter_mm > 0 else float("nan")
        ),
        "sphere": sphere,
        "tracking": tracking,
    }
    return result, points


def _pairwise_cloud_metrics(
    first: np.ndarray,
    second: np.ndarray,
    maximum_points: int,
) -> Dict[str, float]:
    a = _sample_points(first, maximum_points, seed=11)
    b = _sample_points(second, maximum_points, seed=17)
    dist_ab = cKDTree(b).query(a, k=1, workers=-1)[0]
    dist_ba = cKDTree(a).query(b, k=1, workers=-1)[0]
    distances = np.r_[dist_ab, dist_ba] * 1000.0
    return {
        "median_mm": float(np.median(distances)),
        "rms_mm": float(np.sqrt(np.mean(distances ** 2))),
        "p95_mm": float(np.percentile(distances, 95)),
    }


def summarize_repeats(
    scans: List[Dict[str, Any]],
    clouds: List[np.ndarray],
    true_diameter_mm: float,
    maximum_pair_points: int,
) -> Dict[str, Any]:
    diameters = np.asarray([
        scan["sphere"]["diameter_mm"] for scan in scans
    ], dtype=np.float64)
    centers = np.asarray([
        scan["sphere"]["center_m"] for scan in scans
    ], dtype=np.float64)
    pairs = []
    for first_index, second_index in combinations(range(len(scans)), 2):
        cloud_metrics = _pairwise_cloud_metrics(
            clouds[first_index],
            clouds[second_index],
            maximum_pair_points,
        )
        pairs.append({
            "first": scans[first_index]["name"],
            "second": scans[second_index]["name"],
            "diameter_difference_mm": float(abs(
                diameters[first_index] - diameters[second_index])),
            "center_distance_mm": float(np.linalg.norm(
                centers[first_index] - centers[second_index]) * 1000.0),
            **cloud_metrics,
        })
    return {
        "scan_count": len(scans),
        "mean_diameter_mm": float(np.mean(diameters)),
        "diameter_std_mm": (
            float(np.std(diameters, ddof=1)) if len(diameters) > 1 else 0.0
        ),
        "diameter_range_mm": float(np.ptp(diameters)),
        "mean_bias_mm": float(np.mean(diameters) - true_diameter_mm),
        "mean_bias_percent": float(
            (np.mean(diameters) - true_diameter_mm)
            / true_diameter_mm * 100.0
        ),
        "pairwise": pairs,
    }


def analyze_common_overlap(
    scans: List[Dict[str, Any]],
    clouds: List[np.ndarray],
    true_diameter_mm: float,
    inlier_threshold_m: float,
    percentile: float,
    maximum_pair_points: int,
) -> Dict[str, Any]:
    """在板坐标系共同覆盖盒内重新拟合，消除手动停止行程差异。"""
    lower = np.max(np.stack([
        np.percentile(cloud, percentile, axis=0) for cloud in clouds
    ]), axis=0)
    upper = np.min(np.stack([
        np.percentile(cloud, 100.0 - percentile, axis=0)
        for cloud in clouds
    ]), axis=0)
    if np.any(upper <= lower):
        raise RuntimeError("多次扫描没有有效的共同覆盖区域。")

    common_scans: List[Dict[str, Any]] = []
    common_clouds: List[np.ndarray] = []
    for scan, cloud in zip(scans, clouds):
        mask = np.all((cloud >= lower) & (cloud <= upper), axis=1)
        cropped = cloud[mask]
        if len(cropped) < 100:
            raise RuntimeError(
                f"{scan['name']} 在共同覆盖区域内只有 {len(cropped)} 点。")
        sphere = fit_hemisphere(cropped, inlier_threshold_m)
        diameter = float(sphere["diameter_mm"])
        common_scan = {
            "name": scan["name"],
            "sphere": sphere,
            "diameter_error_mm": diameter - true_diameter_mm,
            "diameter_error_percent": (
                (diameter - true_diameter_mm)
                / true_diameter_mm * 100.0
            ),
            "retained_point_ratio": float(len(cropped) / len(cloud)),
        }
        common_scans.append(common_scan)
        common_clouds.append(cropped)

    return {
        "method": "shared_board_frame_axis_aligned_percentile_box",
        "percentile": percentile,
        "lower_bound_m": lower.tolist(),
        "upper_bound_m": upper.tolist(),
        "scans": common_scans,
        "repeatability": summarize_repeats(
            common_scans,
            common_clouds,
            true_diameter_mm,
            maximum_pair_points,
        ),
    }


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.{digits}f}" if np.isfinite(number) else "—"


def _tracking_value(tracking: Dict[str, Any], key: str) -> str:
    value = tracking.get(key)
    if value is None:
        return "—"
    return str(value)


def render_markdown(report: Dict[str, Any]) -> str:
    scans = report["scans"]
    true_diameter = report["true_diameter_mm"]
    lines = [
        f"# {report['title']}",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 半球真实直径：**{true_diameter:.3f} mm**",
        f"- 球面内点阈值：±{report['settings']['inlier_threshold_mm']:.3f} mm",
        "- 配准约定：所有重复扫描直接使用 ChArUco 板坐标，**不执行 ICP**。",
        "",
        "## 结论",
        "",
    ]
    if len(scans) == 1:
        scan = scans[0]
        sphere = scan["sphere"]
        lines.extend([
            f"- 拟合直径：**{sphere['diameter_mm']:.3f} mm**。",
            f"- 相对真实值：**{scan['diameter_error_mm']:+.3f} mm "
            f"({scan['diameter_error_percent']:+.2f}%)**。",
            f"- 球面形状残差：RMS **{sphere['residual_rms_mm']:.3f} mm**，"
            f"|残差| P95 **{sphere['residual_p95_abs_mm']:.3f} mm**。",
            f"- 球面内点率：**{sphere['inlier_ratio']*100:.2f}%**。",
        ])
    else:
        repeat = report["repeatability"]
        common = report.get("common_overlap")
        lines.extend([
            f"- {len(scans)} 次拟合直径均值："
            f"**{repeat['mean_diameter_mm']:.3f} mm**。",
            f"- 绝对尺寸平均偏差：**{repeat['mean_bias_mm']:+.3f} mm "
            f"({repeat['mean_bias_percent']:+.2f}%)**。",
            f"- 直径重复性（样本标准差）："
            f"**{repeat['diameter_std_mm']:.3f} mm**；"
            f"极差 **{repeat['diameter_range_mm']:.3f} mm**。",
        ])
        if common:
            common_repeat = common["repeatability"]
            lines.extend([
                f"- 自动共同覆盖区域重复性：标准差 "
                f"**{common_repeat['diameter_std_mm']:.3f} mm**，"
                f"极差 **{common_repeat['diameter_range_mm']:.3f} mm**。"
                "该值用于消除手动停止行程不同带来的覆盖偏差。",
            ])

    lines.extend([
        "",
        "## 单次扫描结果",
        "",
        "| 扫描 | 点数 | 拟合直径 (mm) | 直径误差 (mm) | 误差 (%) | "
        "残差中位数 (mm) | RMS (mm) | P95 (mm) | 内点率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for scan in scans:
        sphere = scan["sphere"]
        lines.append(
            f"| {scan['name']} | {sphere['point_count']} | "
            f"{_fmt(sphere['diameter_mm'])} | "
            f"{scan['diameter_error_mm']:+.3f} | "
            f"{scan['diameter_error_percent']:+.2f}% | "
            f"{_fmt(sphere['residual_median_abs_mm'])} | "
            f"{_fmt(sphere['residual_rms_mm'])} | "
            f"{_fmt(sphere['residual_p95_abs_mm'])} | "
            f"{sphere['inlier_ratio']*100:.2f}% |"
        )

    lines.extend([
        "",
        "## 形状残差通过率",
        "",
        "| 扫描 | |r| < 0.5 mm | |r| < 1.0 mm | X跨度 (mm) | "
        "Y跨度 (mm) | Z跨度 (mm) |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for scan in scans:
        sphere = scan["sphere"]
        span = sphere["bbox_span_mm"]
        lines.append(
            f"| {scan['name']} | "
            f"{sphere['fraction_abs_lt_0_5mm']*100:.1f}% | "
            f"{sphere['fraction_abs_lt_1_0mm']*100:.1f}% | "
            f"{span[0]:.2f} | {span[1]:.2f} | {span[2]:.2f} |"
        )

    lines.extend([
        "",
        "## ChArUco 导轨跟踪",
        "",
        "| 扫描 | 有效帧/总帧 | center RMS (mm) | 最大中心残差 (mm) | "
        "最大旋转偏差 (°) | 重投影 (px) | actual/nominal | accepted |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for scan in scans:
        tracking = scan["tracking"]
        valid = tracking.get("valid_detection_frames")
        total = tracking.get("total_frames")
        frames = f"{valid}/{total}" if valid is not None else "—"
        lines.append(
            f"| {scan['name']} | {frames} | "
            f"{_fmt(tracking.get('center_fit_rms_mm'))} | "
            f"{_fmt(tracking.get('max_center_residual_mm'))} | "
            f"{_fmt(tracking.get('max_rotation_deviation_deg'))} | "
            f"{_fmt(tracking.get('mean_reprojection_error_px'))} | "
            f"{_fmt(tracking.get('actual_over_nominal_distance'))} | "
            f"{_tracking_value(tracking, 'accepted')} |"
        )

    common = report.get("common_overlap")
    if common:
        common_repeat = common["repeatability"]
        lines.extend([
            "",
            "## 共同覆盖区域重复性（主要判定指标）",
            "",
            f"- 边缘百分位裁剪：每次点云先去除各轴两端 "
            f"{common['percentile']:.1f}%，再取板坐标公共交集。",
            f"- 共同区域直径均值："
            f"**{common_repeat['mean_diameter_mm']:.3f} mm**",
            f"- 共同区域直径样本标准差："
            f"**{common_repeat['diameter_std_mm']:.3f} mm**",
            f"- 共同区域直径极差："
            f"**{common_repeat['diameter_range_mm']:.3f} mm**",
            "",
            "| 扫描 | 保留点数 | 保留比例 | 拟合直径 (mm) | "
            "直径误差 (mm) | RMS (mm) | P95 (mm) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for scan in common["scans"]:
            sphere = scan["sphere"]
            lines.append(
                f"| {scan['name']} | {sphere['point_count']} | "
                f"{scan['retained_point_ratio']*100:.1f}% | "
                f"{sphere['diameter_mm']:.3f} | "
                f"{scan['diameter_error_mm']:+.3f} | "
                f"{sphere['residual_rms_mm']:.3f} | "
                f"{sphere['residual_p95_abs_mm']:.3f} |"
            )
        lines.extend([
            "",
            "### 共同区域两两点云差异（板坐标系，不做 ICP）",
            "",
            "| 扫描对 | 直径差 (mm) | 球心距离 (mm) | "
            "双向最近邻中位数 (mm) | RMS (mm) | P95 (mm) |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for pair in common_repeat["pairwise"]:
            lines.append(
                f"| {pair['first']} – {pair['second']} | "
                f"{pair['diameter_difference_mm']:.3f} | "
                f"{pair['center_distance_mm']:.3f} | "
                f"{pair['median_mm']:.3f} | {pair['rms_mm']:.3f} | "
                f"{pair['p95_mm']:.3f} |"
            )

    if report.get("repeatability"):
        repeat = report["repeatability"]
        lines.extend([
            "",
            "## 完整点云重复性（受起止行程影响）",
            "",
            f"- 直径均值：**{repeat['mean_diameter_mm']:.3f} mm**",
            f"- 直径样本标准差：**{repeat['diameter_std_mm']:.3f} mm**",
            f"- 直径极差：**{repeat['diameter_range_mm']:.3f} mm**",
            "",
            "### 两两点云差异（板坐标系，不做 ICP）",
            "",
            "| 扫描对 | 直径差 (mm) | 球心距离 (mm) | "
            "双向最近邻中位数 (mm) | RMS (mm) | P95 (mm) |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for pair in repeat["pairwise"]:
            lines.append(
                f"| {pair['first']} – {pair['second']} | "
                f"{pair['diameter_difference_mm']:.3f} | "
                f"{pair['center_distance_mm']:.3f} | "
                f"{pair['median_mm']:.3f} | {pair['rms_mm']:.3f} | "
                f"{pair['p95_mm']:.3f} |"
            )

    calibration = report.get("calibration") or {}
    if calibration:
        lines.extend(["", "## 标定记录", ""])
        intrinsic = calibration.get("intrinsic") or {}
        laser = calibration.get("laser_plane") or {}
        lines.extend([
            f"- 内参重投影误差："
            f"**{_fmt(intrinsic.get('reproj_error_px'))} px**",
            f"- 激光平面拟合 RMS："
            f"**{_fmt(float(laser.get('fit_rms_m', float('nan')))*1000.0)} mm**",
        ])

    lines.extend([
        "",
        "## 判读说明",
        "",
        "- 坐标轴跨度只是当前可见区域的包围盒，不等于半球直径。",
        "- 球面 RMS/P95 描述点到拟合球面的形状一致性，不等于真实尺寸误差。",
        "- 直径偏差稳定而重复性好，通常表示激光平面、中心响应或深度尺度存在系统误差。",
        "- 扫描未覆盖接近赤道的两侧区域时，球面半径属于外推结果，对杂点和 ROI 更敏感。",
        "- JSON 文件保存全部原始统计值，可用于后续自动趋势比较。",
        "",
    ])
    return "\n".join(lines)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="固定的半球尺寸、形状残差与多次扫描重复性验证")
    parser.add_argument(
        "--inputs", nargs="+", required=True,
        help="一个或多个 cloud.ply/cloud_clean.ply")
    parser.add_argument(
        "--diameter-mm", type=float, required=True,
        help="卡尺或标准件给出的真实直径 mm")
    parser.add_argument("--out-md", required=True, help="Markdown 报告")
    parser.add_argument("--out-json", default=None, help="JSON 报告")
    parser.add_argument(
        "--inlier-mm", type=float, default=2.5,
        help="球面残差内点阈值，默认 2.5 mm")
    parser.add_argument(
        "--pair-max-points", type=int, default=50000,
        help="两两最近邻比较每个点云最多采样点数")
    parser.add_argument(
        "--common-overlap-percentile", type=float, default=2.0,
        help="共同覆盖盒计算时各轴两端裁剪百分位，默认 2")
    parser.add_argument(
        "--disable-common-overlap", action="store_true",
        help="关闭多次扫描共同覆盖区域拟合")
    parser.add_argument("--intrinsic", default=None)
    parser.add_argument("--laser-plane", default=None)
    parser.add_argument("--title", default="半球重建验证报告")
    args = parser.parse_args()

    if args.diameter_mm <= 0:
        raise ValueError("--diameter-mm 必须大于 0")
    if not 0.0 <= args.common_overlap_percentile < 25.0:
        raise ValueError("--common-overlap-percentile 必须在 [0, 25) 范围")
    missing = [path for path in args.inputs if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"找不到点云: {missing}")

    scans: List[Dict[str, Any]] = []
    clouds: List[np.ndarray] = []
    for index, path in enumerate(args.inputs):
        name = _scan_name(path, index)
        scan, points = analyze_scan(
            path,
            name,
            args.diameter_mm,
            args.inlier_mm * 1e-3,
        )
        scans.append(scan)
        clouds.append(points)
        print(
            f"[{name}] D={scan['sphere']['diameter_mm']:.3f} mm, "
            f"error={scan['diameter_error_mm']:+.3f} mm, "
            f"RMS={scan['sphere']['residual_rms_mm']:.3f} mm")

    from datetime import datetime

    report: Dict[str, Any] = {
        "schema_version": 1,
        "title": args.title,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "true_diameter_mm": args.diameter_mm,
        "settings": {
            "method": "ransac_initial_soft_l1_sphere_no_icp",
            "inlier_threshold_mm": args.inlier_mm,
            "pair_max_points": args.pair_max_points,
            "common_overlap_percentile": args.common_overlap_percentile,
        },
        "scans": scans,
        "calibration": {
            "intrinsic_path": args.intrinsic,
            "intrinsic": _load_yaml(args.intrinsic),
            "laser_plane_path": args.laser_plane,
            "laser_plane": _load_yaml(args.laser_plane),
        },
    }
    if len(scans) > 1:
        report["repeatability"] = summarize_repeats(
            scans,
            clouds,
            args.diameter_mm,
            args.pair_max_points,
        )
        if not args.disable_common_overlap:
            report["common_overlap"] = analyze_common_overlap(
                scans,
                clouds,
                args.diameter_mm,
                args.inlier_mm * 1e-3,
                args.common_overlap_percentile,
                args.pair_max_points,
            )
            common_repeat = report["common_overlap"]["repeatability"]
            print(
                "[共同覆盖] "
                f"std={common_repeat['diameter_std_mm']:.3f} mm, "
                f"range={common_repeat['diameter_range_mm']:.3f} mm")

    out_md = os.path.abspath(args.out_md)
    out_json = os.path.abspath(
        args.out_json or os.path.splitext(out_md)[0] + ".json")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as stream:
        stream.write(render_markdown(report))
    with open(out_json, "w", encoding="utf-8") as stream:
        json.dump(_json_ready(report), stream, ensure_ascii=False, indent=2)
    print(f"Markdown: {out_md}")
    print(f"JSON: {out_json}")


if __name__ == "__main__":
    main()
