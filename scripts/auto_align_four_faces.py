"""Constrained four-face registration from strong board-frame initialization."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.four_face_alignment import (COLORS, FACES, PAIRS, apply_se3,
                                     build_pair_correspondences,
                                     estimate_normals, globally_align,
                                     load_cloud, se3_matrix, voxel_points,
                                     write_json)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="板坐标强初值下的四面小 SE(3) 联合配准")
    parser.add_argument("--input-root",
                        default="ceshi/rail/two_faces/input")
    parser.add_argument("--out-dir",
                        default="ceshi/rail/two_faces/output/auto_alignment")
    parser.add_argument("--voxel-mm", type=float, default=1.5)
    parser.add_argument("--distance-levels-mm", default="15,8,4")
    parser.add_argument("--normal-angle-deg", type=float, default=18.0)
    parser.add_argument("--min-correspondences", type=int, default=100)
    parser.add_argument("--min-coverage", type=float, default=0.02)
    parser.add_argument("--max-final-rmse-mm", type=float, default=3.0)
    parser.add_argument("--max-translation-mm", type=float, default=15.0)
    parser.add_argument("--max-rotation-deg", type=float, default=5.0)
    parser.add_argument("--huber-scale", type=float, default=1.0)
    parser.add_argument("--merge-voxel-mm", type=float, default=0.5)
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def parse_levels(value: str) -> list[float]:
    levels = [float(item.strip()) * 1e-3 for item in value.split(",")
              if item.strip()]
    if not levels or any(item <= 0 for item in levels):
        raise ValueError("--distance-levels-mm 必须是逗号分隔的正数")
    return levels


def final_metrics(
    clouds: dict[str, np.ndarray],
    vectors: dict[str, np.ndarray],
    voxel: float,
    distance: float,
    normal_angle: float,
) -> dict[str, dict[str, float]]:
    sampled = {face: voxel_points(clouds[face], voxel) for face in FACES}
    metrics = {}
    for first, second in PAIRS:
        first_points = apply_se3(sampled[first], vectors[first])
        second_points = apply_se3(sampled[second], vectors[second])
        first_normals = estimate_normals(first_points, voxel * 3.0)
        second_normals = estimate_normals(second_points, voxel * 3.0)
        _, _, pair_metrics = build_pair_correspondences(
            first_points, first_normals, second_points, second_normals,
            distance, normal_angle)
        metrics[f"{first}_{second}"] = pair_metrics
    return metrics


def main() -> int:
    args = parse_args()
    levels = parse_levels(args.distance_levels_mm)
    input_root = resolve(args.input_root)
    out_dir = resolve(args.out_dir)
    aligned_dir = out_dir / "aligned"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    cloud_objects = {
        face: load_cloud(input_root / face / "cloud_clean.ply")
        for face in FACES
    }
    clouds = {
        face: np.asarray(cloud_objects[face].points).copy() for face in FACES
    }
    vectors, history = globally_align(
        clouds=clouds,
        distance_levels=levels,
        voxel=args.voxel_mm * 1e-3,
        normal_angle_deg=args.normal_angle_deg,
        max_translation=args.max_translation_mm * 1e-3,
        max_rotation_deg=args.max_rotation_deg,
        huber_scale=args.huber_scale,
    )
    metrics = final_metrics(
        clouds, vectors, args.voxel_mm * 1e-3, levels[-1],
        args.normal_angle_deg)
    finite_pairs = [
        value for value in metrics.values()
        if value["count"] > 0 and np.isfinite(value["rmse_m"])
    ]
    total_correspondences = sum(value["count"] for value in finite_pairs)
    overall_rmse_m = (
        float(np.sqrt(sum(
            value["count"] * value["rmse_m"] ** 2
            for value in finite_pairs
        ) / total_correspondences))
        if total_correspondences else float("inf")
    )
    failures = []
    for pair, value in metrics.items():
        if value["count"] < args.min_correspondences:
            failures.append(f"{pair}:correspondences")
        if min(value["coverage_first"],
               value["coverage_second"]) < args.min_coverage:
            failures.append(f"{pair}:coverage")
        if value["rmse_m"] * 1000.0 > args.max_final_rmse_mm:
            failures.append(f"{pair}:rmse")
    transform_metrics = {}
    for face in FACES:
        translation_mm = float(np.linalg.norm(vectors[face][3:]) * 1000.0)
        rotation_deg = float(np.degrees(np.linalg.norm(vectors[face][:3])))
        transform_metrics[face] = {
            "rotation_vector_rad": vectors[face][:3].tolist(),
            "rotation_deg": rotation_deg,
            "translation_m": vectors[face][3:].tolist(),
            "translation_norm_mm": translation_mm,
            "matrix": se3_matrix(vectors[face]).tolist(),
        }
        if translation_mm >= args.max_translation_mm * 0.999:
            failures.append(f"{face}:translation_hard_limit")
        if rotation_deg >= args.max_rotation_deg * 0.999:
            failures.append(f"{face}:rotation_hard_limit")
    accepted = not failures

    colored = o3d.geometry.PointCloud()
    merged = o3d.geometry.PointCloud()
    for face in FACES:
        aligned_points = apply_se3(clouds[face], vectors[face])
        aligned = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(aligned_points))
        o3d.io.write_point_cloud(
            str(aligned_dir / f"{face}_aligned.ply"), aligned)
        merged += aligned
        colored_face = o3d.geometry.PointCloud(aligned)
        colored_face.paint_uniform_color(COLORS[face])
        colored += colored_face
    candidate = merged.voxel_down_sample(args.merge_voxel_mm * 1e-3)
    comparison_path = out_dir / "comparison_colored.ply"
    candidate_path = out_dir / "merged_candidate.ply"
    o3d.io.write_point_cloud(str(comparison_path), colored)
    o3d.io.write_point_cloud(str(candidate_path), candidate)

    report = {
        "method": "board_initialized_normal_consistent_bidirectional_global_se3",
        "reference_frame": "charuco_board",
        "gauge": "face1_fixed",
        "scale_changed": False,
        "constraints": {
            "distance_levels_mm": [item * 1000.0 for item in levels],
            "normal_angle_deg": args.normal_angle_deg,
            "min_correspondences": args.min_correspondences,
            "min_coverage": args.min_coverage,
            "max_translation_mm": args.max_translation_mm,
            "max_rotation_deg": args.max_rotation_deg,
        },
        "iterations": history,
        "final_pair_metrics": metrics,
        "loop_closure": {
            "all_four_pairs_included": all(
                f"{first}_{second}" in metrics for first, second in PAIRS
            ),
            "closing_pair": "face4_face1",
            "closing_pair_metrics": metrics["face4_face1"],
            "total_correspondences": total_correspondences,
            "overall_rmse_mm": overall_rmse_m * 1000.0,
        },
        "transforms": transform_metrics,
        "quality": {
            "accepted": accepted,
            "failures": failures,
            "rejected_output_is_diagnostic_candidate": not accepted,
        },
        "outputs": {
            "aligned_dir": str(aligned_dir),
            "comparison_colored": str(comparison_path),
            "merged_candidate": str(candidate_path),
        },
        "safety_note": (
            "仅匹配法向一致的相邻扫描共享条带；未执行全云无约束 ICP。"
            "拒绝结果不可作为测量结果。"
        ),
    }
    report_path = out_dir / "auto_alignment_report.json"
    write_json(report_path, report)
    print(f"accepted={accepted}")
    for pair, value in metrics.items():
        print(f"  {pair}: n={value['count']}, "
              f"coverage={min(value['coverage_first'], value['coverage_second']):.4f}, "
              f"rmse={value['rmse_m'] * 1000.0:.3f} mm")
    print(f"报告: {report_path}")
    if not accepted:
        print("[安全拒绝] " + ", ".join(failures))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
