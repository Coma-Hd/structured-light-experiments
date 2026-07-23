"""Complete a four-sided block without moving measured points."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.four_face_alignment import (FACES, PAIRS, fit_four_planes, load_cloud,
                                     sample_closed_cuboid, write_json,
                                     xy_corners)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="由四侧实测主平面生成明确标记的合成闭合方木块")
    parser.add_argument("--input-root",
                        default="ceshi/rail/two_faces/input")
    parser.add_argument("--out-dir",
                        default="ceshi/rail/two_faces/output/cuboid_completion")
    parser.add_argument("--plane-threshold-mm", type=float, default=1.5)
    parser.add_argument("--ransac-iterations", type=int, default=1500)
    parser.add_argument("--sample-spacing-mm", type=float, default=1.5)
    parser.add_argument("--z-percentile-low", type=float, default=2.0)
    parser.add_argument("--z-percentile-high", type=float, default=98.0)
    parser.add_argument("--orthogonalize", action="store_true")
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def orthogonal_rectangle(corners: list[np.ndarray]) -> list[np.ndarray]:
    """Return the PCA-oriented rectangle nearest the ordered four corners."""
    points = np.asarray(corners)
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    axes = vh
    coordinates = (points - center) @ axes.T
    half = (np.percentile(coordinates, 75, axis=0)
            - np.percentile(coordinates, 25, axis=0))
    candidates = np.array([
        [-half[0], -half[1]], [half[0], -half[1]],
        [half[0], half[1]], [-half[0], half[1]],
    ]) @ axes + center
    # Select cyclic direction/start that best follows face1-face2... order.
    possibilities = []
    for reverse in (False, True):
        values = candidates[::-1] if reverse else candidates
        for shift in range(4):
            ordered = np.roll(values, shift, axis=0)
            possibilities.append((np.sum((ordered - points) ** 2), ordered))
    return [point.copy() for point in min(possibilities, key=lambda item: item[0])[1]]


def main() -> int:
    args = parse_args()
    input_root = resolve(args.input_root)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    objects = {
        face: load_cloud(input_root / face / "cloud_clean.ply")
        for face in FACES
    }
    clouds = {face: np.asarray(objects[face].points).copy() for face in FACES}
    fitted = fit_four_planes(
        clouds, args.plane_threshold_mm * 1e-3, args.ransac_iterations)
    z_reference = float(np.median(np.concatenate(
        [points[:, 2] for points in clouds.values()])))
    corner_map = xy_corners(
        {face: fitted[face]["plane"] for face in FACES}, z_reference)
    corner_keys = [f"{first}_{second}" for first, second in PAIRS]
    measured_corners = [corner_map[key] for key in corner_keys]
    corners = (orthogonal_rectangle(measured_corners)
               if args.orthogonalize else measured_corners)

    per_face_z = {
        face: [
            float(np.percentile(clouds[face][:, 2], args.z_percentile_low)),
            float(np.percentile(clouds[face][:, 2], args.z_percentile_high)),
        ]
        for face in FACES
    }
    z_min = max(value[0] for value in per_face_z.values())
    z_max = min(value[1] for value in per_face_z.values())
    if z_max <= z_min:
        z_min = float(np.median([value[0] for value in per_face_z.values()]))
        z_max = float(np.median([value[1] for value in per_face_z.values()]))
    if z_max <= z_min:
        raise RuntimeError("四面没有有效共同 Z 范围")

    synthetic, vertices, triangles = sample_closed_cuboid(
        corners, z_min, z_max, args.sample_spacing_mm * 1e-3)
    measured = np.vstack([clouds[face] for face in FACES])
    measured_cloud = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(measured))
    measured_cloud.paint_uniform_color((0.15, 0.55, 0.95))
    synthetic_cloud = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(synthetic))
    synthetic_cloud.paint_uniform_color((0.95, 0.35, 0.10))
    combined = measured_cloud + synthetic_cloud
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(triangles))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color((0.95, 0.60, 0.15))

    combined_path = out_dir / "measured_blue_synthetic_orange.ply"
    synthetic_path = out_dir / "synthetic_only.ply"
    mesh_path = out_dir / "completed_cuboid_mesh.ply"
    o3d.io.write_point_cloud(str(combined_path), combined)
    o3d.io.write_point_cloud(str(synthetic_path), synthetic_cloud)
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)

    side_lengths = [
        float(np.linalg.norm(corners[(index + 1) % 4] - corners[index]))
        for index in range(4)
    ]
    height = z_max - z_min
    footprint_area = 0.5 * abs(sum(
        corners[index][0] * corners[(index + 1) % 4][1]
        - corners[(index + 1) % 4][0] * corners[index][1]
        for index in range(4)))
    report = {
        "method": "robust_four_plane_cuboid_completion",
        "measured_points_moved": False,
        "synthetic_is_measurement": False,
        "synthetic_warning": (
            "橙色点和网格由拟合平面生成，仅用于几何补全/可视化，"
            "不是传感器实测数据。"
        ),
        "orthogonalization": {
            "requested": bool(args.orthogonalize),
            "applied": bool(args.orthogonalize),
            "default_preserves_measured_angles": not args.orthogonalize,
        },
        "planes": {
            face: {
                "model": fitted[face]["plane"].tolist(),
                "rms_mm": fitted[face]["rms_m"] * 1000.0,
                "inlier_ratio": float(fitted[face]["mask"].mean()),
            }
            for face in FACES
        },
        "measured_xy_corners_m": {
            key: value.tolist() for key, value in corner_map.items()
        },
        "output_xy_corners_m": {
            key: corners[index].tolist()
            for index, key in enumerate(corner_keys)
        },
        "dimensions": {
            "side_lengths_mm": [value * 1000.0 for value in side_lengths],
            "height_mm": height * 1000.0,
            "footprint_area_mm2": footprint_area * 1e6,
        },
        "z_range": {
            "common_m": [z_min, z_max],
            "per_face_robust_m": per_face_z,
            "percentiles": [args.z_percentile_low, args.z_percentile_high],
        },
        "synthetic": {
            "point_count": int(len(synthetic)),
            "spacing_mm": args.sample_spacing_mm,
            "side_area_mm2": sum(side_lengths) * height * 1e6,
            "closed_mesh_area_mm2": (
                sum(side_lengths) * height + 2.0 * footprint_area) * 1e6,
            "mesh_vertices": int(len(vertices)),
            "mesh_triangles": int(len(triangles)),
        },
        "outputs": {
            "measured_plus_synthetic": str(combined_path),
            "synthetic_only": str(synthetic_path),
            "closed_mesh": str(mesh_path),
        },
    }
    report_path = out_dir / "cuboid_completion_report.json"
    write_json(report_path, report)
    print(f"共同 Z: {z_min * 1000:.3f} .. {z_max * 1000:.3f} mm")
    print(f"合成点数: {len(synthetic)}, mesh triangles: {len(triangles)}")
    print(f"报告: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
