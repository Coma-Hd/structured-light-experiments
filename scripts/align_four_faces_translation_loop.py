"""用四条共享棱锚点联合求解四面板内平移，并闭合 Face4/Face1。

本工具只修正 ChArUco 板坐标系中的 X/Y 平移；不旋转、不缩放、不修改
单面内部几何。四个平移的均值被约束为零，避免任意移动整个模型。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FACES = ("face1", "face2", "face3", "face4")
PAIRS = (
    ("face1", "face2"),
    ("face2", "face3"),
    ("face3", "face4"),
    ("face4", "face1"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="四条共享棱约束下的四面板内平移闭环"
    )
    parser.add_argument(
        "--input-root",
        default="ceshi/rail/two_faces/input",
    )
    parser.add_argument(
        "--anchors-root",
        default="ceshi/rail/two_faces",
    )
    parser.add_argument(
        "--out-dir",
        default="ceshi/rail/two_faces/output/translation_loop",
    )
    parser.add_argument("--voxel-mm", type=float, default=0.5)
    parser.add_argument("--sor-neighbors", type=int, default=20)
    parser.add_argument("--sor-std-ratio", type=float, default=2.5)
    parser.add_argument("--max-translation-mm", type=float, default=25.0)
    parser.add_argument("--max-anchor-rmse-mm", type=float, default=5.0)
    return parser.parse_args()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_cloud(path: Path) -> o3d.geometry.PointCloud:
    cloud = o3d.io.read_point_cloud(str(path))
    if len(cloud.points) == 0:
        raise RuntimeError(f"点云为空或无法读取: {path}")
    return cloud


def _load_anchor_pair(
    path: Path,
    expected_target: str,
    expected_source: str,
) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise RuntimeError(f"缺少共享棱锚点: {path}")
    doc: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    target_name = str(doc.get("target_name", expected_target)).lower()
    source_name = str(doc.get("source_name", expected_source)).lower()
    if target_name != expected_target or source_name != expected_source:
        raise RuntimeError(
            f"锚点面名不匹配: {path}; "
            f"需要 {expected_target}/{expected_source}, "
            f"实际 {target_name}/{source_name}"
        )
    target = doc.get("target_points") or doc.get("target_face1_points")
    source = doc.get("source_points") or doc.get("source_face2_points")
    target_points = np.asarray(target, dtype=np.float64)
    source_points = np.asarray(source, dtype=np.float64)
    if (
        target_points.ndim != 2
        or target_points.shape[1] != 3
        or source_points.shape != target_points.shape
        or len(target_points) < 2
        or not np.isfinite(target_points).all()
        or not np.isfinite(source_points).all()
    ):
        raise RuntimeError(
            f"锚点必须是数量相同的至少两对三维点: {path}"
        )
    return target_points, source_points


def _solve_xy_translations(
    anchor_data: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
) -> dict[str, np.ndarray]:
    face_index = {face: index for index, face in enumerate(FACES)}
    rows: list[np.ndarray] = []
    values: list[float] = []

    for (target_face, source_face), (target, source) in anchor_data.items():
        target_index = face_index[target_face]
        source_index = face_index[source_face]
        for target_point, source_point in zip(target, source):
            for axis in range(2):
                row = np.zeros(2 * len(FACES), dtype=np.float64)
                row[2 * target_index + axis] = 1.0
                row[2 * source_index + axis] = -1.0
                rows.append(row)
                values.append(float(source_point[axis] - target_point[axis]))

    # Gauge constraint: keep the mean correction at zero in board X/Y.
    # A high weight fixes only the common translation and does not favor Face1.
    gauge_weight = 100.0
    for axis in range(2):
        row = np.zeros(2 * len(FACES), dtype=np.float64)
        for face_index_value in range(len(FACES)):
            row[2 * face_index_value + axis] = gauge_weight
        rows.append(row)
        values.append(0.0)

    matrix = np.vstack(rows)
    rhs = np.asarray(values, dtype=np.float64)
    solution, _, rank, _ = np.linalg.lstsq(matrix, rhs, rcond=None)
    if rank < 2 * len(FACES):
        raise RuntimeError("四边约束矩阵秩不足，检查四份锚点是否完整")

    translations = {}
    for face, index in face_index.items():
        translations[face] = np.array(
            [solution[2 * index], solution[2 * index + 1], 0.0],
            dtype=np.float64,
        )
    return translations


def _anchor_report(
    anchor_data: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    translations: dict[str, np.ndarray],
) -> tuple[dict[str, Any], float]:
    report: dict[str, Any] = {}
    all_final_residuals = []
    for (target_face, source_face), (target, source) in anchor_data.items():
        initial_residuals = np.linalg.norm(target - source, axis=1)
        corrected_target = target + translations[target_face]
        corrected_source = source + translations[source_face]
        final_vectors = corrected_target - corrected_source
        final_residuals = np.linalg.norm(final_vectors, axis=1)
        all_final_residuals.extend(final_residuals.tolist())
        key = f"{target_face}_{source_face}"
        report[key] = {
            "pairs": int(len(target)),
            "initial_rmse_mm": float(
                np.sqrt(np.mean(initial_residuals ** 2)) * 1000.0
            ),
            "final_rmse_mm": float(
                np.sqrt(np.mean(final_residuals ** 2)) * 1000.0
            ),
            "final_xy_rmse_mm": float(
                np.sqrt(np.mean(np.sum(final_vectors[:, :2] ** 2, axis=1)))
                * 1000.0
            ),
            "final_z_rmse_mm": float(
                np.sqrt(np.mean(final_vectors[:, 2] ** 2)) * 1000.0
            ),
        }
    overall = float(
        np.sqrt(np.mean(np.square(all_final_residuals))) * 1000.0
    )
    return report, overall


def _transform_from_translation(translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = translation
    return transform


def _translated_copy(
    cloud: o3d.geometry.PointCloud,
    translation: np.ndarray,
) -> o3d.geometry.PointCloud:
    copied = o3d.geometry.PointCloud(cloud)
    copied.translate(translation)
    return copied


def main() -> int:
    args = _parse_args()
    input_root = _resolve(args.input_root)
    anchors_root = _resolve(args.anchors_root)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    aligned_dir = out_dir / "aligned"
    aligned_dir.mkdir(parents=True, exist_ok=True)

    clouds = {
        face: _load_cloud(input_root / face / "cloud_clean.ply")
        for face in FACES
    }
    anchor_data = {}
    anchor_paths = {}
    for target_face, source_face in PAIRS:
        path = anchors_root / (
            f"anchors_loop_{target_face}_{source_face}.json"
        )
        anchor_paths[f"{target_face}_{source_face}"] = str(path)
        anchor_data[(target_face, source_face)] = _load_anchor_pair(
            path,
            target_face,
            source_face,
        )

    translations = _solve_xy_translations(anchor_data)
    pair_report, overall_anchor_rmse_mm = _anchor_report(
        anchor_data, translations
    )
    translation_norms_mm = {
        face: float(np.linalg.norm(value) * 1000.0)
        for face, value in translations.items()
    }
    max_translation_mm = max(translation_norms_mm.values())
    accepted = (
        max_translation_mm <= float(args.max_translation_mm)
        and overall_anchor_rmse_mm <= float(args.max_anchor_rmse_mm)
    )

    colors = {
        "face1": (1.0, 0.2, 0.2),
        "face2": (0.2, 1.0, 0.2),
        "face3": (0.2, 0.4, 1.0),
        "face4": (1.0, 0.8, 0.1),
    }
    aligned = {}
    colored = o3d.geometry.PointCloud()
    merged = o3d.geometry.PointCloud()
    for face in FACES:
        aligned[face] = _translated_copy(clouds[face], translations[face])
        o3d.io.write_point_cloud(
            str(aligned_dir / f"{face}_aligned.ply"),
            aligned[face],
        )
        colored_face = o3d.geometry.PointCloud(aligned[face])
        colored_face.paint_uniform_color(colors[face])
        colored += colored_face
        merged += aligned[face]

    raw_points = int(len(merged.points))
    merged = merged.voxel_down_sample(float(args.voxel_mm) * 1e-3)
    after_voxel = int(len(merged.points))
    if args.sor_neighbors > 1 and after_voxel > args.sor_neighbors:
        merged, _ = merged.remove_statistical_outlier(
            nb_neighbors=int(args.sor_neighbors),
            std_ratio=float(args.sor_std_ratio),
        )

    comparison_path = out_dir / "loop_comparison_colored.ply"
    candidate_path = out_dir / "loop_merged_candidate.ply"
    o3d.io.write_point_cloud(str(comparison_path), colored)
    o3d.io.write_point_cloud(str(candidate_path), merged)

    transforms = {
        f"T_{face}_corrected_from_board": (
            _transform_from_translation(translations[face]).tolist()
        )
        for face in FACES
    }
    report = {
        "method": "four_edge_translation_only_global_loop",
        "coordinate_frame": "charuco_board",
        "rotation_changed": False,
        "scale_changed": False,
        "z_translation_changed": False,
        "anchors": anchor_paths,
        "translations_mm": {
            face: (translations[face] * 1000.0).tolist()
            for face in FACES
        },
        "translation_norms_mm": translation_norms_mm,
        "pair_anchor_metrics": pair_report,
        "overall_anchor_rmse_mm": overall_anchor_rmse_mm,
        "quality": {
            "max_translation_mm": float(args.max_translation_mm),
            "max_anchor_rmse_mm": float(args.max_anchor_rmse_mm),
            "accepted": accepted,
        },
        "transforms": transforms,
        "points": {
            "raw": raw_points,
            "after_voxel": after_voxel,
            "final": int(len(merged.points)),
        },
        "outputs": {
            "comparison": str(comparison_path),
            "candidate": str(candidate_path),
            "aligned_dir": str(aligned_dir),
        },
        "warning": (
            "该结果只修正板平面内平移。若单面几何、尺度、激光平面或"
            "旋转本身错误，不能用本工具掩盖。"
        ),
    }
    report_path = out_dir / "loop_alignment_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[四边平移闭环]")
    for face in FACES:
        values = translations[face] * 1000.0
        print(
            f"  {face}: dx={values[0]:.3f} mm, "
            f"dy={values[1]:.3f} mm, norm={translation_norms_mm[face]:.3f} mm"
        )
    for pair, metrics in pair_report.items():
        print(
            f"  {pair}: anchor RMSE "
            f"{metrics['initial_rmse_mm']:.3f} -> "
            f"{metrics['final_rmse_mm']:.3f} mm"
        )
    print(f"  overall anchor RMSE: {overall_anchor_rmse_mm:.3f} mm")
    print(f"  accepted: {accepted}")
    print(f"  comparison: {comparison_path}")
    print(f"  candidate:  {candidate_path}")
    print(f"  report:     {report_path}")
    if not accepted:
        print(
            "[拒绝] 修正量或最终锚点误差超过门限。"
            "检查是否点错共享棱、顶部/底部顺序是否相反。"
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
