"""用三组相邻面变换把 Face1～Face4 链式合并到 Face1 坐标系。

Face1/Face2 复用已经验收的两面配准；Face2/Face3 和 Face3/Face4
调用现有双平面 + 锚点 + ICP 流程。没有 Face4/Face1 共同观测时不强制闭环。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import open3d as o3d


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="手动共享棱约束下的 Face1～Face4 链式配准"
    )
    parser.add_argument(
        "--config",
        default="ceshi/rail/two_faces/config.yaml",
    )
    parser.add_argument(
        "--anchors-23",
        default="ceshi/rail/two_faces/anchors_face2_face3.json",
    )
    parser.add_argument(
        "--anchors-34",
        default="ceshi/rail/two_faces/anchors_face3_face4.json",
    )
    parser.add_argument(
        "--pair12-dir",
        default="ceshi/rail/two_faces/output",
        help="已经验收的 Face1/Face2 输出目录",
    )
    parser.add_argument(
        "--out-dir",
        default="ceshi/rail/two_faces/output/chain",
    )
    parser.add_argument("--voxel-mm", type=float, default=0.5)
    parser.add_argument("--sor-neighbors", type=int, default=20)
    parser.add_argument("--sor-std-ratio", type=float, default=2.5)
    return parser.parse_args()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_transform(path: Path) -> np.ndarray:
    doc = _load_json(path)
    matrix = doc.get("T_face2_to_face1")
    if matrix is None:
        matrix = doc.get("transformation")
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise RuntimeError(f"无效的 4x4 变换: {path}")
    return transform


def _require_accepted_report(path: Path, pair_name: str) -> Dict[str, Any]:
    report = _load_json(path)
    final = (report.get("registration") or {}).get("final") or {}
    if not bool(final.get("accepted", False)):
        raise RuntimeError(f"{pair_name} 配准未通过质量门限: {path}")
    return report


def _run_pair(
    config: Path,
    target: Path,
    source: Path,
    anchors: Path,
    out_dir: Path,
    pair_name: str,
) -> tuple[np.ndarray, Dict[str, Any]]:
    if not anchors.is_file():
        raise RuntimeError(
            f"缺少 {pair_name} 手动锚点: {anchors}\n"
            "请先运行 4_pick_shared_edge.ps1。"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "6_merge_two_faces.py"),
        "--config",
        str(config),
        "--target",
        str(target),
        "--source",
        str(source),
        "--anchors",
        str(anchors),
        "--out-dir",
        str(out_dir),
    ]
    print(f"\n[链式配准] 开始 {pair_name}")
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{pair_name} 配准失败（exit code {completed.returncode}）"
        )
    transform = _load_transform(out_dir / "transform_face2_to_face1.json")
    report = _require_accepted_report(
        out_dir / "registration_report.json", pair_name
    )
    return transform, report


def _load_cloud(path: Path) -> o3d.geometry.PointCloud:
    cloud = o3d.io.read_point_cloud(str(path))
    if len(cloud.points) == 0:
        raise RuntimeError(f"点云为空或无法读取: {path}")
    return cloud


def _copy_and_transform(
    cloud: o3d.geometry.PointCloud, transform: np.ndarray
) -> o3d.geometry.PointCloud:
    copied = o3d.geometry.PointCloud(cloud)
    copied.transform(transform)
    return copied


def _colored_copy(
    cloud: o3d.geometry.PointCloud, color: tuple[float, float, float]
) -> o3d.geometry.PointCloud:
    copied = o3d.geometry.PointCloud(cloud)
    copied.paint_uniform_color(color)
    return copied


def _matrix_doc(matrix: np.ndarray) -> list[list[float]]:
    return matrix.astype(float).tolist()


def main() -> int:
    args = _parse_args()
    if args.voxel_mm <= 0:
        raise SystemExit("--voxel-mm 必须大于 0")

    config = _resolve(args.config)
    anchors23 = _resolve(args.anchors_23)
    anchors34 = _resolve(args.anchors_34)
    pair12_dir = _resolve(args.pair12_dir)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = {
        face: _resolve(
            f"ceshi/rail/two_faces/input/{face}/cloud_clean.ply"
        )
        for face in ("face1", "face2", "face3", "face4")
    }
    for face, path in inputs.items():
        if not path.is_file():
            raise SystemExit(f"缺少 {face} 点云: {path}")

    pair12_report = _require_accepted_report(
        pair12_dir / "registration_report.json", "Face1/Face2"
    )
    transform21 = _load_transform(
        pair12_dir / "transform_face2_to_face1.json"
    )
    transform32, pair23_report = _run_pair(
        config,
        inputs["face2"],
        inputs["face3"],
        anchors23,
        out_dir / "pair23",
        "Face2/Face3",
    )
    transform43, pair34_report = _run_pair(
        config,
        inputs["face3"],
        inputs["face4"],
        anchors34,
        out_dir / "pair34",
        "Face3/Face4",
    )

    identity = np.eye(4, dtype=np.float64)
    transform31 = transform21 @ transform32
    transform41 = transform31 @ transform43
    transforms = {
        "face1": identity,
        "face2": transform21,
        "face3": transform31,
        "face4": transform41,
    }

    aligned_dir = out_dir / "aligned"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    aligned = {}
    input_counts = {}
    for face, path in inputs.items():
        source = _load_cloud(path)
        input_counts[face] = int(len(source.points))
        aligned[face] = _copy_and_transform(source, transforms[face])
        o3d.io.write_point_cloud(
            str(aligned_dir / f"{face}_aligned_to_face1.ply"),
            aligned[face],
        )

    colored = o3d.geometry.PointCloud()
    colors = {
        "face1": (1.0, 0.2, 0.2),
        "face2": (0.2, 1.0, 0.2),
        "face3": (0.2, 0.4, 1.0),
        "face4": (1.0, 0.8, 0.1),
    }
    merged = o3d.geometry.PointCloud()
    for face in ("face1", "face2", "face3", "face4"):
        merged += aligned[face]
        colored += _colored_copy(aligned[face], colors[face])

    raw_points = int(len(merged.points))
    merged = merged.voxel_down_sample(float(args.voxel_mm) * 1e-3)
    after_voxel = int(len(merged.points))
    if args.sor_neighbors > 1 and after_voxel > args.sor_neighbors:
        merged, _ = merged.remove_statistical_outlier(
            nb_neighbors=int(args.sor_neighbors),
            std_ratio=float(args.sor_std_ratio),
        )

    merged_path = out_dir / "chain_merged_clean.ply"
    colored_path = out_dir / "chain_comparison_colored.ply"
    o3d.io.write_point_cloud(str(merged_path), merged)
    o3d.io.write_point_cloud(str(colored_path), colored)

    report = {
        "method": "manual_shared_edge_chain_no_loop_closure",
        "reference_frame": "face1",
        "inputs": {face: str(path) for face, path in inputs.items()},
        "input_counts": input_counts,
        "pair_reports": {
            "face1_face2": pair12_report["registration"]["final"],
            "face2_face3": pair23_report["registration"]["final"],
            "face3_face4": pair34_report["registration"]["final"],
        },
        "transforms": {
            "T_face2_to_face1": _matrix_doc(transform21),
            "T_face3_to_face2": _matrix_doc(transform32),
            "T_face4_to_face3": _matrix_doc(transform43),
            "T_face3_to_face1": _matrix_doc(transform31),
            "T_face4_to_face1": _matrix_doc(transform41),
        },
        "merged": {
            "raw_points": raw_points,
            "after_voxel_points": after_voxel,
            "final_points": int(len(merged.points)),
            "voxel_mm": float(args.voxel_mm),
        },
        "outputs": {
            "merged": str(merged_path),
            "colored_comparison": str(colored_path),
            "aligned_dir": str(aligned_dir),
        },
        "closure": {
            "face4_face1_observed": False,
            "accepted": None,
            "warning": (
                "Face4/Face1 没有共同可见区域，因此不能验证或强制闭环；"
                "当前结果是 Face1-Face2-Face3-Face4 的链式配准。"
            ),
        },
    }
    report_path = out_dir / "chain_registration_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n[完成] 四面链式配准（未执行 Face4/Face1 闭环）")
    print(f"  合并点云: {merged_path}")
    print(f"  彩色检查: {colored_path}")
    print(f"  总报告:   {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
