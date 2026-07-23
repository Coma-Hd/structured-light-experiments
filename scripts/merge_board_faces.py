"""合并已处于同一 ChArUco 板坐标系的多面点云，不执行 ICP。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reconstruct import write_ply  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="直接合并同一固定ChArUco板坐标系中的2～4份点云"
    )
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--voxel-mm", type=float, default=0.5)
    parser.add_argument("--sor-neighbors", type=int, default=20)
    parser.add_argument("--sor-std-ratio", type=float, default=2.5)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if len(args.inputs) < 2:
        print("[错误] 至少需要两份点云")
        return 2
    if args.voxel_mm <= 0:
        print("[错误] --voxel-mm 必须大于0")
        return 2

    point_groups = []
    input_counts = {}
    for value in args.inputs:
        path = str(Path(value).resolve())
        if not os.path.isfile(path):
            print(f"[错误] 找不到点云：{path}")
            return 2
        input_cloud = o3d.io.read_point_cloud(path)
        points = np.asarray(input_cloud.points, dtype=np.float64)
        if len(points) == 0:
            print(f"[错误] 空点云：{path}")
            return 2
        point_groups.append(points)
        input_counts[path] = int(len(points))
        print(f"[读取] {path}: {len(points)} 点")

    merged_raw = np.vstack(point_groups)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(merged_raw)
    cloud = cloud.voxel_down_sample(float(args.voxel_mm) * 1e-3)
    after_voxel = len(cloud.points)
    if args.sor_neighbors > 1 and after_voxel > args.sor_neighbors:
        cloud, _ = cloud.remove_statistical_outlier(
            nb_neighbors=int(args.sor_neighbors),
            std_ratio=float(args.sor_std_ratio),
        )
    merged = np.asarray(cloud.points, dtype=np.float64)
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_ply(str(output), merged)

    report = {
        "method": "fixed_charuco_board_direct_merge_no_icp",
        "inputs": input_counts,
        "raw_points": int(len(merged_raw)),
        "after_voxel_points": int(after_voxel),
        "final_points": int(len(merged)),
        "voxel_mm": float(args.voxel_mm),
        "sor_neighbors": int(args.sor_neighbors),
        "sor_std_ratio": float(args.sor_std_ratio),
        "warning": (
            "该合并假定所有扫描期间物体与同一块ChArUco板完全固定。"
            "若板或物体移动，不能用ICP掩盖该错误。"
        ),
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[完成] 合并点云：{output}")
    print(f"[完成] 报告：{report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
