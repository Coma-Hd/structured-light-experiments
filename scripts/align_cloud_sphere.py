"""把 cloud_clean 对齐到拟合球局部坐标，便于斜视扫描看曲率。

用法:
    python scripts/align_cloud_sphere.py --in ceshi/rail/output/cloud_clean.ply
    python scripts/align_cloud_sphere.py --in ... --out ... --inspect-out ...
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.postprocess import align_points_to_sphere_frame, fit_sphere_ransac  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="对齐点云到拟合球局部坐标")
    ap.add_argument("--in", dest="in_ply", required=True, help="输入 cloud_clean.ply")
    ap.add_argument("--out", dest="out_ply", default=None, help="输出 aligned ply")
    ap.add_argument(
        "--inspect-out",
        default=None,
        help="inspect_cloud 输出前缀（默认与 out 同目录 inspect_aligned）",
    )
    args = ap.parse_args()

    try:
        import open3d as o3d
    except ImportError as exc:
        raise SystemExit("需要 open3d") from exc

    pcd = o3d.io.read_point_cloud(args.in_ply)
    pts = np.asarray(pcd.points)
    if len(pts) < 30:
        raise SystemExit(f"点数过少: {len(pts)}")

    aligned, center, radius = align_points_to_sphere_frame(pts)
    _, _, res = fit_sphere_ransac(pts)
    print(
        f"sphere align: R={radius*1000:.1f}mm center_mm={np.round(center*1000,1)} "
        f"res_std={res.std()*1000:.2f}mm n={len(pts)}"
    )

    out_ply = args.out_ply
    if out_ply is None:
        base_dir = os.path.dirname(os.path.abspath(args.in_ply))
        out_ply = os.path.join(base_dir, "cloud_clean_aligned.ply")

    out_pcd = o3d.geometry.PointCloud()
    out_pcd.points = o3d.utility.Vector3dVector(aligned)
    if pcd.has_colors():
        out_pcd.colors = pcd.colors
    os.makedirs(os.path.dirname(os.path.abspath(out_ply)), exist_ok=True)
    o3d.io.write_point_cloud(out_ply, out_pcd)
    print(f"已保存: {out_ply}")

    inspect_out = args.inspect_out
    if inspect_out is None:
        base_dir = os.path.dirname(os.path.abspath(out_ply))
        inspect_out = os.path.join(base_dir, "inspect_aligned")

    # reuse inspect_cloud CLI
    import subprocess

    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "inspect_cloud.py"),
        "--in",
        out_ply,
        "--out",
        inspect_out,
    ]
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
