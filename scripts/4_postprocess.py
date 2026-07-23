"""阶段 9：点云后处理、高度图、可选 mesh 入口。

用法:
    python scripts/4_postprocess.py
    python scripts/4_postprocess.py --in output/cloud.ply
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, resolve_path  # noqa: E402
from src.postprocess import save_height_map  # noqa: E402
from src.reconstruct import read_ply_xyz  # noqa: E402


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description="点云后处理与建模")
    ap.add_argument("--in", dest="in_ply", default=None, help="输入点云 ply")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    if args.config:
        cfg = load_config(args.config)

    out_dir = resolve_path(cfg, cfg["paths"]["output"])
    in_ply = args.in_ply or os.path.join(out_dir, "cloud.ply")
    clean_ply = os.path.join(out_dir, "cloud_clean.ply")
    hm_npy = os.path.join(out_dir, "height_map.npy")
    hm_png = os.path.join(out_dir, "height_map.png")
    mesh_path = os.path.join(out_dir, "mesh.ply")

    print(f"[后处理] 输入: {in_ply}")

    # open3d 可选：装了就走完整滤波/法向/mesh，没装则用纯 numpy 出高度图
    try:
        import open3d  # noqa: F401
        has_o3d = True
    except Exception:
        has_o3d = False

    if has_o3d:
        from src.postprocess import process_point_cloud, reconstruct_mesh
        pcd = process_point_cloud(cfg, in_ply, clean_ply)
        pts = np.asarray(pcd.points)
    else:
        print("警告: 未安装 open3d，跳过滤波/法向/mesh，仅用 numpy 生成高度图。")
        from src.postprocess import crop_roi_points
        pts = read_ply_xyz(in_ply)
        pts = crop_roi_points(pts, cfg)

    if pts is None or len(pts) == 0:
        raise SystemExit(
            "后处理失败: ROI 裁剪后 0 点。"
            "若 roi.mode=manual，请按 cloud.ply 的 XYZ 重设包围盒，或改回 roi.mode=auto；"
            "也可临时设 roi.enabled: false。"
        )

    save_height_map(cfg, pts, hm_npy, hm_png)
    if has_o3d:
        from src.postprocess import reconstruct_mesh
        reconstruct_mesh(cfg, pcd, mesh_path)


if __name__ == "__main__":
    main()
