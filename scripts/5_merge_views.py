"""阶段 8b：多朝向扫描融合入口。

把工件绕竖直轴转若干个朝向、每个朝向扫一组图（板贴工件、一起转），
本脚本对每个朝向各自重建到板坐标系，再融合成一个完整点云。

用法:
    # 4 个朝向，CharUco 共坐标系直接拼接（推荐，免配准）
    python scripts/5_merge_views.py --views data/scan_v0 data/scan_v1 data/scan_v2 data/scan_v3

    # 翻面后不共坐标系，用 ICP 合并
    python scripts/5_merge_views.py --views data/front data/back --method icp

    # 顺便出网格
    python scripts/5_merge_views.py --views data/scan_v0 data/scan_v1 --mesh poisson
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, resolve_path  # noqa: E402
from src.multi_view import reconstruct_views, merge_views  # noqa: E402
from src.reconstruct import write_ply  # noqa: E402


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description="多朝向扫描融合")
    ap.add_argument("--views", nargs="+", required=True, help="各朝向扫描目录（空格分隔）")
    ap.add_argument("--method", choices=["charuco", "icp"], default="charuco",
                    help="charuco: 共板坐标系直接拼接(默认) | icp: 跨坐标系配准")
    ap.add_argument("--intrinsic", default=None)
    ap.add_argument("--laser", default=None)
    ap.add_argument("--out", default=None, help="合并后原始点云 ply")
    ap.add_argument("--mesh", choices=["none", "poisson", "bpa"], default="none")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    if args.config:
        cfg = load_config(args.config)

    intrinsic = args.intrinsic or resolve_path(cfg, cfg["paths"]["camera_intrinsic"])
    laser = args.laser or resolve_path(cfg, cfg["paths"]["laser_plane"])
    out_dir = resolve_path(cfg, cfg["paths"]["output"])
    out_ply = args.out or os.path.join(out_dir, "cloud_merged.ply")
    clean_ply = os.path.join(out_dir, "cloud_merged_clean.ply")

    missing = [v for v in args.views if not os.path.isdir(v)]
    if missing:
        print(f"错误: 以下朝向目录不存在: {missing}")
        sys.exit(1)

    print(f"[多朝向融合] {len(args.views)} 个朝向, 方式={args.method}")
    views = reconstruct_views(cfg, args.views, intrinsic, laser, out_dir)
    merged = merge_views(views, method=args.method)
    write_ply(out_ply, merged)
    print(f"\n合并原始点云: {len(merged)} 点 -> {out_ply}")

    # 去板 ROI + 滤波（复用后处理）
    try:
        import open3d  # noqa: F401
        from src.postprocess import process_point_cloud, reconstruct_mesh
    except Exception:
        print("未安装 open3d，跳过滤波/网格。合并点云已保存。")
        return

    pcd = process_point_cloud(cfg, out_ply, clean_ply)
    print(f"去板/滤波后: {len(pcd.points)} 点 -> {clean_ply}")

    if args.mesh != "none":
        cfg.setdefault("postprocess", {})["mesh_method"] = args.mesh
        mesh_path = os.path.join(out_dir, f"mesh_merged.{'obj' if args.mesh=='bpa' else 'ply'}")
        reconstruct_mesh(cfg, pcd, mesh_path)


if __name__ == "__main__":
    main()
