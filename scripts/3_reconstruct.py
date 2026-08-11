"""阶段 8：逐帧重建融合入口。

用法:
    python scripts/3_reconstruct.py
    python scripts/3_reconstruct.py --images data/scan --out output/cloud.ply
    python scripts/3_reconstruct.py --config ceshi/rail/config.yaml --pose-source rail
    python scripts/3_reconstruct.py --pose-source rail --positions ceshi/rail/scan/positions.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, resolve_path  # noqa: E402
from src.reconstruct import reconstruct  # noqa: E402


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description="逐帧线激光重建")
    ap.add_argument("--images", default=None, help="扫描帧目录")
    ap.add_argument("--intrinsic", default=None)
    ap.add_argument("--laser", default=None, help="激光平面 yaml")
    ap.add_argument("--out", default=None, help="输出点云 ply")
    ap.add_argument("--config", default=None)
    ap.add_argument(
        "--pose-source",
        choices=["charuco", "rail", "turntable"],
        default=None,
        help="charuco=固定板位姿; rail=导轨平移; turntable=已标定转轴旋转",
    )
    ap.add_argument("--positions", default=None,
                    help="rail 或 charuco rail_fit 的 positions.csv")
    ap.add_argument("--angles", default=None,
                    help="turntable 的 angles.csv")
    args = ap.parse_args()
    if args.config:
        cfg = load_config(args.config)

    image_dir = args.images or resolve_path(cfg, cfg["paths"]["scan_images"])
    intrinsic = args.intrinsic or resolve_path(cfg, cfg["paths"]["camera_intrinsic"])
    laser = args.laser or resolve_path(cfg, cfg["paths"]["laser_plane"])
    out_dir = resolve_path(cfg, cfg["paths"]["output"])
    out_ply = args.out or os.path.join(out_dir, "cloud.ply")

    print(f"[重建] 扫描目录: {image_dir}")
    reconstruct(
        cfg, image_dir, intrinsic, laser, out_ply,
        pose_source=args.pose_source,
        positions_file=args.angles or args.positions,
    )


if __name__ == "__main__":
    main()
