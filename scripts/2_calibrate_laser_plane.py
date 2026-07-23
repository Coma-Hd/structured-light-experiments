"""阶段 4：激光平面标定入口。

用法:
    python scripts/2_calibrate_laser_plane.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calib_laser_plane import calibrate_laser_plane  # noqa: E402
from src.config import load_config, resolve_path  # noqa: E402


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description="激光平面标定")
    ap.add_argument("--images", default=None, help="激光标定图片目录")
    ap.add_argument("--intrinsic", default=None, help="相机内参 yaml")
    ap.add_argument("--out", default=None, help="输出激光平面 yaml")
    ap.add_argument("--config", default=None)
    ap.add_argument(
        "--filename-prefix",
        default=None,
        help="只使用文件名以此前缀开头的同一批标定图",
    )
    args = ap.parse_args()
    if args.config:
        cfg = load_config(args.config)
    if args.filename_prefix:
        cfg.setdefault("laser_calibration", {})[
            "filename_prefix"
        ] = args.filename_prefix

    image_dir = args.images or resolve_path(cfg, cfg["paths"]["laser_images"])
    intrinsic = args.intrinsic or resolve_path(cfg, cfg["paths"]["camera_intrinsic"])
    out_path = args.out or resolve_path(cfg, cfg["paths"]["laser_plane"])

    print(f"[激光平面标定] 图片目录: {image_dir}")
    calibrate_laser_plane(cfg, image_dir, intrinsic, out_path)


if __name__ == "__main__":
    main()
