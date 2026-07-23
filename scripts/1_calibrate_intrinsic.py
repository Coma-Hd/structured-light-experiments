"""阶段 3：相机内参标定入口。

用法:
    python scripts/1_calibrate_intrinsic.py
    python scripts/1_calibrate_intrinsic.py --images data/intrinsic --out output/camera_intrinsic.yaml
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calib_intrinsic import calibrate_intrinsic  # noqa: E402
from src.config import load_config, resolve_path  # noqa: E402


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description="相机内参标定")
    ap.add_argument("--images", default=None, help="标定图片目录")
    ap.add_argument("--out", default=None, help="输出 yaml 路径")
    ap.add_argument("--config", default=None, help="配置文件路径")
    args = ap.parse_args()
    if args.config:
        cfg = load_config(args.config)

    image_dir = args.images or resolve_path(cfg, cfg["paths"]["intrinsic_images"])
    out_path = args.out or resolve_path(cfg, cfg["paths"]["camera_intrinsic"])
    min_corners = int(cfg.get("gating", {}).get("min_charuco_corners", 6))

    print(f"[内参标定] 图片目录: {image_dir}")
    calibrate_intrinsic(cfg, image_dir, out_path, min_corners=min_corners)


if __name__ == "__main__":
    main()
