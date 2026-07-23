"""调试工具：可视化激光中心线提取效果（阶段 5 前的自检）。

用法:
    python scripts/debug_laser.py --image data/scan/xxx.png
    python scripts/debug_laser.py --config ... --image ... --roi-json ... --distance-mm 40
"""
import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config  # noqa: E402
from src.io_utils import imread_color, imwrite_unicode  # noqa: E402
from src.keyframe_roi import (  # noqa: E402
    load_keyframe_roi_file,
    load_keyframe_roi_from_cfg,
    roi_override_for_distance_m,
)
from src.laser_center import draw_centers, extract_laser_centers  # noqa: E402
from src.rail_poses import load_rail_positions, lookup_distance, resolve_positions_path  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="激光中心提取可视化")
    ap.add_argument("--image", required=True)
    ap.add_argument("--save", default=None, help="保存可视化结果路径")
    ap.add_argument("--config", default=None, help="配置文件，默认使用项目根目录 config.yaml")
    ap.add_argument("--roi-json", default=None, help="关键帧 ROI JSON（可选）")
    ap.add_argument("--distance-mm", type=float, default=None,
                    help="该帧导轨行程 mm；缺省时尝试从 positions.csv 查")
    ap.add_argument("--positions", default=None, help="positions.csv（查行程用）")
    args = ap.parse_args()
    cfg = load_config(args.config)

    img = imread_color(args.image)
    if img is None:
        print(f"读不了图片: {args.image}")
        return

    roi_override = None
    kf = None
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.roi_json:
        kf = load_keyframe_roi_file(args.roi_json)
    else:
        try:
            kf = load_keyframe_roi_from_cfg(cfg, project_root=project_root)
        except FileNotFoundError:
            kf = None

    if kf is not None:
        dist_mm = args.distance_mm
        if dist_mm is None:
            # try positions.csv next to image or from config
            scan_dir = os.path.dirname(os.path.abspath(args.image))
            pos_name = args.positions or (cfg.get("rail") or {}).get(
                "positions_file", "positions.csv"
            )
            try:
                if args.positions and os.path.isfile(args.positions):
                    pos_path = args.positions
                else:
                    pos_path = resolve_positions_path(scan_dir, pos_name)
                positions = load_rail_positions(pos_path, distance_unit="mm")
                s_m = lookup_distance(positions, args.image)
                if s_m is not None:
                    dist_mm = float(s_m) * 1000.0
            except Exception as exc:
                print(f"警告: 无法从 positions 查行程 ({exc})")
        if dist_mm is None:
            print("警告: 未提供 distance-mm 且查不到行程，关键帧 ROI 退回静态 image_roi")
        else:
            roi_override = roi_override_for_distance_m(kf, dist_mm / 1000.0)
            print(
                f"keyframe ROI @ {dist_mm:.1f} mm: "
                f"x[{roi_override['x_min']:.3f},{roi_override['x_max']:.3f}] "
                f"y[{roi_override['y_min']:.3f},{roi_override['y_max']:.3f}]"
            )

    centers = extract_laser_centers(img, cfg, image_roi_override=roi_override)
    print(
        f"提取到 {centers.shape[0]} 个激光中心点 "
        f"(method={cfg['laser']['method']}, "
        f"score_mode={cfg['laser'].get('score_mode', 'blue_minus_max')})"
    )
    vis = draw_centers(img, centers)
    if roi_override is not None and roi_override.get("enabled", False):
        h, w = img.shape[:2]
        if roi_override.get("normalized", True):
            x0 = int(round(float(roi_override["x_min"]) * w))
            x1 = int(round(float(roi_override["x_max"]) * w))
            y0 = int(round(float(roi_override["y_min"]) * h))
            y1 = int(round(float(roi_override["y_max"]) * h))
        else:
            x0 = int(round(float(roi_override["x_min"])))
            x1 = int(round(float(roi_override["x_max"])))
            y0 = int(round(float(roi_override["y_min"])))
            y1 = int(round(float(roi_override["y_max"])))
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)

    if args.save:
        if not imwrite_unicode(args.save, vis):
            print(f"保存失败: {args.save}")
            return
        print(f"已保存: {args.save}")
    else:
        cv2.imshow("laser centers", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
