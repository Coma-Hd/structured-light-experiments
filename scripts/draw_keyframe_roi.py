"""在扫描序列上选 3～5 张关键帧，拖拽框选物体面，保存 keyframe ROI JSON。

用法:
    python scripts/draw_keyframe_roi.py --config ceshi/rail/keyframe_roi/config.yaml
    python scripts/draw_keyframe_roi.py --images ceshi/rail/scan/球测试-单曲面 --out ceshi/rail/keyframe_roi/roi_keyframes.json --n 5

操作:
    鼠标拖拽画矩形
    ENTER / SPACE  确认当前帧并进入下一张
    r              重画当前帧
    s              跳过当前帧（不写入）
    q / ESC        退出（已确认的帧仍会保存，若至少 1 帧）
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, resolve_path  # noqa: E402
from src.io_utils import imread_color  # noqa: E402
from src.keyframe_roi import pick_keyframe_indices, save_keyframe_roi_file  # noqa: E402
from src.rail_poses import load_rail_positions, resolve_positions_path  # noqa: E402


class RectDrawer:
    def __init__(self, base_bgr: np.ndarray, title: str):
        self.base = base_bgr
        self.title = title
        self.dragging = False
        self.p0: Optional[Tuple[int, int]] = None
        self.p1: Optional[Tuple[int, int]] = None
        self.done_rect: Optional[Tuple[int, int, int, int]] = None

    def _clamp(self, x: int, y: int) -> Tuple[int, int]:
        h, w = self.base.shape[:2]
        return max(0, min(w - 1, x)), max(0, min(h - 1, y))

    def _draw(self) -> None:
        vis = self.base.copy()
        rect = self.done_rect
        if self.dragging and self.p0 is not None and self.p1 is not None:
            x0, y0 = self.p0
            x1, y1 = self.p1
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 2)
        elif rect is not None:
            x0, y0, x1, y1 = rect
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        tip = "drag ROI | ENTER=next | r=redraw | s=skip | q=quit"
        cv2.putText(vis, tip, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, tip, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow(self.title, vis)

    def _on_mouse(self, event, x, y, flags, _param) -> None:
        x, y = self._clamp(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.p0 = (x, y)
            self.p1 = (x, y)
            self.done_rect = None
            self._draw()
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.p1 = (x, y)
            self._draw()
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            self.p1 = (x, y)
            assert self.p0 is not None and self.p1 is not None
            x0, y0 = self.p0
            x1, y1 = self.p1
            if abs(x1 - x0) < 3 or abs(y1 - y0) < 3:
                self.done_rect = None
            else:
                self.done_rect = (
                    min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
                )
            self._draw()

    def run(self) -> Optional[Tuple[int, int, int, int]]:
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.title, self._on_mouse)
        self._draw()
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 32):  # Enter / Space
                if self.done_rect is None:
                    print("  请先拖拽画出 ROI 再确认")
                    continue
                cv2.destroyWindow(self.title)
                return self.done_rect
            if key in (ord("r"), ord("R")):
                self.done_rect = None
                self.p0 = self.p1 = None
                self._draw()
            if key in (ord("s"), ord("S")):
                cv2.destroyWindow(self.title)
                return None
            if key in (ord("q"), ord("Q"), 27):
                cv2.destroyWindow(self.title)
                raise KeyboardInterrupt


def load_distance_mm_map(scan_dir: str, positions_csv: str) -> Dict[str, float]:
    """basename -> distance_mm (not meters)."""
    pos_path = resolve_positions_path(scan_dir, positions_csv)
    # read raw mm from csv for JSON friendliness
    out: Dict[str, float] = {}
    with open(pos_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"empty positions: {pos_path}")
        fields = {c.lower().strip(): c for c in reader.fieldnames}
        img_col = None
        for k in ("image", "file", "filename", "name", "img"):
            if k in fields:
                img_col = fields[k]
                break
        dist_col = fields.get("distance_mm") or fields.get("distance") or fields.get("s_mm")
        if img_col is None or dist_col is None:
            # fallback meters map * 1000
            meters = load_rail_positions(pos_path, distance_unit="mm")
            return {k: v * 1000.0 for k, v in meters.items()}
        for row in reader:
            name = os.path.basename(str(row[img_col]).strip())
            out[name] = float(row[dist_col])
    return out


def list_scan_pngs(scan_dir: str) -> List[str]:
    names = sorted(
        f for f in os.listdir(scan_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    )
    return [os.path.join(scan_dir, n) for n in names]


def main():
    ap = argparse.ArgumentParser(description="关键帧画框 → ROI JSON")
    ap.add_argument("--config", default=None)
    ap.add_argument("--images", default=None, help="扫描图目录")
    ap.add_argument("--positions", default=None, help="positions.csv")
    ap.add_argument("--out", default=None, help="输出 JSON")
    ap.add_argument("--n", type=int, default=5, help="关键帧数量 3～5 推荐")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    scan_dir = args.images or resolve_path(cfg, cfg["paths"]["scan_images"])
    pos_name = args.positions or (cfg.get("rail") or {}).get("positions_file", "positions.csv")
    if args.positions and os.path.isfile(args.positions):
        # allow full path: use basename lookup via resolve
        pos_name = args.positions

    out_json = args.out
    if out_json is None:
        kf = (cfg.get("laser") or {}).get("keyframe_roi") or {}
        out_json = kf.get("path") or os.path.join(
            resolve_path(cfg, cfg["paths"]["output"]), "roi_keyframes.json"
        )
        if not os.path.isabs(out_json):
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            out_json = os.path.join(root, out_json)

    paths = list_scan_pngs(scan_dir)
    if not paths:
        raise SystemExit(f"扫描目录无图片: {scan_dir}")

    # positions path for distance map
    if os.path.isfile(str(pos_name)):
        dist_map = load_distance_mm_map(scan_dir, os.path.basename(pos_name))
        # if user passed full path, prefer reading that file
        dist_map = {}
        with open(pos_name, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fields = {c.lower().strip(): c for c in (reader.fieldnames or [])}
            img_col = fields.get("image") or fields.get("file") or fields.get("filename")
            dist_col = fields.get("distance_mm") or fields.get("distance")
            for row in reader:
                dist_map[os.path.basename(str(row[img_col]).strip())] = float(row[dist_col])
    else:
        dist_map = load_distance_mm_map(scan_dir, pos_name)

    n_keys = max(3, min(int(args.n), 5, len(paths)))
    indices = pick_keyframe_indices(len(paths), n_keys)
    print(f"扫描: {scan_dir}")
    print(f"关键帧数: {n_keys}  索引: {indices}")
    print(f"输出: {out_json}")

    keyframes = []
    try:
        for k, idx in enumerate(indices):
            path = paths[idx]
            name = os.path.basename(path)
            if name not in dist_map:
                print(f"警告: {name} 不在 positions.csv，跳过")
                continue
            img = imread_color(path)
            if img is None:
                print(f"读图失败: {path}")
                continue
            h, w = img.shape[:2]
            title = f"keyframe {k+1}/{len(indices)}  {name}  s={dist_map[name]:.1f}mm"
            print(f"\n[{k+1}/{len(indices)}] {name}  distance_mm={dist_map[name]}")
            drawer = RectDrawer(img, title)
            rect = drawer.run()
            if rect is None:
                print("  已跳过")
                continue
            x0, y0, x1, y1 = rect
            roi = {
                "x_min": x0 / w,
                "x_max": x1 / w,
                "y_min": y0 / h,
                "y_max": y1 / h,
            }
            print(
                f"  ROI norm: "
                f"x[{roi['x_min']:.3f},{roi['x_max']:.3f}] "
                f"y[{roi['y_min']:.3f},{roi['y_max']:.3f}]"
            )
            keyframes.append({
                "image": name,
                "distance_mm": float(dist_map[name]),
                "roi": roi,
            })
    except KeyboardInterrupt:
        print("\n用户中止，保存已确认的关键帧…")

    cv2.destroyAllWindows()
    if len(keyframes) < 1:
        raise SystemExit("没有确认任何关键帧，未写入 JSON")
    save_keyframe_roi_file(out_json, keyframes, normalized=True)
    print(f"\n已保存 {len(keyframes)} 个关键帧 -> {out_json}")


if __name__ == "__main__":
    main()
