"""生成 ChArUco 标定板图片用于打印（按 config.yaml 参数）。

打印后务必用卡尺复核方格实际边长，并回填 config.yaml。
建议贴到硬底板(亚克力/铝)上防止翘曲。

用法:
    python scripts/generate_board.py --out output/charuco_board.png --dpi 300
"""
import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.charuco import CharucoTarget  # noqa: E402
from src.config import CharucoConfig, load_config  # noqa: E402


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description="生成 ChArUco 标定板")
    ap.add_argument("--out", default="output/charuco_board.png")
    ap.add_argument("--dpi", type=int, default=300, help="打印分辨率")
    ap.add_argument("--margin", type=int, default=20, help="白边像素")
    args = ap.parse_args()

    cc = CharucoConfig.from_cfg(cfg)
    target = CharucoTarget(cc)

    # 物理尺寸(m) -> 像素： px = length(m) * (dpi / 0.0254)
    px_per_m = args.dpi / 0.0254
    w = int(round(cc.squares_x * cc.square_length * px_per_m))
    h = int(round(cc.squares_y * cc.square_length * px_per_m))

    board = target.board
    if hasattr(board, "generateImage"):
        img = board.generateImage((w, h), marginSize=args.margin)
    else:  # 旧 API
        img = board.draw((w, h), marginSize=args.margin)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cv2.imwrite(args.out, img)
    print(f"已生成标定板: {args.out}  ({w}x{h}px @ {args.dpi}dpi)")
    print(f"方格 {cc.squares_x}x{cc.squares_y}, 方格边长 {cc.square_length*1000:.1f}mm, "
          f"标记 {cc.marker_length*1000:.1f}mm, 字典 {cc.dictionary}")
    print("打印后请用卡尺复核实际边长并回填 config.yaml！")


if __name__ == "__main__":
    main()
