"""点云体检工具：看点云到底长什么样，定位问题在哪一段。

用法:
    python scripts/inspect_cloud.py --in ceshi/output/cloud.ply
    python scripts/inspect_cloud.py --in ceshi/output/cloud.ply --out ceshi/output/inspect

输出:
    - 终端打印: 点数、xyz 范围、各轴分位数、z 直方图
    - 三张投影图 PNG: XY(俯视) / XZ(侧视) / YZ(正视)，直接看形状
      (若未装 matplotlib 则只打印文字统计)
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reconstruct import read_ply_xyz  # noqa: E402


def load_points(path: str) -> np.ndarray:
    """Read ASCII or binary PLY. Open3D cloud_clean.ply is usually binary."""
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points, dtype=np.float64)
        if pts.size > 0:
            return pts.reshape(-1, 3)
    except Exception:
        pass
    return read_ply_xyz(path)


def text_hist(vals: np.ndarray, bins: int = 30, width: int = 50) -> None:
    lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo:
        print(f"    (所有值相同 = {lo:.4f})")
        return
    counts, edges = np.histogram(vals, bins=bins)
    cmax = counts.max()
    for i in range(bins):
        bar = "#" * int(round(counts[i] / cmax * width)) if cmax else ""
        print(f"    [{edges[i]*1000:8.2f},{edges[i+1]*1000:8.2f}] mm | {counts[i]:7d} {bar}")


def main():
    ap = argparse.ArgumentParser(description="点云体检")
    ap.add_argument("--in", dest="in_ply", required=True, help="输入点云 ply")
    ap.add_argument("--out", dest="out_prefix", default=None,
                    help="投影图输出前缀 (默认: 输入同目录/inspect)")
    args = ap.parse_args()

    if not os.path.isfile(args.in_ply):
        print(f"错误: 找不到 {args.in_ply}")
        sys.exit(1)

    pts = load_points(args.in_ply)
    n = pts.shape[0]
    print(f"\n点云: {args.in_ply}")
    print(f"点数: {n}")
    if n == 0:
        sys.exit(0)

    names = ["X", "Y", "Z"]
    print("\n各轴范围与分位数 (单位 mm):")
    for i, nm in enumerate(names):
        v = pts[:, i] * 1000.0
        pcts = np.percentile(v, [1, 25, 50, 75, 99])
        print(f"  {nm}: min={v.min():9.2f}  max={v.max():9.2f}  "
              f"span={v.max()-v.min():8.2f} | "
              f"p1={pcts[0]:.1f} p25={pcts[1]:.1f} p50={pcts[2]:.1f} "
              f"p75={pcts[3]:.1f} p99={pcts[4]:.1f}")

    print("\nZ 方向直方图 (板面应在 Z≈0 处形成一个高峰；工件是另一群):")
    text_hist(pts[:, 2])

    # 板面点占比估计: |z| < 2mm 认为是板面
    near_board = np.abs(pts[:, 2]) < 0.002
    print(f"\n|Z|<2mm 的点(疑似板面): {int(near_board.sum())} "
          f"({near_board.mean()*100:.1f}%)")

    out_prefix = args.out_prefix or os.path.join(
        os.path.dirname(os.path.abspath(args.in_ply)), "inspect")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("\n(未安装 matplotlib，跳过投影图。pip install matplotlib 可开启)")
        return

    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)
    combos = [(0, 1, "XY_top"), (0, 2, "XZ_side"), (1, 2, "YZ_front")]
    for a, b, tag in combos:
        plt.figure(figsize=(8, 8))
        plt.scatter(pts[:, a] * 1000, pts[:, b] * 1000, s=0.5,
                    c=pts[:, 2] * 1000, cmap="jet")
        plt.xlabel(f"{names[a]} (mm)")
        plt.ylabel(f"{names[b]} (mm)")
        plt.title(f"{tag}  ({n} 点)")
        plt.axis("equal")
        plt.colorbar(label="Z (mm)")
        path = f"{out_prefix}_{tag}.png"
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"已保存投影图: {path}")


if __name__ == "__main__":
    main()
