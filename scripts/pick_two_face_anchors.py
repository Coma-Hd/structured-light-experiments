"""交互点选两面点云中的同名共享棱点，生成粗配准 anchors.json。

建议在两份点云中按相同顺序选择：
  1. 共享棱顶部
  2. 共享棱底部
  3. 可选：共享条带上的同一显著角点

Open3D 操作：
  Shift + 鼠标左键：选点
  Shift + 鼠标右键：撤销
  Q：完成当前点云
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.two_face_registration import load_cloud  # noqa: E402


def pick_points(pcd, title: str):
    import open3d as o3d

    print("")
    print(title)
    print("  Shift + 左键: 选择点")
    print("  Shift + 右键: 撤销")
    print("  Q: 完成当前点云")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=title, width=1100, height=800)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    indices = list(vis.get_picked_points())
    pts = np.asarray(pcd.points)
    return indices, pts[indices] if indices else np.empty((0, 3))


def main() -> None:
    ap = argparse.ArgumentParser(description="点选两份相邻点云的共享棱对应点")
    ap.add_argument("--target", required=True, help="基准点云 cloud_clean.ply")
    ap.add_argument("--source", required=True, help="待对齐点云 cloud_clean.ply")
    ap.add_argument("--out", required=True, help="输出 anchors.json")
    ap.add_argument("--target-name", default="face1", help="基准面的显示名称")
    ap.add_argument("--source-name", default="face2", help="待对齐面的显示名称")
    args = ap.parse_args()

    target = load_cloud(args.target)
    source = load_cloud(args.source)

    target_idx, target_pts = pick_points(
        target,
        f"{args.target_name.upper()} target: "
        "依次点共享棱顶部、底部（然后 Q）",
    )
    if len(target_pts) < 1:
        raise SystemExit(f"{args.target_name} 未选择任何点")

    source_idx, source_pts = pick_points(
        source,
        f"{args.source_name.upper()} source: "
        "按完全相同顺序点同名点（然后 Q）",
    )
    if len(source_pts) != len(target_pts):
        raise SystemExit(
            "对应点数量不一致: "
            f"{args.target_name}={len(target_pts)}, "
            f"{args.source_name}={len(source_pts)}"
        )

    data = {
        "description": "Corresponding shared-edge anchors; same order in both arrays.",
        "target_name": args.target_name,
        "source_name": args.source_name,
        "target_indices": [int(x) for x in target_idx],
        "source_indices": [int(x) for x in source_idx],
        "target_points": target_pts.astype(float).tolist(),
        "source_points": source_pts.astype(float).tolist(),
    }
    if args.target_name == "face1" and args.source_name == "face2":
        data.update(
            {
                "target_face1_indices": data["target_indices"],
                "source_face2_indices": data["source_indices"],
                "target_face1_points": data["target_points"],
                "source_face2_points": data["source_points"],
            }
        )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"已保存 {len(target_pts)} 对锚点: {args.out}")


if __name__ == "__main__":
    main()
