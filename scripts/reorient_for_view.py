"""把板坐标系点云/OBJ 转成更适合 3D 软件查看的坐标系。

本项目重建输出是 ChArUco 板坐标系:
    X: 标定板横向
    Y: 标定板纵向
    Z: 垂直标定板的深度方向

当标定板竖直放置时，普通 3D 查看器通常希望 Z 是竖直向上。
默认转换:
    X_view = X_board
    Y_view = Z_board
    Z_view = -Y_board

用法:
    python scripts/reorient_for_view.py --in ceshi/output/cloud_clean.ply --out ceshi/output/cloud_clean_view.ply
    python scripts/reorient_for_view.py --in ceshi/output/mesh_planes.obj --out ceshi/output/mesh_planes_view.obj
"""
import argparse
import os
import sys

import numpy as np


def transform_xyz(x: float, y: float, z: float):
    return x, z, -y


def convert_obj(src: str, dst: str):
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with open(src, "r", encoding="utf-8", errors="ignore") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            if line.startswith("v "):
                parts = line.strip().split()
                x, y, z = map(float, parts[1:4])
                xo, yo, zo = transform_xyz(x, y, z)
                rest = " " + " ".join(parts[4:]) if len(parts) > 4 else ""
                fout.write(f"v {xo:.9g} {yo:.9g} {zo:.9g}{rest}\n")
            elif line.startswith("vn "):
                parts = line.strip().split()
                x, y, z = map(float, parts[1:4])
                xo, yo, zo = transform_xyz(x, y, z)
                fout.write(f"vn {xo:.9g} {yo:.9g} {zo:.9g}\n")
            else:
                fout.write(line)


def convert_ascii_ply(src: str, dst: str):
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with open(src, "r", encoding="ascii", errors="ignore") as fin, open(dst, "w", encoding="ascii") as fout:
        header = []
        vertex_count = None
        for line in fin:
            header.append(line)
            parts = line.strip().split()
            if len(parts) == 3 and parts[0] == "element" and parts[1] == "vertex":
                vertex_count = int(parts[2])
            if line.strip() == "end_header":
                break

        if vertex_count is None:
            raise ValueError("PLY header missing vertex count")

        fout.writelines(header)
        for i in range(vertex_count):
            line = fin.readline()
            parts = line.strip().split()
            if len(parts) < 3:
                fout.write(line)
                continue
            x, y, z = map(float, parts[:3])
            xo, yo, zo = transform_xyz(x, y, z)
            rest = " " + " ".join(parts[3:]) if len(parts) > 3 else ""
            fout.write(f"{xo:.9g} {yo:.9g} {zo:.9g}{rest}\n")
        for line in fin:
            fout.write(line)


def convert_ply(src: str, dst: str):
    """Convert ASCII or binary PLY point clouds.

    Open3D writes `cloud_clean.ply` as binary by default, so prefer Open3D here.
    Fall back to the simple ASCII parser for raw project PLY files.
    """
    try:
        import open3d as o3d
    except Exception:
        convert_ascii_ply(src, dst)
        return

    pcd = o3d.io.read_point_cloud(src)
    if len(pcd.points) == 0:
        raise RuntimeError(f"empty or unreadable point cloud: {src}")

    pts = np.asarray(pcd.points)
    pcd.points = o3d.utility.Vector3dVector(
        np.column_stack([pts[:, 0], pts[:, 2], -pts[:, 1]])
    )
    if pcd.has_normals():
        n = np.asarray(pcd.normals)
        pcd.normals = o3d.utility.Vector3dVector(
            np.column_stack([n[:, 0], n[:, 2], -n[:, 1]])
        )

    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    o3d.io.write_point_cloud(dst, pcd, write_ascii=True)


def main():
    ap = argparse.ArgumentParser(description="Reorient board coordinates for 3D viewers")
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    args = ap.parse_args()

    ext = os.path.splitext(args.src)[1].lower()
    if ext == ".obj":
        convert_obj(args.src, args.dst)
    elif ext == ".ply":
        convert_ply(args.src, args.dst)
    else:
        print("Only .ply and .obj are supported", file=sys.stderr)
        sys.exit(1)
    print(f"saved: {args.dst}")


if __name__ == "__main__":
    main()
