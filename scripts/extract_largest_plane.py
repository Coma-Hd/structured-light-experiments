"""从点云中提取最大的有效平面，并输出该平面的点云和 OBJ。

用于“一次只扫一个面”的流程：
    cloud_clean.ply -> largest_face.ply -> largest_face.obj

示例：
    python scripts/extract_largest_plane.py --in ceshi/output/cloud_clean.ply --out-ply ceshi/output/largest_face.ply --out-obj ceshi/output/largest_face.obj
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = normal / (np.linalg.norm(normal) + 1e-12)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(n, u)
    return u, v


def mesh_plane_points(o3d, points: np.ndarray, plane: np.ndarray, alpha: float):
    """把单个平面的点投影到平面后做 2D Delaunay 三角化。"""
    from scipy.spatial import Delaunay

    n = np.asarray(plane[:3], dtype=float)
    n /= np.linalg.norm(n) + 1e-12
    d = float(plane[3])

    signed = points @ n + d
    proj = points - np.outer(signed, n)

    u, v = plane_basis(n)
    uv = np.column_stack([proj @ u, proj @ v])
    tri = Delaunay(uv)
    faces = tri.simplices

    p3 = proj[faces]
    e0 = np.linalg.norm(p3[:, 0] - p3[:, 1], axis=1)
    e1 = np.linalg.norm(p3[:, 1] - p3[:, 2], axis=1)
    e2 = np.linalg.norm(p3[:, 2] - p3[:, 0], axis=1)
    keep = np.maximum.reduce([e0, e1, e2]) <= alpha
    faces = faces[keep]

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(proj)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()
    return mesh


def main() -> None:
    ap = argparse.ArgumentParser(description="提取点云中最大的有效平面")
    ap.add_argument("--in", dest="in_ply", required=True, help="输入点云 PLY，建议使用 cloud_clean.ply")
    ap.add_argument("--out-ply", default="output/largest_face.ply", help="最大平面点云输出")
    ap.add_argument("--out-obj", default="output/largest_face.obj", help="最大平面 OBJ 输出")
    ap.add_argument("--dist", type=float, default=0.004, help="RANSAC 平面内点距离阈值，单位 m")
    ap.add_argument("--min-points", type=int, default=300, help="最少内点数，低于则认为提取失败")
    ap.add_argument("--iters", type=int, default=2000, help="RANSAC 迭代次数")
    ap.add_argument("--cluster-eps", type=float, default=0.006, help="平面内点 DBSCAN 半径，单位 m")
    ap.add_argument("--cluster-min", type=int, default=30, help="DBSCAN 最小点数")
    ap.add_argument("--no-cluster", action="store_true", help="不做聚类，只保留全部平面内点")
    ap.add_argument("--alpha", type=float, default=0.012, help="OBJ 三角最大边长，单位 m")
    args = ap.parse_args()

    try:
        import open3d as o3d
    except Exception:
        print("错误: 需要安装 open3d -> pip install open3d", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(args.in_ply):
        print(f"错误: 找不到输入点云 {args.in_ply}", file=sys.stderr)
        sys.exit(1)

    pcd = o3d.io.read_point_cloud(args.in_ply)
    n0 = len(pcd.points)
    if n0 < args.min_points:
        print(f"错误: 点数太少 {n0} < {args.min_points}", file=sys.stderr)
        sys.exit(1)

    model, inliers = pcd.segment_plane(
        distance_threshold=args.dist,
        ransac_n=3,
        num_iterations=args.iters,
    )
    if len(inliers) < args.min_points:
        print(f"错误: 最大平面内点太少 {len(inliers)} < {args.min_points}", file=sys.stderr)
        sys.exit(1)

    plane_pcd = pcd.select_by_index(inliers)
    before_cluster = len(plane_pcd.points)

    if not args.no_cluster and before_cluster > args.cluster_min:
        labels = np.asarray(plane_pcd.cluster_dbscan(eps=args.cluster_eps, min_points=args.cluster_min))
        valid = labels[labels >= 0]
        if valid.size:
            ids, counts = np.unique(valid, return_counts=True)
            best_id = ids[np.argmax(counts)]
            keep = np.where(labels == best_id)[0]
            plane_pcd = plane_pcd.select_by_index(keep)

    n1 = len(plane_pcd.points)
    if n1 < args.min_points:
        print(f"错误: 聚类后最大面点数太少 {n1} < {args.min_points}", file=sys.stderr)
        sys.exit(1)

    ensure_parent(args.out_ply)
    o3d.io.write_point_cloud(args.out_ply, plane_pcd, write_ascii=True)

    pts = np.asarray(plane_pcd.points)
    mesh = mesh_plane_points(o3d, pts, np.asarray(model, dtype=float), args.alpha)
    ensure_parent(args.out_obj)
    o3d.io.write_triangle_mesh(args.out_obj, mesh)

    a, b, c, d = model
    print(f"输入点数: {n0}")
    print(f"最大平面: 内点 {before_cluster}, 聚类后 {n1}")
    print(f"平面方程: {a:.6f} x + {b:.6f} y + {c:.6f} z + {d:.6f} = 0")
    print(f"已保存点云: {args.out_ply}")
    print(f"已保存 OBJ: {args.out_obj} ({len(mesh.vertices)} 顶点, {len(mesh.triangles)} 面)")


if __name__ == "__main__":
    main()
