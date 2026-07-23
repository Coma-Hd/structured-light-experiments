"""Finish a single-view surface mesh from a cleaned point cloud.

BPA is faithful but noisy. Poisson is smooth but fills holes / adds fake walls.
This script:
  1) lightly smooths the point cloud
  2) builds BPA and/or Poisson meshes
  3) crops Poisson faces that drift away from the original points
  4) applies mild mesh smoothing

Usage:
    python scripts/finish_surface_mesh.py --in ceshi/surface/output/cloud_clean.ply --out-dir ceshi/surface/output
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def smooth_point_cloud(pcd, radius: float, iters: int):
    import open3d as o3d

    pts = np.asarray(pcd.points)
    if len(pts) < 20 or iters <= 0 or radius <= 0:
        return pcd

    tree = o3d.geometry.KDTreeFlann(pcd)
    cur = pts.copy()
    for _ in range(iters):
        nxt = cur.copy()
        for i, p in enumerate(cur):
            _, idx, _ = tree.search_radius_vector_3d(p, radius)
            if len(idx) >= 3:
                nxt[i] = cur[idx].mean(axis=0)
        cur = nxt
        # rebuild tree on updated points for next iteration
        tmp = o3d.geometry.PointCloud()
        tmp.points = o3d.utility.Vector3dVector(cur)
        tree = o3d.geometry.KDTreeFlann(tmp)

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(cur)
    if pcd.has_colors():
        out.colors = pcd.colors
    return out


def crop_mesh_to_cloud(mesh, pcd, max_dist: float):
    import open3d as o3d

    if max_dist <= 0 or len(mesh.vertices) == 0 or len(pcd.points) == 0:
        return mesh
    # Distance is defined on PointCloud, not TriangleMesh.
    query = o3d.geometry.PointCloud()
    query.points = mesh.vertices
    dists = np.asarray(query.compute_point_cloud_distance(pcd))
    keep_v = dists <= max_dist
    if not np.any(keep_v):
        return mesh
    mesh = mesh.select_by_index(np.where(keep_v)[0])
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    return mesh


def build_bpa(pcd, radius_scale: float):
    import open3d as o3d

    d = np.asarray(pcd.compute_nearest_neighbor_distance())
    avg = float(np.mean(d)) if len(d) else 0.002
    radii = [avg * r * radius_scale for r in (1.5, 2.5, 4.0, 6.0, 8.0)]
    return o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )


def build_poisson(pcd, depth: int):
    import open3d as o3d

    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )
    dens = np.asarray(dens)
    if dens.size:
        # drop the lowest-density fringe that Poisson tends to invent
        keep = dens > np.quantile(dens, 0.05)
        mesh = mesh.select_by_index(np.where(keep)[0])
        mesh.remove_unreferenced_vertices()
    return mesh


def finalize_mesh(mesh, smooth_iters: int):
    if len(mesh.triangles) == 0:
        return mesh
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    if smooth_iters > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iters)
    mesh.compute_vertex_normals()
    return mesh


def main() -> None:
    ap = argparse.ArgumentParser(description="Smooth single-view surface meshes")
    ap.add_argument("--in", dest="in_ply", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--voxel", type=float, default=0.001, help="voxel size in meters")
    ap.add_argument("--smooth-radius", type=float, default=0.004)
    ap.add_argument("--smooth-iters", type=int, default=3)
    ap.add_argument("--bpa-radius-scale", type=float, default=1.8)
    ap.add_argument("--poisson-depth", type=int, default=8)
    ap.add_argument("--poisson-crop", type=float, default=0.004,
                    help="keep Poisson faces within this distance of the cloud (m)")
    ap.add_argument("--mesh-smooth", type=int, default=10)
    ap.add_argument("--no-bpa", action="store_true")
    ap.add_argument("--no-poisson", action="store_true")
    args = ap.parse_args()

    try:
        import open3d as o3d
    except Exception:
        print("Need open3d: pip install open3d", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(args.in_ply):
        print(f"Missing input: {args.in_ply}", file=sys.stderr)
        sys.exit(1)

    ensure_dir(args.out_dir)
    pcd = o3d.io.read_point_cloud(args.in_ply)
    n0 = len(pcd.points)
    if n0 < 30:
        print(f"Too few points: {n0}", file=sys.stderr)
        sys.exit(1)

    if args.voxel > 0:
        pcd = pcd.voxel_down_sample(args.voxel)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = smooth_point_cloud(pcd, args.smooth_radius, args.smooth_iters)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=max(args.smooth_radius, 0.004), max_nn=40)
    )
    pcd.orient_normals_consistent_tangent_plane(30)

    smooth_ply = os.path.join(args.out_dir, "cloud_smooth.ply")
    o3d.io.write_point_cloud(smooth_ply, pcd, write_ascii=True)
    print(f"Smoothed cloud: {n0} -> {len(pcd.points)}  saved {smooth_ply}")

    if not args.no_bpa:
        mesh = finalize_mesh(build_bpa(pcd, args.bpa_radius_scale), args.mesh_smooth)
        out = os.path.join(args.out_dir, "surface_bpa_smooth.obj")
        o3d.io.write_triangle_mesh(out, mesh)
        print(f"BPA smooth: {len(mesh.vertices)} verts, {len(mesh.triangles)} faces -> {out}")

    if not args.no_poisson:
        mesh = build_poisson(pcd, args.poisson_depth)
        mesh = crop_mesh_to_cloud(mesh, pcd, args.poisson_crop)
        mesh = finalize_mesh(mesh, args.mesh_smooth)
        out = os.path.join(args.out_dir, "surface_poisson_cropped.obj")
        o3d.io.write_triangle_mesh(out, mesh)
        print(f"Poisson cropped: {len(mesh.vertices)} verts, {len(mesh.triangles)} faces -> {out}")


if __name__ == "__main__":
    main()
