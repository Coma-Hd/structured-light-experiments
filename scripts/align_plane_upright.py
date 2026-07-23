#!/usr/bin/env python3
"""Rigid-only reorient of a near-planar patch so the face lies in XY (Z = normal).

No scaling: only rotation + translation. In-plane yaw is chosen to minimize the
axis-aligned rectangle (so a 100x100 square does not show a ~125 diagonal AABB).

Usage:
    python scripts/align_plane_upright.py --in ceshi/rail/output/largest_face.obj
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / (n + 1e-12)


def rigid_upright(points: np.ndarray) -> tuple[np.ndarray, dict]:
    """Map plane -> XY by rigid transform; align edges via min-area yaw."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    c = pts.mean(axis=0)
    centered = pts - c
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    n = vh[2].copy()
    if n[2] > 0:
        n = -n
    e_z = _unit(n)
    # Temporary in-plane basis from PCA
    e_x0 = vh[0].copy()
    e_x0 = _unit(e_x0 - e_z * float(e_x0 @ e_z))
    if np.linalg.norm(e_x0) < 1e-9:
        e_x0 = _unit(vh[1] - e_z * float(vh[1] @ e_z))
    e_y0 = _unit(np.cross(e_z, e_x0))
    uv = np.column_stack([centered @ e_x0, centered @ e_y0])

    # Search yaw that minimizes AABB area in the plane (rigid, no scale).
    best_area = None
    best_ang = 0.0
    best_wh = None
    for ang in np.linspace(0.0, np.pi / 2, 361):
        ca, sa = np.cos(ang), np.sin(ang)
        p = np.column_stack([ca * uv[:, 0] - sa * uv[:, 1], sa * uv[:, 0] + ca * uv[:, 1]])
        wh = p.max(axis=0) - p.min(axis=0)
        area = float(wh[0] * wh[1])
        if best_area is None or area < best_area:
            best_area = area
            best_ang = float(ang)
            best_wh = wh.copy()

    ca, sa = np.cos(best_ang), np.sin(best_ang)
    e_x = _unit(ca * e_x0 + sa * e_y0)
    e_y = _unit(np.cross(e_z, e_x))
    R = np.stack([e_x, e_y, e_z], axis=0)  # p_view = (p - c) @ R.T
    aligned = centered @ R.T

    # Center XY for viewing; keep mean Z = 0 (on the plane)
    aligned[:, 0] -= aligned[:, 0].mean()
    aligned[:, 1] -= aligned[:, 1].mean()
    aligned[:, 2] -= aligned[:, 2].mean()

    info = {
        "obb_mm": (best_wh * 1000.0).tolist(),
        "yaw_deg": float(np.degrees(best_ang)),
        "thickness_std_mm": float(np.std(aligned[:, 2]) * 1000.0),
    }
    return aligned, info


def load_points(path: str) -> tuple[str, np.ndarray, object | None]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".obj":
        verts = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.split()
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        return "obj", np.asarray(verts, dtype=np.float64), None
    if ext == ".ply":
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points)
        if len(pts) == 0:
            mesh = o3d.io.read_triangle_mesh(path)
            pts = np.asarray(mesh.vertices)
            return "mesh", pts, mesh
        return "pcd", pts, pcd
    raise ValueError("only .obj / .ply supported")


def write_obj_like(src: str, dst: str, new_verts: np.ndarray) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
    vi = 0
    with open(src, "r", encoding="utf-8", errors="ignore") as fin, open(
        dst, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if line.startswith("v "):
                x, y, z = new_verts[vi]
                vi += 1
                fout.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
            elif line.startswith("vn "):
                continue
            else:
                fout.write(line)


def main() -> int:
    ap = argparse.ArgumentParser(description="Rigid upright align (no scale)")
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", default=None)
    args = ap.parse_args()
    src = args.src
    dst = args.dst or f"{os.path.splitext(src)[0]}_upright{os.path.splitext(src)[1]}"

    kind, pts, obj = load_points(src)
    if len(pts) < 10:
        print(f"too few points: {len(pts)}", file=sys.stderr)
        return 1

    # Sanity: rigid transform must preserve pairwise scale (check bbox diagonal).
    d0 = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    aligned, info = rigid_upright(pts)
    d1 = float(np.linalg.norm(aligned.max(0) - aligned.min(0)))
    # Diagonals of AABB can change under rotation; use mean pairwise on sample.
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pts), size=min(500, len(pts)), replace=False)
    a = pts[idx]
    b = aligned[idx]
    # Compare distances from centroid
    ra = np.linalg.norm(a - a.mean(0), axis=1)
    rb = np.linalg.norm(b - b.mean(0), axis=1)
    scale_ratio = float(np.median(rb / np.maximum(ra, 1e-12)))

    print(f"points: {len(aligned)}")
    print(
        f"face OBB (rigid, mm): {info['obb_mm'][0]:.1f} x {info['obb_mm'][1]:.1f}  "
        f"yaw={info['yaw_deg']:.1f} deg  thickness_std={info['thickness_std_mm']:.2f} mm"
    )
    print(f"scale check (median radius ratio, expect ~1): {scale_ratio:.6f}")
    if abs(scale_ratio - 1.0) > 1e-3:
        print("WARNING: unexpected scale change", file=sys.stderr)

    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
    if kind == "obj":
        write_obj_like(src, dst, aligned)
    else:
        import open3d as o3d

        if kind == "mesh" and obj is not None and len(obj.triangles) > 0:
            obj.vertices = o3d.utility.Vector3dVector(aligned)
            obj.compute_vertex_normals()
            o3d.io.write_triangle_mesh(dst, obj)
        else:
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(aligned))
            o3d.io.write_point_cloud(dst, pcd, write_ascii=True)
    print(f"saved: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
