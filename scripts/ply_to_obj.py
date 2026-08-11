"""PLY 点云 -> OBJ 网格。

用法:
    # 默认：读 output/cloud.ply，按 config.yaml 的 roi 裁剪后建网格，存 output/mesh.obj
    python scripts/ply_to_obj.py

    # 指定输入/输出
    python scripts/ply_to_obj.py --in output/cloud_clean.ply --out output/mesh.obj

    # 换重建算法 (bpa 适合表面, poisson 适合封闭实体)
    python scripts/ply_to_obj.py --method poisson

    # 线激光轮廓间隙：MLS 补点 + Poisson，并裁掉四周虚平面
    python scripts/ply_to_obj.py --in output/cloud_clean.ply --out output/mesh_mls_poisson.obj \\
        --method poisson --mls --no-roi --poisson-crop 0.004

    # 对已有 OBJ 补洞: max-edges 80 只补小洞; 0=大洞也补
    python scripts/ply_to_obj.py --fill-mesh output/mesh_mls_bpa.obj \\
        --out output/mesh_mls_bpa_filled.obj --fill-holes-max-edges 0

    # 不做 ROI 裁剪 (直接对原始点云建网格)
    python scripts/ply_to_obj.py --no-roi
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, resolve_path  # noqa: E402
from src.postprocess import roi_mask  # noqa: E402


def estimate_orient_normals(o3d, pcd, radius=0.01, max_nn=30, orient_k=20):
    """估法向并做切平面一致定向。"""
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    pcd.orient_normals_consistent_tangent_plane(orient_k)
    return pcd


def _boundary_loops(triangles):
    """从三角面提取边界闭环(边只被 1 个三角使用)。"""
    edge_count = defaultdict(int)
    for a, b, c in triangles:
        for e in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            edge_count[tuple(sorted(e))] += 1

    adj = defaultdict(list)
    for (a, b), cnt in edge_count.items():
        if cnt == 1:
            adj[a].append(b)
            adj[b].append(a)

    visited = set()
    loops = []
    for start in list(adj.keys()):
        for nb0 in adj[start]:
            e0 = tuple(sorted((start, nb0)))
            if e0 in visited:
                continue
            loop = [start]
            prev, cur = start, nb0
            visited.add(e0)
            ok = False
            for _ in range(len(adj) + 5):
                if cur == start:
                    ok = True
                    break
                loop.append(cur)
                nbs = adj.get(cur, [])
                nxt = None
                for cand in nbs:
                    if cand == prev:
                        continue
                    ek = tuple(sorted((cur, cand)))
                    if ek not in visited:
                        nxt = cand
                        visited.add(ek)
                        break
                if nxt is None:
                    break
                prev, cur = cur, nxt
            if ok and len(loop) >= 3:
                loops.append(loop)
    return loops


def _project_basis(points):
    """Newell 法估计环法向, 返回原点与 2D 基。"""
    n = np.zeros(3, dtype=float)
    for i in range(len(points)):
        p = points[i]
        q = points[(i + 1) % len(points)]
        n[0] += (p[1] - q[1]) * (p[2] + q[2])
        n[1] += (p[2] - q[2]) * (p[0] + q[0])
        n[2] += (p[0] - q[0]) * (p[1] + q[1])
    ln = float(np.linalg.norm(n))
    if ln < 1e-12:
        n = np.array([0.0, 0.0, 1.0])
    else:
        n /= ln
    origin = points.mean(axis=0)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref)
    u /= (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u)
    return origin, u, v, n


def _ear_clip_triangulate(loop_ids, verts):
    """对近似平面孔洞做耳切三角剖分, 返回三角顶点索引列表。"""
    loop_ids = list(loop_ids)
    pts = verts[np.asarray(loop_ids, dtype=int)]
    origin, u, v, _ = _project_basis(pts)
    uv = np.column_stack([(pts - origin) @ u, (pts - origin) @ v])

    idx = list(range(len(loop_ids)))

    def area2(i0, i1, i2):
        a, b, c = uv[i0], uv[i1], uv[i2]
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    # 统一为逆时针后重排, 保证 idx 与 loop_ids/uv 一致
    total = 0.0
    for i in range(len(idx)):
        total += area2(idx[i], idx[(i + 1) % len(idx)], idx[(i + 2) % len(idx)])
    if total < 0:
        loop_ids = list(reversed(loop_ids))
        pts = verts[np.asarray(loop_ids, dtype=int)]
        origin, u, v, _ = _project_basis(pts)
        uv = np.column_stack([(pts - origin) @ u, (pts - origin) @ v])
        idx = list(range(len(loop_ids)))

    def point_in_tri(p, a, b, c):
        c1 = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        c2 = (c[0] - b[0]) * (p[1] - b[1]) - (c[1] - b[1]) * (p[0] - b[0])
        c3 = (a[0] - c[0]) * (p[1] - c[1]) - (a[1] - c[1]) * (p[0] - c[0])
        return (c1 >= -1e-12 and c2 >= -1e-12 and c3 >= -1e-12) or (
            c1 <= 1e-12 and c2 <= 1e-12 and c3 <= 1e-12)

    faces = []
    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        n = len(idx)
        clipped = False
        for i in range(n):
            i0, i1, i2 = idx[(i - 1) % n], idx[i], idx[(i + 1) % n]
            if area2(i0, i1, i2) <= 1e-12:
                continue
            ear = True
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                if point_in_tri(uv[j], uv[i0], uv[i1], uv[i2]):
                    ear = False
                    break
            if not ear:
                continue
            faces.append((loop_ids[i0], loop_ids[i1], loop_ids[i2]))
            del idx[i]
            clipped = True
            break
        if not clipped:
            break
    if len(idx) == 3:
        faces.append((loop_ids[idx[0]], loop_ids[idx[1]], loop_ids[idx[2]]))
    return faces


def _fan_triangulate(loop_ids, verts):
    """大洞回退: 在孔洞局部平面中心加点, 做扇形三角剖分。"""
    loop_ids = list(loop_ids)
    pts = verts[np.asarray(loop_ids, dtype=int)]
    origin, _, _, nrm = _project_basis(pts)
    c = pts.mean(axis=0)
    c = c - np.dot(c - origin, nrm) * nrm
    faces = []
    for i in range(len(loop_ids)):
        a = int(loop_ids[i])
        b = int(loop_ids[(i + 1) % len(loop_ids)])
        faces.append((a, b, -1))  # -1 = 中心顶点占位
    return c, faces


def fill_small_holes(o3d, mesh, max_edges=80, max_perimeter=None, ear_max_edges=200):
    """填充边界洞。

    max_edges<=0 表示不限制边数(大洞也填); >0 时更大环跳过。
    小洞优先耳切; 大洞或耳切失败时用平面中心扇形剖分。
    """
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles, dtype=int)
    if len(tris) == 0:
        return mesh

    loops = _boundary_loops(tris)
    if not loops:
        print("补洞: 未发现边界环")
        return mesh

    new_verts = []
    new_faces = []
    filled_ear = 0
    filled_fan = 0
    skipped = 0
    for loop in loops:
        n_e = len(loop)
        if n_e < 3:
            skipped += 1
            continue
        perim = 0.0
        for i in range(n_e):
            a = verts[loop[i]]
            b = verts[loop[(i + 1) % n_e]]
            perim += float(np.linalg.norm(a - b))
        too_many = max_edges > 0 and n_e > max_edges
        too_long = max_perimeter is not None and max_perimeter > 0 and perim > max_perimeter
        if too_many or too_long:
            skipped += 1
            continue

        faces = None
        if n_e <= ear_max_edges:
            faces = _ear_clip_triangulate(loop, verts)
            # 耳切未完全收完则视为失败
            if faces and len(faces) < n_e - 2:
                faces = None
            if faces:
                new_faces.extend(faces)
                filled_ear += 1
                continue

        center, fan_faces = _fan_triangulate(loop, verts)
        cid = len(verts) + len(new_verts)
        new_verts.append(center)
        for a, b, _ in fan_faces:
            new_faces.append((a, b, cid))
        filled_fan += 1

    if not new_faces:
        print(f"补洞: 无可填洞 (边界环 {len(loops)} 个, 跳过 {skipped})")
        return mesh

    all_verts = verts
    if new_verts:
        all_verts = np.vstack([verts, np.asarray(new_verts, dtype=float)])
    all_tris = np.vstack([tris, np.asarray(new_faces, dtype=int)])
    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(all_verts)
    out.triangles = o3d.utility.Vector3iVector(all_tris)
    out.remove_duplicated_triangles()
    out.remove_degenerate_triangles()
    out.compute_vertex_normals()
    print(
        f"补洞: 耳切 {filled_ear} + 扇形 {filled_fan} "
        f"(+{len(new_faces)} 面, +{len(new_verts)} 顶点), 跳过 {skipped}"
    )
    return out


def crop_mesh_to_cloud(o3d, mesh, pcd, max_dist):
    """去掉远离参考点云的网格顶点(抑制 Poisson 四周补墙/封底)。"""
    if max_dist <= 0 or len(mesh.vertices) == 0 or len(pcd.points) == 0:
        return mesh
    query = o3d.geometry.PointCloud()
    query.points = mesh.vertices
    dists = np.asarray(query.compute_point_cloud_distance(pcd))
    keep = dists <= max_dist
    n_keep = int(np.count_nonzero(keep))
    if n_keep == 0:
        print(f"Poisson 裁剪: 阈值 {max_dist * 1000:.2f}mm 过严，未保留顶点，跳过")
        return mesh
    n0 = len(mesh.vertices)
    mesh = mesh.select_by_index(np.where(keep)[0])
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    print(f"Poisson 点云裁剪: 顶点 {n0} -> {len(mesh.vertices)} "
          f"(阈值 {max_dist * 1000:.2f}mm)")
    return mesh


def trim_poisson_low_density(mesh, dens, quantile=0.05):
    """去掉 Poisson 低密度边缘顶点(四周虚构面常见于低密度区)。"""
    dens = np.asarray(dens)
    if dens.size == 0 or quantile <= 0 or dens.size != len(mesh.vertices):
        return mesh
    thr = float(np.quantile(dens, quantile))
    keep = dens > thr
    n0 = len(mesh.vertices)
    mesh = mesh.select_by_index(np.where(keep)[0])
    mesh.remove_unreferenced_vertices()
    print(f"Poisson 密度裁剪: 顶点 {n0} -> {len(mesh.vertices)} "
          f"(去掉最低 {quantile * 100:.1f}%)")
    return mesh


def _finalize_vertex_subset(mesh, keep, label):
    keep = np.asarray(keep, dtype=bool)
    n_keep = int(np.count_nonzero(keep))
    if n_keep == 0:
        print(f"{label}: 阈值过严，未保留顶点，跳过")
        return mesh
    n0 = len(mesh.vertices)
    if n_keep >= n0:
        print(f"{label}: 顶点 {n0} (无变化)")
        return mesh
    mesh = mesh.select_by_index(np.where(keep)[0])
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    print(f"{label}: 顶点 {n0} -> {len(mesh.vertices)}")
    return mesh


def crop_mesh_by_support(mesh, pcd_ref, radius, min_neighbors):
    """只保留附近有足够原始扫描点支撑的网格顶点。"""
    if radius <= 0 or min_neighbors <= 0 or len(mesh.vertices) == 0:
        return mesh
    from scipy.spatial import cKDTree

    ref = np.asarray(pcd_ref.points)
    verts = np.asarray(mesh.vertices)
    counts = cKDTree(ref).query_ball_point(verts, r=radius, return_length=True)
    keep = np.asarray(counts) >= int(min_neighbors)
    return _finalize_vertex_subset(
        mesh, keep,
        f"Poisson 支撑裁剪(r={radius * 1000:.2f}mm, min={min_neighbors})")


def crop_mesh_by_normal_agree(o3d, mesh, pcd_ref, max_dist, min_dot):
    """去掉与原始点云法向不同向的顶点。

    Poisson 常把开口曲面收成薄壳: 背面/四周立面距点云很近, 但法向相反或侧向。
    这里用有符号点积(不用绝对值)去掉背面。
    """
    if min_dot <= 0 or max_dist <= 0 or len(mesh.vertices) == 0:
        return mesh
    from scipy.spatial import cKDTree

    if not pcd_ref.has_normals():
        estimate_orient_normals(o3d, pcd_ref)
    mesh.compute_vertex_normals()

    ref = np.asarray(pcd_ref.points)
    ref_n = np.asarray(pcd_ref.normals)
    verts = np.asarray(mesh.vertices)
    mesh_n = np.asarray(mesh.vertex_normals)
    # 网格法向可能整体翻转; 先按与点云的平均点积决定是否统一翻转
    dists, idx = cKDTree(ref).query(verts, k=1)
    dots = np.sum(mesh_n * ref_n[idx], axis=1)
    if float(np.mean(dots)) < 0:
        mesh_n = -mesh_n
        dots = -dots
        mesh.vertex_normals = o3d.utility.Vector3dVector(mesh_n)
    keep = (dists <= max_dist) & (dots >= float(min_dot))
    return _finalize_vertex_subset(
        mesh, keep,
        f"Poisson 法向裁剪(dot>={min_dot}, dist<={max_dist * 1000:.2f}mm)")


def mls_upsample(o3d, pcd, search_radius=None, gap=None, gap_factor=5.0,
                 samples=1, voxel=None, max_nn=40):
    """局部平面 MLS 上采样：在稀疏邻边中点处按切平面投影插入新点，填扫描轮廓间隙。

    Open3D 0.19 无内置 MLS，这里用邻域 PCA 平面近似。单位与点云一致（米）。
    """
    pts = np.asarray(pcd.points)
    n = len(pts)
    if n < 10:
        return pcd

    nn = np.asarray(pcd.compute_nearest_neighbor_distance())
    positive = nn[nn > 0]
    avg = float(np.median(positive)) if positive.size else 0.001

    if search_radius is None or search_radius <= 0:
        search_radius = max(avg * 8.0, 1e-6)
    if gap is None or gap <= 0:
        gap = max(avg * float(gap_factor), avg * 2.0)
    min_edge = avg * 1.6
    if gap <= min_edge:
        print(f"MLS: gap({gap:.6f}) <= min_edge({min_edge:.6f})，跳过上采样")
        return pcd

    tree = o3d.geometry.KDTreeFlann(pcd)
    normals = np.zeros((n, 3), dtype=float)
    centroids = np.zeros((n, 3), dtype=float)

    for i in range(n):
        _, idx, _ = tree.search_hybrid_vector_3d(pts[i], search_radius, max_nn)
        if len(idx) < 4:
            _, idx, _ = tree.search_knn_vector_3d(pts[i], min(max_nn, n))
        nb = pts[np.asarray(idx, dtype=int)]
        c = nb.mean(axis=0)
        centered = nb - c
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            nrm = vh[-1]
        except np.linalg.LinAlgError:
            nrm = np.array([0.0, 0.0, 1.0])
        normals[i] = nrm
        centroids[i] = c

    new_pts = []
    samples = max(int(samples), 1)
    for i in range(n):
        _, idx, dist2 = tree.search_radius_vector_3d(pts[i], gap)
        for j, d2 in zip(idx, dist2):
            if j <= i:
                continue
            d = float(d2) ** 0.5
            if d < min_edge or d > gap:
                continue
            nrm = normals[i] + normals[j]
            ln = float(np.linalg.norm(nrm))
            if ln < 1e-12:
                continue
            nrm /= ln
            c = 0.5 * (centroids[i] + centroids[j])
            for s in range(1, samples + 1):
                t = s / (samples + 1)
                m = (1.0 - t) * pts[i] + t * pts[j]
                m = m - np.dot(m - c, nrm) * nrm
                new_pts.append(m)

    if not new_pts:
        print("MLS: 未插入新点 (可增大 --mls-gap / --mls-gap-factor)")
        return pcd

    new_pts = np.asarray(new_pts, dtype=float)
    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(np.vstack([pts, new_pts]))
    if pcd.has_colors():
        # 新点无颜色时只保留几何，避免长度不匹配
        pass

    n_before_voxel = len(merged.points)
    if voxel is not None and voxel > 0:
        merged = merged.voxel_down_sample(voxel)
        print(f"MLS 体素合并: {n_before_voxel} -> {len(merged.points)} 点")

    print(
        f"MLS 上采样: {n} + {len(new_pts)} -> {len(merged.points)} 点 "
        f"(search={search_radius * 1000:.2f}mm, gap={gap * 1000:.2f}mm, "
        f"samples={samples})"
    )
    return merged


def mesh_by_planes(o3d, points, dist=0.003, min_inliers=300, max_planes=8,
                   alpha=0.008):
    """RANSAC 迭代分平面, 每片平面: 内点投影到平面(去噪变平) -> 2D Delaunay
    (按最大边长 alpha 剔除跨洞三角) -> 合并成一个网格。"""
    from scipy.spatial import Delaunay

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    rest = pcd

    all_v = []
    all_f = []
    voff = 0
    for i in range(max_planes):
        if len(rest.points) < min_inliers:
            break
        model, inl = rest.segment_plane(dist, 3, 1000)
        if len(inl) < min_inliers:
            break
        a, b, c, d = model
        n = np.array([a, b, c], dtype=float)
        n /= (np.linalg.norm(n) + 1e-12)
        pts = np.asarray(rest.select_by_index(inl).points)

        # 投影到平面: p' = p - (n·p + d) n
        signed = pts @ n + d
        proj = pts - np.outer(signed, n)

        # 平面内 2D 基
        ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(n, ref); u /= (np.linalg.norm(u) + 1e-12)
        v = np.cross(n, u)
        uv = np.column_stack([proj @ u, proj @ v])

        try:
            tri = Delaunay(uv)
            faces = tri.simplices
        except Exception:
            rest = rest.select_by_index(inl, invert=True)
            continue

        # 按最大边长过滤跨洞/跨缝三角
        p3 = proj[faces]
        e0 = np.linalg.norm(p3[:, 0] - p3[:, 1], axis=1)
        e1 = np.linalg.norm(p3[:, 1] - p3[:, 2], axis=1)
        e2 = np.linalg.norm(p3[:, 2] - p3[:, 0], axis=1)
        keep = (np.maximum.reduce([e0, e1, e2]) <= alpha)
        faces = faces[keep]
        if len(faces) == 0:
            rest = rest.select_by_index(inl, invert=True)
            continue

        all_v.append(proj)
        all_f.append(faces + voff)
        voff += len(proj)
        print(f"  平面{i}: n=[{n[0]:.2f},{n[1]:.2f},{n[2]:.2f}] "
              f"内点{len(inl)} -> {len(faces)} 面")
        rest = rest.select_by_index(inl, invert=True)

    if not all_v:
        raise RuntimeError("未提取到平面, 试着调大 --plane-dist 或调小 --plane-min")

    V = np.vstack(all_v)
    F = np.vstack(all_f)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(V)
    mesh.triangles = o3d.utility.Vector3iVector(F)
    return mesh


def main():
    cfg = load_config()
    out_dir = resolve_path(cfg, cfg["paths"]["output"])

    ap = argparse.ArgumentParser(description="PLY 转 OBJ 网格")
    ap.add_argument("--in", dest="in_ply", default=os.path.join(out_dir, "cloud_clean.ply"),
                    help="输入 ply (默认 output/cloud.ply)")
    ap.add_argument("--out", dest="out_obj", default=os.path.join(out_dir, "mesh.obj"),
                    help="输出 obj (默认 output/mesh.obj)")
    ap.add_argument("--method", choices=["bpa", "poisson", "planes"], default="bpa",
                    help="网格算法: bpa(表面) | poisson(封闭实体) | planes(RANSAC分平面, 每片拟合成平整面片)")
    ap.add_argument("--poisson-depth", type=int, default=9)
    ap.add_argument("--poisson-crop", type=float, default=0.001,
                    help="poisson: 只保留距参考点云不超过该距离的顶点 m "
                         "(默认0.001=1mm; 0=关闭)")
    ap.add_argument("--poisson-density-q", type=float, default=0.15,
                    help="poisson: 去掉密度最低的该比例顶点 (默认0.15; 0=关闭)")
    ap.add_argument("--poisson-support-radius", type=float, default=0.0012,
                    help="poisson: 原始点云支撑半径 m (默认0.0012=1.2mm; 0=关闭)")
    ap.add_argument("--poisson-support-min", type=int, default=2,
                    help="poisson: 支撑半径内至少几个原始点 (默认2)")
    ap.add_argument("--poisson-normal-dot", type=float, default=0.5,
                    help="poisson: 与最近原始点法向有符号 dot 下限, 去掉背面/立面 "
                         "(默认0.5; 0=关闭)")
    ap.add_argument("--no-roi", action="store_true", help="不按 config 的 roi 裁剪")
    ap.add_argument("--simplify", type=int, default=0, metavar="N",
                    help="目标面数: >0 时用二次误差简化把共面小三角合并成大面片(如 --simplify 1500)")
    ap.add_argument("--plane-dist", type=float, default=0.003,
                    help="planes: RANSAC 平面内点距离阈值 m (默认 0.003=3mm)")
    ap.add_argument("--plane-min", type=int, default=300,
                    help="planes: 一个平面的最小内点数 (默认 300)")
    ap.add_argument("--max-planes", type=int, default=8,
                    help="planes: 最多提取几个平面 (默认 8)")
    ap.add_argument("--plane-alpha", type=float, default=0.008,
                    help="planes: Delaunay 三角最大边长 m, 用于剔除跨洞连接 (默认 0.008=8mm)")
    # ---- 去噪(建网格前清理离群点) ----
    ap.add_argument("--sor-nb", type=int, default=20,
                    help="统计滤波邻居数(0=关闭SOR, 默认20)")
    ap.add_argument("--sor-std", type=float, default=2.0,
                    help="统计滤波标准差倍数, 越小越激进(默认2.0)")
    ap.add_argument("--dbscan", action=argparse.BooleanOptionalAction, default=True,
                    help="是否启用 DBSCAN 聚类去噪 (默认开; 用 --no-dbscan 关闭)")
    ap.add_argument("--cluster-eps", type=float, default=0.004,
                    help="DBSCAN 聚类邻域半径 m(默认0.004=4mm)")
    ap.add_argument("--cluster-min", type=int, default=30,
                    help="DBSCAN 一个簇的最小点数(默认30)")
    ap.add_argument("--keep-clusters", type=int, default=1,
                    help="只保留最大的前 N 个簇(默认1=只留最大块; 0=不按簇裁)")
    # ---- MLS 上采样(填线激光轮廓间隙) ----
    ap.add_argument("--mls", action=argparse.BooleanOptionalAction, default=False,
                    help="建网格前做局部平面 MLS 上采样 (默认关; --mls 开启)")
    ap.add_argument("--mls-search", type=float, default=0.0,
                    help="MLS 邻域半径 m, 0=按点距自动 (默认0)")
    ap.add_argument("--mls-gap", type=float, default=0.0,
                    help="MLS 最大填缝边长 m, 0=按 --mls-gap-factor 自动 (默认0)")
    ap.add_argument("--mls-gap-factor", type=float, default=5.0,
                    help="mls-gap 未指定时: gap = median_nn * factor (默认5)")
    ap.add_argument("--mls-samples", type=int, default=1,
                    help="每条稀疏边上插入点数 (默认1=中点; 2=三等分点)")
    ap.add_argument("--mls-voxel", type=float, default=0.0,
                    help="MLS 后体素合并边长 m, 0=不合并 (默认0)")
    # ---- 小洞填充 ----
    ap.add_argument("--fill-mesh", default="",
                    help="对已有 OBJ 只做小洞填充后写出(不重建点云)")
    ap.add_argument("--fill-holes", action=argparse.BooleanOptionalAction, default=False,
                    help="建网格后填充小洞 (默认关; --fill-holes 开启)")
    ap.add_argument("--fill-holes-max-edges", type=int, default=80,
                    help="可填充孔洞的最大边界边数 (默认80; 0=不限制, 大洞也填)")
    ap.add_argument("--fill-holes-max-perimeter", type=float, default=0.0,
                    help="可填充孔洞的最大周长 m, 0=不按周长限制 (默认0)")
    args = ap.parse_args()

    try:
        import open3d as o3d
    except Exception:
        print("错误: 需要安装 open3d -> pip install open3d")
        sys.exit(1)

    def read_triangle_mesh(path):
        import shutil
        import tempfile

        def _from_temp():
            # Windows 下含中文路径时 Open3D 读 OBJ 常失败, 经临时 ASCII 路径重试
            with tempfile.TemporaryDirectory(prefix="mesh_io_") as td:
                tmp = os.path.join(td, "mesh.obj")
                shutil.copy2(path, tmp)
                return o3d.io.read_triangle_mesh(tmp)

        try:
            mesh = o3d.io.read_triangle_mesh(path)
            if len(mesh.vertices) > 0:
                return mesh
        except UnicodeDecodeError:
            pass
        return _from_temp()

    if args.fill_mesh:
        if not os.path.isfile(args.fill_mesh):
            print(f"错误: 找不到网格 {args.fill_mesh}")
            sys.exit(1)
        mesh = read_triangle_mesh(args.fill_mesh)
        if len(mesh.vertices) == 0:
            print(f"错误: 网格为空或无法读取 {args.fill_mesh}")
            sys.exit(1)
        print(f"读入网格 {args.fill_mesh}: {len(mesh.vertices)} 顶点, "
              f"{len(mesh.triangles)} 面")
        max_perim = args.fill_holes_max_perimeter if args.fill_holes_max_perimeter > 0 else None
        mesh = fill_small_holes(
            o3d, mesh,
            max_edges=args.fill_holes_max_edges,
            max_perimeter=max_perim,
        )
        mesh.compute_vertex_normals()
        os.makedirs(os.path.dirname(os.path.abspath(args.out_obj)), exist_ok=True)
        o3d.io.write_triangle_mesh(args.out_obj, mesh)
        print(f"补洞后: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 面")
        print(f"已保存: {args.out_obj}")
        return

    if not os.path.isfile(args.in_ply):
        print(f"错误: 找不到输入点云 {args.in_ply}")
        sys.exit(1)

    pcd = o3d.io.read_point_cloud(args.in_ply)
    n0 = len(pcd.points)
    if n0 == 0:
        print(f"错误: 点云为空 {args.in_ply}")
        sys.exit(1)
    print(f"读入 {args.in_ply}: {n0} 点")

    # ROI 裁剪 (去板面等)
    if not args.no_roi:
        mask = roi_mask(np.asarray(pcd.points), cfg)
        if not mask.all():
            pcd = pcd.select_by_index(np.where(mask)[0])
            print(f"ROI 裁剪: {n0} -> {len(pcd.points)} 点")

    # 去噪: 统计滤波 + 只保留最大点簇, 清掉零散悬空离群点
    if args.sor_nb > 0 and len(pcd.points) > args.sor_nb:
        n1 = len(pcd.points)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=args.sor_nb,
                                                std_ratio=args.sor_std)
        print(f"统计滤波(SOR): {n1} -> {len(pcd.points)} 点")

    if args.dbscan and args.cluster_eps > 0 and args.keep_clusters > 0 and len(pcd.points) > args.cluster_min:
        labels = np.asarray(pcd.cluster_dbscan(eps=args.cluster_eps,
                                               min_points=args.cluster_min))
        valid = labels[labels >= 0]
        if valid.size:
            ids, counts = np.unique(valid, return_counts=True)
            keep_ids = set(ids[np.argsort(counts)[::-1][:args.keep_clusters]])
            sel = np.where(np.isin(labels, list(keep_ids)))[0]
            n2 = len(pcd.points)
            pcd = pcd.select_by_index(sel)
            print(f"聚类去噪: 保留最大 {args.keep_clusters} 簇, {n2} -> {len(pcd.points)} 点")

    if len(pcd.points) < 10:
        print("错误: 裁剪/去噪后点太少，无法建网格。检查 roi/去噪参数。")
        sys.exit(1)

    # 裁剪参考: 用 MLS 前的真实扫描点, 避免按补点把四周虚平面留下
    pcd_ref = o3d.geometry.PointCloud(pcd)

    # 法向 -> 可选 MLS 补点 -> 再估法向 (BPA/Poisson 都需要)
    pcd = estimate_orient_normals(o3d, pcd)
    if args.mls:
        pcd = mls_upsample(
            o3d, pcd,
            search_radius=args.mls_search if args.mls_search > 0 else None,
            gap=args.mls_gap if args.mls_gap > 0 else None,
            gap_factor=args.mls_gap_factor,
            samples=args.mls_samples,
            voxel=args.mls_voxel if args.mls_voxel > 0 else None,
        )
        if len(pcd.points) < 10:
            print("错误: MLS 后点太少，无法建网格。")
            sys.exit(1)
        pcd = estimate_orient_normals(o3d, pcd)

    if args.method == "poisson":
        mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=args.poisson_depth)
        mesh = trim_poisson_low_density(mesh, dens, args.poisson_density_q)
        mesh = crop_mesh_to_cloud(o3d, mesh, pcd_ref, args.poisson_crop)
        mesh = crop_mesh_by_support(
            mesh, pcd_ref, args.poisson_support_radius, args.poisson_support_min)
        # 法向裁剪需要参考点法向; 与建网格前同一套估计
        if args.poisson_normal_dot > 0:
            estimate_orient_normals(o3d, pcd_ref)
        mesh = crop_mesh_by_normal_agree(
            o3d, mesh, pcd_ref, max(args.poisson_crop, 0.001),
            args.poisson_normal_dot)
    elif args.method == "planes":
        mesh = mesh_by_planes(o3d, np.asarray(pcd.points),
                              dist=args.plane_dist, min_inliers=args.plane_min,
                              max_planes=args.max_planes, alpha=args.plane_alpha)
    else:  # bpa
        d = pcd.compute_nearest_neighbor_distance()
        avg = float(np.mean(d))
        radii = [avg * r for r in (1.5, 2.5, 4.0, 6.0)]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii))

    # 清理退化/重复元素，保证简化稳定
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    n_tri0 = len(mesh.triangles)

    # 平面感知简化：共面小三角合并成大面片
    if args.simplify > 0 and n_tri0 > args.simplify:
        mesh = mesh.simplify_quadric_decimation(
            target_number_of_triangles=args.simplify)
        mesh.remove_duplicated_vertices()
        mesh.remove_degenerate_triangles()
        print(f"简化: {n_tri0} -> {len(mesh.triangles)} 面")

    if args.fill_holes:
        max_perim = args.fill_holes_max_perimeter if args.fill_holes_max_perimeter > 0 else None
        mesh = fill_small_holes(
            o3d, mesh,
            max_edges=args.fill_holes_max_edges,
            max_perimeter=max_perim,
        )

    mesh.compute_vertex_normals()
    os.makedirs(os.path.dirname(os.path.abspath(args.out_obj)), exist_ok=True)
    o3d.io.write_triangle_mesh(args.out_obj, mesh)
    print(f"网格({args.method}): {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 面")
    print(f"已保存: {args.out_obj}")


if __name__ == "__main__":
    main()
