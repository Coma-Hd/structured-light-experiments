"""PLY 点云 -> OBJ 网格。

用法:
    # 默认：读 output/cloud.ply，按 config.yaml 的 roi 裁剪后建网格，存 output/mesh.obj
    python scripts/ply_to_obj.py

    # 指定输入/输出
    python scripts/ply_to_obj.py --in output/cloud_clean.ply --out output/mesh.obj

    # 换重建算法 (bpa 适合表面, poisson 适合封闭实体)
    python scripts/ply_to_obj.py --method poisson

    # 不做 ROI 裁剪 (直接对原始点云建网格)
    python scripts/ply_to_obj.py --no-roi
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, resolve_path  # noqa: E402
from src.postprocess import roi_mask  # noqa: E402


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
    args = ap.parse_args()

    try:
        import open3d as o3d
    except Exception:
        print("错误: 需要安装 open3d -> pip install open3d")
        sys.exit(1)

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

    # 法向估计 (BPA/Poisson 都需要)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(20)

    if args.method == "poisson":
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=args.poisson_depth)
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

    mesh.compute_vertex_normals()
    os.makedirs(os.path.dirname(os.path.abspath(args.out_obj)), exist_ok=True)
    o3d.io.write_triangle_mesh(args.out_obj, mesh)
    print(f"网格({args.method}): {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 面")
    print(f"已保存: {args.out_obj}")


if __name__ == "__main__":
    main()
