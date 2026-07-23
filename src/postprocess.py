"""点云后处理与建模（阶段 9）。

- 三维 ROI（manual 包围盒 / auto 主峰+百分位）
- 球面拟合残差 trim（斜视掠射坏边）
- 统计离群点滤波 (SOR)
- 体素降采样
- 法向估计
- 高度图栅格化 (X,Y -> Z)
- 可选 mesh 重建 (Poisson / Ball Pivoting)
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import open3d as o3d
    _HAS_O3D = True
except Exception:  # pragma: no cover
    _HAS_O3D = False


def _require_o3d():
    if not _HAS_O3D:
        raise ImportError("需要 open3d，请先 pip install open3d")


def load_ply(path: str) -> "o3d.geometry.PointCloud":
    _require_o3d()
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        raise RuntimeError(f"点云为空: {path}")
    return pcd


def fit_sphere_lstsq(points: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """最小二乘球面拟合。返回 (center, radius, signed_residuals)。"""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 4:
        raise ValueError("球面拟合至少需要 4 个点")
    A = np.column_stack([2.0 * pts, np.ones(len(pts))])
    b = (pts ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    center = sol[:3]
    radius = float(np.sqrt(max(sol[3] + float((center ** 2).sum()), 0.0)))
    residuals = np.linalg.norm(pts - center, axis=1) - radius
    return center, radius, residuals


def fit_sphere_ransac(
    points: np.ndarray,
    iters: int = 80,
    inlier_k: float = 2.5,
    min_inliers: int = 50,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """稳健球面拟合：RANSAC 种子 + 内点再 lstsq。"""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 4:
        raise ValueError("球面拟合至少需要 4 个点")
    rng = rng or np.random.default_rng(0)

    best_n = -1
    best_center = None
    best_radius = None
    best_res = None

    for _ in range(max(1, int(iters))):
        idx = rng.choice(len(pts), size=4, replace=False)
        try:
            c, r, _ = fit_sphere_lstsq(pts[idx])
        except Exception:
            continue
        if not np.isfinite(r) or r <= 1e-6 or r > 5.0:
            continue
        res = np.linalg.norm(pts - c, axis=1) - r
        scale = max(float(np.median(np.abs(res))) * 1.4826, 1e-4)
        inl = np.abs(res) <= float(inlier_k) * scale
        n_inl = int(inl.sum())
        if n_inl > best_n and n_inl >= min(min_inliers, max(4, len(pts) // 5)):
            best_n = n_inl
            best_center, best_radius, best_res = c, r, res

    if best_center is None:
        return fit_sphere_lstsq(pts)

    scale = max(float(np.median(np.abs(best_res))) * 1.4826, 1e-4)
    inl = np.abs(best_res) <= float(inlier_k) * scale
    if inl.sum() >= 4:
        return fit_sphere_lstsq(pts[inl])
    return best_center, float(best_radius), best_res


def auto_roi_mask(
    points: np.ndarray,
    cfg: Dict,
    verbose: bool = True,
) -> np.ndarray:
    """Z 直方图主峰 + 峰内 XYZ 百分位盒。"""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    roi = cfg.get("roi", {}) or {}
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)

    n_bins = int(roi.get("auto_z_bins", 40))
    pad = float(roi.get("auto_z_pad_m", 0.008))
    lo_pct = float(roi.get("auto_percentile_lo", 2.0))
    hi_pct = float(roi.get("auto_percentile_hi", 98.0))
    min_peak_frac = float(roi.get("auto_min_peak_frac", 0.05))

    z = pts[:, 2]
    z_min, z_max = float(z.min()), float(z.max())
    if z_max <= z_min:
        return np.ones(len(pts), dtype=bool)

    hist, edges = np.histogram(z, bins=max(8, n_bins))
    peak = int(np.argmax(hist))
    # grow contiguous band around peak while bins stay reasonably populated
    thr = max(1, int(hist[peak] * 0.15))
    left = right = peak
    while left > 0 and hist[left - 1] >= thr:
        left -= 1
    while right < len(hist) - 1 and hist[right + 1] >= thr:
        right += 1
    z_lo = float(edges[left]) - pad
    z_hi = float(edges[right + 1]) + pad
    peak_mask = (z >= z_lo) & (z <= z_hi)
    if peak_mask.sum() < max(20, int(min_peak_frac * len(pts))):
        # fallback: densest quartile by keeping points near median Z
        med = float(np.median(z))
        mad = float(np.median(np.abs(z - med))) + 1e-6
        peak_mask = np.abs(z - med) <= 4.0 * mad
        z_lo, z_hi = float(z[peak_mask].min()), float(z[peak_mask].max())

    sub = pts[peak_mask]
    lo = np.percentile(sub, lo_pct, axis=0)
    hi = np.percentile(sub, hi_pct, axis=0)
    # slight pad on XY
    xy_pad = float(roi.get("auto_xy_pad_m", 0.003))
    lo = lo.copy()
    hi = hi.copy()
    lo[:2] -= xy_pad
    hi[:2] += xy_pad
    lo[2] = min(lo[2], z_lo)
    hi[2] = max(hi[2], z_hi)

    m = (
        (pts[:, 0] >= lo[0]) & (pts[:, 0] <= hi[0])
        & (pts[:, 1] >= lo[1]) & (pts[:, 1] <= hi[1])
        & (pts[:, 2] >= lo[2]) & (pts[:, 2] <= hi[2])
    )
    if verbose:
        print(
            "ROI auto: "
            f"x=[{lo[0]:.4f},{hi[0]:.4f}] "
            f"y=[{lo[1]:.4f},{hi[1]:.4f}] "
            f"z=[{lo[2]:.4f},{hi[2]:.4f}] m "
            f"({int(m.sum())}/{len(pts)} pts)"
        )
    return m


def manual_roi_mask(points: np.ndarray, cfg: Dict) -> np.ndarray:
    """按 config 固定包围盒裁剪。"""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    roi = cfg.get("roi", {}) or {}
    m = np.ones(len(pts), dtype=bool)
    bounds = [
        ("x_min", 0, np.greater_equal), ("x_max", 0, np.less_equal),
        ("y_min", 1, np.greater_equal), ("y_max", 1, np.less_equal),
        ("z_min", 2, np.greater_equal), ("z_max", 2, np.less_equal),
    ]
    for key, axis, comp in bounds:
        v = roi.get(key, None)
        if v is not None:
            m &= comp(pts[:, axis], float(v))
    return m


def roi_mask(points: np.ndarray, cfg: Dict, verbose: bool = False) -> np.ndarray:
    """按 config 的 roi 段返回布尔掩码 (导轨/相机坐标系, 单位 m)。未启用则全 True。"""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    roi = cfg.get("roi", {}) or {}
    if not roi.get("enabled", False):
        return np.ones(len(pts), dtype=bool)
    mode = str(roi.get("mode", "manual")).lower().strip()
    if mode == "auto":
        return auto_roi_mask(pts, cfg, verbose=verbose)
    return manual_roi_mask(pts, cfg)


def crop_roi_points(points: np.ndarray, cfg: Dict, verbose: bool = True) -> np.ndarray:
    """对 numpy 点云应用 ROI 裁剪。"""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    mask = roi_mask(pts, cfg, verbose=verbose)
    if verbose and (cfg.get("roi", {}) or {}).get("enabled", False):
        mode = str((cfg.get("roi") or {}).get("mode", "manual"))
        print(f"ROI 裁剪 ({mode}): {len(pts)} -> {int(mask.sum())} 点")
    return pts[mask]


def surface_trim_mask(
    points: np.ndarray,
    cfg: Dict,
    verbose: bool = True,
) -> np.ndarray:
    """按稳健球面残差剔除掠射/飞边。关闭或点数不足时全保留。"""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pp = cfg.get("postprocess", {}) or {}
    k = pp.get("surface_trim_k", None)
    if k is None or float(k) <= 0 or len(pts) < 30:
        return np.ones(len(pts), dtype=bool)

    k = float(k)
    iters = int(pp.get("surface_trim_ransac_iters", 80))
    try:
        center, radius, res = fit_sphere_ransac(pts, iters=iters, inlier_k=k)
    except Exception as exc:
        if verbose:
            print(f"surface_trim: 球面拟合失败，跳过 ({exc})")
        return np.ones(len(pts), dtype=bool)

    scale = max(float(np.median(np.abs(res))) * 1.4826, 1e-4)
    keep = np.abs(res) <= k * scale
    # also drop extreme spatial outliers relative to inlier AABB (outer fringe)
    if keep.sum() >= 20:
        inl = pts[keep]
        lo = np.percentile(inl, 1.0, axis=0)
        hi = np.percentile(inl, 99.0, axis=0)
        pad = float(pp.get("surface_trim_aabb_pad_m", 0.002))
        aabb = (
            (pts[:, 0] >= lo[0] - pad) & (pts[:, 0] <= hi[0] + pad)
            & (pts[:, 1] >= lo[1] - pad) & (pts[:, 1] <= hi[1] + pad)
            & (pts[:, 2] >= lo[2] - pad) & (pts[:, 2] <= hi[2] + pad)
        )
        keep &= aabb

    if verbose:
        print(
            f"surface_trim: sphere R={radius*1000:.1f}mm "
            f"center_mm={np.round(center*1000, 1)} "
            f"res_std={res.std()*1000:.2f}mm "
            f"keep {int(keep.sum())}/{len(pts)} (k={k})"
        )
    if keep.sum() < 20:
        if verbose:
            print("surface_trim: 保留点过少，回退为不剔除")
        return np.ones(len(pts), dtype=bool)
    return keep


def two_plane_cleanup_mask(
    points: np.ndarray,
    cfg: Dict,
    verbose: bool = True,
) -> np.ndarray:
    """删除同时远离两个主平面的混合边缘点，不投影或修改保留点。

    仅适用于确定由两个平面组成的直角件/多面体扫描。未知曲面、真实缺陷
    检测和自由形状物体必须关闭。
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pp = cfg.get("postprocess", {}) or {}
    cleanup = pp.get("two_plane_cleanup", {}) or {}
    if not bool(cleanup.get("enabled", False)):
        return np.ones(len(pts), dtype=bool)
    min_points = int(cleanup.get("min_plane_points", 500))
    if len(pts) < 2 * min_points:
        if verbose:
            print("two_plane_cleanup: 点数不足，跳过")
        return np.ones(len(pts), dtype=bool)

    ransac_threshold = float(
        cleanup.get("ransac_threshold_m", 0.0015)
    )
    max_distance = float(cleanup.get("max_distance_m", 0.002))
    min_angle_deg = float(cleanup.get("min_plane_angle_deg", 45.0))
    min_keep_fraction = float(cleanup.get("min_keep_fraction", 0.70))
    iterations = int(cleanup.get("ransac_iterations", 4000))

    work_points = pts
    first_model = None
    second_model = None
    first_count = second_count = 0
    for _ in range(10):
        if len(work_points) < min_points:
            break
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(work_points)
        model, local_inliers = cloud.segment_plane(
            distance_threshold=ransac_threshold,
            ransac_n=3,
            num_iterations=iterations,
        )
        local_inliers = np.asarray(local_inliers, dtype=np.int64)
        if len(local_inliers) < min_points:
            break
        normal = np.asarray(model[:3], dtype=np.float64)
        normal /= np.linalg.norm(normal) + 1e-12
        if first_model is None:
            first_model = np.asarray(model, dtype=np.float64)
            first_count = len(local_inliers)
        else:
            first_normal = first_model[:3]
            cosine = np.clip(
                abs(float(np.dot(first_normal, normal)))
                / (
                    (np.linalg.norm(first_normal) + 1e-12)
                    * (np.linalg.norm(normal) + 1e-12)
                ),
                0.0,
                1.0,
            )
            angle = float(np.degrees(np.arccos(cosine)))
            if angle >= min_angle_deg:
                second_model = np.asarray(model, dtype=np.float64)
                second_count = len(local_inliers)
                break
        keep_remaining = np.ones(len(work_points), dtype=bool)
        keep_remaining[local_inliers] = False
        work_points = work_points[keep_remaining]

    if first_model is None or second_model is None:
        if verbose:
            print("two_plane_cleanup: 未可靠找到两个不同平面，跳过")
        return np.ones(len(pts), dtype=bool)

    first_normal = first_model[:3]
    second_normal = second_model[:3]
    first_distance = np.abs(
        pts @ first_normal + first_model[3]
    ) / (np.linalg.norm(first_normal) + 1e-12)
    second_distance = np.abs(
        pts @ second_normal + second_model[3]
    ) / (np.linalg.norm(second_normal) + 1e-12)
    keep = np.minimum(first_distance, second_distance) <= max_distance
    keep_fraction = float(keep.mean())
    if keep_fraction < min_keep_fraction:
        if verbose:
            print(
                "two_plane_cleanup: "
                f"仅保留 {keep_fraction*100:.1f}% < "
                f"{min_keep_fraction*100:.1f}%，安全回退为不清理"
            )
        return np.ones(len(pts), dtype=bool)

    cosine = np.clip(
        abs(float(np.dot(first_normal, second_normal)))
        / (
            (np.linalg.norm(first_normal) + 1e-12)
            * (np.linalg.norm(second_normal) + 1e-12)
        ),
        0.0,
        1.0,
    )
    angle = float(np.degrees(np.arccos(cosine)))
    if verbose:
        print(
            "two_plane_cleanup: "
            f"planes={first_count}/{second_count}, angle={angle:.2f}°, "
            f"distance<={max_distance*1000:.1f}mm, "
            f"keep {int(keep.sum())}/{len(pts)}, "
            f"removed {int((~keep).sum())}"
        )
    return keep


def align_points_to_sphere_frame(
    points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """把点云变到拟合球局部坐标：球心为原点，补片主方向对齐 +Z（朝外法向近似）。

    返回 (aligned_points, center, radius)。
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    center, radius, _ = fit_sphere_ransac(pts)
    q = pts - center
    # PCA: thinnest axis ~ radial/normal for a thin shell patch; point +Z toward camera-ish
    # Prefer mean radial direction as +Z (from center to patch)
    mean_dir = q.mean(axis=0)
    n = np.linalg.norm(mean_dir)
    if n < 1e-9:
        z_axis = np.array([0.0, 0.0, 1.0])
    else:
        z_axis = mean_dir / n
    # build orthonormal frame
    tmp = np.array([0.0, 1.0, 0.0]) if abs(z_axis[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(tmp, z_axis)
    x_n = np.linalg.norm(x_axis)
    if x_n < 1e-9:
        tmp = np.array([1.0, 0.0, 0.0])
        x_axis = np.cross(tmp, z_axis)
        x_n = np.linalg.norm(x_axis)
    x_axis /= x_n
    y_axis = np.cross(z_axis, x_axis)
    R = np.stack([x_axis, y_axis, z_axis], axis=0)  # world -> local
    aligned = (R @ q.T).T
    return aligned, center, radius


def process_point_cloud(cfg: Dict, in_ply: str, out_ply: str,
                        verbose: bool = True):
    """滤波 + 降采样 + 法向估计，写出处理后的点云。返回处理后的 pcd。"""
    _require_o3d()
    pp = cfg.get("postprocess", {})
    pcd = load_ply(in_ply)
    n0 = len(pcd.points)

    # ROI 裁剪（去掉远景/背景），保留法向/颜色
    mask = roi_mask(np.asarray(pcd.points), cfg, verbose=verbose)
    if not mask.all():
        pcd = pcd.select_by_index(np.where(mask)[0])
        if verbose:
            mode = str((cfg.get("roi") or {}).get("mode", "manual"))
            print(f"ROI 裁剪 ({mode}): {n0} -> {len(pcd.points)} 点")

    # 球面残差 trim（斜视掠射坏边）
    if len(pcd.points) > 0:
        tmask = surface_trim_mask(np.asarray(pcd.points), cfg, verbose=verbose)
        if not tmask.all():
            n_before = len(pcd.points)
            pcd = pcd.select_by_index(np.where(tmask)[0])
            if verbose:
                print(f"surface_trim 应用: {n_before} -> {len(pcd.points)} 点")

    # 两平面直角件专用：删除棱边混合像素/多路径反光形成的桥接点。
    if len(pcd.points) > 0:
        pmask = two_plane_cleanup_mask(
            np.asarray(pcd.points), cfg, verbose=verbose
        )
        if not pmask.all():
            n_before = len(pcd.points)
            pcd = pcd.select_by_index(np.where(pmask)[0])
            if verbose:
                print(
                    f"two_plane_cleanup 应用: "
                    f"{n_before} -> {len(pcd.points)} 点"
                )

    voxel = float(pp.get("voxel_size", 0.0005))
    if voxel > 0 and len(pcd.points) > 0:
        pcd = pcd.voxel_down_sample(voxel)

    nb = int(pp.get("sor_neighbors", 20))
    std = float(pp.get("sor_std_ratio", 2.0))
    if nb > 0 and len(pcd.points) > 0:
        pcd, _ = pcd.remove_statistical_outlier(nb, std)

    if len(pcd.points) > 0:
        radius = float(pp.get("normal_radius", 0.002))
        max_nn = int(pp.get("normal_max_nn", 30))
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))

    os.makedirs(os.path.dirname(os.path.abspath(out_ply)), exist_ok=True)
    o3d.io.write_point_cloud(out_ply, pcd)
    if verbose:
        print(f"后处理: {n0} -> {len(pcd.points)} 点, 已保存 {out_ply}")
    return pcd


def to_height_map(points: np.ndarray, res: float,
                  pct: float = 0.5, max_cells: int = 4000
                  ) -> Tuple[np.ndarray, Tuple[float, float]]:
    """把点云栅格化成高度图 (X,Y -> Z)。

    返回 (height_map, (min_x, min_y))。空栅格为 NaN。
    对每个栅格取落入点的 Z 均值。

    鲁棒性: 用分位数 [pct, 100-pct] 界定 X/Y 边界并丢弃界外离群点,
    避免个别坏点把栅格撑到几 GB; 若栅格仍超过 max_cells 则自动放大 res。
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    lo_x, hi_x = np.percentile(pts[:, 0], [pct, 100.0 - pct])
    lo_y, hi_y = np.percentile(pts[:, 1], [pct, 100.0 - pct])
    keep = ((pts[:, 0] >= lo_x) & (pts[:, 0] <= hi_x) &
            (pts[:, 1] >= lo_y) & (pts[:, 1] <= hi_y))
    pts = pts[keep]
    if len(pts) == 0:
        return np.full((1, 1), np.nan, dtype=np.float64), (0.0, 0.0)

    min_x, min_y = pts[:, 0].min(), pts[:, 1].min()
    span_x = pts[:, 0].max() - min_x
    span_y = pts[:, 1].max() - min_y
    need = int(max(span_x, span_y) / res) + 1
    if need > max_cells:
        res = max(span_x, span_y) / max_cells
    ix = np.floor((pts[:, 0] - min_x) / res).astype(int)
    iy = np.floor((pts[:, 1] - min_y) / res).astype(int)
    w = int(ix.max()) + 1
    h = int(iy.max()) + 1

    acc = np.zeros((h, w), dtype=np.float64)
    cnt = np.zeros((h, w), dtype=np.int64)
    np.add.at(acc, (iy, ix), pts[:, 2])
    np.add.at(cnt, (iy, ix), 1)

    hm = np.full((h, w), np.nan, dtype=np.float64)
    mask = cnt > 0
    hm[mask] = acc[mask] / cnt[mask]
    return hm, (float(min_x), float(min_y))


def save_height_map(cfg: Dict, points: np.ndarray, out_npy: str,
                    out_png: Optional[str] = None, verbose: bool = True) -> None:
    """生成并保存高度图（.npy 原始值 + 可选 .png 可视化）。"""
    pp = cfg.get("postprocess", {})
    res = float(pp.get("height_map_res", 0.0005))
    hm, origin = to_height_map(points, res)

    os.makedirs(os.path.dirname(os.path.abspath(out_npy)), exist_ok=True)
    np.save(out_npy, hm)

    if out_png is not None:
        _save_height_png(hm, out_png)
    if verbose:
        valid = np.isfinite(hm).sum()
        print(f"高度图: {hm.shape[1]}x{hm.shape[0]} 栅格, 有效 {valid}, "
              f"原点=({origin[0]:.4f},{origin[1]:.4f})m, 分辨率={res*1000:.3f}mm")
        print(f"已保存: {out_npy}")


def _save_height_png(hm: np.ndarray, out_png: str) -> None:
    import cv2
    valid = np.isfinite(hm)
    if not valid.any():
        return
    zmin, zmax = np.nanmin(hm), np.nanmax(hm)
    norm = np.zeros_like(hm)
    if zmax > zmin:
        norm = (hm - zmin) / (zmax - zmin)
    img = np.zeros(hm.shape, dtype=np.uint8)
    img[valid] = (norm[valid] * 255).astype(np.uint8)
    color = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    color[~valid] = 0
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    cv2.imwrite(out_png, color)


def reconstruct_mesh(cfg: Dict, pcd, out_mesh: str, verbose: bool = True):
    """可选：从点云重建 mesh。method: none|poisson|bpa。"""
    _require_o3d()
    pp = cfg.get("postprocess", {})
    method = str(pp.get("mesh_method", "none")).lower()
    if method == "none":
        if verbose:
            print("mesh_method=none，跳过 mesh 重建。")
        return None
    if not pcd.has_normals():
        pcd.estimate_normals()

    if method == "poisson":
        depth = int(pp.get("poisson_depth", 9))
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth)
    elif method == "bpa":
        dists = pcd.compute_nearest_neighbor_distance()
        avg = float(np.mean(dists))
        radii = [avg * r for r in (1.0, 2.0, 4.0)]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii))
    else:
        raise ValueError(f"未知 mesh_method: {method}")

    os.makedirs(os.path.dirname(os.path.abspath(out_mesh)), exist_ok=True)
    o3d.io.write_triangle_mesh(out_mesh, mesh)
    if verbose:
        print(f"mesh({method}): {len(mesh.triangles)} 面, 已保存 {out_mesh}")
    return mesh
