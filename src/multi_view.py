"""多视角点云配准与融合（阶段 8b）。

移植并适配自参考实现 multi_view_fusion.py，配合本项目的 reconstruct 使用。

提供两种融合方式：
  - charuco: 各朝向都用同一块贴在工件上的标定板重建到"板坐标系"，
             因此天然共坐标系，直接拼接即可（推荐，免配准）。
  - icp:     FPFH+RANSAC 粗配准 + Point-to-Plane ICP 精配准，
             用于翻面后不共坐标系的点云合并。
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np

from .reconstruct import reconstruct, write_ply

try:
    import open3d as o3d
    _HAS_O3D = True
except Exception:  # pragma: no cover
    _HAS_O3D = False


def _require_o3d():
    if not _HAS_O3D:
        raise ImportError("多视角 ICP 融合需要 open3d，请先 pip install open3d")


def reconstruct_views(cfg: Dict, view_dirs: List[str], intrinsic_path: str,
                      laser_plane_path: str, out_dir: str,
                      verbose: bool = True) -> List[Tuple[str, np.ndarray]]:
    """对每个朝向目录各跑一次重建，返回 [(名称, 板坐标系点云 Nx3), ...]。

    每个朝向的点云同时写出到 out_dir/views/<名称>.ply 便于单独检查。
    """
    views_dir = os.path.join(out_dir, "views")
    os.makedirs(views_dir, exist_ok=True)

    results: List[Tuple[str, np.ndarray]] = []
    for vdir in view_dirs:
        name = os.path.basename(os.path.normpath(vdir))
        if verbose:
            print(f"\n===== 重建朝向: {name} ({vdir}) =====")
        out_ply = os.path.join(views_dir, f"{name}.ply")
        try:
            pts = reconstruct(cfg, vdir, intrinsic_path, laser_plane_path,
                              out_ply, verbose=verbose)
        except Exception as e:
            print(f"  [跳过] {name} 重建失败: {e}")
            continue
        if pts is not None and len(pts) > 0:
            results.append((name, np.asarray(pts, dtype=np.float64).reshape(-1, 3)))
    if not results:
        raise RuntimeError("没有任何朝向成功重建，检查采集与标定。")
    return results


# ----------------------------------------------------------------------
# 方式 A：CharUco 共坐标系直接拼接
# ----------------------------------------------------------------------

def merge_charuco(views: List[Tuple[str, np.ndarray]],
                  verbose: bool = True) -> np.ndarray:
    """各朝向已在同一板坐标系，直接拼接。"""
    clouds = [pts for _, pts in views]
    merged = np.vstack(clouds)
    if verbose:
        for name, pts in views:
            print(f"  {name}: {len(pts)} 点")
        print(f"CharUco 直接拼接: {len(views)} 个朝向 -> {len(merged)} 点")
    return merged


# ----------------------------------------------------------------------
# 方式 B：FPFH + RANSAC + ICP（跨坐标系，如翻面后）
# ----------------------------------------------------------------------

def _to_o3d(pts: np.ndarray) -> "o3d.geometry.PointCloud":
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64).reshape(-1, 3))
    return pcd


def _preprocess(pcd, voxel):
    down = pcd.voxel_down_sample(voxel)
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))
    return down, fpfh


def _global_reg(src, tgt, src_f, tgt_f, voxel):
    dist = voxel * 1.5
    return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src, tgt, src_f, tgt_f,
        mutual_filter=True,
        max_correspondence_distance=dist,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )


def _icp_refine(src, tgt, init, voxel, max_iter=50):
    dist = voxel * 0.4
    src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
    return o3d.pipelines.registration.registration_icp(
        src, tgt, dist, init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))


def merge_icp(views: List[Tuple[str, np.ndarray]],
              voxel_coarse: float = 0.005, voxel_fine: float = 0.002,
              verbose: bool = True) -> np.ndarray:
    """逐对 FPFH+RANSAC 粗配准 + ICP 精配准，累积合并。"""
    _require_o3d()
    clouds = [_to_o3d(pts) for _, pts in views]
    merged = clouds[0]
    for i in range(1, len(clouds)):
        src = clouds[i]
        tgt = merged
        sd, sf = _preprocess(src, voxel_coarse)
        td, tf = _preprocess(tgt, voxel_coarse)
        gr = _global_reg(sd, td, sf, tf, voxel_coarse)
        sf2, _ = _preprocess(src, voxel_fine)
        tf2, _ = _preprocess(tgt, voxel_fine)
        rr = _icp_refine(sf2, tf2, gr.transformation, voxel_fine)
        aligned = o3d.geometry.PointCloud(src)
        aligned.transform(rr.transformation)
        merged = merged + aligned
        if verbose:
            print(f"  {views[i][0]}: fitness={rr.fitness:.4f} RMSE={rr.inlier_rmse*1000:.3f}mm")
    return np.asarray(merged.points, dtype=np.float64)


def merge_views(views: List[Tuple[str, np.ndarray]], method: str = "charuco",
                voxel_coarse: float = 0.005, voxel_fine: float = 0.002,
                verbose: bool = True) -> np.ndarray:
    method = (method or "charuco").lower()
    if method == "charuco":
        return merge_charuco(views, verbose=verbose)
    if method == "icp":
        return merge_icp(views, voxel_coarse, voxel_fine, verbose=verbose)
    raise ValueError(f"未知融合方式: {method} (可选 charuco | icp)")
