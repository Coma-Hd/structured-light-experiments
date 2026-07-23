"""ChArUco 标定板封装：创建、检测、位姿估计。

兼容 OpenCV 新旧 aruco API：
- 新 (>=4.7): cv2.aruco.CharucoBoard(...) / CharucoDetector
- 旧 (<4.7):  cv2.aruco.CharucoBoard_create(...) / detectMarkers + interpolateCornersCharuco
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import CharucoConfig


def _get_dictionary(name: str):
    dic_id = getattr(cv2.aruco, name, None)
    if dic_id is None:
        raise ValueError(f"未知的 ArUco 字典: {name}")
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(dic_id)
    return cv2.aruco.Dictionary_get(dic_id)


def _create_board(cfg: CharucoConfig, dictionary):
    size = (cfg.squares_x, cfg.squares_y)
    # 新 API
    if hasattr(cv2.aruco, "CharucoBoard"):
        try:
            return cv2.aruco.CharucoBoard(
                size, cfg.square_length, cfg.marker_length, dictionary)
        except TypeError:
            pass
    # 旧 API
    return cv2.aruco.CharucoBoard_create(
        cfg.squares_x, cfg.squares_y,
        cfg.square_length, cfg.marker_length, dictionary)


def _board_object_points(board) -> np.ndarray:
    """返回所有棋盘内角点在板坐标系下的 3D 坐标 (N,3)。"""
    if hasattr(board, "getChessboardCorners"):
        return np.asarray(board.getChessboardCorners(), dtype=np.float64)
    return np.asarray(board.chessboardCorners, dtype=np.float64)


@dataclass
class CharucoDetection:
    corners: np.ndarray      # (N,2) 亚像素图像角点
    ids: np.ndarray          # (N,1) 角点 id
    obj_points: np.ndarray   # (N,3) 对应板坐标系 3D 点

    @property
    def count(self) -> int:
        return 0 if self.ids is None else int(len(self.ids))


class CharucoTarget:
    """封装一块 ChArUco 板的检测与位姿估计。"""

    def __init__(self, cfg: CharucoConfig):
        self.cfg = cfg
        self.dictionary = _get_dictionary(cfg.dictionary)
        self.board = _create_board(cfg, self.dictionary)
        self._all_obj = _board_object_points(self.board)
        # 新式 CharucoDetector：同时准备「新排布」和「旧版(legacy)排布」两种，
        # 自动选能识别到角点的那种（市售板常为 legacy，OpenCV>=4.6 默认新排布）。
        self._detectors = []          # [(legacy_flag, detector)]
        self._pref_idx = 0            # 记住上次成功的那种，避免每帧都两种都跑
        if hasattr(cv2.aruco, "CharucoDetector"):
            for legacy in (False, True):
                try:
                    board = _create_board(cfg, self.dictionary)
                    if hasattr(board, "setLegacyPattern"):
                        board.setLegacyPattern(legacy)
                    elif legacy:
                        continue  # 该版本不支持 legacy 切换，跳过重复项
                    self._detectors.append((legacy, cv2.aruco.CharucoDetector(board)))
                except Exception:
                    pass

        # 回退路径：ArUco 标记检测器 + 每个标记 4 角的板坐标。
        # 某些 OpenCV 版本 detectBoard 无法把标记插值成棋盘角点(返回 0)，
        # 此时直接用标记角点构建 3D-2D 对应，一样能标定/求位姿。
        self._aruco_detector = None
        if hasattr(cv2.aruco, "ArucoDetector"):
            try:
                self._aruco_detector = cv2.aruco.ArucoDetector(
                    self.dictionary, cv2.aruco.DetectorParameters())
            except Exception:
                self._aruco_detector = None

        # 候选 id->3D 布局：OpenCV 自带 + 两种错列(行优先，偶数行 偶/奇 列起)。
        # 市售板的 marker-id 排布常与 OpenCV 生成的错开一列，首帧用单应残差自动选中。
        self._marker_obj = self._build_marker_obj_map()   # OpenCV getObjPoints
        self._cand_maps = []
        if self._marker_obj:
            self._cand_maps.append(self._marker_obj)
        if self.cfg.squares_x % 2 == 0:
            self._cand_maps.append(self._build_formula_map(0))
            self._cand_maps.append(self._build_formula_map(1))
        # 兼容“每个网格都是一个 ArUco 标记”的 GridBoard 实体板。
        # 此时 square_length 是标记中心间距，marker_length 是标记边长。
        self._cand_maps.append(self._build_grid_formula_map())
        self._locked_map = None      # 锁定选中的布局，后续帧不再重选

    def _build_marker_obj_map(self) -> dict:
        """id -> (4,3) 该标记四角在板坐标系的 3D 坐标（取自 OpenCV board）。"""
        mp: dict = {}
        try:
            if hasattr(self.board, "getObjPoints"):
                obj_list = self.board.getObjPoints()
                ids = self.board.getIds()
            else:
                obj_list = self.board.objPoints
                ids = self.board.ids
            ids = np.asarray(ids).flatten()
            for i, mid in enumerate(ids):
                mp[int(mid)] = np.asarray(obj_list[i], dtype=np.float64).reshape(-1, 3)
        except Exception:
            pass
        return mp

    def _build_formula_map(self, even_offset: int) -> dict:
        """按「行优先、每行 squares_x//2 个」的错列布局构造 id->(4,3)。

        even_offset=1 复现 OpenCV 排布(偶数行在奇数列)，=0 为其镜像错列
        (偶数行在偶数列，很多市售板如此)。角点顺序 TL,TR,BR,BL 与 OpenCV 一致。
        """
        mp: dict = {}
        per_row = self.cfg.squares_x // 2
        if per_row <= 0:
            return mp
        sq = float(self.cfg.square_length)
        mk = float(self.cfg.marker_length)
        inset = (sq - mk) / 2.0
        for mid in range(per_row * self.cfg.squares_y):
            row = mid // per_row
            j = mid % per_row
            col = 2 * j + (even_offset if row % 2 == 0 else 1 - even_offset)
            x0 = col * sq + inset
            y0 = row * sq + inset
            mp[mid] = np.array([[x0, y0, 0.0], [x0 + mk, y0, 0.0],
                                [x0 + mk, y0 + mk, 0.0], [x0, y0 + mk, 0.0]],
                               dtype=np.float64)
        return mp

    def _build_grid_formula_map(self) -> dict:
        """构造标准 ArUco GridBoard 的行优先 id->四角布局。

        物理板若为 12x9 个标记、中心间距 30 mm、标记边长 24 mm，
        则配置仍使用 squares_x=12, squares_y=9,
        square_length=0.030, marker_length=0.024。
        """
        mp: dict = {}
        pitch = float(self.cfg.square_length)
        marker = float(self.cfg.marker_length)
        for row in range(self.cfg.squares_y):
            for col in range(self.cfg.squares_x):
                marker_id = row * self.cfg.squares_x + col
                x0 = col * pitch
                y0 = row * pitch
                mp[marker_id] = np.array([
                    [x0, y0, 0.0],
                    [x0 + marker, y0, 0.0],
                    [x0 + marker, y0 + marker, 0.0],
                    [x0, y0 + marker, 0.0],
                ], dtype=np.float64)
        return mp

    def _select_map(self, m_corners, m_ids):
        """用单应(与内参无关)残差，从候选布局里挑与实体板最吻合的一种。"""
        best = None
        best_err = float("inf")
        for cand in self._cand_maps:
            bxy, ixy = [], []
            for c, mid in zip(m_corners, m_ids):
                o = cand.get(int(mid))
                if o is None:
                    continue
                bxy.append(o[:4, :2])
                ixy.append(np.asarray(c, dtype=np.float64).reshape(-1, 2)[:4])
            if len(bxy) < 4:
                continue
            b = np.concatenate(bxy).astype(np.float32)
            im = np.concatenate(ixy).astype(np.float32)
            H, _ = cv2.findHomography(b, im, cv2.RANSAC, 5.0)
            if H is None:
                continue
            proj = cv2.perspectiveTransform(b.reshape(-1, 1, 2), H).reshape(-1, 2)
            err = float(np.sqrt(np.mean(np.sum((proj - im) ** 2, axis=1))))
            if err < best_err:
                best_err = err
                best = cand
        return best

    def _detect_markers(self, gray: np.ndarray):
        if self._aruco_detector is not None:
            corners, ids, _ = self._aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary)
        return corners, ids

    def _detect_via_markers(self, gray: np.ndarray) -> Optional[CharucoDetection]:
        """用 ArUco 标记角点直接构建对应点（detectBoard 失败时的回退）。"""
        if not self._cand_maps:
            return None
        m_corners, m_ids = self._detect_markers(gray)
        if m_ids is None or len(m_ids) == 0:
            return None
        m_ids = np.asarray(m_ids).flatten()
        if self._locked_map is None:
            self._locked_map = self._select_map(m_corners, m_ids)
            if self._locked_map is None:
                return None
        obj_map = self._locked_map
        img_pts, obj_pts, ids_out = [], [], []
        for c, mid in zip(m_corners, m_ids):
            mo = obj_map.get(int(mid))
            if mo is None or mo.shape[0] < 4:
                continue
            c = np.asarray(c, dtype=np.float64).reshape(-1, 2)
            if c.shape[0] < 4:
                continue
            img_pts.append(c[:4])
            obj_pts.append(mo[:4])
            ids_out.extend([int(mid)] * 4)
        if not img_pts:
            return None
        corners = np.concatenate(img_pts, axis=0)
        obj = np.concatenate(obj_pts, axis=0)
        ids = np.asarray(ids_out, dtype=np.int32).reshape(-1, 1)
        return CharucoDetection(corners=corners, ids=ids, obj_points=obj)

    def detect(self, image: np.ndarray) -> Optional[CharucoDetection]:
        """检测 ChArUco 角点。失败返回 None。

        优先用 detectBoard 拿棋盘角点(定位更准)；若拿不到(某些 OpenCV
        版本插值失败)，回退到直接用 ArUco 标记角点。
        """
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ch_corners = None
        ch_ids = None
        if self._detectors:
            order = [self._pref_idx] + [i for i in range(len(self._detectors))
                                        if i != self._pref_idx]
            for i in order:
                try:
                    cc, ci, _mc, _mi = self._detectors[i][1].detectBoard(gray)
                except Exception:
                    continue
                if ci is not None and len(ci) >= 4:
                    ch_corners, ch_ids = cc, ci
                    self._pref_idx = i
                    break

        if ch_ids is not None and len(ch_ids) >= 4:
            ch_corners = np.asarray(ch_corners, dtype=np.float64).reshape(-1, 2)
            ch_ids = np.asarray(ch_ids, dtype=np.int32).reshape(-1, 1)
            obj = self._all_obj[ch_ids.flatten()]
            return CharucoDetection(corners=ch_corners, ids=ch_ids, obj_points=obj)

        return self._detect_via_markers(gray)

    def estimate_pose(self, det: CharucoDetection, K: np.ndarray,
                      dist: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """求解板 -> 相机 位姿，返回 (rvec, tvec)。至少需要 4 个角点。"""
        if det is None or det.count < 4:
            return None
        obj = det.obj_points.reshape(-1, 1, 3).astype(np.float64)
        img = det.corners.reshape(-1, 1, 2).astype(np.float64)
        # ChArUco/GridBoard 的点全部共面。solvePnPRansac 在只看到局部板时
        # 容易由最小子集选中错误的平面分支；全点 ITERATIVE 对该场景更稳定，
        # 后续由整体重投影误差和跨帧直线轨迹负责拒绝异常位姿。
        ok, rvec, tvec = cv2.solvePnP(
            obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        return rvec, tvec

    def reproj_error(self, det: CharucoDetection, rvec: np.ndarray,
                     tvec: np.ndarray, K: np.ndarray, dist: np.ndarray) -> float:
        """位姿重投影 RMS 误差 (px)。"""
        proj, _ = cv2.projectPoints(det.obj_points, rvec, tvec, K, dist)
        proj = proj.reshape(-1, 2)
        err = np.linalg.norm(proj - det.corners, axis=1)
        return float(np.sqrt(np.mean(err ** 2)))
