"""诊断工具：为什么 ChArUco 角点识别为 0。

对一张图片：
1. 遍历所有 ArUco 字典，用 detectMarkers 看各能认出多少个标记 → 定位正确字典；
2. 用当前 config 的板参数，分别按「新排布 / 旧版legacy排布」跑 ChArUco 角点插值，看各得到多少角点；
3. 可选保存可视化(画出检测到的标记)。

用法:
    python scripts/debug_charuco.py --image data/scan/xxx.png
    python scripts/debug_charuco.py --image data/scan/xxx.png --save out_vis.png
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CharucoConfig, load_config  # noqa: E402
from src.charuco import _create_board, _get_dictionary  # noqa: E402


def _all_dict_names():
    return sorted(n for n in dir(cv2.aruco)
                  if n.startswith("DICT_") and isinstance(getattr(cv2.aruco, n), int))


def _get_dictionary_by_name(name):
    dic_id = getattr(cv2.aruco, name)
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(dic_id)
    return cv2.aruco.Dictionary_get(dic_id)


def _detect_markers(gray, dictionary):
    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    n = 0 if ids is None else len(ids)
    return corners, ids, n


def main():
    ap = argparse.ArgumentParser(description="ChArUco 检测诊断")
    ap.add_argument("--image", required=True)
    ap.add_argument("--save", default=None, help="保存标记可视化图")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"读不了图片: {args.image}")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"图片尺寸: {img.shape[1]}x{img.shape[0]}")

    cfg = load_config()
    cc = CharucoConfig.from_cfg(cfg)
    print(f"\n当前 config: squares={cc.squares_x}x{cc.squares_y}, "
          f"square={cc.square_length}, marker={cc.marker_length}, dict={cc.dictionary}")

    # ---- 1. 遍历所有字典 ----
    print("\n=== 各字典能检测到的标记数 (只列出 >0 的) ===")
    results = []
    for name in _all_dict_names():
        try:
            d = _get_dictionary_by_name(name)
            _, _, n = _detect_markers(gray, d)
            if n > 0:
                results.append((n, name))
        except Exception:
            pass
    if not results:
        print("  所有字典都检测不到任何标记！")
        print("  → 多半是: 图像模糊/过曝、标记太小/太斜、或这不是标准 ArUco 板。")
    else:
        for n, name in sorted(results, reverse=True):
            flag = "  <== 当前config" if name == cc.dictionary else ""
            print(f"  {name:20s} : {n} 个{flag}")

    # ---- 2. 用当前板参数跑 ChArUco 插值(新/旧排布) ----
    dictionary = _get_dictionary(cc.dictionary)

    # 先看用「当前字典」检测到的标记 ID 范围（判断和板生成的 id 是否吻合）
    m_corners, m_ids, _n = _detect_markers(gray, dictionary)
    if m_ids is not None and len(m_ids) > 0:
        ids_flat = np.asarray(m_ids).flatten()
        print(f"\n检测到标记 {len(ids_flat)} 个, id 范围 [{ids_flat.min()}, {ids_flat.max()}]")
        print(f"  id 列表: {sorted(ids_flat.tolist())}")
        expect_max = (cc.squares_x * cc.squares_y) // 2 - 1
        print(f"  12x9 板期望 id 应在 [0, {expect_max}] 内")

    print("\n=== detectBoard(新API) 角点数 ===")
    if hasattr(cv2.aruco, "CharucoDetector"):
        for legacy in (False, True):
            try:
                board = _create_board(cc, dictionary)
                if hasattr(board, "setLegacyPattern"):
                    board.setLegacyPattern(legacy)
                elif legacy:
                    continue
                det = cv2.aruco.CharucoDetector(board)
                ch_c, ch_i, _, _ = det.detectBoard(gray)
                n = 0 if ch_i is None else len(ch_i)
                print(f"  legacy={legacy}: {n} 个角点")
            except Exception as e:  # noqa: BLE001
                print(f"  legacy={legacy}: 出错 {e}")
    else:
        print("  当前 OpenCV 无 CharucoDetector (旧版 API)")

    # ---- 2b. 老接口 interpolateCornersCharuco(常更稳) ----
    print("\n=== interpolateCornersCharuco(老API) 角点数 ===")
    if hasattr(cv2.aruco, "interpolateCornersCharuco") and m_ids is not None and len(m_ids) > 0:
        for legacy in (False, True):
            try:
                board = _create_board(cc, dictionary)
                if hasattr(board, "setLegacyPattern"):
                    board.setLegacyPattern(legacy)
                elif legacy:
                    continue
                retval, ch_c, ch_i = cv2.aruco.interpolateCornersCharuco(
                    m_corners, m_ids, gray, board)
                n = 0 if ch_i is None else len(ch_i)
                print(f"  legacy={legacy}: {n} 个角点 (retval={retval})")
            except Exception as e:  # noqa: BLE001
                print(f"  legacy={legacy}: 出错 {e}")
    else:
        print("  该 OpenCV 无 interpolateCornersCharuco 或无标记")

    # ---- 3. 可视化 ----
    if args.save and results:
        best_name = sorted(results, reverse=True)[0][1]
        d = _get_dictionary_by_name(best_name)
        corners, ids, _ = _detect_markers(gray, d)
        vis = img.copy()
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)
        cv2.imwrite(args.save, vis)
        print(f"\n已保存(用字典 {best_name} 画标记): {args.save}")


if __name__ == "__main__":
    main()
