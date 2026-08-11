"""一次框选物体，自动生成整段扫描的逐帧动态掩码。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, resolve_path  # noqa: E402
from src.io_utils import imread_color, imwrite_unicode  # noqa: E402
from src.object_mask import (  # noqa: E402
    cleanup_object_mask,
    dilate_mask,
    grabcut_from_rectangle,
    track_and_refine_mask,
)


def list_images(directory: str) -> List[str]:
    names = sorted(
        name for name in os.listdir(directory)
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
    return [os.path.join(directory, name) for name in names]


def overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = image.copy()
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    return result


def select_reference_mask(
    image: np.ndarray,
    title: str,
    grabcut_iterations: int,
    morphology_px: int,
    smoothing_px: int,
    brush_px: int,
) -> np.ndarray:
    while True:
        rect = cv2.selectROI(
            title, image, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(title)
        x, y, width, height = [int(value) for value in rect]
        if width < 5 or height < 5:
            raise KeyboardInterrupt
        mask = grabcut_from_rectangle(
            image, (x, y, x + width, y + height),
            iterations=grabcut_iterations,
            morphology_px=morphology_px,
            smoothing_px=smoothing_px,
        )
        editing = {"mask": mask.copy(), "button": None}

        def redraw() -> None:
            preview = overlay_mask(image, editing["mask"])
            tip = (
                "L-drag=add  R-drag=erase  ENTER=accept  "
                "R key=redraw box  Q=quit"
            )
            cv2.putText(
                preview, tip, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                preview, tip, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (0, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow("object mask preview", preview)

        def paint(event, px, py, flags, _param) -> None:
            if event == cv2.EVENT_LBUTTONDOWN:
                editing["button"] = "add"
            elif event == cv2.EVENT_RBUTTONDOWN:
                editing["button"] = "erase"
            elif event in (
                cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP
            ):
                editing["button"] = None
                return
            if (
                event == cv2.EVENT_MOUSEMOVE
                and editing["button"] is None
            ):
                return
            value = 1 if editing["button"] == "add" else 0
            if editing["button"] is not None:
                cv2.circle(
                    editing["mask"], (int(px), int(py)),
                    max(1, int(brush_px)), int(value), -1)
                redraw()

        cv2.namedWindow("object mask preview", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("object mask preview", paint)
        redraw()
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 32):
                cv2.destroyWindow("object mask preview")
                return cleanup_object_mask(
                    editing["mask"],
                    morphology_px=morphology_px,
                    smoothing_px=smoothing_px,
                )
            if key in (ord("r"), ord("R")):
                cv2.destroyWindow("object mask preview")
                break
            if key in (ord("q"), ord("Q"), 27):
                cv2.destroyWindow("object mask preview")
                raise KeyboardInterrupt


def save_mask(path: str, mask: np.ndarray) -> None:
    if not imwrite_unicode(path, (mask > 0).astype(np.uint8) * 255):
        raise RuntimeError(f"保存物体掩码失败: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="参考帧框选一次物体并自动跟踪逐帧掩码")
    parser.add_argument("--config", required=True)
    parser.add_argument("--images", default=None)
    parser.add_argument("--out", default=None, help="掩码 manifest JSON")
    parser.add_argument("--reference-index", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    image_dir = args.images or resolve_path(
        cfg, cfg["paths"]["scan_images"])
    setting = (cfg.get("laser") or {}).get("object_mask") or {}
    out_manifest = args.out or str(setting.get("manifest", "")).strip()
    if not out_manifest:
        raise ValueError("缺少 laser.object_mask.manifest")
    if not os.path.isabs(out_manifest):
        project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
        out_manifest = os.path.join(project_root, out_manifest)
    out_manifest = os.path.abspath(out_manifest)
    mask_dir = os.path.join(os.path.dirname(out_manifest), "masks")
    os.makedirs(mask_dir, exist_ok=True)

    paths = list_images(image_dir)
    if not paths:
        raise FileNotFoundError(f"扫描目录没有图像: {image_dir}")
    reference_index = (
        int(args.reference_index)
        if args.reference_index is not None
        else len(paths) // 2
    )
    reference_index = max(0, min(reference_index, len(paths) - 1))
    grabcut_iterations = int(setting.get("grabcut_iterations", 3))
    tracking_grabcut_iterations = int(
        setting.get("tracking_grabcut_iterations", 1))
    margin_px = int(setting.get("dilate_px", 8))
    morphology_px = int(setting.get("morphology_px", 7))
    smoothing_px = int(setting.get("smoothing_px", 9))
    correction_brush_px = int(
        setting.get("correction_brush_px", 8))
    min_tracking_ratio = float(
        setting.get("min_tracking_inlier_ratio", 0.30))
    min_tracking_inliers = int(
        setting.get("min_tracking_inlier_count", 20))
    max_area_change = float(setting.get("max_area_change_ratio", 0.35))

    reference_image = imread_color(paths[reference_index])
    if reference_image is None:
        raise RuntimeError(f"读不了参考帧: {paths[reference_index]}")
    print(
        f"参考帧 [{reference_index}/{len(paths)-1}]: "
        f"{os.path.basename(paths[reference_index])}")
    print("请用矩形完整框住物体，尽量少包含标定板；随后确认绿色轮廓。")
    try:
        reference_mask = select_reference_mask(
            reference_image,
            "drag one box around the complete object",
            grabcut_iterations,
            morphology_px,
            smoothing_px,
            correction_brush_px,
        )
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        raise SystemExit("用户取消，未生成物体掩码")
    cv2.destroyAllWindows()

    raw_masks: Dict[int, np.ndarray] = {reference_index: reference_mask}
    confidences: Dict[int, float] = {reference_index: 1.0}
    inlier_counts: Dict[int, int] = {reference_index: 0}

    def track_range(indices: List[int], previous_index: int) -> None:
        previous_image = imread_color(paths[previous_index])
        previous_mask = raw_masks[previous_index]
        assert previous_image is not None
        for sequence_number, index in enumerate(indices, start=1):
            current_image = imread_color(paths[index])
            if current_image is None:
                raise RuntimeError(f"读不了扫描图: {paths[index]}")
            # 关闭逐帧 GrabCut 时，所有帧都直接相对人工确认的参考帧跟踪。
            # 相邻帧位移小于 0.5 px 时，逐帧二值掩码仿射会反复舍入而不移动；
            # 直接参考帧变换既保留累计亚像素位移，也不会逐帧吸入棋盘区域。
            tracking_image = (
                reference_image
                if tracking_grabcut_iterations <= 0
                else previous_image
            )
            tracking_mask = (
                reference_mask
                if tracking_grabcut_iterations <= 0
                else previous_mask
            )
            mask, confidence, inlier_count = track_and_refine_mask(
                tracking_image,
                current_image,
                tracking_mask,
                grabcut_iterations=tracking_grabcut_iterations,
                morphology_px=morphology_px,
                smoothing_px=smoothing_px,
            )
            old_area = max(int(previous_mask.sum()), 1)
            area_change = abs(int(mask.sum()) - old_area) / old_area
            if (
                confidence < min_tracking_ratio
                and inlier_count < min_tracking_inliers
            ):
                raise RuntimeError(
                    f"物体掩码跟踪置信度过低: "
                    f"{os.path.basename(paths[index])}, "
                    f"inlier={confidence:.3f}, "
                    f"inlier_count={inlier_count}")
            if area_change > max_area_change:
                raise RuntimeError(
                    f"物体掩码面积突变: "
                    f"{os.path.basename(paths[index])}, "
                    f"change={area_change*100:.1f}%")
            raw_masks[index] = mask
            confidences[index] = confidence
            inlier_counts[index] = inlier_count
            previous_image = current_image
            previous_mask = mask
            if sequence_number % 50 == 0 or index in (0, len(paths) - 1):
                print(
                    f"  跟踪 {os.path.basename(paths[index])}: "
                    f"inlier={confidence:.3f}, "
                    f"inlier_count={inlier_count}")

    track_range(
        list(range(reference_index + 1, len(paths))), reference_index)
    track_range(
        list(range(reference_index - 1, -1, -1)), reference_index)

    frames: Dict[str, Dict] = {}
    for index, path in enumerate(paths):
        extraction_mask = dilate_mask(raw_masks[index], margin_px)
        mask_name = os.path.splitext(os.path.basename(path))[0] + ".png"
        mask_path = os.path.join(mask_dir, mask_name)
        save_mask(mask_path, extraction_mask)
        image_stat = os.stat(path)
        frames[os.path.basename(path)] = {
            "mask": os.path.relpath(
                mask_path, os.path.dirname(out_manifest)).replace("\\", "/"),
            "tracking_inlier_ratio": float(confidences[index]),
            "tracking_inlier_count": int(inlier_counts[index]),
            "area_px": int(extraction_mask.sum()),
            "reference": bool(index == reference_index),
            "image_size_bytes": int(image_stat.st_size),
            "image_mtime_ns": int(image_stat.st_mtime_ns),
        }

    payload = {
        "version": 1,
        "method": "grabcut_reference_lk_affine_grabcut_tracking",
        "image_dir": os.path.abspath(image_dir),
        "reference_image": os.path.basename(paths[reference_index]),
        "dilate_px": margin_px,
        "frames": frames,
    }
    with open(out_manifest, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    preview_path = os.path.join(
        os.path.dirname(out_manifest), "object_mask_reference.png")
    imwrite_unicode(
        preview_path,
        overlay_mask(reference_image, dilate_mask(
            reference_mask, margin_px)),
    )
    print(f"物体掩码完成: {len(frames)} 帧")
    print(f"清单: {out_manifest}")
    print(f"参考预览: {preview_path}")


if __name__ == "__main__":
    main()
