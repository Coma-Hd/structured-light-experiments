"""机械臂平稳拖动时的自动连拍。

与 shot（停稳拍一张）不同：空格开始后按时间间隔连续存图，适合慢速平滑拖动。
可选：仅在检测到足够 ChArUco 角点时保存，减少无效帧。

用法（测试包根目录）:
    .\\.venv\\Scripts\\python.exe ceshi/上机械臂/continuous_arm_capture.py \\
      --config ceshi/上机械臂/arm_scan.yaml \\
      --out ceshi/上机械臂/data/scan \\
      --cam 1 --width 800 --height 600 \\
      --interval-ms 250 --countdown 3
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.charuco import CharucoTarget  # noqa: E402
from src.config import CharucoConfig, load_config  # noqa: E402
from src.io_utils import imwrite_unicode  # noqa: E402
from src.laser_center import blue_laser_score  # noqa: E402

AUTO_EXPOSURE_ON = 0.75
AUTO_EXPOSURE_OFF = 0.25


def set_manual_exposure(cap):
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, AUTO_EXPOSURE_OFF)


def set_auto_exposure(cap):
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, AUTO_EXPOSURE_ON)


def _sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main() -> int:
    ap = argparse.ArgumentParser(description="机械臂平稳拖动自动连拍")
    ap.add_argument("--out", required=True, help="保存目录")
    ap.add_argument("--config", default=None)
    ap.add_argument("--cam", type=int, default=1)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--prefix", default="img")
    ap.add_argument("--exposure", type=float, default=-4.0,
                    help="手动曝光值（默认 -4；设置后关闭自动曝光）")
    ap.add_argument("--gain", type=float, default=10.0, help="增益")
    ap.add_argument("--brightness", type=float, default=None, help="亮度(brightness)")
    ap.add_argument(
        "--auto-exposure",
        action="store_true",
        help="强制开启自动曝光（与 --exposure 互斥，优先本项）",
    )
    ap.add_argument(
        "--interval-ms",
        type=float,
        default=250.0,
        help="目标存图间隔毫秒（默认 250 ≈ 4 Hz）",
    )
    ap.add_argument(
        "--countdown",
        type=float,
        default=3.0,
        help="按空格后倒计时秒数，到 0 自动开始连拍；0=立即开始",
    )
    ap.add_argument(
        "--min-corners",
        type=int,
        default=None,
        help="最少角点数才保存；默认读配置 gating.min_charuco_corners",
    )
    ap.add_argument(
        "--require-board",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="仅在板可见且角点足够时保存（默认开；--no-require-board 关闭）",
    )
    ap.add_argument(
        "--min-sharpness",
        type=float,
        default=20.0,
        help="拉普拉斯方差下限，过低视为运动模糊并跳过；0=关闭",
    )
    ap.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="最多保存张数，0=不限制",
    )
    ap.add_argument(
        "--clear-output",
        action="store_true",
        help="开始前清空输出目录中的 img_*.png",
    )
    args = ap.parse_args()

    cfg_path = args.config or os.path.join(_HERE, "arm_scan.yaml")
    cfg = load_config(cfg_path)
    gating = cfg.get("gating", {}) or {}
    min_corners = int(
        args.min_corners
        if args.min_corners is not None
        else gating.get("min_charuco_corners", 8)
    )
    interval_s = max(0.05, float(args.interval_ms) / 1000.0)

    os.makedirs(args.out, exist_ok=True)
    if args.clear_output:
        removed = 0
        for name in os.listdir(args.out):
            if name.startswith(f"{args.prefix}_") and name.lower().endswith(
                (".png", ".jpg", ".jpeg", ".bmp")
            ):
                os.remove(os.path.join(args.out, name))
                removed += 1
        print(f"[清理] 删除旧图 {removed} 张")

    try:
        target = CharucoTarget(CharucoConfig.from_cfg(cfg))
    except Exception as exc:  # noqa: BLE001
        target = None
        print(f"[警告] ChArUco 初始化失败: {exc}")
        if args.require_board:
            raise RuntimeError("require-board 已开启，但检测器不可用") from exc

    cap = cv2.VideoCapture(args.cam)
    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"无法打开摄像头 {args.cam}")
        return 1

    act_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    act_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[分辨率] 实际采集 = {act_w}x{act_h}")
    if args.width > 0 and args.height > 0 and (act_w, act_h) != (args.width, args.height):
        print(
            f"[警告] 请求 {args.width}x{args.height}，实际 {act_w}x{act_h}。"
            " 内参/激光平面/扫描必须同分辨率。"
        )

    def _get(prop, fallback):
        v = cap.get(prop)
        return v if v not in (0.0, -1.0) else fallback

    auto_exposure = True
    if args.auto_exposure:
        set_auto_exposure(cap)
        auto_exposure = True
    else:
        set_manual_exposure(cap)
        cap.set(cv2.CAP_PROP_EXPOSURE, float(args.exposure))
        auto_exposure = False
    if args.gain is not None:
        cap.set(cv2.CAP_PROP_GAIN, float(args.gain))
    if args.brightness is not None:
        cap.set(cv2.CAP_PROP_BRIGHTNESS, float(args.brightness))

    exposure = (
        float(args.exposure)
        if args.exposure is not None
        else _get(cv2.CAP_PROP_EXPOSURE, -4.0)
    )
    gain = (
        float(args.gain)
        if args.gain is not None
        else _get(cv2.CAP_PROP_GAIN, 0.0)
    )
    brightness = (
        float(args.brightness)
        if args.brightness is not None
        else _get(cv2.CAP_PROP_BRIGHTNESS, 128.0)
    )

    print()
    print("[机械臂连续连拍]")
    print(f"  输出目录     : {args.out}")
    print(f"  存图间隔     : {args.interval_ms:.0f} ms")
    print(f"  要求板可见   : {bool(args.require_board)} (min_corners={min_corners})")
    print(f"  模糊阈值     : {args.min_sharpness} (0=关闭)")
    print(
        f"  初始曝光     : AE={'ON' if auto_exposure else 'OFF'}  "
        f"exp={exposure:.1f}  gain={gain:.1f}  bright={brightness:.1f}"
    )
    print("操作:")
    print("  空格 / s  : 开始倒计时连拍 / 停止连拍")
    print("  l         : 激光响应叠加")
    print("  c         : ChArUco 角点叠加")
    print("  a         : 自动曝光 开/关")
    print("  e / d     : 曝光 +/-（会切到手动曝光）")
    print("  r / f     : 增益 +/-")
    print("  b / v     : 亮度 +/-")
    print("  q / ESC   : 退出")
    print("拖动提示: 慢速平稳，保持板与物体激光同时入画；停稳时也可继续录。")
    print()

    saved = 0
    skipped_board = 0
    skipped_blur = 0
    recording = False
    countdown_until = 0.0
    next_save_t = 0.0
    show_laser = False
    show_charuco = target is not None
    detect_interval = 3
    frame_idx = 0
    last_corners = None
    last_count = 0
    last_sharp = 0.0
    last_reason = "idle"

    while True:
        ok, frame = cap.read()
        if not ok:
            print("读取失败")
            break

        now = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if show_charuco and target is not None and frame_idx % detect_interval == 0:
            try:
                det = target.detect(frame)
                if det is not None:
                    last_corners = det.corners
                    last_count = int(det.count)
                else:
                    last_corners = None
                    last_count = 0
            except Exception:  # noqa: BLE001
                last_corners = None
                last_count = 0

        # 倒计时结束 -> 进入录制
        if countdown_until > 0.0 and now >= countdown_until:
            countdown_until = 0.0
            recording = True
            next_save_t = now
            print(f"[REC] 开始连拍 -> {args.out}")

        if recording and now >= next_save_t:
            board_ok = (not args.require_board) or (last_count >= min_corners)
            sharp = _sharpness(gray)
            last_sharp = sharp
            blur_ok = (args.min_sharpness <= 0.0) or (sharp >= args.min_sharpness)
            if not board_ok:
                skipped_board += 1
                last_reason = f"skip_board({last_count})"
            elif not blur_ok:
                skipped_blur += 1
                last_reason = f"skip_blur({sharp:.1f})"
            else:
                name = f"{args.prefix}_{int(now * 1000)}.png"
                path = os.path.join(args.out, name)
                if not imwrite_unicode(path, frame):
                    raise RuntimeError(f"图像保存失败: {path}")
                saved += 1
                last_reason = f"saved#{saved}"
                print(
                    f"  保存 #{saved}: {name}  corners={last_count}  "
                    f"sharp={sharp:.1f}"
                )
                if args.max_frames > 0 and saved >= args.max_frames:
                    print(f"[REC] 已达 max-frames={args.max_frames}，自动停止")
                    recording = False
            next_save_t = now + interval_s

        vis = frame.copy()
        if show_laser:
            laser_cfg = cfg.get("laser", {}) or {}
            score = blue_laser_score(
                frame,
                laser_cfg.get("score_mode", "blue_guided_intensity"),
                blue_gate_threshold=float(laser_cfg.get("blue_gate_threshold", 5.0)),
                blue_gate_expand_px=int(laser_cfg.get("blue_gate_expand_px", 0)),
            )
            heat = cv2.applyColorMap(score.astype("uint8"), cv2.COLORMAP_JET)
            vis = cv2.addWeighted(vis, 0.5, heat, 0.5, 0)

        if show_charuco and last_corners is not None:
            for x, y in last_corners:
                cv2.circle(vis, (int(round(x)), int(round(y))), 4, (0, 0, 255), -1)

        h, w = vis.shape[:2]
        ae_txt = "AUTO" if auto_exposure else "MANUAL"
        cv2.putText(
            vis,
            f"saved={saved}  skip_board={skipped_board}  skip_blur={skipped_blur}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            vis,
            f"AE={ae_txt} exp={exposure:.1f} gain={gain:.1f} bright={brightness:.1f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        corner_color = (0, 255, 0) if last_count >= min_corners else (0, 165, 255)
        cv2.putText(
            vis,
            f"charuco={last_count}/{min_corners}  sharp={last_sharp:.1f}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            corner_color,
            2,
        )
        cv2.putText(
            vis,
            f"{last_reason}  interval={args.interval_ms:.0f}ms",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

        if countdown_until > 0.0:
            remain = max(0.0, countdown_until - now)
            cv2.putText(
                vis,
                f"START IN {remain:.1f}s  keep board visible",
                (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
            )
        elif recording:
            if (frame_idx // 8) % 2 == 0:
                cv2.circle(vis, (w - 40, 40), 14, (0, 0, 255), -1)
            cv2.putText(
                vis,
                "REC  move slowly",
                (w - 260, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
        else:
            cv2.putText(
                vis,
                "SPACE=start continuous  Q=quit",
                (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                2,
            )

        cv2.imshow("arm continuous capture", vis)
        frame_idx += 1

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            if recording:
                print(f"录制结束(退出): 保存 {saved} 张")
            break
        if key == ord("l"):
            show_laser = not show_laser
        elif key == ord("c"):
            show_charuco = (not show_charuco) and target is not None
        elif key == ord("a"):
            auto_exposure = not auto_exposure
            if auto_exposure:
                set_auto_exposure(cap)
                print("自动曝光: 开")
            else:
                set_manual_exposure(cap)
                cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
                print(f"自动曝光: 关, 曝光={exposure:.1f}")
        elif key in (ord("e"), ord("d")):
            if auto_exposure:
                auto_exposure = False
                set_manual_exposure(cap)
                print("已自动切到手动曝光")
            exposure += 1.0 if key == ord("e") else -1.0
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            print(f"曝光={exposure:.1f}")
        elif key in (ord("r"), ord("f")):
            gain += 1.0 if key == ord("r") else -1.0
            gain = max(0.0, gain)
            cap.set(cv2.CAP_PROP_GAIN, gain)
            print(f"增益={gain:.1f}")
        elif key in (ord("b"), ord("v")):
            brightness += 5.0 if key == ord("b") else -5.0
            cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
            print(f"亮度={brightness:.1f}")
        elif key in (ord(" "), ord("s")):
            if recording or countdown_until > 0.0:
                recording = False
                countdown_until = 0.0
                print(
                    f"[REC] 停止: saved={saved}  "
                    f"skip_board={skipped_board}  skip_blur={skipped_blur}"
                )
            else:
                if args.countdown <= 0:
                    recording = True
                    next_save_t = time.time()
                    print(f"[REC] 开始连拍 -> {args.out}")
                else:
                    countdown_until = time.time() + float(args.countdown)
                    print(f"[REC] {args.countdown:.1f}s 后开始，请开始平稳拖动")

    cap.release()
    cv2.destroyAllWindows()
    print()
    print(
        f"[完成] 保存 {saved} 张 -> {args.out}  "
        f"(跳过板不足 {skipped_board}, 模糊 {skipped_blur})"
    )
    if saved == 0:
        print("[警告] 未保存任何图片。检查板是否入画、间隔是否过短、拖动是否过快。")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
