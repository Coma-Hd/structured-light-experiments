"""图像采集小工具（可选）。

用摄像头预览并保存图片到指定目录，用于内参/激光/扫描三种采集。

两种采集模式(用 --mode 选择):
    shot   : 单张拍摄。按一次 空格/s 存一张。用于内参标定(逐个姿态拍)。[默认]
    record : 连续录制。按 空格/s 开始，再按一次结束，录制期间每帧自动保存
             (可用 --every 隔帧保存)。用于激光平面标定 / 扫描。

按键:
    空格 / s : shot 模式=拍一张; record 模式=开始/结束录制
    l        : 切换激光响应叠加显示(蓝激光调试)
    c        : 切换 ChArUco 角点检测叠加(看板识别到多少)
    a        : 切换自动曝光 开/关
    e / d    : 曝光 +/-        (需先关自动曝光)
    r / f    : 增益(gain) +/-
    b / v    : 亮度(brightness) +/-
    q / ESC  : 退出(会自动停止录制)

用法:
    # 内参标定: 单张拍摄
    python scripts/capture.py --out data/intrinsic
    # 激光平面标定: 连续录制
    python scripts/capture.py --out data/laser_plane --mode record
    # 扫描: 连续录制, 每3帧存一张
    python scripts/capture.py --out data/scan --mode record --every 3
    python scripts/capture.py --out data/scan --cam 1 --exposure -5 --gain 1 --mode record
"""
import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CharucoConfig, load_config  # noqa: E402
from src.charuco import CharucoTarget  # noqa: E402
from src.io_utils import imwrite_unicode  # noqa: E402
from src.laser_center import blue_laser_score  # noqa: E402

# 自动曝光属性在不同后端取值不同，OpenCV 常见约定：0.25=手动, 0.75=自动
AUTO_EXPOSURE_ON = 0.75
AUTO_EXPOSURE_OFF = 0.25


def set_manual_exposure(cap):
    """切到手动曝光模式（尽力而为，不同相机/后端支持程度不同）。"""
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, AUTO_EXPOSURE_OFF)


def set_auto_exposure(cap):
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, AUTO_EXPOSURE_ON)


def main():
    ap = argparse.ArgumentParser(description="图像采集")
    ap.add_argument("--out", required=True, help="保存目录")
    ap.add_argument("--cam", type=int, default=0, help="摄像头索引")
    ap.add_argument("--width", type=int, default=0, help="设置采集宽度(可选)")
    ap.add_argument("--height", type=int, default=0, help="设置采集高度(可选)")
    ap.add_argument("--prefix", default="img", help="文件名前缀")
    ap.add_argument("--mode", choices=["shot", "record"], default="shot",
                    help="shot=单张拍摄(空格存一张,内参标定用) | record=连续录制(激光平面/扫描用)")
    ap.add_argument("--every", type=int, default=1,
                    help="录制时每隔几帧保存一张(默认1=每帧都存)")
    ap.add_argument("--exposure", type=float, default=-5.0,
                    help="手动曝光值(默认 -5；设置后关闭自动曝光)")
    ap.add_argument("--gain", type=float, default=1.0, help="增益(gain，默认 1)")
    ap.add_argument("--brightness", type=float, default=None, help="亮度(brightness)")
    ap.add_argument("--auto-exposure", action="store_true",
                    help="强制开启自动曝光(与 --exposure 互斥, 优先本项)")
    ap.add_argument("--config", default=None, help="配置文件，影响 ChArUco 和激光预览")
    args = ap.parse_args()
    cfg = load_config(args.config)

    os.makedirs(args.out, exist_ok=True)
    cap = cv2.VideoCapture(args.cam)
    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"无法打开摄像头 {args.cam}")
        return

    # 回读相机实际给出的分辨率（cap.set 只是请求，相机可能静默回退到别的模式）
    act_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    act_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[分辨率] 实际采集 = {act_w}x{act_h}")
    if args.width > 0 and args.height > 0 and (act_w, act_h) != (args.width, args.height):
        print(f"[警告] 相机未采用请求的 {args.width}x{args.height}，实际回退为 {act_w}x{act_h}！")
        print("       内参标定 / 激光平面标定 / 扫描 必须都用这个实际分辨率，否则重建会出错。")

    # 初始曝光/增益/亮度设置
    auto_exposure = True
    if args.auto_exposure:
        set_auto_exposure(cap)
        auto_exposure = True
    elif args.exposure is not None:
        set_manual_exposure(cap)
        cap.set(cv2.CAP_PROP_EXPOSURE, args.exposure)
        auto_exposure = False
    if args.gain is not None:
        cap.set(cv2.CAP_PROP_GAIN, args.gain)
    if args.brightness is not None:
        cap.set(cv2.CAP_PROP_BRIGHTNESS, args.brightness)

    # 追踪当前数值（相机不一定能回读，回读失败就用我们自己的初值）
    def _get(prop, fallback):
        v = cap.get(prop)
        return v if v not in (0.0, -1.0) else fallback

    exposure = args.exposure if args.exposure is not None else _get(cv2.CAP_PROP_EXPOSURE, -5.0)
    gain = args.gain if args.gain is not None else _get(cv2.CAP_PROP_GAIN, 0.0)
    brightness = args.brightness if args.brightness is not None else _get(cv2.CAP_PROP_BRIGHTNESS, 128.0)

    # ChArUco 检测器（用于实时叠加板识别情况）
    try:
        target = CharucoTarget(CharucoConfig.from_cfg(cfg))
    except Exception as e:  # noqa: BLE001
        target = None
        print(f"[警告] ChArUco 检测器初始化失败, 检测叠加不可用: {e}")

    if args.mode == "shot":
        print("[模式] shot 单张拍摄: 空格/s 存一张")
    else:
        print("[模式] record 连续录制: 空格/s 开始/结束")
    print("空格/s 拍摄或录制 | l 激光叠加 | c 板检测 | a 自动曝光 | e/d 曝光 | r/f 增益 | b/v 亮度 | q/ESC 退出")
    every = max(1, int(args.every))
    show_laser = False
    show_charuco = target is not None
    detect_interval = 5          # 每隔几帧跑一次检测(13MP 检测慢, 降频保流畅)
    frame_idx = 0
    last_corners = None          # 缓存最近一次检测到的角点
    last_count = 0
    count = 0                    # 已保存总张数
    recording = False            # 是否处于连续录制状态
    rec_frame_idx = 0            # 当前录制段内的帧计数(用于隔帧保存)
    rec_saved = 0                # 当前录制段已保存张数
    while True:
        ok, frame = cap.read()
        if not ok:
            print("读取失败")
            break

        # 录制中：按 every 隔帧保存原始帧(保存 frame 而非带叠加的 vis)
        if recording:
            if rec_frame_idx % every == 0:
                name = f"{args.prefix}_{int(time.time()*1000)}.png"
                path = os.path.join(args.out, name)
                if not imwrite_unicode(path, frame):
                    raise RuntimeError(f"图像保存失败: {path}")
                count += 1
                rec_saved += 1
            rec_frame_idx += 1

        vis = frame.copy()
        if show_laser:
            score_mode = cfg.get("laser", {}).get("score_mode", "blue_minus_max")
            blue_gate_threshold = float(
                cfg.get("laser", {}).get("blue_gate_threshold", 20.0)
            )
            blue_gate_expand_px = int(
                cfg.get("laser", {}).get("blue_gate_expand_px", 0)
            )
            score = blue_laser_score(
                frame,
                score_mode,
                blue_gate_threshold=blue_gate_threshold,
                blue_gate_expand_px=blue_gate_expand_px,
            )
            heat = cv2.applyColorMap(score.astype("uint8"), cv2.COLORMAP_JET)
            vis = cv2.addWeighted(vis, 0.5, heat, 0.5, 0)

        if show_charuco and target is not None:
            if frame_idx % detect_interval == 0:
                try:
                    det = target.detect(frame)
                    if det is not None:
                        last_corners = det.corners
                        last_count = det.count
                    else:
                        last_corners = None
                        last_count = 0
                except Exception:  # noqa: BLE001
                    last_corners = None
                    last_count = 0
            if last_corners is not None:
                for (x, y) in last_corners:
                    cv2.circle(vis, (int(round(x)), int(round(y))), 5,
                               (0, 0, 255), -1)

        ae_txt = "AUTO" if auto_exposure else "MANUAL"
        cv2.putText(vis, f"saved={count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if recording:
            # 红点 + REC 提示，闪烁便于分辨
            if (frame_idx // 8) % 2 == 0:
                cv2.circle(vis, (vis.shape[1] - 40, 40), 14, (0, 0, 255), -1)
            cv2.putText(vis, f"REC {rec_saved}", (vis.shape[1] - 200, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(vis,
                    f"AE={ae_txt} exp={exposure:.1f} gain={gain:.1f} bright={brightness:.1f}",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if show_charuco:
            ch_color = (0, 255, 0) if last_count >= 12 else (0, 165, 255)
            cv2.putText(vis, f"charuco corners={last_count}", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, ch_color, 2)
        cv2.imshow("capture", vis)
        frame_idx += 1

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            if recording:
                print(f"录制结束(退出): 本段保存 {rec_saved} 张")
            break
        elif key == ord("l"):
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
            if args.mode == "shot":
                # 单张拍摄: 存当前原始帧(不带叠加)
                name = f"{args.prefix}_{int(time.time()*1000)}.png"
                path = os.path.join(args.out, name)
                if not imwrite_unicode(path, frame):
                    raise RuntimeError(f"图像保存失败: {path}")
                count += 1
                print(f"拍摄 #{count}: {name}")
            else:
                recording = not recording
                if recording:
                    rec_frame_idx = 0
                    rec_saved = 0
                    print(f"[REC] start -> {args.out} (every {every} frames)")
                else:
                    print(f"[REC] stop: saved {rec_saved} this segment (total {count})")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
