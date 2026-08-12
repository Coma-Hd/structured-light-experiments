# 线激光 3D 重建 · 代码与操作文档

基于 **450nm 蓝色线激光器 + RGB 相机 + ChArUco 标定板** 的线结构光三维重建。

本文档既讲清每个脚本/模块做什么、输入输出是什么，也给出**从标定到出模型的完整操作命令**。照着[从零到出模型的完整操作](#从零到出模型的完整操作)一节走一遍即可跑通。

## 新电脑快速安装

系统要求：Windows 10/11、64 位 Python 3.10 或 3.11、Git LFS，以及可正常工作的 USB 相机驱动。

```powershell
git clone https://github.com/Coma-Hd/structured-light-experiments.git
cd structured-light-experiments
powershell -ExecutionPolicy Bypass -File .\install.ps1 -Action setup
```

安装脚本会创建 `.venv`、安装 `requirements.txt`、下载 LFS 点云并检查主程序。之后先阅读 [`ceshi/导轨/导轨.md`](ceshi/导轨/导轨.md)，再按顺序运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\ceshi\导轨\0_capture.ps1 -ClearOutput
powershell -ExecutionPolicy Bypass -File .\ceshi\导轨\1_draw_check.ps1
powershell -ExecutionPolicy Bypass -File .\ceshi\导轨\2_rebuild.ps1
```

四面拼接流程位于 [`ceshi/rail/two_faces`](ceshi/rail/two_faces/)，完整采集、重建与拼接顺序见 [`拼接README.md`](ceshi/rail/two_faces/拼接README.md)。相关 PowerShell 脚本同样会自动使用 `.venv`，四面流程使用根目录 `output/` 中的专用标定文件。

仅更换电脑且相机、镜头、激光、支架、分辨率、曝光和增益完全不变时，可以先验证现有标定。任何光机位置或成像参数变化，都必须按对应平台文档重新标定。

---


## 实验三大类

仓库实验区已整理为三类平台，详见 [ceshi/README.md](ceshi/README.md)：

- **导轨**：`ceshi/导轨`
- **转台**：`ceshi/转台`
- **手动机械臂**：`ceshi/手动机械臂`

过程文档：`ceshi/排错与优化文档`。`data/` 中的采集图片不纳入版本库。

## 目录

- [一句话原理](#一句话原理)
- [端到端数据流](#端到端数据流)
- [坐标系与数学约定](#坐标系与数学约定)
- [目录结构](#目录结构)
- [脚本逐个详解与操作命令](#脚本逐个详解与操作命令)
- [核心库逐模块讲解](#核心库逐模块讲解)
- [config.yaml 参数说明](#configyaml-参数说明)
- [ROI 裁剪怎么设](#roi-裁剪怎么设)
- [多视角融合出完整模型](#多视角融合出完整模型)
- [从零到出模型的完整操作](#从零到出模型的完整操作)
- [文件格式约定](#文件格式约定)
- [常见问题与红线](#常见问题与红线)
- [现场操作速查](#现场操作速查)

---

## 一句话原理

相机和线激光**刚性固定**后，激光在空间中形成一个**固定的光平面**。相机每帧只能拍到「激光光平面 ∩ 工件表面」的**一条 3D 轮廓线**。

没有导轨，所以用 **ChArUco 板的视觉位姿代替位移传感器**：把工件和标定板一起移动，每帧反推出标定板位姿，就能把每帧的激光轮廓统一到同一个「板坐标系」里，累积成完整点云。

```
         相机 ───→ 看
           \
            \   激光光平面（固定不动）
             | /
             |/________ 工件截面（每帧只得到这一条线上的点）
             │
        移动板+工件，让工件不同区域依次穿过激光平面
```

---

## 端到端数据流

```
        ┌────────────── 离线标定（各做一次，动了相机/激光/分辨率才重做）──────────────┐

 data/intrinsic/*.png ──[1_calibrate_intrinsic]──► output/camera_intrinsic.yaml  (K, dist)
                                        │
                                        ▼ 用到 K,dist
 data/laser_plane/*.png ─[2_calibrate_laser_plane]─► output/laser_plane.yaml    (a,b,c,d)

        ┌────────────── 在线扫描重建（每次扫工件都跑）──────────────┐

 data/scan/*.png ──[3_reconstruct]──► output/cloud.ply     (板坐标系点云)
                                │ 用到 K,dist,激光平面
                                ▼
                  [4_postprocess] ──► output/cloud_clean.ply + height_map.npy/.png
                                ▼
                  [ply_to_obj]   ──► output/mesh.obj   (点云转网格)

        ┌────────────── 多视角（可选，出完整模型）──────────────┐

 多个朝向的 scan 目录 ──[5_merge_views 或手动拼接]──► 合并点云 ──► mesh.obj
```

**内参**和**激光平面**是一次性标定产物；之后每次扫描都复用它们。**三者(内参/激光平面/扫描)的采集分辨率必须完全一致**，否则重建全错（见[红线](#常见问题与红线)）。

---

## 坐标系与数学约定

所有几何函数在 `src/geometry.py`。理解这几条就通了：

**相机坐标系**：光心在原点，光轴 +Z 朝前，单位统一为**米(m)**。

**板 → 相机 位姿** `(R, t)`（`solvePnP` 得到）：`P_cam = R · P_board + t`

**平面表示**：`[a,b,c,d]`，满足 `a·x + b·y + c·z + d = 0`，法向 `n=(a,b,c)` 已单位化。

**像素 → 相机射线**（`pixels_to_rays`）：`undistortPoints` 去畸变得归一化坐标 `(xn,yn)`，射线方向 `dir=(xn,yn,1)`。

**射线 ∩ 平面**（`ray_plane_intersect_masked`）：`s = -d/(n·dir)`，`P_cam = s·dir`（`s>0` 且分母非 0 才有效）。

**相机系 → 板坐标系**（`transform_cam_to_board`，拼接核心）：`P_board = Rᵀ·(P_cam − t)`

> 无论板怎么移动，每帧的点经过这一步都回到**同一个板坐标系**，于是自然对齐——这就是无导轨也能拼点云、也能多视角融合的原因。

---


## 实验三大类

仓库实验区已整理为三类平台，详见 [ceshi/README.md](ceshi/README.md)：

- **导轨**：`ceshi/导轨`
- **转台**：`ceshi/转台`
- **手动机械臂**：`ceshi/手动机械臂`

过程文档：`ceshi/排错与优化文档`。`data/` 中的采集图片不纳入版本库。

## 目录结构

```
Sturctured light/
├── config.yaml                     # 全局参数（唯一常改的文件）
├── requirements.txt
├── src/                            # 核心库（算法都在这里）
│   ├── config.py                   # 读 config.yaml
│   ├── io_utils.py                 # 内参/激光平面 yaml 读写
│   ├── charuco.py                  # ChArUco 检测 + 位姿估计（兼容 OpenCV 新旧 API）
│   ├── laser_center.py             # 蓝激光中心线提取
│   ├── geometry.py                 # 所有几何数学（反投影/求交/拟合/变换）
│   ├── calib_intrinsic.py          # 内参标定
│   ├── calib_laser_plane.py        # 激光平面标定
│   ├── reconstruct.py              # 逐帧重建融合 + PLY 读写
│   ├── multi_view.py               # 多朝向重建与融合(charuco 直拼 / ICP)
│   └── postprocess.py              # 滤波/ROI/高度图/mesh
├── scripts/                        # 命令行入口（薄封装，调用 src）
│   ├── generate_board.py           # 生成 ChArUco 板打印图
│   ├── capture.py                  # 摄像头采集（shot 单张 / record 录制）
│   ├── debug_laser.py              # 激光提取可视化自检
│   ├── debug_charuco.py            # ChArUco 检测自检
│   ├── 1_calibrate_intrinsic.py    # ① 内参标定
│   ├── 2_calibrate_laser_plane.py  # ② 激光平面标定
│   ├── 3_reconstruct.py            # ③ 逐帧重建
│   ├── 4_postprocess.py            # ④ 后处理（清理+高度图）
│   ├── 5_merge_views.py            # ⑤ 多朝向融合（可选）
│   └── ply_to_obj.py               # 点云转网格（bpa/poisson/planes + 简化 + 去噪）
├── data/{intrinsic,laser_plane,scan}/   # 三类输入图片
└── output/                         # 所有产物
```

设计原则：**`scripts/` 只做参数解析和路径拼接，逻辑全在 `src/`**。

---

## 脚本逐个详解与操作命令

> 通用：`--cam` 相机索引，`--width/--height` 分辨率，`--exposure/--gain/--brightness` 相机参数。所有脚本支持 `--help`。

### `generate_board.py` — 生成标定板打印图（一次）

生成 ChArUco 板 PNG，打印后**用卡尺复核实际边长**并回填 `config.yaml`。

```bash
python scripts/generate_board.py --out output/charuco_board.png --dpi 300
```

### `capture.py` — 摄像头采集（两种模式）

预览画面并保存图片。**关键是 `--mode`**：

| 模式 | 空格/s 行为 | 用途 |
|------|------------|------|
| `shot`（默认） | 按一次存一张原始帧 | **内参标定**（逐姿态单拍） |
| `record` | 按一次开始录制、再按结束，期间每帧自动存（配 `--every` 隔帧） | **激光平面标定、扫描** |

窗口内按键：`空格/s` 拍摄或录制 · `l` 激光响应叠加 · `c` ChArUco 角点叠加 · `a` 自动曝光开关 · `e/d` 曝光± · `r/f` 增益± · `b/v` 亮度± · `q/ESC` 退出。

```bash
# 内参：单张拍摄
python scripts/capture.py --out data/intrinsic --cam 1 --width 800 --height 600 --exposure -4 --gain 20
# 激光平面：录制
python scripts/capture.py --out data/laser_plane --mode record --cam 1 --width 800 --height 600 --exposure -4 --gain 20
# 扫描：录制，每3帧存一张
python scripts/capture.py --out data/scan --mode record --every 3 --cam 1 --width 800 --height 600 --exposure -4 --gain 20
```

> `shot` 模式存的是**原始帧**（不含 `l`/`c` 叠加），标定安全；只有 `record` 才会自动保存。

### `debug_laser.py` / `debug_charuco.py` — 自检

- `debug_laser.py`：可视化激光中心线提取效果，用来定工作距离、调激光阈值。
  ```bash
  python scripts/debug_laser.py --image data/scan/xxx.png
  ```
- `debug_charuco.py`：排查 ChArUco 检测（识别到多少角点、位姿是否稳）。

### `1_calibrate_intrinsic.py` — ① 相机内参标定

读 `data/intrinsic/*.png`，每张检测 ChArUco 角点 → `cv2.calibrateCamera` → 保存 `output/camera_intrinsic.yaml`。**验收：重投影误差 < 1px（越小越好，理想 < 0.5px）。**

```bash
python scripts/1_calibrate_intrinsic.py
```

### `2_calibrate_laser_plane.py` — ② 激光平面标定

读 `data/laser_plane/*.png`（每张须同时含 ChArUco + 激光线），用内参解板位姿得板平面 → 激光射线 ∩ 板平面得 3D 激光点 → RANSAC 拟合出**激光平面**，存 `output/laser_plane.yaml`。**验收：拟合 RMS ≈ 0.1–0.5mm。**

```bash
python scripts/2_calibrate_laser_plane.py
```

> 改了内参（或分辨率）后**必须重跑本步**；若激光平面图分辨率没变，不用重拍，直接重跑即可。

### `3_reconstruct.py` — ③ 逐帧重建

读 `data/scan/*.png` + 内参 + 激光平面，逐帧：解位姿(质量门控) → 提激光中心 → 射线 ∩ 激光平面得相机系点 → 变换到板坐标系 → 累积，输出 `output/cloud.ply`。跳过位姿误差过大的帧。

```bash
python scripts/3_reconstruct.py
```

### `4_postprocess.py` — ④ 后处理

读 `cloud.ply`：ROI 裁剪（按 `config.yaml` 的 `roi` 段）→ 体素降采样 → 统计滤波(SOR) → 输出 `cloud_clean.ply`；再栅格化出 `height_map.npy/.png`（高度图对离群点鲁棒，会用分位数界定边界，不会因坏点 OOM）。

```bash
python scripts/4_postprocess.py
```

### `ply_to_obj.py` — 点云转网格（重点）

把 PLY 转成 OBJ 网格，集成了 **ROI 裁剪 → 去噪 → 网格化 → 简化** 一条龙。

**三种网格算法 `--method`：**
- `bpa`（默认）：Ball Pivoting，贴合表面。
- `poisson`：泊松重建，适合较封闭的实体。
- `planes`：**RANSAC 分平面**，每个平面把内点投影到平面(去噪变平)后三角化——适合方块/台阶这类由平面构成的工件，出来的面最“硬”。

**去噪（建网格前）：**
- 统计滤波 `--sor-nb 20 --sor-std 2.0`（`--sor-std` 调小更激进）。
- DBSCAN 聚类去噪 `--dbscan`（默认开，`--no-dbscan` 关）：`--cluster-eps 0.004` 邻域、`--keep-clusters 1` 只留最大簇。**单视角**留 1 可自动甩掉板面/飞点；**多视角**要调大到覆盖各面的簇数或直接 `--no-dbscan`。

**简化 `--simplify N`：** 目标面数，二次误差简化把共面小三角合并成大面片。

**其它：** `--no-roi` 不按 config 裁剪（输入已是干净点云时用）；`--plane-dist/--plane-min/--max-planes/--plane-alpha` 调 `planes` 方法。

```bash
# 默认：cloud.ply → ROI 裁剪 → BPA → mesh.obj
python scripts/ply_to_obj.py

# 干净点云 + BPA + 简化到 1500 面
python scripts/ply_to_obj.py --in output/cloud_clean.ply --out output/mesh.obj --method bpa --no-roi --simplify 1500

# RANSAC 分平面 + 简化（方块类工件推荐）
python scripts/ply_to_obj.py --in output/cloud_clean.ply --out output/mesh.obj --method planes --no-roi --simplify 1200

# 关掉聚类去噪
python scripts/ply_to_obj.py --in output/cloud_clean.ply --out output/mesh.obj --method bpa --no-roi --no-dbscan
```

### `5_merge_views.py` — ⑤ 多朝向融合（可选）

对多个朝向的扫描目录各自重建到板坐标系再融合。`--method charuco` 共板坐标系直接拼接（推荐，免配准）；`--method icp` 跨坐标系时用 ICP。

```bash
python scripts/5_merge_views.py --views data/scan_v0 data/scan_v1 data/scan_v2 data/scan_v3
python scripts/5_merge_views.py --views data/front data/back --method icp --mesh poisson
```

---

## 核心库逐模块讲解

### `charuco.py` — 标定板检测与位姿
`CharucoTarget.detect(image)` → 亚像素角点 + 3D 板坐标；`estimate_pose` 用 `solvePnP` 求板→相机位姿；`reproj_error` 供帧质量门控。**自动适配 OpenCV 新旧 API**（新版 `CharucoDetector`，旧版回退），并对非标准板布局用单应残差自动锁定正确的 marker 布局。

### `laser_center.py` — 蓝激光中心线提取
`blue_laser_score(bgr)`：`score = B − max(R,G)` 凸显蓝激光。`extract_laser_centers`：`centroid`（逐列/行亮度质心，快）或 `steger`（Hessian 亚像素脊线，精度高）。

### `geometry.py` — 几何数学核心
`pixels_to_rays` / `ray_plane_intersect_masked` / `board_plane_in_camera` / `transform_cam_to_board` / `fit_plane_lstsq` / `fit_plane_ransac` / `plane_point_distance`。

### `reconstruct.py` — 逐帧重建融合
含 `reconstruct(...)` 与纯 numpy 的 `write_ply/read_ply_xyz`（ASCII PLY）。

### `postprocess.py` — 后处理
`roi_mask/crop_roi_points`（ROI 裁剪）、`process_point_cloud`（降采样+SOR+法向，需 open3d）、`to_height_map/save_height_map`（栅格化高度图，纯 numpy，对离群点鲁棒）、`reconstruct_mesh`（Poisson/BPA）。

### `multi_view.py` — 多视角
`reconstruct_views` 逐目录重建；`merge_views` 用 charuco 直拼或 FPFH+RANSAC+ICP 融合。

---

## config.yaml 参数说明

| 参数 | 作用 | 何时改 |
|------|------|--------|
| `charuco.*` | 方格数/边长/字典 | **必改**，与实物一致，否则位姿全错 |
| `laser.method` | `centroid`/`steger` | 要更高精度切 `steger` |
| `laser.score_threshold` | 蓝激光响应阈值 | 环境光强/误检多→调高；浅色工件抓不到激光→调低 |
| `laser.min_intensity` | 单列/行最小累计强度 | 同上 |
| `laser.scan_axis` | `column`/`row` | 激光近水平用 `column`，近竖直用 `row` |
| `laser.steger_sigma` | Steger 高斯尺度 | ≈ 激光线半宽(像素) |
| `plane_fit.ransac_threshold` | 激光平面内点阈值(m) | 拟合噪声大时调 |
| `gating.max_reproj_error` | 位姿门控(px) | 重影明显→调小更严格 |
| `roi.*` | ROI 裁剪范围（板坐标系,米） | 见下节 |
| `postprocess.voxel_size` | 体素降采样(m) | 点太密/太稀时调 |
| `postprocess.height_map_res` | 高度图分辨率(m/px) | 按精度需求 |
| `postprocess.mesh_method` | `none`/`poisson`/`bpa` | 第一版建议 `none`，先看高度图 |

> 所有物理长度默认单位**米(m)**：20mm 写 `0.02`。

---

## ROI 裁剪怎么设

ROI 在 `config.yaml` 的 `roi` 段，全部是**标定板坐标系、单位米**：

- **x / y**：沿板面的两条边——“在板面上框哪块矩形”。范围约 0~0.40 / 0~0.30（400×300 板）。
- **z**：**垂直板面的方向，板面恒在 z=0**。`z_min ≤ z ≤ z_max` 之间的点才保留，`null`=该方向不限制。

常见摆法：
- **板竖直在后、工件在板前**（当前）：工件在 **z<0** 侧。用 `z_max`（如 `-0.02`）切掉板面(z≈0)；用 `z_min`（如 `-0.115`）切掉**比工件更靠近相机**的悬空噪点（z 越负=越靠近相机）。
- **板平放、工件坐在板上**：工件在 **z>0** 侧，改用 `z_min`（如 `0.002`）切板面。

> 判断阈值别靠猜：跑一次重建后看 `cloud.ply` 的 z 直方图，把阈值切在“板面层”和“工件层”之间的空档。也可以干脆用 `ply_to_obj.py` 的 DBSCAN 聚类去噪代替 z 阈值来剥离板面。

---

## 多视角融合出完整模型

单个视角只能拿到工件朝向相机的那一两个面。要更完整，转动**工件+板（一起转，保持相对固定）**换几个朝向各扫一组，靠板坐标系自动对齐后拼接。

**已验证流程（三视角：正面/前左/前右）：**
1. 各朝向分别采集到独立目录并重建、后处理，得到各自的 `cloud_clean.ply`。
2. 各视角统一裁掉板面与深噪（如只留 `-0.145 ≤ z ≤ -0.02`）+ SOR，然后**直接拼接**（`merged = A + B + C`）。多视角拼接**不要用“只留最大簇”**，否则会删掉其它视角。
3. 对合并点云跑 `ply_to_obj.py --method planes --no-dbscan --simplify N` 得到平整面片模型。

也可直接用 `5_merge_views.py` 一条命令完成（重建+融合+可选网格）。

> 实测：三视角在板坐标系里能拼成立方体的“正面 + 左右侧面”三面，RANSAC 稳定分出对应平面，对齐误差约几毫米级。要闭合立方体再补一个俯拍顶面视角即可。

---

## 从零到出模型的完整操作

```bash
# 0. 环境
pip install -r requirements.txt

# 1. (一次) 生成并打印标定板，卡尺复核后回填 config.yaml 的 charuco 段
python scripts/generate_board.py --out output/charuco_board.png --dpi 300

# 2. 内参标定（关激光，单张拍 15~30 张不同姿态）
python scripts/capture.py --out data/intrinsic --cam 1 --width 800 --height 600 --exposure -4 --gain 20
python scripts/1_calibrate_intrinsic.py          # 验收: 重投影误差 < 1px

# 3. 激光平面标定（开激光，录制 10~20 个姿态，激光落在板白色区）
python scripts/capture.py --out data/laser_plane --mode record --cam 1 --width 800 --height 600 --exposure -4 --gain 20
python scripts/2_calibrate_laser_plane.py        # 验收: 拟合 RMS ≈ 0.1~0.5mm

# 4. 扫描工件（工件+板一起，沿垂直激光线方向平移；录制隔帧存）
python scripts/capture.py --out data/scan --mode record --every 3 --cam 1 --width 800 --height 600 --exposure -4 --gain 20

# 5. 重建 + 后处理
python scripts/3_reconstruct.py
python scripts/4_postprocess.py

# 6. 出网格（方块类推荐 planes，先看 cloud_clean 再定 ROI/去噪参数）
python scripts/ply_to_obj.py --in output/cloud_clean.ply --out output/mesh.obj --method planes --no-roi --simplify 1200

# 7. (可选) 多视角：换朝向重复第 4~5 步到不同目录，再融合
python scripts/5_merge_views.py --views data/scan_v0 data/scan_v1 data/scan_v2 --mesh bpa
```

**⚠️ 三个采集分辨率（内参 / 激光平面 / 扫描）必须完全一致**（本项目用 800×600）。改分辨率就得重标内参和激光平面。

---

## 文件格式约定

**`camera_intrinsic.yaml`**：`image_width/height`、`reproj_error_px`、`K{fx,fy,cx,cy}`、`camera_matrix(3×3)`、`dist_coeffs[k1,k2,p1,p2,k3]`。

**`laser_plane.yaml`**：`plane_abcd:[a,b,c,d]`（相机系）、`fit_rms_m`（验收看这个）、`num_points`。

**`cloud.ply`**：ASCII PLY，**板坐标系** xyz，单位米。

**`height_map.npy`**：二维数组，值为 Z 高度，空栅格 `NaN`。

---

## 常见问题与红线

- **红线：内参/激光平面/扫描三者分辨率必须一致。** 不一致会导致重投影误差大、帧被全部跳过，或点云尺度炸到米/公里级。
- **点云尺度异常（z 达几米/公里）**：激光平面标定坏了或与内参不匹配 → 重标激光平面。
- **点云里有整排棋盘格/板面**：ROI 没生效或没设 z 阈值 → 设 `z_max`/`z_min`，或用 `ply_to_obj` 的 DBSCAN 去噪。
- **高度图报内存错(OOM)**：离群点把栅格撑爆——现已改鲁棒(分位数界定+栅格上限)，但根因是没裁 ROI，建议开 ROI 或聚类去噪。
- **浅色工件抓不到激光**：`score=B−max(R,G)` 对比度低 → 调低 `laser.score_threshold`/`min_intensity`，或给工件喷哑光/贴激光友好涂层。
- **改了相机/激光支架**：位姿关系变了，内参可复用（只要没换镜头/分辨率），但**激光平面必须重标**。

---

## 现场操作速查

1. **确认线激光**：对白墙是一条线，不是点。
2. **刚性固定**：相机与激光相对位置标定后绝不能动；碰了要重标激光平面。
3. **锁死相机参数**：关自动曝光/白平衡，固定焦距，标定与扫描全程一致。
4. **三角角 20°–45°**，工件保持在标定工作距离附近。
5. **激光平面标定**：让激光落在标定板**白色区域**（黑格吸光会断线）。
6. **扫描是规划平移**，不是多角度拍照：沿垂直激光线方向分步移动 2–5mm。
7. **验收**：先扫已知尺寸方块/台阶块，对比卡尺再扫真实工件。
8. **安全**：450nm 蓝光戴对应波长防护镜，勿直视反射。

| 阶段 | 验收指标 |
|------|----------|
| 内参标定 | 重投影误差 < 1px（理想 < 0.5px） |
| 激光平面 | 拟合 RMS ≈ 0.1–0.5mm |
| 拼接 | 移动前后同一物体点对齐，无明显重影 |
| 最终 | 已知物体尺寸误差在可接受范围 |


