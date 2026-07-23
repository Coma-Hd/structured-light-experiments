# 电动导轨结构光多面扫描：八次互补扫描与四次固定板扫描

本文档只保留两条独立路线：

```text
路线 A：八次互补扫描
标定板主要用于确定每次导轨运动方向和位移尺度；
每条物理棱使用两次互补扫描、共享棱锚点和 ICP；
四条棱最后还需要全局闭环。

路线 B：四次固定板绝对定位（当前优先）
标定板同时确定导轨轨迹和物体公共坐标系；
四个物理面各扫描一次后直接合并；
正常四次，某一面失败时最多补拍一次，共五次。
```

两条路线不能混用输出。采集前必须先确定路线。

---

## 1. 先区分物理面、共享棱和扫描批次

### 1.1 物理面与共享棱

物体四个侧面固定记为：

```text
P1、P2、P3、P4
```

四条共享棱固定记为：

```text
E12：P1 与 P2 的共享棱
E23：P2 与 P3 的共享棱
E34：P3 与 P4 的共享棱
E41：P4 与 P1 的共享棱
```

它们是物体本身的几何对象，不代表拍摄次数。

### 1.2 扫描批次

扫描批次表示实际启动一次电机并记录一批图片。

同一个物理面可以出现在多个批次中。例如：

```text
E12-A：P1 主面 + 部分 P2
E12-B：部分 P1 + P2 主面

E23-A：P2 主面 + 部分 P3
E23-B：部分 P2 + P3 主面
```

`E12-B` 和 `E23-A` 是两次不同扫描，虽然都以 P2 为主面。

### 1.3 当前 `face1～face4` 目录

当前脚本中的：

```text
face1、face2、face3、face4
```

是四次固定板路线的四个扫描槽位，不能自动表达八个批次。

八次路线应明确命名为：

```text
edge12_a、edge12_b
edge23_a、edge23_b
edge34_a、edge34_b
edge41_a、edge41_b
```

不要再用“Face3”同时表示第三个物理面和第三次拍摄。

---



## 2. 如何选择路线



### 2.1 路线 A：八次互补扫描

适用情况：

```text
只稳定信任 ChArUco 给出的单次导轨方向和位移尺度；
不依赖转位前后板坐标的绝对一致性完成最终拼接；
希望每条物理棱都像最初 P1/P2 一样双向互补拍摄；
允许手动点选共享棱并执行 ICP；
允许采集八次并做最终闭环优化。
```



### 2.2 路线 B：四次固定板绝对定位

适用情况：

```text
物体能够可靠固定在同一块 ChArUco 板上；
物体与板在所有扫描期间不会相对移动；
每次转位只移动整条导轨、相机和激光组件；
每次都能看到足够且二维分布良好的 ChArUco 角点；
希望四次扫描后直接在板坐标系合并。
```

路线 B 是当前推荐方案。

### 2.3 本质区别

路线 A：

```text
ChArUco = 单次轨迹方向和尺度
最终相邻面关系 = 互补扫描 + 共享棱锚点 + ICP
```

路线 B：

```text
ChArUco = 单次轨迹方向和尺度 + 全部扫描的公共物体坐标系
最终相邻面关系 = 同一块固定板的绝对坐标
```

---



## 3. 两条路线共同的标定与机械要求



### 3.1 ChArUco 板

当前规格：

```text
横向方格：12
纵向方格：9
单格边长：30 mm
ArUco 边长：24 mm
字典：DICT_5X5_100
```

必须用卡尺确认实体打印尺寸。打印缩放错误会造成三维尺度和位姿系统偏差，但拟合 RMS 仍可能看起来正常。

### 3.2 相机设置

```text
相机：cam 0
分辨率：800 × 600
曝光初值：-3
增益：20
关闭自动曝光和自动增益
焦距、对焦、缩放和分辨率保持不变
```

移动普通背景不需要重新标定。以下变化必须重新标定：

```text
改变焦距或对焦
改变分辨率
更换相机
移动镜头
相机与激光器之间发生相对位移或转动
重新锁紧松动的激光头或相机支架
```



### 3.3 整套移动与相对移动

允许：

```text
相机 + 激光器 + 导轨作为一个刚体整体转位
```

ChArUco 会重新测量相机相对标定板的位姿，激光平面仍固定在相机坐标系中。

不允许：

```text
相机固定而激光头偏转
激光器固定而相机支架偏转
线缆拉扯导致两者相对姿态变化
支架重新锁紧后继续使用旧激光平面
```

ChArUco 只跟踪相机，无法检测激光器相对相机是否移动。因此 `accepted=true` 不能证明激光平面仍有效。

---



## 4. 相机内参标定

关闭激光，采集 20～30 张不同位置、距离和倾角的 ChArUco 图像：

```powershell
cd "C:\Users\Administrator\Desktop\Sturctured light pipeline"

python scripts/capture.py `
  --config config.yaml `
  --out data/intrinsic `
  --mode shot `
  --cam 0 `
  --width 800 `
  --height 600 `
  --exposure -3 `
  --gain 20
```

每个姿态只保存一张，覆盖画面中心、四角、多个距离和约 15°～40° 倾角。

执行：

```powershell
python scripts/1_calibrate_intrinsic.py `
  --config config.yaml `
  --images data/intrinsic `
  --out output/camera_intrinsic.yaml
```

验收：

```text
reproj_error_px < 1.0 px
理想值 < 0.5 px
```

内参一旦改变，所有激光平面都必须重新标定。

---



## 5. 激光平面标定与版本管理



### 5.1 标定

保持正式扫描的相机设置，采集 15～30 个不同位置、距离和倾角：

```powershell
python scripts/capture.py `
  --config config.yaml `
  --out data/laser_plane `
  --mode shot `
  --cam 0 `
  --width 800 `
  --height 600 `
  --exposure -3 `
  --gain 20
```

激光必须落在标定板有效区域，不要混入板外地面、墙面或物体上的激光。

执行：

```powershell
python scripts/2_calibrate_laser_plane.py `
  --config config.yaml `
  --images data/laser_plane `
  --intrinsic output/camera_intrinsic.yaml `
  --out output/laser_plane.yaml
```

目录中存在多批图片时必须使用 `--filename-prefix`，不能混合不同安装状态。

验收：

```text
fit_rms_m 理想为 0.0001～0.0005 m
内点率 >= 65%
有效三维激光点数量充足
使用已知平面、直角块或正方体做实物验证
```

RMS 合格不等于没有系统偏差。

### 5.2 过曝后呈白紫色的激光核心

当前扫描图中，激光并不是“中心纯蓝、两侧变暗”，而是：

```text
纯蓝光晕在一侧形成很强的 B-G 响应
真实高亮核心因过曝呈白色或白紫色
```

旧 `blue_minus_green` 会偏向纯蓝光晕。以 Face1 中间帧为例：

```text
旧中心相对最亮白紫核心偏约 3 px
该位置 3.5 px 横向误差对应约 6.8 mm 三维点误差
其中深度方向约 5.9 mm
```

`blue_weighted_intensity` 在白色标定板上接近亮核，但在木材上仍会因
`B-G` 不对称而偏向纯蓝光晕。程序已经提供会饱和的蓝色软门控：

```text
score_mode = blue_gated_intensity
blue_gate_threshold = 5
blue_gate = clip(max(B-G, 0) / 5, 0, 1)
score = mean(B,G,R) × blue_gate

steger_sigma = 2.0
score_blur_sigma = 0
smooth_window = 0
```

`B-G` 达到门限后不再继续增加权重，最终由亮度定位白紫核心。Face1
木材抽样帧中，旧响应相对亮核中位偏左约 `2.80 px`，当前响应约
`-0.01 px`，绝对偏差中位数约 `0.46 px`。

当前正式 gated 激光平面由本次刚性安装的数据重新标定，拟合 RMS
约 `0.731 mm`。不能把其他安装状态或其他中心响应的激光平面直接用于
四面重建；即使单面更平，也可能产生很大的绝对深度偏差。

最重要的约束：

```text
激光平面标定和正式扫描必须使用完全相同的：
score_mode
blue_gate_threshold
score_threshold
steger_sigma
score_blur_sigma
smooth_window
```

改变中心定义后，旧 `laser_plane.yaml` 必须作废并重新标定。不能只修改扫描配置后直接用旧激光平面重建。

### 5.3 中途碰到相机或激光头

如果前三面已经完成，Face4 前相机与激光器相对位置发生变化：

```text
1. 保留 Face1～Face3 已生成的 cloud_clean.ply。
2. 备份旧激光平面。
3. 锁紧相机、激光头和支架。
4. 在当前安装状态重新标定激光平面。
5. 仅让 Face4 配置引用新激光平面。
6. 丢弃并重拍 Face4。
```

建议版本名：

```text
output/laser_plane_before_face4.yaml
output/laser_plane_face4.yaml
```

只要物体与底部 ChArUco 板之间没有相对移动，新标定重建的 Face4 仍可进入相同板坐标系，与前三面连接。

不要用新激光平面重新重建旧 Face1～Face3。重新标定最好使用另一块标定板，不要为了标定取下物体。

---



## 6. 连续采集共同操作

当前参数：

```text
导轨名义速度：1.0 mm/s
目标保存周期：0.1 s
目标抽帧间隔：0.1 mm
曝光初值：-3
增益：20
手动开始和停止
```

采集脚本按空间步距计算保存周期：

```text
保存周期(s) = StepMm / VelocityMmS
```

因此在 `VelocityMmS=1.0` 时，`StepMm=0.1` 表示约每 `0.1 s` 保存一张、相邻扫描轮廓名义间隔 `0.1 mm`。若终端提示保存或系统延迟超过采样周期，应先降低图像写盘压力或检查相机实际帧率，不能把跳帧数据当作等间距采样。

预览快捷键：

```text
E / D：提高 / 降低曝光
R / F：提高 / 降低增益
B / V：提高 / 降低亮度
L：激光识别热力图
C：ChArUco 检查
空格：开始或结束
S / Q：结束
```

操作顺序：

```text
1. 预览确认激光识别和 ChArUco PASS。
2. 启动电机并等待速度稳定。
3. 激光到达扫描区域前按空格开始。
4. 中途保持连续匀速。
5. 完整扫过目标面和需要保留的邻面条带。
6. 电机仍在运动时按空格结束。
7. 录制结束后再停止电机。
```

ROI 必须覆盖物体真实激光和需要的邻面条带，同时排除板面激光、背景蓝光和反光。

---



# 路线 A：八次互补扫描



## A1. 为什么需要八次

一个共享棱需要两个互补视角：

```text
A 视角：前一个物理面为主面，后一个面为条带
B 视角：后一个物理面为主面，前一个面为条带
```

完整序列：

```text
E12-A：P1 主面 + 部分 P2
E12-B：部分 P1 + P2 主面

E23-A：P2 主面 + 部分 P3
E23-B：部分 P2 + P3 主面

E34-A：P3 主面 + 部分 P4
E34-B：部分 P3 + P4 主面

E41-A：P4 主面 + 部分 P1
E41-B：部分 P4 + P1 主面
```

因此：

```text
4 条共享棱 × 每条 2 次 = 8 次
```

前两次只能解决 E12，不能同时代表 E23、E34 和 E41。

## A2. 标定板在八次路线中的作用

板放在物体下面并参与每次轨迹拟合，用于：

```text
测量导轨三维方向
修正名义速度与真实位移尺度
平滑连续扫描轨迹
拒绝明显异常帧
```

但最终拼接不依赖八次板位姿的绝对一致性：

```text
板坐标只作为初值和诊断；
每条共享棱最终由互补扫描、同名锚点和 ICP 验证。
```



## A3. 八批数据必须独立保存

推荐目录：

```text
ceshi/rail/eight_scans/edge12_a/
ceshi/rail/eight_scans/edge12_b/
ceshi/rail/eight_scans/edge23_a/
ceshi/rail/eight_scans/edge23_b/
ceshi/rail/eight_scans/edge34_a/
ceshi/rail/eight_scans/edge34_b/
ceshi/rail/eight_scans/edge41_a/
ceshi/rail/eight_scans/edge41_b/
```

每批包含：

```text
img_*.png
positions.csv
continuous_capture_report.json
独立 ROI
独立重建输出
```

当前 `0_capture_continuous.ps1` 只支持 `face1～face4`，不能连续保存八批。八次路线应调用底层采集脚本并明确输出目录：

```powershell
python scripts/continuous_rail_capture.py `
  --out ceshi/rail/eight_scans/edge12_a `
  --config ceshi/rail/two_faces/face1_scan.yaml `
  --velocity-mm-s 1.0 `
  --step-mm 0.1 `
  --cam 0 `
  --width 800 `
  --height 600 `
  --exposure -3 `
  --gain 20 `
  --clear-output
```

其余批次改变 `--out`，并为每批准备对应扫描配置、ROI 和输出路径，避免覆盖。

## A4. 每条共享棱独立配准

以 E12 为例，重建后得到：

```text
edge12_a/cloud_clean.ply
edge12_b/cloud_clean.ply
```

点选共享棱：

```powershell
python scripts/pick_two_face_anchors.py `
  --target ceshi/rail/eight_scans/edge12_a/cloud_clean.ply `
  --source ceshi/rail/eight_scans/edge12_b/cloud_clean.ply `
  --target-name edge12_a `
  --source-name edge12_b `
  --out ceshi/rail/eight_scans/anchors_edge12.json
```

两份点云按同一物理顺序选择：

```text
1. E12 顶部同名点
2. E12 底部同名点
3. 可选：共享条带中的可靠显著点
```

执行：

```powershell
python scripts/6_merge_two_faces.py `
  --config ceshi/rail/two_faces/config.yaml `
  --target ceshi/rail/eight_scans/edge12_a/cloud_clean.ply `
  --source ceshi/rail/eight_scans/edge12_b/cloud_clean.ply `
  --anchors ceshi/rail/eight_scans/anchors_edge12.json `
  --out-dir ceshi/rail/eight_scans/output/edge12
```

E23、E34、E41 重复相同步骤。

每一对必须通过：

```text
fitness >= 0.50
RMSE <= 4 mm
anchor RMSE <= 10 mm
ICP 相对初值旋转变化 <= 15°
共享棱无上下反转
主面与邻面没有错误交换
```

不要使用 `AllowLowQuality` 把失败结果带入最终模型。

## A5. 最后必须做四边全局闭环

四条棱分别得到：

```text
Pair12：P1 + P2
Pair23：P2 + P3
Pair34：P3 + P4
Pair41：P4 + P1
```

边对之间通过完整物理面重叠：

```text
Pair12 与 Pair23 共享 P2
Pair23 与 Pair34 共享 P3
Pair34 与 Pair41 共享 P4
Pair41 与 Pair12 共享 P1
```

最终应把四个边对作为闭环图联合优化。不能只顺序累乘后忽略最后的 P1 闭环，否则误差会集中在最后一条边。

当前 `6_merge_face_chain.ps1` 是四扫描顺序链，不符合八次互补扫描模型，不应作为八次路线最终工具。

当前代码可完成每条棱的两面配准，但四边全局位姿图优化仍需单独实现。未完成前，四个局部边对只能分别验收，不能称为可靠闭环模型。

## A6. 八次路线验收

每条边：

```text
顶部和底部锚点顺序一致
共享棱没有双线
互补扫描主面重合
没有用大对应距离强行吸附
```

全局闭环：

```text
P1 在 Pair12 与 Pair41 中一致
P2 在 Pair12 与 Pair23 中一致
P3 在 Pair23 与 Pair34 中一致
P4 在 Pair34 与 Pair41 中一致
四条边约束同时成立
```

---



# 路线 B：四次固定板绝对定位



## B1. 固定方式

```text
物体固定在同一块 ChArUco 板上
物体相对板不能滑动或重新放置
板不能弯曲
四次只转动整条导轨 + 相机 + 激光器
```

物体可以遮挡板中央，但需要：

```text
配置最低 6 点，建议 8～12 点以上
角点凸包面积 >= 画面 0.2%
每次至少 8 个合格位姿
有效位姿沿导轨覆盖 >= 30 mm
```

预览显示 `PASS` 后再采集。

## B2. 四次扫描

```text
face1：主要覆盖 P1
face2：主要覆盖 P2
face3：主要覆盖 P3
face4：主要覆盖 P4
```

每次尽量保留相邻面少量条带用于接缝检查，但不要求八次路线那样反向再拍一次。

## B3. 单个面完整命令

将 `$Face` 依次改为 `face1`、`face2`、`face3`、`face4`：

```powershell
cd "C:\Users\Administrator\Desktop\Sturctured light pipeline"

$Face = "face1"

powershell -ExecutionPolicy Bypass `
  -File ceshi/rail/two_faces/0_capture_continuous.ps1 `
  -Face $Face `
  -VelocityMmS 1.0 `
  -StepMm 0.2 `
  -Exposure -3 `
  -ClearOutput

powershell -ExecutionPolicy Bypass `
  -File ceshi/rail/two_faces/2_draw_check_face.ps1 `
  -Face $Face `
  -Keyframes 5 `
  -Samples 9

powershell -ExecutionPolicy Bypass `
  -File ceshi/rail/two_faces/3_rebuild_face.ps1 `
  -Face $Face
```

输出：

```text
work/<face>/output/cloud.ply
work/<face>/output/cloud_clean.ply
work/<face>/output/cloud_charuco_tracking.yaml
input/<face>/cloud_clean.ply
```

四份 `face1_scan.yaml`～`face4_scan.yaml` 当前使用：

```yaml
postprocess:
  voxel_size: 0.0002  # 0.2 mm
```

`voxel_size` 只控制后处理降采样，不改变原始扫描轮廓间距。已有 `0.5 mm` 步距的图片可以直接重新重建以减少点合并，但不会凭空变成 `0.1 mm` 采样；只有重新采集时使用 `-StepMm 0.1` 才会增加扫描方向的真实轮廓密度。

如果只修改了后处理参数，不需要重新采集或重画 ROI。原始图片、`positions.csv`、采集报告和 ROI 均保留时可批量重新重建：

```powershell
foreach ($Face in "face1","face2","face3","face4") {
    powershell -ExecutionPolicy Bypass `
      -File ceshi/rail/two_faces/3_rebuild_face.ps1 `
      -Face $Face
}
```

循环中必须使用 `-Face $Face`，不能写死为 `-Face face1`。

## B4. 每面轨迹报告

必须检查：

```text
accepted = true
inlier_pose_frames >= 8
pose_span_mm >= 30
mean_reprojection_error_px < 2
建议 mean_reprojection_error_px 约 1 或更低
center_fit_rms_mm 建议 < 2
max_rotation_deviation_deg < 1.5
actual_over_nominal_distance 合理
```

这些门限检查单次扫描内部稳定性，但不能排除整批固定偏移，也不能检测激光器相对相机是否偏转。

## B5. 四面直接合并

```powershell
powershell -ExecutionPolicy Bypass `
  -File ceshi/rail/two_faces/5_merge_board_faces.ps1 `
  -Faces face1,face2,face3,face4 `
  -VoxelMm 0.2
```

输出：

```text
ceshi/rail/two_faces/output/board_merged.ply
ceshi/rail/two_faces/output/board_merged.json
```

该脚本只做：

```text
读取四份板坐标点云
体素降采样
统计离群点过滤
```

它不执行 ICP，不强制正方体夹角，也不强制闭环。接缝是在暴露标定、机械或识别问题。

## B5.1 正式自动流程：诊断 → 受约束配准 → 可选方木补全

四面重建后，默认先运行只读诊断：

```powershell
powershell -ExecutionPolicy Bypass `
  -File ceshi/rail/two_faces/9_analyze_four_faces.ps1 `
  -PoseCompareSamples 9
```

默认输出诊断 JSON 和四色比较 PLY。可用 `-SkipPoseComparison` 跳过逐帧 PnP 与导轨拟合位姿比较，用 `-SkipColoredPly` 不写彩色 PLY。

诊断通过且相邻扫描确实有共享数据后，再运行通用受约束自动配准：

```powershell
powershell -ExecutionPolicy Bypass `
  -File ceshi/rail/two_faces/10_auto_align_four_faces.ps1 `
  -VoxelMm 1.5 `
  -DistanceLevelsMm "15,8,4" `
  -NormalAngleDeg 18 `
  -MinCorrespondences 100 `
  -MinCoverage 0.02 `
  -MaxTranslationMm 15 `
  -MaxRotationDeg 5 `
  -MaxFinalRmseMm 3 `
  -MergeVoxelMm 0.5
```

该算法以 ChArUco 板坐标为强初值，只在四组相邻面的法向一致共享条带上建立双向对应，并联合求解有硬上限的小范围 SE(3) 修正。**禁止对四份完整点云执行无约束 ICP**：主平面互相垂直，全云最近邻会把主面吸到错误的邻面，得到视觉闭合但几何错误的结果。

采集时每个扫描必须同时覆盖“当前主面 + 下一面的连续条带”，邻面条带沿共享棱法向应有约 **10～20 mm** 连续宽度。只拍到主面、只剩一条稀疏棱线或四面之间完全不重叠，都没有足够几何约束。当前旧数据缺少这种连续共享条带，自动配准会以退出码 `2` **安全拒绝**；这不是脚本失败，更不能把拒绝时写出的候选点云当作成功结果。

仅当工件确为方木块且需要闭合外观时，可在实测结果之外生成明确标记的合成几何：

```powershell
powershell -ExecutionPolicy Bypass `
  -File ceshi/rail/two_faces/11_complete_cuboid.ps1 `
  -SampleSpacingMm 1.5

# 只有明确需要强制正交截面时才增加：
# -Orthogonalize
```

补全输出中的橙色点和网格由拟合平面生成，**是合成数据，不是传感器测量**；实测蓝色点不会被移动。不得用合成面参与精度、缺陷或尺寸测量。

输出目录：

```text
output/auto_diagnostics/
output/auto_alignment/
output/cuboid_completion/
```

## B5.2 手动 fallback：四条共享棱平移闭环

`7_pick_four_edges.ps1` 和 `8_align_four_faces_loop.ps1` 保留为自动流程因数据条件不满足而无法使用时的人工 fallback，不再是默认正式路径。

只有满足以下条件才执行本节：

```text
四面使用同一套中心响应参数，并与激光平面标定参数完全一致
使用同一份重新标定的激光平面重建
物体与 ChArUco 板在四次扫描中没有移动
四个单面形状和方向正确，只剩稳定的小范围板内平移接缝
```

该步骤不是 ICP，也不把物体强制修成正方体。它只联合求解四个面的 ChArUco 板坐标 `X/Y` 平移：

```text
Face1 / Face2 的共享棱相等
Face2 / Face3 的共享棱相等
Face3 / Face4 的共享棱相等
Face4 / Face1 的共享棱相等
四个面的平均修正量为零
```

旋转、尺度、板坐标 Z 和每个面的内部形状均保持不变。最后一条 `Face4 / Face1` 约束会参与同一次最小二乘求解，因此误差由四面共同分摊，不会顺序累积到 Face4。

### 第一步：点选四条共享棱

```powershell
cd "C:\Users\Administrator\Desktop\Sturctured light pipeline"

powershell -ExecutionPolicy Bypass `
  -File ceshi/rail/two_faces/7_pick_four_edges.ps1
```

程序依次打开：

```text
Face1 / Face2
Face2 / Face3
Face3 / Face4
Face4 / Face1
```

每一对都要在两份点云中按完全相同顺序点：

```text
1. 同一条物理共享棱的顶部
2. 同一条物理共享棱的底部
```

操作为 `Shift + 左键` 点选、`Shift + 右键` 撤销、`Q` 完成。四条棱共形成 8 对对应点，也就是 16 次点选。不能把相邻但不同的角点当成对应点，也不能把顶部和底部顺序颠倒。

生成：

```text
anchors_loop_face1_face2.json
anchors_loop_face2_face3.json
anchors_loop_face3_face4.json
anchors_loop_face4_face1.json
```



### 第二步：联合求解并输出候选结果

```powershell
powershell -ExecutionPolicy Bypass `
  -File ceshi/rail/two_faces/8_align_four_faces_loop.ps1 `
  -VoxelMm 0.2 `
  -MaxTranslationMm 25 `
  -MaxAnchorRmseMm 5
```

输出：

```text
output/translation_loop/aligned/face1_aligned.ply
output/translation_loop/aligned/face2_aligned.ply
output/translation_loop/aligned/face3_aligned.ply
output/translation_loop/aligned/face4_aligned.ply
output/translation_loop/loop_comparison_colored.ply
output/translation_loop/loop_merged_candidate.ply
output/translation_loop/loop_alignment_report.json
```

先看彩色比较点云，再看报告中的：

```text
translations_mm
pair_anchor_metrics
overall_anchor_rmse_mm
quality.accepted
```

默认质量门限：

```text
任一面的板内平移修正 <= 25 mm
闭环后的整体锚点 RMSE <= 5 mm
```

任一门限失败都会拒绝结果。此时应先检查共享棱是否点错；如果点选正确，说明仍存在旋转、尺度、Z 高度或单面形变，不能继续放宽门限强行闭合。

## B6. 为什么正常四次、最多五次

正常：

```text
P1、P2、P3、P4 各一次 = 4 次
```

某一面失败：

```text
保留其他三个面
修复机械、标定或识别问题
丢弃失败面
只补拍该面一次
```

因此最多五次。第五次是失败扫描的替代，不是新物理面或额外闭环面。

如果物体相对底板已经移动，则公共坐标关系失效，不能只补拍一个面。

---



## 7. 当前数据诊断



### 7.1 四份轨迹

```text
Face1：272 帧，span 141.54 mm，center RMS 0.865 mm，reproj 0.999 px
Face2：251 帧，span 125.02 mm，center RMS 0.713 mm，reproj 1.042 px
Face3：222 帧，span 110.55 mm，center RMS 0.704 mm，reproj 0.978 px
Face4：158 帧，span 78.54 mm，center RMS 0.751 mm，reproj 1.073 px
```

四份单次轨迹内部均通过。

### 7.2 相邻点云距离

```text
Face1 → Face2：1% 分位约 0.30 mm
Face2 → Face3：1% 分位约 0.37 mm
Face3 → Face4：1% 分位约 4.45 mm
Face4 → Face1：1% 分位约 2.00 mm
```

主要异常是 Face4。

### 7.3 已排除

Face4 使用逐帧 ChArUco 位姿重建后仍然相同，基本排除 `rail_fit` 平滑和导轨直线拟合。

Face4 原始与清理后到 Face3 的 1% 距离分别约：

```text
raw：4.48 mm
clean：4.49 mm
```

因此双平面清理不是整体错位根因。

### 7.4 最可能原因

```text
1. Face4 前相机与激光器发生相对位移或转动。
2. 物体相对底板被碰动。
3. Face4 激光中心提取到反光或错误响应。
4. Face4 的板可见区域和角度产生跨扫描系统偏差。
```

当前建议按路线 B：

```text
锁紧相机、激光头和支架
保留前三面和旧激光平面
重新标定当前安装状态的激光平面
让 Face4 使用新平面
丢弃并重拍 Face4
再次直接板坐标合并
```



### 7.5 7月21日重新采集后的四面整体外扩

新一批四面轨迹仍全部通过，而且导轨轴在板坐标中大致按四个方向分布，说明板坐标轴和轨迹方向不是当前首要问题。

当前四面对之间的最近邻距离：

```text
Face1 → Face2：最小 14.57 mm，1% 分位 17.06 mm
Face2 → Face3：最小 19.86 mm，1% 分位 23.93 mm
Face3 → Face4：最小 11.84 mm，1% 分位 13.96 mm
Face4 → Face1：最小 8.48 mm，1% 分位 10.60 mm
```

四个面都沿各自观察方向向外散开，而不是只有某一个面发生随机旋转。这更符合“激光三角测量存在共同深度偏差”的特征。

俯视图中把四面分别向内平移确实可以在视觉上闭合，但不能靠目测分别拖动四面作为正式修正，否则会掩盖激光中心或激光平面的系统误差。只有先统一提取参数、重新标定和重新重建，确认只剩小范围刚性接缝后，才能使用 B5.1 的四边约束平移闭环。

当前正确顺序：

```text
1. 使用与当前激光平面标定完全一致的响应检查所有 laser_check 红点。
2. 使用同一响应参数重新标定激光平面。
3. 用新激光平面重新重建四面。
4. 再执行板坐标直接合并并测量四条接缝。
5. 只有统一提取与重新标定后仍有固定小偏差，才研究受约束的全局偏移估计。
```



### 7.6 重新标定和重建后的残余接缝

使用同一中心响应重新标定激光平面并重建四面后，若整体结果仍有约数毫米到十余毫米的稳定接缝，不再使用顺序 ICP，也不单独拖动某一个面；按 B5.1 点选四条物理共享棱，并一次性求解四面的板内平移闭环。

该闭环只处理残余平移。如果报告拒绝结果，或彩色比较点云仍呈现角度不一致、上下错层、尺度不同，应返回激光平面和逐帧 ChArUco 位姿排查。

---



## 8. 棱边卷曲与双平面清理

激光横跨真实棱边时，散射、过曝和多路径反光可能产生多个响应峰，平滑后形成弯曲“桥”。

Face3、Face4 当前配置：

```yaml
postprocess:
  two_plane_cleanup:
    enabled: true
    ransac_threshold_m: 0.0015
    max_distance_m: 0.002
    min_plane_points: 500
    min_plane_angle_deg: 45.0
    min_keep_fraction: 0.70
    ransac_iterations: 4000
```

它删除同时远离两个主平面超过 2 mm 的点，不投影保留点、不强制 90°，失败时自动回退。

只适合明确双平面直角件；自由曲面、圆柱、雕刻和缺陷检测必须关闭。

它不能修复：

```text
整面平移或转动
激光平面错误造成的深度形变
物体相对底板移动
```

---



## 9. 常见失败



### 9.1 `accepted=true` 但四面错开

继续检查：

```text
相机—激光相对位姿
物体—底板相对位姿
激光中心识别
不同扫描使用的激光平面版本
跨扫描板位姿系统误差
```



### 9.2 单面几何已经错误

不要执行 ICP。先修复：

```text
内参
激光平面
激光中心
ROI
导轨轨迹
机械振动
```

ICP 不能修复尺度和非刚性形变。

### 9.3 共享条带消失

检查 ROI、auto padding、SOR 和 `two_plane_cleanup` 是否删除了稀疏邻面条带。

### 9.4 禁止用大范围 ICP 掩盖错误

禁止：

```text
使用 10～30 mm 大对应距离强行吸附
质量门限失败后直接 AllowLowQuality
为了闭环把未知物体强制修成正方体
混用不同激光安装状态的数据
```

---



## 10. 最终执行清单



### 10.1 八次路线

```text
[ ] 明确 P1～P4 和 E12/E23/E34/E41
[ ] 建立八个独立扫描目录
[ ] 完成 E12-A/B、E23-A/B、E34-A/B、E41-A/B
[ ] 四条棱分别点选顶部和底部锚点
[ ] 四个局部边对全部通过质量门限
[ ] 用公共物理面建立四边全局闭环
[ ] 闭环后检查四个面和四条棱
```



### 10.2 四次路线

```text
[ ] 物体固定在同一块 ChArUco 板上
[ ] 相机、激光和导轨保持刚性
[ ] 每次采到主面和下一面 10～20 mm 连续重叠条带
[ ] Face1～Face4 分别采集、ROI、重建并通过报告
[ ] 执行 5_merge_board_faces.ps1
[ ] 执行 9_analyze_four_faces.ps1 并检查诊断
[ ] 执行 10_auto_align_four_faces.ps1，并确认 accepted = true
[ ] 需要方木外观补全时才执行 11_complete_cuboid.ps1
[ ] 自动流程无法使用时，才以 7/8 手动点选作为 fallback
[ ] 检查彩色比较点云、四条棱和整体形状
[ ] 若只有一个面失败，只修复并补拍该面一次
```

---



## 11. 当前推荐结论

物体与底板能够可靠固定时，优先路线 B：

```text
四次固定板绝对定位
失败面最多补拍一次
直接板坐标合并
自动诊断后做共享条带受约束配准
不用全云无约束 ICP 掩盖误差
手动四边平移闭环仅作 fallback
```

无法信任跨扫描板绝对位置、但能利用板测量单次轨迹时，选择路线 A：

```text
八次互补扫描
每条棱两次
共享棱锚点
每条棱独立配准
最后做四边全局闭环
```

当前旧数据没有按“主面 + 下一面 10～20 mm 连续条带”采集，`10_auto_align_four_faces.ps1` 预计会安全拒绝。重新采集后默认执行 `9_analyze_four_faces.ps1` 和 `10_auto_align_four_faces.ps1`；`7_pick_four_edges.ps1` / `8_align_four_faces_loop.ps1` 仅作 fallback。不要使用 `6_merge_face_chain.ps1` 顺序强行连接四面。