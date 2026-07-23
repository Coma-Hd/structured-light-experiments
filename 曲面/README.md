# 曲面单面扫描

该目录是独立的曲面扫描工作区。曲面采集图片、位置记录、ROI、检查图、标定副本和重建结果都保存在本目录下，不再引用 `ceshi/rail` 的数据路径。

## 目录

```text
ceshi/曲面/
  curve_scan.yaml
  0_capture.ps1
  1_draw_check.ps1
  2_rebuild.ps1
  calibration/
    camera_intrinsic.yaml
    laser_plane.yaml
  data/calibration/      # 仅在重新采集标定数据时使用
  data/scan/             # 运行采集后自动生成
  work/                  # ROI 和激光检查图
  output/                # 点云和检查报告
```

## 默认相机参数

曲面扫描、内参采集和激光平面标定统一使用：

```text
Exposure = -5
Gain = 1
Resolution = 800 x 600
```

采集开始后不要再修改曝光或增益。当前 `calibration` 中的 YAML 是此前标定结果的副本；切换到上述新参数后，至少必须重新标定激光平面。若相机分辨率、镜头、焦距或对焦发生变化，也必须重新标定内参。

## 相机内参标定

在项目根目录执行：

```powershell
python scripts/capture.py `
  --config ceshi/曲面/curve_scan.yaml `
  --out ceshi/曲面/data/calibration/intrinsic `
  --mode shot `
  --cam 0 `
  --width 800 `
  --height 600 `
  --exposure -5 `
  --gain 1
```

移动 ChArUco 板拍摄约 20～30 张清晰图片，覆盖画面中心、四角、远近和不同倾角。避免运动模糊、反光和严重过曝。采集完成后：

```powershell
python scripts/1_calibrate_intrinsic.py `
  --config ceshi/曲面/curve_scan.yaml `
  --images ceshi/曲面/data/calibration/intrinsic `
  --out ceshi/曲面/calibration/camera_intrinsic.yaml
```

确认输出分辨率为 `800 x 600`，并检查重投影误差。

## 激光平面标定

保持相机、激光和镜头完全不动。使用与正式扫描完全相同的曝光、增益和激光中心参数：

```powershell
python scripts/capture.py `
  --config ceshi/曲面/curve_scan.yaml `
  --out ceshi/曲面/data/calibration/laser_plane `
  --mode shot `
  --cam 0 `
  --width 800 `
  --height 600 `
  --exposure -5 `
  --gain 1
```

打开激光，在不同距离和倾角放置 ChArUco 板，每个姿态静止后拍一张；激光必须落在实体标定板内。建议至少 15～20 个有效姿态，并覆盖正式扫描的工作距离。然后执行：

```powershell
python scripts/2_calibrate_laser_plane.py `
  --config ceshi/曲面/curve_scan.yaml `
  --images ceshi/曲面/data/calibration/laser_plane `
  --intrinsic ceshi/曲面/calibration/camera_intrinsic.yaml `
  --out ceshi/曲面/calibration/laser_plane.yaml
```

不得混用旧曝光、旧增益、旧激光响应或其他安装状态的标定图片。标定完成后检查 `fit_rms_m`，并优先使用未参与拟合的板姿态做深度留出验证。

## 曲面扫描顺序

在项目根目录执行：

```powershell
powershell -ExecutionPolicy Bypass `
  -File ceshi/曲面/0_capture.ps1 `
  -VelocityMmS 1.0 `
  -StepMm 0.2 `
  -Exposure -5 `
  -Gain 1 `
  -ClearOutput
```

采集完成后重新绘制本次曲面的 ROI：

```powershell
powershell -ExecutionPolicy Bypass `
  -File ceshi/曲面/1_draw_check.ps1
```

逐张检查 `work/laser_check`：

- 绿色 ROI 覆盖完整曲面。
- 红点贴合激光亮核。
- 红点不能进入标定板、背景或反光。
- 若同一图像行出现多段独立激光，当前单分支 Steger 可能漏掉其中一段，应停止重建并先升级提取算法。

检查通过后：

```powershell
powershell -ExecutionPolicy Bypass `
  -File ceshi/曲面/2_rebuild.ps1
```

最终查看：

```text
ceshi/曲面/output/cloud_clean.ply
ceshi/曲面/output/inspect_clean/
```

## 约束

- 曲面和 ChArUco 板在扫描过程中必须保持刚性固定。
- 保持当前相机、激光安装状态不变。
- `score_mode`、门限和 Steger 参数必须与激光平面标定一致。
- `calibration` 中保存本曲面流程正式使用的标定结果，不能再引用或混入其他目录的 YAML。
- 当前配置关闭三维自动裁剪、平面提取、双平面清理和网格生成，避免把真实曲率误删或压平。
- 单方向看不到的凹槽、背面和激光阴影不能由本次单面扫描恢复。
