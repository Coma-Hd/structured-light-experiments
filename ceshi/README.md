> **核心代码不在本目录的 .ps1 里。**  
> 算法与实现在仓库根目录：[`src/`](../src/)、[`scripts/`](../scripts/)。  
> 完整说明见 [`代码位置.md`](../代码位置.md)。

# 实验工作区

本目录按三种扫描平台组织，各自独立配置与流程脚本：

| 类别 | 目录 | 说明 |
|------|------|------|
| 导轨 | [`导轨/`](导轨/) | 电机导轨连续扫描（曲面/半球等） |
| 转台 | [`转台/`](转台/) | 转台轴标定与多角度扫描 |
| 手动机械臂 | [`手动机械臂/`](手动机械臂/) | 机械臂末端手持位姿采集与重建 |

排错与优化过程文档见 [`排错与优化文档/`](排错与优化文档/)。

## 通用约定

- 每个类别目录内含：`*.yaml` 配置、`0_*.ps1` 起的流程脚本、`calibration/`、`data/`、`work/`、`output/`。
- 运行脚本时请在仓库根目录已完成 `install.ps1 -Action setup`；脚本会自动定位到仓库根并调用 `scripts/`。
- **不上传 `data/` 中的采集图片**（仓库已忽略常见图片格式）；本地采集后图片仍可放在 `data/` 供脚本使用。

## 快速入口

### 导轨

```powershell
powershell -ExecutionPolicy Bypass -File .\ceshi\导轨\0_capture.ps1 -ClearOutput
powershell -ExecutionPolicy Bypass -File .\ceshi\导轨\1_draw_check.ps1
powershell -ExecutionPolicy Bypass -File .\ceshi\导轨\2_rebuild.ps1
```

详见 [`导轨/导轨.md`](导轨/导轨.md)。

### 转台

```powershell
powershell -ExecutionPolicy Bypass -File .\ceshi\转台\0_axis_capture.ps1
powershell -ExecutionPolicy Bypass -File .\ceshi\转台\1_calibrate_axis.ps1
powershell -ExecutionPolicy Bypass -File .\ceshi\转台\2_capture.ps1 -ClearOutput
powershell -ExecutionPolicy Bypass -File .\ceshi\转台\3_draw_check.ps1
powershell -ExecutionPolicy Bypass -File .\ceshi\转台\4_rebuild.ps1
```

详见 [`转台/转台扫描.md`](转台/转台扫描.md)。

### 手动机械臂

```powershell
powershell -ExecutionPolicy Bypass -File .\ceshi\手动机械臂\0_capture_continuous.ps1 -ClearOutput
powershell -ExecutionPolicy Bypass -File .\ceshi\手动机械臂\1_check_poses.ps1
powershell -ExecutionPolicy Bypass -File .\ceshi\手动机械臂\2_draw_check.ps1
powershell -ExecutionPolicy Bypass -File .\ceshi\手动机械臂\3_rebuild.ps1
```

详见 [`手动机械臂/机械臂末端手持.md`](手动机械臂/机械臂末端手持.md)。

