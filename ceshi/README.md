# Structured Light Experiments

本仓库整理并保存两个结构光扫描实验工作区：

- [`曲面`](曲面/)：曲面单面扫描、检查与点云重建。
- [`四面拼接`](rail/two_faces/)：电动导轨多面扫描、四面点云对齐与拼接。

两个目录均保留了脚本、配置、实验记录、中间结果和 `.ply` 点云。点云文件由 Git LFS 管理，克隆仓库后请执行：

```powershell
git lfs pull
```

仓库根目录现已包含 `scripts/`、`src/`、依赖清单和安装脚本。请先在根目录运行 `.\install.ps1 -Action setup`，再进入对应工作区操作。
