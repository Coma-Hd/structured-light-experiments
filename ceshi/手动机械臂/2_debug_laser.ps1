# 兼容旧入口：转发到带关键帧 ROI 画框与激光抽检的 2_draw_check.ps1
param(
    [int]$Samples = 9,
    [switch]$ReuseRoi
)

$ErrorActionPreference = "Stop"
& powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "2_draw_check.ps1") `
    -Samples $Samples `
    -ReuseRoi:$ReuseRoi
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}