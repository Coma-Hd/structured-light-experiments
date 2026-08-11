param(
    [ValidateSet("shot", "record")]
    [string]$Mode = "shot",
    [int]$Every = 1,
    [int]$Cam = 1,
    [int]$Width = 800,
    [int]$Height = 600,
    [double]$Exposure = -4,
    [double]$Gain = 10,
    [switch]$ClearOutput
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $ProjectRoot

$ArmRoot = $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment is missing. Run .\install.ps1 -Action setup from $ProjectRoot."
}

$Config = Join-Path $ArmRoot "arm_scan.yaml"
$ScanDir = Join-Path $ArmRoot "data\scan"

$RoiFile = Join-Path $ArmRoot "work\roi_keyframes.json"
if ($ClearOutput -and (Test-Path -LiteralPath $ScanDir)) {
    Remove-Item -LiteralPath $ScanDir -Recurse -Force
    Write-Host "Cleared old scan folder: $ScanDir"
}
if ($ClearOutput -and (Test-Path -LiteralPath $RoiFile)) {
    Remove-Item -LiteralPath $RoiFile -Force
    Write-Host "Removed stale keyframe ROI: $RoiFile"
}
New-Item -ItemType Directory -Force -Path $ScanDir | Out-Null

$Arguments = @(
    "scripts/capture.py",
    "--config", $Config,
    "--out", $ScanDir,
    "--mode", $Mode,
    "--cam", $Cam,
    "--width", $Width,
    "--height", $Height,
    "--exposure", $Exposure,
    "--gain", $Gain,
    "--every", $Every
)

Write-Host "Arm / handheld capture"
Write-Host "  mode     : $Mode"
Write-Host "  output   : $ScanDir"
Write-Host "  camera   : $Cam  ${Width}x${Height}"
Write-Host ""
Write-Host "Keep object + ChArUco board fixed. Move only the arm tip."
Write-Host "Recommended: drag -> stop -> SPACE to shoot (shot mode)."
Write-Host "Press C in preview to verify board corners before capturing."
Write-Host "Press Q / ESC to finish."
Write-Host ""

& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Arm capture failed with exit code $LASTEXITCODE."
}

$Count = @(Get-ChildItem -Path $ScanDir -Filter "img_*.png" -ErrorAction SilentlyContinue).Count
Write-Host ""
Write-Host "Saved $Count images to $ScanDir"
Write-Host "Next:"
Write-Host "  powershell -ExecutionPolicy Bypass -File `"$ArmRoot\1_check_poses.ps1`""
