param(
    [int]$Cam = 1,
    [int]$Width = 800,
    [int]$Height = 600,
    [double]$Exposure = -4,
    [double]$Gain = 10,
    [double]$Brightness = -1,
    [switch]$AutoExposure,
    [double]$IntervalMs = 250,
    [double]$Countdown = 3,
    [int]$MinCorners = 0,
    [double]$MinSharpness = 20,
    [int]$MaxFrames = 0,
    [switch]$AllowNoBoard,
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
$Script = Join-Path $ArmRoot "continuous_arm_capture.py"
New-Item -ItemType Directory -Force -Path $ScanDir | Out-Null
if ($ClearOutput -and (Test-Path -LiteralPath $RoiFile)) {
    Remove-Item -LiteralPath $RoiFile -Force
    Write-Host "Removed stale keyframe ROI: $RoiFile"
}

$Arguments = @(
    $Script,
    "--config", $Config,
    "--out", $ScanDir,
    "--cam", $Cam,
    "--width", $Width,
    "--height", $Height,
    "--gain", $Gain,
    "--interval-ms", $IntervalMs,
    "--countdown", $Countdown,
    "--min-sharpness", $MinSharpness,
    "--max-frames", $MaxFrames
)

if ($AutoExposure) {
    $Arguments += "--auto-exposure"
} else {
    $Arguments += @("--exposure", $Exposure)
}
if ($Brightness -ge 0) {
    $Arguments += @("--brightness", $Brightness)
}
if ($MinCorners -gt 0) {
    $Arguments += @("--min-corners", $MinCorners)
}
if ($AllowNoBoard) {
    $Arguments += "--no-require-board"
} else {
    $Arguments += "--require-board"
}
if ($ClearOutput) {
    $Arguments += "--clear-output"
}

Write-Host "Arm continuous capture (smooth drag + auto burst)"
Write-Host "  output      : $ScanDir"
Write-Host "  interval    : $IntervalMs ms"
Write-Host "  countdown   : $Countdown s"
Write-Host "  require board: $(-not $AllowNoBoard)"
Write-Host ""
Write-Host "1) Aim so board + object laser are both visible"
Write-Host "2) Tune camera if needed: a=AE  e/d=exposure  r/f=gain  b/v=brightness"
Write-Host "3) Press SPACE -> countdown -> drag the arm slowly and smoothly"
Write-Host "4) Press SPACE again to stop, or Q to quit"
Write-Host "   Overlay keys: l=laser  c=charuco"
Write-Host ""

& $Python @Arguments
$code = $LASTEXITCODE
if ($code -ne 0 -and $code -ne 2) {
    throw "Continuous capture failed with exit code $code."
}
if ($code -eq 2) {
    throw "No images were saved. Keep the board visible and move more slowly."
}

$Count = @(Get-ChildItem -Path $ScanDir -Filter "img_*.png" -ErrorAction SilentlyContinue).Count
Write-Host ""
Write-Host "Saved $Count images to $ScanDir"
Write-Host "Next:"
Write-Host "  powershell -ExecutionPolicy Bypass -File `"$ArmRoot\1_check_poses.ps1`""
