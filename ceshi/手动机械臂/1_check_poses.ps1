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
$Intrinsic = Join-Path $ArmRoot "calibration\camera_intrinsic.yaml"
$OutDir = Join-Path $ArmRoot "output"
$OutPrefix = Join-Path $OutDir "pose_check"

if (-not (Test-Path -LiteralPath $Intrinsic)) {
    throw "Missing $Intrinsic. Calibrate camera intrinsic into this folder first."
}
$Images = @(Get-ChildItem -Path $ScanDir -Filter "img_*.png" -ErrorAction SilentlyContinue)
if ($Images.Count -eq 0) {
    throw "No scan images in $ScanDir. Run 0_capture.ps1 first."
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host "Checking per-frame ChArUco poses (no rail/turntable required)..."
& $Python (Join-Path $ArmRoot "check_poses.py") `
    --config $Config `
    --images $ScanDir `
    --intrinsic $Intrinsic `
    --out $OutPrefix
$code = $LASTEXITCODE
if ($code -ne 0 -and $code -ne 2) {
    throw "Pose check failed with exit code $code."
}
if ($code -eq 2) {
    Write-Host ""
    Write-Host "Pose coverage is weak. Capture more viewpoints with the board visible,"
    Write-Host "then re-run this script before rebuild."
    exit 2
}

Write-Host ""
Write-Host "Next: draw keyframe ROI and check laser centers"
Write-Host "  powershell -ExecutionPolicy Bypass -File `"$ArmRoot\2_draw_check.ps1`""
