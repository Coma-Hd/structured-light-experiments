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
$LaserPlane = Join-Path $ArmRoot "calibration\laser_plane.yaml"
$RoiFile = Join-Path $ArmRoot "work\roi_keyframes.json"
$OutDir = Join-Path $ArmRoot "output"
$RawPly = Join-Path $OutDir "cloud.ply"
$CleanPly = Join-Path $OutDir "cloud_clean.ply"

if (-not (Test-Path -LiteralPath $Intrinsic)) {
    throw "Missing $Intrinsic"
}
if (-not (Test-Path -LiteralPath $LaserPlane)) {
    throw "Missing $LaserPlane. Calibrate laser plane for this camera-laser mount first."
}
if (-not (Test-Path -LiteralPath $RoiFile)) {
    throw "Missing $RoiFile. Run 2_draw_check.ps1 to draw keyframe ROI first."
}
$Images = @(Get-ChildItem -Path $ScanDir -Filter "img_*.png" -ErrorAction SilentlyContinue)
if ($Images.Count -eq 0) {
    throw "No scan images in $ScanDir. Run 0_capture.ps1 first."
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host "Reconstructing with pose_source=charuco (per-frame PnP)..."
& $Python scripts/3_reconstruct.py `
    --config $Config `
    --images $ScanDir `
    --intrinsic $Intrinsic `
    --laser $LaserPlane `
    --out $RawPly `
    --pose-source charuco
if ($LASTEXITCODE -ne 0) {
    throw "Arm reconstruction failed."
}

& $Python scripts/4_postprocess.py --config $Config --in $RawPly
if ($LASTEXITCODE -ne 0) {
    throw "Postprocess failed."
}

& $Python scripts/inspect_cloud.py --in $RawPly --out "$OutDir/inspect_raw"
if (Test-Path -LiteralPath $CleanPly) {
    & $Python scripts/inspect_cloud.py --in $CleanPly --out "$OutDir/inspect_clean"
}

Write-Host ""
Write-Host "Arm reconstruction completed:"
Write-Host "  $RawPly"
if (Test-Path -LiteralPath $CleanPly) {
    Write-Host "  $CleanPly"
}
Write-Host "  $OutDir/inspect_raw"
