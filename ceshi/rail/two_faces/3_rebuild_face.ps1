param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("face1", "face2", "face3", "face4")]
    [string]$Face
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment is missing. Run .\install.ps1 -Action setup from $ProjectRoot."
}

$Config = "ceshi/rail/two_faces/${Face}_scan.yaml"
$ScanDir = "ceshi/rail/scan/two_faces_$Face"
$PositionsCsv = "$ScanDir/positions.csv"
$CaptureReport = "$ScanDir/continuous_capture_report.json"
$WorkDir = "ceshi/rail/two_faces/work/$Face"
$RoiJson = "$WorkDir/roi_keyframes.json"
$OutDir = "$WorkDir/output"
$InputDir = "ceshi/rail/two_faces/input/$Face"
$RawPly = "$OutDir/cloud.ply"
$CleanPly = "$OutDir/cloud_clean.ply"

if (-not (Test-Path $RoiJson)) {
    throw "Missing $RoiJson. Run 2_draw_check_face.ps1 first."
}
if (-not (Test-Path $PositionsCsv)) {
    throw "Missing $PositionsCsv."
}
if (-not (Test-Path $CaptureReport)) {
    throw "Missing $CaptureReport. Capture may still be running; wait until it finishes."
}
# Windows PowerShell 5.1 defaults to the system ANSI code page. Capture reports
# are written as UTF-8 and contain Chinese text, so decode them explicitly.
$Capture = Get-Content -Raw -Encoding UTF8 -Path $CaptureReport | ConvertFrom-Json
$ExpectedImages = [int]$Capture.saved_images
$ActualImages = @(Get-ChildItem -Path $ScanDir -Filter "img_*.png").Count
$PositionRows = @(Import-Csv -Path $PositionsCsv).Count
if (
    $ExpectedImages -le 0 -or
    $ActualImages -ne $ExpectedImages -or
    $PositionRows -ne $ExpectedImages
) {
    throw (
        "Incomplete or changing capture for ${Face}: " +
        "report=$ExpectedImages, png=$ActualImages, positions=$PositionRows. " +
        "Do not rebuild while capture with -ClearOutput is running."
    )
}
New-Item -ItemType Directory -Force -Path $OutDir, $InputDir | Out-Null

& $Python scripts/3_reconstruct.py `
    --config $Config `
    --images $ScanDir `
    --out $RawPly `
    --positions $PositionsCsv
if ($LASTEXITCODE -ne 0) {
    throw "Reconstruction failed for $Face"
}

& $Python scripts/4_postprocess.py --config $Config --in $RawPly
if ($LASTEXITCODE -ne 0) {
    throw "Postprocess failed for $Face"
}

& $Python scripts/inspect_cloud.py --in $RawPly --out "$OutDir/inspect_raw"
& $Python scripts/inspect_cloud.py --in $CleanPly --out "$OutDir/inspect_clean"

Copy-Item $CleanPly "$InputDir/cloud_clean.ply" -Force

Write-Host ""
Write-Host "Prepared registration input:"
Write-Host "  $InputDir/cloud_clean.ply"
Write-Host "Verify the neighbor-face strip still exists in cloud_clean before registration."
