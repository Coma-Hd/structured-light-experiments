$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $ProjectRoot

$CurveRoot = $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment is missing. Run .\install.ps1 -Action setup from $ProjectRoot."
}
$Config = Join-Path $CurveRoot "curve_scan.yaml"
$ScanDir = Join-Path $CurveRoot "data\scan"
$PositionsCsv = "$ScanDir/positions.csv"
$CaptureReport = "$ScanDir/continuous_capture_report.json"
$RoiFile = Join-Path $CurveRoot "work\roi_keyframes.json"
$OutDir = Join-Path $CurveRoot "output"
$RawPly = "$OutDir/cloud.ply"
$CleanPly = "$OutDir/cloud_clean.ply"

if (-not (Test-Path $RoiFile)) {
    throw "Missing $RoiFile. Run 1_draw_check.ps1 first."
}
if (-not (Test-Path $PositionsCsv)) {
    throw "Missing $PositionsCsv."
}
if (-not (Test-Path $CaptureReport)) {
    throw "Missing $CaptureReport. Capture may not have completed."
}

$Capture = Get-Content -Raw -Encoding UTF8 -Path $CaptureReport |
    ConvertFrom-Json
$ExpectedImages = [int]$Capture.saved_images
$ActualImages = @(
    Get-ChildItem -Path $ScanDir -Filter "img_*.png"
).Count
$PositionRows = @(Import-Csv -Path $PositionsCsv).Count
if (
    $ExpectedImages -le 0 -or
    $ActualImages -ne $ExpectedImages -or
    $PositionRows -ne $ExpectedImages
) {
    throw (
        "Incomplete capture: report=$ExpectedImages, " +
        "png=$ActualImages, positions=$PositionRows."
    )
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

& $Python scripts/3_reconstruct.py `
    --config $Config `
    --images $ScanDir `
    --out $RawPly `
    --positions $PositionsCsv
if ($LASTEXITCODE -ne 0) {
    throw "Curve reconstruction failed."
}

& $Python scripts/4_postprocess.py --config $Config --in $RawPly
if ($LASTEXITCODE -ne 0) {
    throw "Curve postprocess failed."
}

& $Python scripts/inspect_cloud.py --in $RawPly --out "$OutDir/inspect_raw"
if (Test-Path $CleanPly) {
    & $Python scripts/inspect_cloud.py --in $CleanPly --out "$OutDir/inspect_clean"
}

Write-Host ""
Write-Host "Curve reconstruction completed:"
Write-Host "  $RawPly"
Write-Host "  $CleanPly"
Write-Host "  $OutDir/inspect_raw"
Write-Host "  $OutDir/inspect_clean"
