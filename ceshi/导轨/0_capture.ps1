param(
    [double]$VelocityMmS = 1.0,
    [double]$StepMm = 0.2,
    [double]$StartMm = 0.0,
    [double]$MaxTravelMm = 0.0,
    [int]$Cam = 0,
    [int]$Width = 800,
    [int]$Height = 600,
    [double]$Exposure = -4,
    [double]$Gain = 10,
    [switch]$ClearOutput
)

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
$ObjectMaskDir = Join-Path $CurveRoot "work\object_masks"
$RoiFile = Join-Path $CurveRoot "work\roi_keyframes.json"
$Arguments = @(
    "scripts/continuous_rail_capture.py",
    "--out", $ScanDir,
    "--config", $Config,
    "--velocity-mm-s", $VelocityMmS,
    "--step-mm", $StepMm,
    "--start-mm", $StartMm,
    "--max-travel-mm", $MaxTravelMm,
    "--cam", $Cam,
    "--width", $Width,
    "--height", $Height,
    "--exposure", $Exposure,
    "--gain", $Gain
)
if ($ClearOutput) {
    $Arguments += "--clear-output"
    if (Test-Path -LiteralPath $ObjectMaskDir) {
        Remove-Item -LiteralPath $ObjectMaskDir -Recurse -Force
        Write-Host "  removed stale object masks: $ObjectMaskDir"
    }
    if (Test-Path -LiteralPath $RoiFile) {
        Remove-Item -LiteralPath $RoiFile -Force
        Write-Host "  removed stale keyframe ROI: $RoiFile"
    }
}

Write-Host "Curve capture"
Write-Host "  output   : $ScanDir"
Write-Host "  velocity : $VelocityMmS mm/s"
Write-Host "  spacing  : $StepMm mm"
Write-Host ""
Write-Host "Keep the curved object and ChArUco board rigidly fixed."
Write-Host "Press SPACE to start and SPACE/S/Q to finish while the rail is moving."

& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Curve capture failed with exit code $LASTEXITCODE."
}

$PositionsCsv = "$ScanDir/positions.csv"
if (-not (Test-Path $PositionsCsv)) {
    throw "Capture finished without $PositionsCsv."
}

Write-Host ""
Write-Host "Next:"
Write-Host "  powershell -ExecutionPolicy Bypass -File `"$CurveRoot\1_draw_check.ps1`""
