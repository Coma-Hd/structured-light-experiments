param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("face1", "face2", "face3", "face4")]
    [string]$Face,
    [double]$VelocityMmS = 1.0,
    [double]$StepMm = 0.5,
    [double]$StartMm = 0.0,
    [double]$MaxTravelMm = 0.0,
    [int]$Cam = 0,
    [int]$Width = 800,
    [int]$Height = 600,
    [double]$Exposure = -3,
    [double]$Gain = 20,
    [switch]$ClearOutput
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment is missing. Run .\install.ps1 -Action setup from $ProjectRoot."
}

$ScanDir = "ceshi/rail/scan/two_faces_$Face"
$Config = "ceshi/rail/two_faces/${Face}_scan.yaml"
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
}

Write-Host ""
Write-Host "Continuous rail capture: $Face"
Write-Host "  velocity: $VelocityMmS mm/s"
Write-Host "  spacing : $StepMm mm"
Write-Host "  output  : $ScanDir"
Write-Host ""
Write-Host "1. Start the motor before the object and wait for stable speed."
Write-Host "2. Press SPACE before the laser reaches the wanted scan start."
Write-Host "3. Press SPACE/S/Q while the motor is still moving to finish."
Write-Host "4. Stop the motor only after capture has finished."
Write-Host "Preview keys: E/D exposure, R/F gain, B/V brightness, L laser, C ChArUco."
Write-Host ""

& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Continuous capture failed for $Face (exit code $LASTEXITCODE)"
}

$PositionsCsv = "$ScanDir/positions.csv"
if (-not (Test-Path $PositionsCsv)) {
    throw "Capture finished without $PositionsCsv"
}

Write-Host ""
Write-Host "Capture prepared:"
Write-Host "  $ScanDir"
Write-Host "  $PositionsCsv"
Write-Host ""
Write-Host "Next:"
Write-Host "  powershell -ExecutionPolicy Bypass -File ceshi/rail/two_faces/2_draw_check_face.ps1 -Face $Face -Keyframes 5 -Samples 9"
