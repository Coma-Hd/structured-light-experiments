param(
    [Parameter(Mandatory = $true)]
    [string]$MotorPort,
    [Parameter(Mandatory = $true)]
    [double]$AngularVelocityDegS,
    [double]$StepDeg = 0.2,
    [double]$StartDeg = 0.0,
    [double]$MaxRotationDeg = 360.0,
    [int]$Cam = 0,
    [int]$Width = 800,
    [int]$Height = 600,
    [double]$Exposure = -4,
    [double]$Gain = 10,
    [int]$WarmupFrames = 5,
    [switch]$ClearOutput
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $ProjectRoot

$Root = $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment is missing. Run .\install.ps1 -Action setup."
}
$Axis = Join-Path $Root "calibration\turntable_axis.yaml"
if (-not (Test-Path -LiteralPath $Axis)) {
    throw "Missing $Axis. Run 0_axis_capture.ps1 and 1_calibrate_axis.ps1 first."
}
$ScanDir = Join-Path $Root "data\scan"
$RoiJson = Join-Path $Root "work\roi_keyframes.json"
$Arguments = @(
    "scripts/continuous_turntable_capture.py",
    "--out", $ScanDir,
    "--motor-port", $MotorPort,
    "--angular-velocity-deg-s", $AngularVelocityDegS,
    "--step-deg", $StepDeg,
    "--start-deg", $StartDeg,
    "--max-rotation-deg", $MaxRotationDeg,
    "--cam", $Cam,
    "--width", $Width,
    "--height", $Height,
    "--exposure", $Exposure,
    "--gain", $Gain,
    "--warmup-frames", $WarmupFrames
)
if ($ClearOutput) {
    $Arguments += "--clear-output"
    if (Test-Path -LiteralPath $RoiJson) {
        Remove-Item -LiteralPath $RoiJson -Force
        Write-Host "  removed stale keyframe ROI: $RoiJson"
    }
}

Write-Host "Turntable object scan"
Write-Host "  output        : $ScanDir"
Write-Host "  motor port    : $MotorPort"
Write-Host "  angular speed : $AngularVelocityDegS deg/s"
Write-Host "  angle spacing : $StepDeg deg"
Write-Host ""
Write-Host "Keep camera and laser fixed. The object must be rigidly fixed to the turntable."
Write-Host "The program will connect, detect motion, capture, and stop the motor."

& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Turntable capture failed with exit code $LASTEXITCODE."
}

$Angles = Join-Path $ScanDir "angles.csv"
if (-not (Test-Path -LiteralPath $Angles)) {
    throw "Capture finished without $Angles."
}

Write-Host ""
Write-Host "Next:"
Write-Host "  powershell -ExecutionPolicy Bypass -File `"$Root\3_draw_check.ps1`""
