param(
    [Parameter(Mandatory = $true)]
    [string]$MotorPort,
    [Parameter(Mandatory = $true)]
    [double]$AngularVelocityDegS,
    [double]$StepDeg = 2.0,
    [double]$StartDeg = 0.0,
    [double]$MaxRotationDeg = 180.0,
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
$OutDir = Join-Path $Root "data\calibration\turntable_axis"
$Arguments = @(
    "scripts/continuous_turntable_capture.py",
    "--out", $OutDir,
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
}

Write-Host "Turntable axis calibration capture"
Write-Host "  output        : $OutDir"
Write-Host "  motor port    : $MotorPort"
Write-Host "  angular speed : $AngularVelocityDegS deg/s"
Write-Host "  angle spacing : $StepDeg deg"
Write-Host ""
Write-Host "Fix the ChArUco board rigidly on the turntable, offset from the axis."
Write-Host "The program will connect, detect motion, capture, and stop the motor."

& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Turntable axis capture failed with exit code $LASTEXITCODE."
}

Write-Host ""
Write-Host "Next:"
Write-Host "  powershell -ExecutionPolicy Bypass -File `"$Root\1_calibrate_axis.ps1`""
