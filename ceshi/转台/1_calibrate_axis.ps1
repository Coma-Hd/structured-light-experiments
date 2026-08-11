$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $ProjectRoot

$Root = $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment is missing. Run .\install.ps1 -Action setup."
}
$Config = Join-Path $Root "turntable_scan.yaml"
$Images = Join-Path $Root "data\calibration\turntable_axis"
$Angles = Join-Path $Images "angles.csv"
$Intrinsic = Join-Path $Root "calibration\camera_intrinsic.yaml"
$Out = Join-Path $Root "calibration\turntable_axis.yaml"

if (-not (Test-Path -LiteralPath $Angles)) {
    throw "Missing $Angles. Run 0_axis_capture.ps1 first."
}

& $Python scripts/calibrate_turntable_axis.py `
    --config $Config `
    --images $Images `
    --angles $Angles `
    --intrinsic $Intrinsic `
    --out $Out
$CalibrationExitCode = $LASTEXITCODE
if ($CalibrationExitCode -eq 3) {
    throw "Turntable axis calibration quality is NOT acceptable. Review the quality report above."
}
if ($CalibrationExitCode -ne 0) {
    throw "Turntable axis calibration failed."
}

Write-Host ""
Write-Host "Axis calibration completed: $Out"
Write-Host "Next:"
Write-Host "  powershell -ExecutionPolicy Bypass -File `"$Root\2_capture.ps1`" -MotorPort <COMx> -AngularVelocityDegS <value> -ClearOutput"
