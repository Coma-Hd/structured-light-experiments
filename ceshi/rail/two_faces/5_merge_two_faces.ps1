param(
    [switch]$AllowLowQuality
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $ProjectRoot

$Config = "ceshi/rail/two_faces/config.yaml"
$Face1 = "ceshi/rail/two_faces/input/face1/cloud_clean.ply"
$Face2 = "ceshi/rail/two_faces/input/face2/cloud_clean.ply"
$OutDir = "ceshi/rail/two_faces/output"

if (-not (Test-Path $Face1)) { throw "Missing face1: $Face1" }
if (-not (Test-Path $Face2)) { throw "Missing face2: $Face2" }

$Args = @(
    "scripts/6_merge_two_faces.py",
    "--config", $Config,
    "--target", $Face1,
    "--source", $Face2,
    "--out-dir", $OutDir
)
if ($AllowLowQuality) { $Args += "--allow-low-quality" }

python @Args
if ($LASTEXITCODE -ne 0) {
    throw "Two-face registration failed or did not pass quality thresholds."
}

Write-Host ""
Write-Host "Two-face merge done. Check:"
Write-Host "  1. $OutDir/comparison_colored.ply"
Write-Host "  2. $OutDir/aligned/face2_aligned.ply"
Write-Host "  3. $OutDir/cloud_merged_clean.ply"
Write-Host "  4. $OutDir/registration_report.json"
