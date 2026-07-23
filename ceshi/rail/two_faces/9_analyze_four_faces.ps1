param(
    [int]$PoseCompareSamples = 9,
    [switch]$SkipPoseComparison,
    [switch]$SkipColoredPly
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $ProjectRoot

$Arguments = @(
    "scripts/analyze_four_face_alignment.py",
    "--pose-compare-samples", "$PoseCompareSamples"
)
if ($SkipPoseComparison) {
    $Arguments += "--no-pose-mode-comparison"
}
if (-not $SkipColoredPly) {
    $Arguments += "--write-colored-ply"
}

& python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Four-face diagnostics failed."
}

Write-Host ""
Write-Host "Diagnostics completed. Check:"
Write-Host "  ceshi/rail/two_faces/output/auto_diagnostics/four_face_diagnostics.json"
if (-not $SkipColoredPly) {
    Write-Host "  ceshi/rail/two_faces/output/auto_diagnostics/four_faces_colored.ply"
}
