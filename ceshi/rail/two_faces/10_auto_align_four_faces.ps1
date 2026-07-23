param(
    [double]$VoxelMm = 1.5,
    [string]$DistanceLevelsMm = "15,8,4",
    [double]$NormalAngleDeg = 18.0,
    [int]$MinCorrespondences = 100,
    [double]$MinCoverage = 0.02,
    [double]$MaxTranslationMm = 15.0,
    [double]$MaxRotationDeg = 5.0,
    [double]$MaxFinalRmseMm = 3.0,
    [double]$MergeVoxelMm = 0.5
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment is missing. Run .\install.ps1 -Action setup from $ProjectRoot."
}

& $Python scripts/auto_align_four_faces.py `
    --voxel-mm $VoxelMm `
    --distance-levels-mm $DistanceLevelsMm `
    --normal-angle-deg $NormalAngleDeg `
    --min-correspondences $MinCorrespondences `
    --min-coverage $MinCoverage `
    --max-translation-mm $MaxTranslationMm `
    --max-rotation-deg $MaxRotationDeg `
    --max-final-rmse-mm $MaxFinalRmseMm `
    --merge-voxel-mm $MergeVoxelMm
$ExitCode = $LASTEXITCODE

if ($ExitCode -eq 2) {
    Write-Host ""
    Write-Warning "SAFE REJECTION: quality gates failed. Candidate clouds are diagnostic only, not measurement results."
    Write-Host "Report: ceshi/rail/two_faces/output/auto_alignment/auto_alignment_report.json"
    exit 2
}
if ($ExitCode -ne 0) {
    throw "Constrained four-face alignment failed with exit code $ExitCode."
}

Write-Host ""
Write-Host "Constrained alignment accepted. Check:"
Write-Host "  ceshi/rail/two_faces/output/auto_alignment/comparison_colored.ply"
Write-Host "  ceshi/rail/two_faces/output/auto_alignment/merged_candidate.ply"
Write-Host "  ceshi/rail/two_faces/output/auto_alignment/auto_alignment_report.json"
