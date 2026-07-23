param(
    [double]$VoxelMm = 0.5,
    [double]$MaxTranslationMm = 25.0,
    [double]$MaxAnchorRmseMm = 5.0
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment is missing. Run .\install.ps1 -Action setup from $ProjectRoot."
}

$AnchorFiles = @(
    "ceshi/rail/two_faces/anchors_loop_face1_face2.json",
    "ceshi/rail/two_faces/anchors_loop_face2_face3.json",
    "ceshi/rail/two_faces/anchors_loop_face3_face4.json",
    "ceshi/rail/two_faces/anchors_loop_face4_face1.json"
)
foreach ($AnchorFile in $AnchorFiles) {
    if (-not (Test-Path $AnchorFile)) {
        throw "Missing $AnchorFile. Run 7_pick_four_edges.ps1 first."
    }
}

& $Python scripts/align_four_faces_translation_loop.py `
    --voxel-mm $VoxelMm `
    --max-translation-mm $MaxTranslationMm `
    --max-anchor-rmse-mm $MaxAnchorRmseMm
if ($LASTEXITCODE -ne 0) {
    throw "Four-edge translation loop failed its quality gates."
}

Write-Host ""
Write-Host "Four-edge translation loop accepted. Check:"
Write-Host "  ceshi/rail/two_faces/output/translation_loop/loop_comparison_colored.ply"
Write-Host "  ceshi/rail/two_faces/output/translation_loop/loop_merged_candidate.ply"
Write-Host "  ceshi/rail/two_faces/output/translation_loop/loop_alignment_report.json"
