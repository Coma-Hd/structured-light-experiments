param(
    [double]$VoxelMm = 0.5
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment is missing. Run .\install.ps1 -Action setup from $ProjectRoot."
}

$Anchors23 = "ceshi/rail/two_faces/anchors_face2_face3.json"
$Anchors34 = "ceshi/rail/two_faces/anchors_face3_face4.json"
$Pair12Transform = "ceshi/rail/two_faces/output/transform_face2_to_face1.json"
$Pair12Report = "ceshi/rail/two_faces/output/registration_report.json"

if (-not (Test-Path $Anchors23)) {
    throw "Missing $Anchors23. Pick Face2-Face3 shared-edge anchors first."
}
if (-not (Test-Path $Anchors34)) {
    throw "Missing $Anchors34. Pick Face3-Face4 shared-edge anchors first."
}
if (-not (Test-Path $Pair12Transform)) {
    throw "Missing accepted Face1-Face2 transform: $Pair12Transform"
}
if (-not (Test-Path $Pair12Report)) {
    throw "Missing accepted Face1-Face2 report: $Pair12Report"
}

& $Python scripts/merge_face_chain.py `
    --anchors-23 $Anchors23 `
    --anchors-34 $Anchors34 `
    --voxel-mm $VoxelMm
if ($LASTEXITCODE -ne 0) {
    throw "Four-face chain registration failed or a pair did not pass quality gates."
}

Write-Host ""
Write-Host "Four-face chain registration done. Check:"
Write-Host "  ceshi/rail/two_faces/output/chain/chain_comparison_colored.ply"
Write-Host "  ceshi/rail/two_faces/output/chain/chain_merged_clean.ply"
Write-Host "  ceshi/rail/two_faces/output/chain/chain_registration_report.json"
Write-Host "Note: Face4-Face1 has no shared observation and was not force-closed."
