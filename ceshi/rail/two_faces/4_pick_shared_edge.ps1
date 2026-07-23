param(
    [ValidateSet("face1", "face2", "face3", "face4")]
    [string]$TargetFace = "face1",
    [ValidateSet("face1", "face2", "face3", "face4")]
    [string]$SourceFace = "face2",
    [string]$Out = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $ProjectRoot

if ($TargetFace -eq $SourceFace) {
    throw "TargetFace and SourceFace must be different."
}
$Target = "ceshi/rail/two_faces/input/$TargetFace/cloud_clean.ply"
$Source = "ceshi/rail/two_faces/input/$SourceFace/cloud_clean.ply"
if (-not $Out) {
    if ($TargetFace -eq "face1" -and $SourceFace -eq "face2") {
        $Out = "ceshi/rail/two_faces/anchors.json"
    } else {
        $Out = "ceshi/rail/two_faces/anchors_${TargetFace}_${SourceFace}.json"
    }
}

if (-not (Test-Path $Target)) { throw "Missing $Target" }
if (-not (Test-Path $Source)) { throw "Missing $Source" }

Write-Host "Pick corresponding points in the same order:"
Write-Host "  1. shared-edge top"
Write-Host "  2. shared-edge bottom"
Write-Host "Use Shift+LeftClick, then Q."
Write-Host "Target: $TargetFace"
Write-Host "Source: $SourceFace"

python scripts/pick_two_face_anchors.py `
    --target $Target `
    --source $Source `
    --target-name $TargetFace `
    --source-name $SourceFace `
    --out $Out
if ($LASTEXITCODE -ne 0) {
    throw "Anchor picking failed."
}

Write-Host "Saved: $Out"
if ($TargetFace -eq "face1" -and $SourceFace -eq "face2") {
    Write-Host "Next: powershell -ExecutionPolicy Bypass -File ceshi/rail/two_faces/5_merge_two_faces.ps1"
} else {
    Write-Host "Complete face2-face3 and face3-face4 anchors, then run 6_merge_face_chain.ps1."
}
