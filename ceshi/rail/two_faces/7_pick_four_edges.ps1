$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $ProjectRoot

$Pairs = @(
    @("face1", "face2"),
    @("face2", "face3"),
    @("face3", "face4"),
    @("face4", "face1")
)

Write-Host "Four-edge loop anchor picking"
Write-Host "For every cloud, pick the SAME physical edge points in this order:"
Write-Host "  1. shared-edge top"
Write-Host "  2. shared-edge bottom"
Write-Host "Use Shift+LeftClick, then Q."
Write-Host ""

foreach ($Pair in $Pairs) {
    $TargetFace = $Pair[0]
    $SourceFace = $Pair[1]
    $Out = "ceshi/rail/two_faces/anchors_loop_${TargetFace}_${SourceFace}.json"

    Write-Host "============================================================"
    Write-Host "Picking $TargetFace <-> $SourceFace"
    Write-Host "Output: $Out"
    Write-Host "============================================================"

    & "$PSScriptRoot/4_pick_shared_edge.ps1" `
        -TargetFace $TargetFace `
        -SourceFace $SourceFace `
        -Out $Out
    if ($LASTEXITCODE -ne 0) {
        throw "Anchor picking failed for $TargetFace/$SourceFace."
    }
}

Write-Host ""
Write-Host "All four edge anchor files are ready."
Write-Host "Next:"
Write-Host "  powershell -ExecutionPolicy Bypass -File ceshi/rail/two_faces/8_align_four_faces_loop.ps1"
