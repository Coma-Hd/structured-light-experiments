param(
    [double]$SampleSpacingMm = 1.5,
    [switch]$Orthogonalize
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $ProjectRoot

$Arguments = @(
    "scripts/complete_cuboid_from_planes.py",
    "--sample-spacing-mm", "$SampleSpacingMm"
)
if ($Orthogonalize) {
    $Arguments += "--orthogonalize"
}

& python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Cuboid completion failed."
}

Write-Host ""
Write-Warning "The completion is synthetic fitted geometry, not sensor measurement."
Write-Host "Check: ceshi/rail/two_faces/output/cuboid_completion/"
