param(
    [string[]]$Faces = @("face1", "face2", "face3", "face4"),
    [double]$VoxelMm = 0.5
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment is missing. Run .\install.ps1 -Action setup from $ProjectRoot."
}

# 使用 powershell -File 时，逗号分隔值会作为一个字符串传入，而不是数组。
# 同时兼容 "-Faces face1,face2,face3,face4" 和真正的 string[] 输入。
$AllowedFaces = @("face1", "face2", "face3", "face4")
$Faces = @(
    foreach ($FaceArgument in $Faces) {
        foreach ($FaceName in ($FaceArgument -split ",")) {
            $NormalizedFace = $FaceName.Trim().ToLowerInvariant()
            if ($NormalizedFace) {
                $NormalizedFace
            }
        }
    }
)
$InvalidFaces = @($Faces | Where-Object { $_ -notin $AllowedFaces })
if ($InvalidFaces.Count -gt 0) {
    throw "Invalid Faces: $($InvalidFaces -join ', '). Allowed: $($AllowedFaces -join ', ')"
}
if ($Faces.Count -eq 0) {
    throw "Faces cannot be empty."
}

$Inputs = @()
foreach ($Face in $Faces) {
    $Cloud = "ceshi/rail/two_faces/input/$Face/cloud_clean.ply"
    if (-not (Test-Path $Cloud)) {
        throw "Missing $Cloud. Rebuild $Face first."
    }
    $TrackingReport = "ceshi/rail/two_faces/work/$Face/output/cloud_charuco_tracking.yaml"
    if (-not (Test-Path $TrackingReport)) {
        throw "Missing $TrackingReport. $Face was not reconstructed with ChArUco rail_fit."
    }
    $Inputs += $Cloud
}

$OutDir = "ceshi/rail/two_faces/output"
$OutPly = "$OutDir/board_merged.ply"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$Arguments = @(
    "scripts/merge_board_faces.py",
    "--inputs"
) + $Inputs + @(
    "--out", $OutPly,
    "--voxel-mm", $VoxelMm
)

& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Board-coordinate cloud merge failed (exit code $LASTEXITCODE)"
}

Write-Host ""
Write-Host "Merged without ICP because every face is already in the same board coordinate system:"
Write-Host "  $OutPly"
