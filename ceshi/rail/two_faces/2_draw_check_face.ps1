param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("face1", "face2", "face3", "face4")]
    [string]$Face,
    [int]$Keyframes = 5,
    [int]$Samples = 9
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment is missing. Run .\install.ps1 -Action setup from $ProjectRoot."
}

$Config = "ceshi/rail/two_faces/${Face}_scan.yaml"
$ScanDir = "ceshi/rail/scan/two_faces_$Face"
$PositionsCsv = "$ScanDir/positions.csv"
$WorkDir = "ceshi/rail/two_faces/work/$Face"
$RoiJson = "$WorkDir/roi_keyframes.json"
$CheckDir = "$WorkDir/laser_check"
New-Item -ItemType Directory -Force -Path $CheckDir | Out-Null

if (-not (Test-Path $PositionsCsv)) {
    throw "Missing $PositionsCsv. Run 1_make_positions.ps1 first."
}
if ($Keyframes -lt 3) { $Keyframes = 3 }
if ($Keyframes -gt 5) { $Keyframes = 5 }

& $Python scripts/draw_keyframe_roi.py `
    --config $Config `
    --images $ScanDir `
    --positions $PositionsCsv `
    --out $RoiJson `
    --n $Keyframes
if ($LASTEXITCODE -ne 0) {
    throw "ROI drawing failed for $Face"
}

$Images = @(Get-ChildItem -Path $ScanDir -Filter *.png | Sort-Object Name)
if ($Samples -lt 1) { $Samples = 1 }
if ($Samples -gt $Images.Count) { $Samples = $Images.Count }
$Indices = @()
if ($Samples -eq 1) {
    $Indices += 0
} else {
    for ($i = 0; $i -lt $Samples; $i++) {
        $Indices += [int][math]::Round($i * ($Images.Count - 1) / ($Samples - 1))
    }
}

foreach ($idx in $Indices) {
    $Image = $Images[$idx]
    $Save = "$CheckDir/check_$('{0:D3}' -f $idx).png"
    & $Python scripts/debug_laser.py `
        --config $Config `
        --image $Image.FullName `
        --save $Save `
        --roi-json $RoiJson `
        --positions $PositionsCsv
    if ($LASTEXITCODE -ne 0) {
        throw "Laser check failed: $($Image.FullName)"
    }
}

Write-Host ""
Write-Host "Check images: $CheckDir"
Write-Host "Green ROI must cover the same physical faces; red points must not enter background."
Write-Host "Keep the current main face, both shared edges, and visible neighbor-face strips."
