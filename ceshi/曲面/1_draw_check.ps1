param(
    [int]$Keyframes = 5,
    [int]$Samples = 9
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $ProjectRoot

$CurveRoot = $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment is missing. Run .\install.ps1 -Action setup from $ProjectRoot."
}
$Config = Join-Path $CurveRoot "curve_scan.yaml"
$ScanDir = Join-Path $CurveRoot "data\scan"
$PositionsCsv = "$ScanDir/positions.csv"
$WorkDir = Join-Path $CurveRoot "work"
$RoiJson = "$WorkDir/roi_keyframes.json"
$CheckDir = "$WorkDir/laser_check"

if (-not (Test-Path $PositionsCsv)) {
    throw "Missing $PositionsCsv. Run 0_capture.ps1 first."
}
New-Item -ItemType Directory -Force -Path $CheckDir | Out-Null

if ($Keyframes -lt 3) { $Keyframes = 3 }
if ($Keyframes -gt 5) { $Keyframes = 5 }
& $Python scripts/draw_keyframe_roi.py `
    --config $Config `
    --images $ScanDir `
    --positions $PositionsCsv `
    --out $RoiJson `
    --n $Keyframes
if ($LASTEXITCODE -ne 0) {
    throw "Curve ROI drawing failed."
}

$Images = @(Get-ChildItem -Path $ScanDir -Filter "img_*.png" | Sort-Object Name)
if ($Images.Count -eq 0) {
    throw "No scan images in $ScanDir."
}
if ($Samples -lt 1) { $Samples = 1 }
if ($Samples -gt $Images.Count) { $Samples = $Images.Count }

$Indices = @()
if ($Samples -eq 1) {
    $Indices += 0
} else {
    for ($i = 0; $i -lt $Samples; $i++) {
        $Indices += [int][math]::Round(
            $i * ($Images.Count - 1) / ($Samples - 1)
        )
    }
}

foreach ($Index in $Indices) {
    $Image = $Images[$Index]
    $Save = "$CheckDir/check_$('{0:D3}' -f $Index).png"
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
Write-Host "Check: $CheckDir"
Write-Host "The green ROI must cover the complete curved surface."
Write-Host "Red centers must follow the laser core and must not enter the board/background."
Write-Host "If one image row contains several separate laser segments, stop before rebuilding."
Write-Host ""
Write-Host "Next:"
Write-Host "  powershell -ExecutionPolicy Bypass -File `"$CurveRoot\2_rebuild.ps1`""
