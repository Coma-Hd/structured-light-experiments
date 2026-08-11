param(
    [int]$Keyframes = 5,
    [int]$Samples = 12
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $ProjectRoot

$Root = $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment is missing. Run .\install.ps1 -Action setup."
}
$Config = Join-Path $Root "turntable_scan.yaml"
$ScanDir = Join-Path $Root "data\scan"
$Angles = Join-Path $ScanDir "angles.csv"
$WorkDir = Join-Path $Root "work"
$RoiJson = Join-Path $WorkDir "roi_keyframes.json"
$CheckDir = Join-Path $WorkDir "laser_check"

if (-not (Test-Path -LiteralPath $Angles)) {
    throw "Missing $Angles. Run 2_capture.ps1 first."
}
New-Item -ItemType Directory -Force -Path $CheckDir | Out-Null

if ($Keyframes -lt 3) { $Keyframes = 3 }
if ($Keyframes -gt 5) { $Keyframes = 5 }

& $Python scripts/draw_keyframe_roi.py `
    --config $Config `
    --images $ScanDir `
    --angles $Angles `
    --out $RoiJson `
    --n $Keyframes
if ($LASTEXITCODE -ne 0) {
    throw "Turntable keyframe ROI generation failed."
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
    $Save = Join-Path $CheckDir "check_$('{0:D3}' -f $Index).png"
    & $Python scripts/debug_laser.py `
        --config $Config `
        --image $Image.FullName `
        --save $Save `
        --angles $Angles
    if ($LASTEXITCODE -ne 0) {
        throw "Laser check failed: $($Image.FullName)"
    }
}

Write-Host ""
Write-Host "Check: $CheckDir"
Write-Host "The green interpolated ROI must contain only the required object laser."
Write-Host "Red centers must follow the laser core and avoid the turntable/background."
Write-Host ""
Write-Host "Next:"
Write-Host "  powershell -ExecutionPolicy Bypass -File `"$Root\4_rebuild.ps1`""
