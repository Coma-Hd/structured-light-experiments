param(
    [int]$Keyframes = 5,
    [int]$Samples = 9,
    [switch]$ReuseRoi
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $ProjectRoot

$ArmRoot = $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment is missing. Run .\install.ps1 -Action setup from $ProjectRoot."
}

$Config = Join-Path $ArmRoot "arm_scan.yaml"
$ScanDir = Join-Path $ArmRoot "data\scan"
$WorkDir = Join-Path $ArmRoot "work"
$RoiFile = Join-Path $WorkDir "roi_keyframes.json"
$CheckDir = Join-Path $WorkDir "laser_check"

$Images = @(Get-ChildItem -Path $ScanDir -Filter "img_*.png" -ErrorAction SilentlyContinue | Sort-Object Name)
if ($Images.Count -eq 0) {
    throw "No scan images in $ScanDir. Run 0_capture.ps1 or 0_capture_continuous.ps1 first."
}
New-Item -ItemType Directory -Force -Path $CheckDir | Out-Null

if ($ReuseRoi) {
    if (-not (Test-Path -LiteralPath $RoiFile)) {
        throw "Cannot reuse ROI because the keyframe file is missing: $RoiFile"
    }
    Write-Host "Reusing existing keyframe ROI: $RoiFile"
} else {
    Write-Host "Draw ROI on $Keyframes keyframes (interpolated by frame_index)..."
    Write-Host "Drag a box around the object laser. ENTER=next  r=redraw  s=skip  q=quit"
    & $Python scripts/draw_keyframe_roi.py `
        --config $Config `
        --images $ScanDir `
        --parameter frame_index `
        --out $RoiFile `
        --n $Keyframes
    if ($LASTEXITCODE -ne 0) {
        throw "Arm keyframe ROI selection failed."
    }
}

if ($Samples -lt 1) { $Samples = 1 }
if ($Samples -gt $Images.Count) { $Samples = $Images.Count }

$Indices = @()
if ($Samples -eq 1) {
    $Indices = @(0)
} else {
    for ($i = 0; $i -lt $Samples; $i++) {
        $Indices += [int][Math]::Round($i * ($Images.Count - 1) / ($Samples - 1))
    }
    $Indices = @($Indices | Select-Object -Unique)
}

Write-Host "Writing laser-center checks with interpolated ROI..."
foreach ($Index in $Indices) {
    $Image = $Images[$Index]
    $Save = Join-Path $CheckDir ("check_{0:D3}.png" -f $Index)
    & $Python scripts/debug_laser.py `
        --config $Config `
        --image $Image.FullName `
        --save $Save `
        --roi-json $RoiFile `
        --frame-index $Index
    if ($LASTEXITCODE -ne 0) {
        throw "Laser check failed: $($Image.Name)"
    }
    Write-Host "  saved $Save"
}

Write-Host ""
Write-Host "Check: $CheckDir"
Write-Host "The interpolated ROI must cover the object laser in every check image."
Write-Host "Red centers should follow the laser core and avoid board/background."
Write-Host ""
Write-Host "Next:"
Write-Host "  powershell -ExecutionPolicy Bypass -File `"$ArmRoot\3_rebuild.ps1`""
