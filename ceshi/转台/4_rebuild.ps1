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
$CaptureReport = Join-Path $ScanDir "continuous_capture_report.json"
$RoiJson = Join-Path $Root "work\roi_keyframes.json"
$Axis = Join-Path $Root "calibration\turntable_axis.yaml"
$OutDir = Join-Path $Root "output"
$RawPly = Join-Path $OutDir "cloud.ply"
$CleanPly = Join-Path $OutDir "cloud_clean.ply"

foreach ($Required in @($Angles, $CaptureReport, $RoiJson, $Axis)) {
    if (-not (Test-Path -LiteralPath $Required)) {
        throw "Missing $Required. Complete the earlier steps first."
    }
}

$Capture = Get-Content -Raw -Encoding UTF8 -Path $CaptureReport |
    ConvertFrom-Json
$ExpectedImages = [int]$Capture.saved_images
$ImageFiles = @(
    Get-ChildItem -Path $ScanDir -Filter "img_*.png" | Sort-Object Name
)
$AngleData = @(Import-Csv -Path $Angles)
$ActualImages = $ImageFiles.Count
$AngleRows = $AngleData.Count
if ($ExpectedImages -le 0 -or $AngleRows -ne $ExpectedImages) {
    throw (
        "Incomplete capture: report=$ExpectedImages, " +
        "png=$ActualImages, angles=$AngleRows."
    )
}

# A keyboard stop or external file cleanup can leave only a trailing suffix of
# reported images missing. The remaining prefix is still a valid partial-angle
# scan. Missing images in the middle are unsafe because they create angle holes.
if ($ActualImages -le 0) {
    throw "Capture contains no PNG images."
}
if ($ActualImages -gt $AngleRows) {
    throw "Capture has more PNG files than angle rows."
}
for ($i = 0; $i -lt $ActualImages; $i++) {
    if ($ImageFiles[$i].Name -ne $AngleData[$i].image) {
        throw (
            "Capture has a missing or mismatched image inside the sequence " +
            "at index ${i}: png=$($ImageFiles[$i].Name), " +
            "angles=$($AngleData[$i].image)."
        )
    }
}
if ($ActualImages -lt $ExpectedImages) {
    $MissingCount = $ExpectedImages - $ActualImages
    $LastAngle = [double]$AngleData[$ActualImages - 1].angle_deg
    Write-Warning (
        "The final $MissingCount reported images are absent. " +
        "Rebuilding the valid $ActualImages-image prefix through " +
        "$('{0:F3}' -f $LastAngle) deg."
    )
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

& $Python scripts/3_reconstruct.py `
    --config $Config `
    --images $ScanDir `
    --out $RawPly `
    --pose-source turntable `
    --angles $Angles
if ($LASTEXITCODE -ne 0) {
    throw "Turntable reconstruction failed."
}

& $Python scripts/4_postprocess.py --config $Config --in $RawPly
if ($LASTEXITCODE -ne 0) {
    throw "Turntable postprocess failed."
}

& $Python scripts/inspect_cloud.py --in $RawPly --out "$OutDir\inspect_raw"
if (Test-Path -LiteralPath $CleanPly) {
    & $Python scripts/inspect_cloud.py --in $CleanPly --out "$OutDir\inspect_clean"
}

Write-Host ""
Write-Host "Turntable reconstruction completed:"
Write-Host "  $RawPly"
Write-Host "  $CleanPly"
Write-Host "  $OutDir\inspect_raw"
Write-Host "  $OutDir\inspect_clean"
