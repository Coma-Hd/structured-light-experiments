param(
    [Parameter(Mandatory = $true)]
    [double]$DiameterMm,
    [string[]]$Inputs = @(),
    [string]$InputList = "",
    [string]$ReportName = "",
    [double]$InlierMm = 2.5,
    [double]$CommonOverlapPercentile = 2.0
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $ProjectRoot

$CurveRoot = $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment is missing. Run install.ps1 -Action setup."
}

if (-not [string]::IsNullOrWhiteSpace($InputList)) {
    $Inputs = @(
        $InputList -split ";" |
            ForEach-Object { $_.Trim() } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
}

if ($Inputs.Count -eq 0) {
    $Clean = Join-Path $CurveRoot "output\cloud_clean.ply"
    $Raw = Join-Path $CurveRoot "output\cloud.ply"
    if (Test-Path -LiteralPath $Clean) {
        $Inputs = @($Clean)
    } elseif (Test-Path -LiteralPath $Raw) {
        $Inputs = @($Raw)
    } else {
        throw "No current cloud.ply/cloud_clean.ply. Use -Inputs to select saved scans."
    }
}

$ResolvedInputs = @()
foreach ($InputPath in $Inputs) {
    if ([System.IO.Path]::IsPathRooted($InputPath)) {
        $Resolved = $InputPath
    } else {
        $Resolved = Join-Path $ProjectRoot $InputPath
    }
    if (-not (Test-Path -LiteralPath $Resolved)) {
        throw "Missing point cloud: $Resolved"
    }
    $ResolvedInputs += (Resolve-Path -LiteralPath $Resolved).Path
}

$ValidationDir = Join-Path $CurveRoot "output\validation"
New-Item -ItemType Directory -Force -Path $ValidationDir | Out-Null
if ([string]::IsNullOrWhiteSpace($ReportName)) {
    $ReportName = if ($ResolvedInputs.Count -gt 1) {
        "hemisphere_repeatability"
    } else {
        "hemisphere_validation"
    }
}
$OutMd = Join-Path $ValidationDir ($ReportName + ".md")
$OutJson = Join-Path $ValidationDir ($ReportName + ".json")
$Intrinsic = Join-Path $CurveRoot "calibration\camera_intrinsic.yaml"
$LaserPlane = Join-Path $CurveRoot "calibration\laser_plane.yaml"

$Arguments = @(
    "scripts\validate_hemisphere.py",
    "--inputs"
) + $ResolvedInputs + @(
    "--diameter-mm", $DiameterMm.ToString(
        [System.Globalization.CultureInfo]::InvariantCulture),
    "--inlier-mm", $InlierMm.ToString(
        [System.Globalization.CultureInfo]::InvariantCulture),
    "--common-overlap-percentile", $CommonOverlapPercentile.ToString(
        [System.Globalization.CultureInfo]::InvariantCulture),
    "--out-md", $OutMd,
    "--out-json", $OutJson,
    "--intrinsic", $Intrinsic,
    "--laser-plane", $LaserPlane
)

& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Hemisphere validation failed."
}

Write-Host ""
Write-Host "Hemisphere validation completed:"
Write-Host "  $OutMd"
Write-Host "  $OutJson"
