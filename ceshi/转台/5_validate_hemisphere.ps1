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

$Root = $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment is missing. Run .\install.ps1 -Action setup."
}
if (-not [string]::IsNullOrWhiteSpace($InputList)) {
    $Inputs = @(
        $InputList -split ";" |
            ForEach-Object { $_.Trim() } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
}
if ($Inputs.Count -eq 0) {
    $Clean = Join-Path $Root "output\cloud_clean.ply"
    $Raw = Join-Path $Root "output\cloud.ply"
    if (Test-Path -LiteralPath $Clean) {
        $Inputs = @($Clean)
    } elseif (Test-Path -LiteralPath $Raw) {
        $Inputs = @($Raw)
    } else {
        throw "No current cloud.ply/cloud_clean.ply."
    }
}

$ResolvedInputs = @()
foreach ($InputPath in $Inputs) {
    $Resolved = if ([System.IO.Path]::IsPathRooted($InputPath)) {
        $InputPath
    } else {
        Join-Path $ProjectRoot $InputPath
    }
    if (-not (Test-Path -LiteralPath $Resolved)) {
        throw "Missing point cloud: $Resolved"
    }
    $ResolvedInputs += (Resolve-Path -LiteralPath $Resolved).Path
}

$ValidationDir = Join-Path $Root "output\validation"
New-Item -ItemType Directory -Force -Path $ValidationDir | Out-Null
if ([string]::IsNullOrWhiteSpace($ReportName)) {
    $ReportName = if ($ResolvedInputs.Count -gt 1) {
        "turntable_repeatability"
    } else {
        "turntable_validation"
    }
}
$OutMd = Join-Path $ValidationDir ($ReportName + ".md")
$OutJson = Join-Path $ValidationDir ($ReportName + ".json")
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
    "--intrinsic", (Join-Path $Root "calibration\camera_intrinsic.yaml"),
    "--laser-plane", (Join-Path $Root "calibration\laser_plane.yaml")
)

& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Turntable hemisphere validation failed."
}
Write-Host ""
Write-Host "Validation completed:"
Write-Host "  $OutMd"
Write-Host "  $OutJson"
