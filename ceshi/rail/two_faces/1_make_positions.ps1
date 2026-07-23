param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("face1", "face2", "face3", "face4")]
    [string]$Face,
    [double]$StepMm = 0.5,
    [double]$StartMm = 0.0
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $ProjectRoot

$ScanDir = "ceshi/rail/scan/two_faces_$Face"
$PositionsCsv = Join-Path $ScanDir "positions.csv"
$Images = @(Get-ChildItem -Path $ScanDir -Filter *.png -ErrorAction SilentlyContinue | Sort-Object Name)
if ($Images.Count -eq 0) {
    throw "No PNG images in $ScanDir. Capture this face first."
}

$Lines = New-Object System.Collections.Generic.List[string]
$Lines.Add("image,distance_mm") | Out-Null
for ($i = 0; $i -lt $Images.Count; $i++) {
    $Distance = $StartMm + $i * $StepMm
    $Lines.Add("$($Images[$i].Name),$Distance") | Out-Null
}
$Lines | Set-Content -Path $PositionsCsv -Encoding UTF8

Write-Host "Wrote: $PositionsCsv ($($Images.Count) frames)"
Write-Host "Assumed constant step: $StepMm mm"
Write-Host "IMPORTANT: replace distance_mm with real rail readings if motion was not exact."
