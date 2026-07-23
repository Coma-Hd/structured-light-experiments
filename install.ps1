param(
    [ValidateSet("setup", "check")]
    [string]$Action = "setup"
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Venv = Join-Path $Root ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"
Set-Location $Root

function Get-SystemPython {
    $Py = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $Py) {
        return [pscustomobject]@{
            Executable = $Py.Source
            Arguments = @("-3")
        }
    }

    $Command = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $Command) {
        throw "Python was not found. Install Python 3.10 or 3.11 x64 and enable 'Add Python to PATH'."
    }
    return [pscustomobject]@{
        Executable = $Command.Source
        Arguments = @()
    }
}

function Assert-LastExitCode {
    param([string]$Message)
    if ($LASTEXITCODE -ne 0) {
        throw $Message
    }
}

function Invoke-Check {
    if (-not (Test-Path -LiteralPath $Python)) {
        throw "Virtual environment is missing. Run: .\install.ps1 -Action setup"
    }

    $Required = @(
        "requirements.txt",
        "src\__init__.py",
        "src\config.py",
        "src\reconstruct.py",
        "scripts\continuous_rail_capture.py",
        "scripts\draw_keyframe_roi.py",
        "scripts\debug_laser.py",
        "scripts\3_reconstruct.py",
        "scripts\4_postprocess.py",
        "scripts\inspect_cloud.py",
        "scripts\merge_board_faces.py",
        "scripts\align_four_faces_translation_loop.py",
        "scripts\auto_align_four_faces.py",
        "ceshi\曲面\curve_scan.yaml",
        "ceshi\曲面\calibration\camera_intrinsic.yaml",
        "ceshi\曲面\calibration\laser_plane.yaml",
        "output\camera_intrinsic.yaml",
        "output\laser_plane.yaml",
        "ceshi\rail\two_faces\face1_scan.yaml",
        "ceshi\rail\two_faces\face2_scan.yaml",
        "ceshi\rail\two_faces\face3_scan.yaml",
        "ceshi\rail\two_faces\face4_scan.yaml",
        "ceshi\rail\two_faces\0_capture_continuous.ps1",
        "ceshi\rail\two_faces\3_rebuild_face.ps1",
        "ceshi\rail\two_faces\5_merge_board_faces.ps1"
    )
    $Missing = @(
        $Required |
            Where-Object { -not (Test-Path -LiteralPath (Join-Path $Root $_)) }
    )
    if ($Missing.Count -gt 0) {
        throw "Required files are missing:`n$($Missing -join "`n")"
    }

    $env:PYTHONDONTWRITEBYTECODE = "1"
    & $Python -c "import cv2, matplotlib, numpy, open3d, scipy, yaml; from src.config import load_config; curve=load_config('ceshi/曲面/curve_scan.yaml'); faces=[load_config(f'ceshi/rail/two_faces/face{i}_scan.yaml') for i in range(1,5)]; assert curve['paths']['camera_intrinsic']=='ceshi/曲面/calibration/camera_intrinsic.yaml'; assert all(f['paths']['camera_intrinsic']=='output/camera_intrinsic.yaml' and f['paths']['laser_plane']=='output/laser_plane.yaml' for f in faces); print('Dependencies, curve and four-face configurations OK'); print('OpenCV', cv2.__version__); print('Open3D', open3d.__version__)"
    Assert-LastExitCode "Dependency or configuration check failed."

    & $Python -c "from pathlib import Path; import py_compile,tempfile; files=list(Path('src').glob('*.py'))+list(Path('scripts').glob('*.py')); td=tempfile.TemporaryDirectory(); [py_compile.compile(str(p),cfile=str(Path(td.name)/(str(i)+'.pyc')),doraise=True) for i,p in enumerate(files)]; print('Python compile OK:', len(files), 'files')"
    Assert-LastExitCode "Python source compilation failed."

    Write-Host ""
    Write-Host "Installation check passed."
    Write-Host "Hardware is not checked automatically. Confirm camera index, 800x600 resolution, exposure -5 and gain 1 before scanning."
}

if ($Action -eq "setup") {
    if (-not (Test-Path -LiteralPath $Python)) {
        $System = Get-SystemPython
        & $System.Executable @($System.Arguments) -m venv $Venv
        Assert-LastExitCode "Failed to create the virtual environment."
    }

    & $Python -m pip install --upgrade pip
    Assert-LastExitCode "Failed to upgrade pip."
    & $Python -m pip install -r (Join-Path $Root "requirements.txt")
    Assert-LastExitCode "Failed to install Python dependencies."

    $GitLfs = Get-Command git-lfs -ErrorAction SilentlyContinue
    if ($null -ne $GitLfs) {
        git lfs install --local
        Assert-LastExitCode "Git LFS initialization failed."
        git lfs pull
        Assert-LastExitCode "Git LFS download failed."
    } else {
        Write-Warning "Git LFS is not installed. Install it and run 'git lfs pull' to download point clouds."
    }

    @(
        "ceshi\曲面\data\scan",
        "ceshi\曲面\data\calibration\intrinsic",
        "ceshi\曲面\data\calibration\laser_plane",
        "ceshi\曲面\work",
        "ceshi\曲面\output",
        "ceshi\rail\scan\two_faces_face1",
        "ceshi\rail\scan\two_faces_face2",
        "ceshi\rail\scan\two_faces_face3",
        "ceshi\rail\scan\two_faces_face4"
    ) | ForEach-Object {
        New-Item -ItemType Directory -Path (Join-Path $Root $_) -Force | Out-Null
    }

    Invoke-Check
    Write-Host ""
    Write-Host "Setup completed. Next read: ceshi\曲面\README.md"
    exit 0
}

Invoke-Check
