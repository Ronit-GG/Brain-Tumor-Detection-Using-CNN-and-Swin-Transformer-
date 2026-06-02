# =====================================================================
#  Brain Tumor Detection — Windows PowerShell setup script
# ---------------------------------------------------------------------
#  What this does:
#    1. Creates a Python 3.11 virtual environment at .\.venv
#    2. Activates it
#    3. Upgrades pip / setuptools / wheel
#    4. Installs PyTorch 2.12 + torchvision with CUDA 12.6 wheels
#       (auto-falls back to CPU wheels if --cpu is passed)
#    5. Installs the rest of requirements.txt
#    6. Runs the environment diagnostic
#
#  Usage:
#    PS C:\Brain Tumor Detection> .\setup.ps1
#    PS C:\Brain Tumor Detection> .\setup.ps1 -CPU      # CPU-only install
#
#  If you get a "running scripts is disabled" error, run *once*:
#    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# =====================================================================

[CmdletBinding()]
param(
    [switch]$CPU
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host " Brain Tumor Detection — environment bootstrap"            -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan

# --- 1. Locate Python 3.11 -------------------------------------------
$py = & py -3.11 -c "import sys; print(sys.executable)" 2>$null
if (-not $py) {
    Write-Host "[FAIL] Python 3.11 not found." -ForegroundColor Red
    Write-Host "       Install it from https://www.python.org/downloads/release/python-3119/"
    exit 1
}
Write-Host "[ OK ] Found Python 3.11 at: $py" -ForegroundColor Green

# --- 2. Create venv ---------------------------------------------------
if (-not (Test-Path ".\.venv")) {
    Write-Host "[INFO] Creating virtual environment at .\.venv ..." -ForegroundColor Yellow
    & py -3.11 -m venv .venv
} else {
    Write-Host "[ OK ] Virtual environment already exists at .\.venv" -ForegroundColor Green
}

# --- 3. Activate ------------------------------------------------------
$activate = ".\.venv\Scripts\Activate.ps1"
if (-not (Test-Path $activate)) {
    Write-Host "[FAIL] Could not find $activate" -ForegroundColor Red
    exit 1
}
. $activate
Write-Host "[ OK ] Virtual environment activated." -ForegroundColor Green

# --- 4. Upgrade pip toolchain ----------------------------------------
Write-Host "[INFO] Upgrading pip / setuptools / wheel ..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel | Out-Host

# --- 5. Install PyTorch ----------------------------------------------
if ($CPU) {
    Write-Host "[INFO] Installing PyTorch (CPU-only) ..." -ForegroundColor Yellow
    python -m pip install "torch==2.12.0" "torchvision==0.27.0" `
        --index-url https://download.pytorch.org/whl/cpu | Out-Host
} else {
    Write-Host "[INFO] Installing PyTorch + CUDA 12.6 wheels ..." -ForegroundColor Yellow
    python -m pip install "torch==2.12.0" "torchvision==0.27.0" `
        --index-url https://download.pytorch.org/whl/cu126 | Out-Host
}

# --- 6. Install everything else --------------------------------------
Write-Host "[INFO] Installing remaining packages from requirements.txt ..." -ForegroundColor Yellow
python -m pip install -r requirements.txt | Out-Host

# --- 7. Diagnostic ---------------------------------------------------
Write-Host ""
Write-Host "[INFO] Running environment diagnostic ..." -ForegroundColor Yellow
python -m src.check_env

Write-Host ""
Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host " Setup complete."                                          -ForegroundColor Cyan
Write-Host " To re-activate later:  .\.venv\Scripts\Activate.ps1"      -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan
