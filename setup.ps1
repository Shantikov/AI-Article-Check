[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$backendRoot = Join-Path $projectRoot "backend"
$venvRoot = Join-Path $backendRoot ".venv"
$venvPython = Join-Path $venvRoot "Scripts\python.exe"
$requirements = Join-Path $backendRoot "requirements.txt"

if (-not (Test-Path $requirements)) {
    throw "Could not find backend\requirements.txt. Keep setup.ps1 in the project root."
}

$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvCommand) {
    throw "uv is not installed or is not available in PATH. Install uv, reopen PowerShell, and run setup again."
}

Push-Location $backendRoot
try {
    if (-not (Test-Path $venvPython)) {
        Write-Host "Creating a Python 3.12 environment..." -ForegroundColor Cyan
        & $uvCommand.Source venv --python 3.12 $venvRoot
        if ($LASTEXITCODE -ne 0) {
            throw "Could not create the Python environment."
        }
    } else {
        Write-Host "Using the existing environment: $venvRoot" -ForegroundColor DarkCyan
    }

    Write-Host "Installing runtime dependencies..." -ForegroundColor Cyan
    & $uvCommand.Source pip install --python $venvPython -r $requirements
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency installation failed."
    }

    Write-Host "Checking calibration and preparing the detector model..." -ForegroundColor Cyan
    & $venvPython -m app.preflight --download-model
    if ($LASTEXITCODE -ne 0) {
        throw "Backend preflight failed."
    }
} finally {
    Pop-Location
}

Write-Host "Setup completed. Use start.cmd to run AI Article Check." -ForegroundColor Green
