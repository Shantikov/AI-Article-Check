[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$backendRoot = Join-Path $projectRoot "backend"
$venvPython = Join-Path $backendRoot ".venv\Scripts\python.exe"
$manifestPath = Join-Path $projectRoot "extension\manifest.json"

if (-not (Test-Path $venvPython)) {
    throw "Python environment is missing. Run setup.cmd once before starting the backend."
}
if (-not (Test-Path $manifestPath)) {
    throw "Could not find extension\manifest.json. Keep start.ps1 in the project root."
}

$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
$expectedVersion = [string]$manifest.version
$healthUrl = "http://127.0.0.1:8787/health"

try {
    $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2
} catch {
    $health = $null
}

if ($health -and $health.status -eq "ok") {
    $expectedSeries = (($expectedVersion -split "\.")[0..1] -join ".")
    $runningSeries = (([string]$health.version -split "\.")[0..1] -join ".")
    if ($runningSeries -ne $expectedSeries) {
        throw "Backend v$($health.version) is already using port 8787, but this extension is v$expectedVersion. Stop the old backend with Ctrl+C first."
    }
    Write-Host "Compatible AI Article Check backend v$($health.version) is already running." -ForegroundColor Green
    Write-Host $healthUrl
    return
}

if (Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue) {
    $listeners = @(Get-NetTCPConnection -LocalPort 8787 -State Listen -ErrorAction SilentlyContinue)
    if ($listeners.Count -gt 0) {
        $processIds = ($listeners | Select-Object -ExpandProperty OwningProcess -Unique) -join ", "
        throw "Port 8787 is occupied by process $processIds. Stop that process before starting AI Article Check."
    }
}

Push-Location $backendRoot
try {
    & $venvPython -m app.preflight
    if ($LASTEXITCODE -ne 0) {
        throw "Backend preflight failed. Run setup.cmd to repair the installation."
    }

    Write-Host "Starting AI Article Check v$expectedVersion..." -ForegroundColor Cyan
    Write-Host "Keep this window open. Press Ctrl+C to stop the server." -ForegroundColor Yellow
    Write-Host $healthUrl
    & $venvPython -m uvicorn app.main:app --host 127.0.0.1 --port 8787
} finally {
    Pop-Location
}
