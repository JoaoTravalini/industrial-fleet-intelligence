param(
    [switch]$NoPostgresStart,
    [switch]$WarmCopilot
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "[platform] $Message"
}

function Write-Warn {
    param([string]$Message)
    Write-Warning "[platform] $Message"
}

function Assert-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command not found on PATH: $Name"
    }
}

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

if (-not (Test-Path (Join-Path $RepoRoot "pyproject.toml")) -or -not (Test-Path (Join-Path $RepoRoot "apps\web\package.json"))) {
    throw "This script must be run from the Industrial Fleet Intelligence repository."
}

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Project virtual environment interpreter not found at .venv\Scripts\python.exe."
}

Write-Step "Verifying project Python interpreter."
& $Python --version

Assert-Command "docker"
Write-Step "Verifying Docker CLI and Linux container mode."
$DockerOsType = docker info --format "{{.OSType}}" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "Docker Engine is not reachable. Start Docker Desktop and try again."
}
if ($DockerOsType -and $DockerOsType.Trim().ToLowerInvariant() -ne "linux") {
    throw "Docker is not using Linux containers. Switch Docker Desktop to Linux containers."
}

if (-not $NoPostgresStart) {
    Write-Step "Starting PostgreSQL with Docker Compose."
    docker compose up -d postgres
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to start PostgreSQL through Docker Compose."
    }
}

Write-Step "Checking PostgreSQL Compose status."
docker compose ps postgres

$Ollama = Get-Command "ollama" -ErrorAction SilentlyContinue
if ($Ollama) {
    Write-Step "Checking local Ollama availability."
    ollama list | Out-Host
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "Ollama CLI exists, but the local Ollama service is not responding. The dashboard can still run without Copilot."
    } elseif ($WarmCopilot) {
        $WarmupScript = Join-Path $RepoRoot "scripts\warm_copilot_model.py"
        if (Test-Path $WarmupScript) {
            Write-Step "Warming the configured local Copilot model."
            & $Python $WarmupScript
        } else {
            Write-Warn "Copilot warmup script was not found; skipping warmup."
        }
    }
} else {
    Write-Warn "Ollama CLI was not found. Core dashboard startup can continue; Copilot will be unavailable."
}

$LogsDir = Join-Path $RepoRoot "logs"
New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null
$PidFile = Join-Path $LogsDir "local-platform-pids.json"

Write-Step "Starting FastAPI backend in a separate PowerShell process."
$ApiCommand = "Set-Location '$RepoRoot'; & '$Python' -m fastapi dev apps/api/main.py --host 127.0.0.1 --port 8000"
$ApiProcess = Start-Process powershell.exe -ArgumentList @("-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $ApiCommand) -PassThru -WindowStyle Normal

Write-Step "Starting Vite dashboard in a separate PowerShell process."
$WebRoot = Join-Path $RepoRoot "apps\web"
$WebCommand = "Set-Location '$WebRoot'; npm run dev -- --host 127.0.0.1"
$WebProcess = Start-Process powershell.exe -ArgumentList @("-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $WebCommand) -PassThru -WindowStyle Normal

@{
    api_pid = $ApiProcess.Id
    web_pid = $WebProcess.Id
    repo_root = $RepoRoot.Path
} | ConvertTo-Json | Set-Content -Path $PidFile -Encoding UTF8

Write-Step "Backend: http://127.0.0.1:8000"
Write-Step "API docs: http://127.0.0.1:8000/docs"
Write-Step "Dashboard: http://127.0.0.1:5173"
Write-Step "Process IDs saved to logs\local-platform-pids.json."
