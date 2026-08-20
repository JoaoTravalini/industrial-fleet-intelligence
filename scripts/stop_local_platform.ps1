$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "[platform] $Message"
}

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$PidFile = Join-Path $RepoRoot "logs\local-platform-pids.json"

if (-not (Test-Path $PidFile)) {
    Write-Step "No local platform PID file found. Nothing to stop."
    exit 0
}

$Pids = Get-Content -Path $PidFile -Raw | ConvertFrom-Json
$Stopped = @()

foreach ($ProcessId in @($Pids.api_pid, $Pids.web_pid)) {
    if ($null -eq $ProcessId) {
        continue
    }

    $Process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($Process) {
        Write-Step "Stopping process $ProcessId ($($Process.ProcessName))."
        Stop-Process -Id $ProcessId
        $Stopped += $ProcessId
    }
}

Remove-Item -Path $PidFile -Force

if ($Stopped.Count -eq 0) {
    Write-Step "No matching developer API/frontend processes were running."
} else {
    Write-Step "Stopped developer API/frontend processes: $($Stopped -join ', ')."
}

Write-Step "Docker containers and volumes were not removed."
