<#
.SYNOPSIS
  Build and run the Kangri TTS webservice Docker container.

.EXAMPLE
  # Everything defaults to this machine's usual paths (project, wavs, character mapping)
  .\buildDocker.ps1 -Gpu

.EXAMPLE
  # Opt out of a default mount (e.g. skip character-mapping, so synthesize-file returns
  # its "not configured" error instead of mounting a possibly-stale file)
  .\buildDocker.ps1 -Gpu -CharacterMappingDir ""

.EXAMPLE
  # Just rebuild the image without starting a container
  .\buildDocker.ps1 -BuildOnly
#>
param(
    # Host path to the buildCombinedDataset project (src/, checkpoints/, data/, model_init*/,
    # tokenizer*/). Required for every endpoint.
    [string]$ProjectPath = "C:\vscode\buildCombinedDataset",

    # Host path to the folder of raw reference wavs (FCBH/wavs). Optional -- only needed
    # for the "compare" option of POST /api/v1/tts/synthesize/. Pass -WavDir "" to opt out.
    [string]$WavDir = "C:\Users\pete_\Dropbox\NTprogress\PahariAudio\KangriWordDownloads\FCBH\wavs",

    # Host path to the folder containing characterMapping.json. Optional -- only needed
    # for POST /api/v1/tts/synthesize-file/. Pass -CharacterMappingDir "" to opt out.
    [string]$CharacterMappingDir = "C:\Users\pete_\Dropbox\NTprogress\PahariAudio\KangriWordDownloads\FCBH",

    [int]$Port = 8000,
    [string]$ApiKey = "",
    [ValidateSet("a", "b")]
    [string]$DefaultVariant = "b",
    [switch]$Gpu,
    [switch]$Detached,
    [switch]$BuildOnly,
    # Container name -- fixed so it shows up predictably in Docker Desktop and can be
    # stopped/started/restarted there instead of re-running this script.
    [string]$ContainerName = "kangri-tts",
    # Opt back into throwaway behavior: the container is auto-removed (`--rm`) when it
    # stops. WITHOUT this (the default), the container persists after stopping so you can
    # restart it from Docker Desktop; a repeat run of this script replaces it in place.
    [switch]$Ephemeral
)

$ErrorActionPreference = "Stop"
$ImageName = "kangri-tts"
$RepoRoot = Split-Path -Parent $PSScriptRoot  # docker/ -> project root

if (-not (Test-Path $ProjectPath)) {
    throw "ProjectPath not found: $ProjectPath"
}

Write-Host "Building Docker image '$ImageName' (build context: $RepoRoot)..."
docker build -f (Join-Path $PSScriptRoot "Dockerfile") -t $ImageName $RepoRoot
if ($LASTEXITCODE -ne 0) { throw "docker build failed" }

if ($BuildOnly) {
    Write-Host "Build complete (skipped run due to -BuildOnly)."
    exit 0
}

# A persistent container can't share its name with a leftover one, so clear any
# existing container of this name first (running or stopped). This makes a repeat run
# of the script a clean "replace", while a plain `docker start $ContainerName` (or the
# Docker Desktop restart button) reuses the existing one without touching this script.
$existing = docker ps -aq --filter "name=^/$ContainerName$"
if ($existing) {
    Write-Host "Removing existing container '$ContainerName' ($existing)..."
    docker rm -f $ContainerName | Out-Null
}

$dockerArgs = @("run")
if ($Ephemeral) { $dockerArgs += "--rm" }
$dockerArgs += @("--name", $ContainerName)
if ($Detached) { $dockerArgs += "-d" } else { $dockerArgs += "-it" }
$dockerArgs += @("-p", "${Port}:8000")
$dockerArgs += @("-e", "PORT=8000")
$dockerArgs += @("-e", "DEFAULT_VARIANT=$DefaultVariant")
$dockerArgs += @("-v", "${ProjectPath}:/app/project:ro")

if ($ApiKey) {
    $dockerArgs += @("-e", "API_KEY=$ApiKey")
}
if ($WavDir) {
    if (-not (Test-Path $WavDir)) { throw "WavDir not found: $WavDir" }
    $dockerArgs += @("-e", "WAV_DIR=/app/wavs", "-v", "${WavDir}:/app/wavs:ro")
}
if ($CharacterMappingDir) {
    if (-not (Test-Path $CharacterMappingDir)) { throw "CharacterMappingDir not found: $CharacterMappingDir" }
    $dockerArgs += @("-e", "CHARACTER_MAPPING_DIR=/app/character-mapping", "-v", "${CharacterMappingDir}:/app/character-mapping:ro")
}
if ($Gpu) {
    $dockerArgs += @("--gpus", "device=0")
}
$dockerArgs += $ImageName

Write-Host "Running: docker $($dockerArgs -join ' ')"
docker @dockerArgs

if ($Detached) {
    if (-not $Ephemeral) {
        Write-Host "`nContainer '$ContainerName' is running detached and will persist when stopped."
        Write-Host "Restart it later from Docker Desktop, or:  docker start $ContainerName"
        Write-Host "Stop (without removing) with:            docker stop $ContainerName"
    }
    Write-Host "Web UI: http://localhost:$Port/"
} else {
    Write-Host "`nRunning in the foreground (Ctrl-C to stop)."
    if (-not $Ephemeral) {
        Write-Host "The stopped container will remain in Docker Desktop for restart as '$ContainerName'."
    }
    Write-Host "To view the web UI, in another PowerShell window run:"
    Write-Host "  Start-Process http://localhost:$Port/"
}
