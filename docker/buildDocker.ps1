<#
.SYNOPSIS
  Build and run the Kangri TTS webservice Docker container.

.EXAMPLE
  # Minimal: just the project path (default), GPU, default port 8000
  .\buildDocker.ps1 -Gpu

.EXAMPLE
  # Full: enable compare mode and the synthesize-file endpoint too
  .\buildDocker.ps1 -Gpu `
    -WavDir "C:\Users\pete_\Dropbox\NTprogress\PahariAudio\KangriWordDownloads\FCBH\wavs" `
    -CharacterMappingDir "C:\My Paratext 9 Projects\xnr\shared\milestone-markers"

.EXAMPLE
  # Just rebuild the image without starting a container
  .\buildDocker.ps1 -BuildOnly
#>
param(
    # Host path to the buildCombinedDataset project (src/, checkpoints/, data/, model_init*/,
    # tokenizer*/). Required for every endpoint.
    [string]$ProjectPath = "C:\vscode\buildCombinedDataset",

    # Host path to the folder of raw reference wavs (FCBH/wavs). Optional -- only needed
    # for the "compare" option of POST /api/v1/tts/synthesize/.
    [string]$WavDir = "",

    # Host path to the folder containing characterMapping.json. Optional -- only needed
    # for POST /api/v1/tts/synthesize-file/.
    [string]$CharacterMappingDir = "",

    [int]$Port = 8000,
    [string]$ApiKey = "",
    [ValidateSet("a", "b")]
    [string]$DefaultVariant = "b",
    [switch]$Gpu,
    [switch]$Detached,
    [switch]$BuildOnly
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

$dockerArgs = @("run", "--rm")
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

if (-not $Detached) {
    Write-Host "`nTo view the web UI, in another PowerShell window run:"
    Write-Host "  Start-Process http://localhost:$Port/"
}
