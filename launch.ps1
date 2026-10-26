param([switch]$Cpu, [switch]$Repair)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$setupArgs = @('setup_runtime.py')
if ($Cpu) { $setupArgs += '--cpu' }
if ($Repair) { $setupArgs += '--repair' }
python @setupArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if (-not $env:HF_HOME) { $env:HF_HOME = Join-Path $PSScriptRoot '.model-cache' }
Write-Host 'Open http://localhost:8791 — weights load only when you choose Start Qwen.'
& (Join-Path $PSScriptRoot '.venv\Scripts\python.exe') 'explorer/server.py'
exit $LASTEXITCODE
