param(
    [switch]$Demo,
    [int]$Port = 8765,
    [int]$DeviceIndex = 0,
    [ValidateSet('MSMF', 'DSHOW', 'ANY')][string]$Api = 'MSMF',
    [string]$PythonExe = ''
)

$pokerRepo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $pokerRepo
if (-not $PythonExe) {
    $pokerCandidates = @(
        $env:POKERSENSE_PYTHON,
        (Join-Path $pokerRepo '.venv\Scripts\python.exe'),
        (Join-Path $env:USERPROFILE '.workbuddy\binaries\python\envs\default\Scripts\python.exe')
    )
    $PythonExe = $pokerCandidates | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -First 1
}
if (-not $PythonExe) { throw 'Set POKERSENSE_PYTHON or pass -PythonExe to an installed Python environment.' }
$pokerOptional = $env:POKERSENSE_OSS_PATH
if (-not $pokerOptional) {
    $pokerOptional = Join-Path $env:USERPROFILE '.codex\runtimes\pokersense-oss-20260908'
}
$pokerSearchPaths = @((Join-Path $pokerRepo 'src'), $pokerRepo)
if (Test-Path -LiteralPath $pokerOptional) { $pokerSearchPaths += $pokerOptional }
$env:PYTHONPATH = $pokerSearchPaths -join [IO.Path]::PathSeparator
$env:PYTHONUTF8 = '1'
if ($Demo) {
    Write-Output "WPK simulation only: http://127.0.0.1:$Port"
    & $PythonExe -m tools.wpk_demo --port $Port
} else {
    Write-Output "WPK capture-card source ($Api, device $DeviceIndex): http://127.0.0.1:$Port"
    & $PythonExe -m poker_engine.desktop.server --source capture-card --device-index $DeviceIndex --api $Api --port $Port
}
exit $LASTEXITCODE
