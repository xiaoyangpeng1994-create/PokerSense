@echo off
rem USABLE-001 U2 offline trial launcher (version aa-trial-u2.0).
rem
rem Uses the local Python this project already runs on. It does NOT install
rem anything, does NOT change global environment variables or proxies, does NOT
rem open a capture device, and writes nothing outside launch\u2-trial\state.
rem
rem Optional first argument: the preferred port (default 8791). When that port is
rem already taken nothing is killed - the launcher reports it and picks a free one.

setlocal
set "HERE=%~dp0"
set "PYTHON_BIN=%PYTHON_BIN%"
if "%PYTHON_BIN%"=="" set "PYTHON_BIN=C:\Users\Administrator\.codex\runtimes\pokersense-v6-clean-20260908\Scripts\python.exe"
if not exist "%PYTHON_BIN%" (
  echo Local Python not found at "%PYTHON_BIN%".
  echo Set PYTHON_BIN to the interpreter this project already uses, then retry.
  echo Nothing was started and nothing was modified.
  exit /b 2
)
set "PORT=%~1"
if "%PORT%"=="" set "PORT=8791"
"%PYTHON_BIN%" "%HERE%trial_launcher.py" --port %PORT%
endlocal
