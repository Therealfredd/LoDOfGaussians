@echo off
setlocal
cd /d "%~dp0"

rem Prefer a real CPython 3.10/3.11 via the py launcher; fall back to PATH.
set "LAUNCH="
for %%V in (3.10 3.11) do (
    if not defined LAUNCH (
        py -%%V -c "import tkinter" >nul 2>&1 && set "LAUNCH=py -%%V"
    )
)
if not defined LAUNCH (
    python -c "import tkinter" >nul 2>&1 && set "LAUNCH=python"
)

if not defined LAUNCH (
    echo.
    echo   Could not find a Python installation with tkinter.
    echo.
    echo   Install Python 3.10 ^(64-bit^) from https://www.python.org/downloads/
    echo   and tick "tcl/tk and IDLE" plus "Add python.exe to PATH".
    echo.
    pause
    exit /b 1
)

start "" %LAUNCH% "%~dp0lod_trainer.py"
