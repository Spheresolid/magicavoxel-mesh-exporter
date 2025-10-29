@echo off
setlocal

:: Locate Python
set PYTHON_CMD=
for %%P in (python3 python) do (
    where %%P >nul 2>nul && set PYTHON_CMD=%%P
)

if not defined PYTHON_CMD (
    echo Python not found. Please install Python and add it to PATH.
    pause
    exit /b 1
)

:: Check and run optional empty layer checker
if exist "CheckEmptyLayers.py" (
    echo Running CheckEmptyLayers.py...
    %PYTHON_CMD% CheckEmptyLayers.py
) else (
    echo CheckEmptyLayers.py not detected; skipping it.
)

:: Main exporter
echo Running ExportMeshes.py...
%PYTHON_CMD% ExportMeshes.py

pause
endlocal
