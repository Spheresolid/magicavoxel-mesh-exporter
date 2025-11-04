@echo off
setlocal ENABLEDELAYEDEXPANSION

:: Force Python to run unbuffered so child scripts stream output immediately
set "PYTHONUNBUFFERED=1"

:: If double-clicked (no args) re-launch into an interactive child window that stays open.
:: To run in-place from an existing console, call: RunMagicaExport.bat --child
if "%~1"=="" (
    start "" cmd /k "%~f0" --child
    exit /b 0
)

:: Interactive child execution from here
set "RC=0"
set "SCRIPT_DIR=%~dp0"

:: Locate Python
set "PYTHON_CMD="
for %%P in (python3 python) do (
    where %%P >nul 2>nul && set "PYTHON_CMD=%%P"
)

if not defined PYTHON_CMD (
    echo Python not found. Please install Python and add it to PATH.
    set "RC=1"
    goto :finish
)

:: Ensure deps folder is used for local installs and available to Python
set "DEPS_DIR=%SCRIPT_DIR%deps"
set "PYTHONPATH=%DEPS_DIR%;%PYTHONPATH%"

:: Run dependency checker/installer if present (unbuffered)
if exist "%SCRIPT_DIR%Ensure_deps.py" (
    echo Checking local Python dependencies...
    call "%PYTHON_CMD%" -u "%SCRIPT_DIR%Ensure_deps.py"
    set "RC=%ERRORLEVEL%"
    if "%RC%"=="3" (
        echo Dependency installation failed. Please inspect pip output and retry.
        set "RC=1"
        goto :finish
    ) else if "%RC%"=="2" (
        echo User declined to install missing packages into ./deps.
        set /p CONTINUE_WITHOUT=Continue without installing missing packages? [y/N]
        if /I "%CONTINUE_WITHOUT%" NEQ "y" (
            echo Aborting per user request.
            set "RC=1"
            goto :finish
        )
    )
) else (
    echo Ensure_deps.py not found; skipping dependency auto-check.
)

:: Optional: run empty layer checker if present
if exist "%SCRIPT_DIR%CheckEmptyLayers.py" (
    echo Running CheckEmptyLayers.py...
    call "%PYTHON_CMD%" -u "%SCRIPT_DIR%CheckEmptyLayers.py"
    if errorlevel 1 (
        echo CheckEmptyLayers.py returned an error; continuing.
    )
) else (
    echo CheckEmptyLayers.py not detected; skipping it.
)

:: Main exporter (exports .vox -> exported_meshes/<voxbasename>/)
echo Running ExportMeshes.py...
if exist "%SCRIPT_DIR%ExportMeshes.py" (
    call "%PYTHON_CMD%" -u "%SCRIPT_DIR%ExportMeshes.py"
    if errorlevel 1 (
        echo ExportMeshes.py failed.
        set "RC=1"
        goto :finish
    )
) else (
    echo ExportMeshes.py not found; aborting.
    set "RC=1"
    goto :finish
)

:: Ensure FinalMapping reports exist (run FinalizeMapping once)
if exist "%SCRIPT_DIR%FinalizeMapping.py" (
    echo Running FinalizeMapping.py to generate FinalMapping reports...
    call "%PYTHON_CMD%" -u "%SCRIPT_DIR%FinalizeMapping.py"
    if errorlevel 1 (
        echo FinalizeMapping.py reported errors; check reports\ for details.
    )
) else (
    echo FinalizeMapping.py not found; skipping.
)

:: Optional: compare expected vs actual exports
if exist "%SCRIPT_DIR%CompareExports.py" (
    echo Running CompareExports.py...
    call "%PYTHON_CMD%" -u "%SCRIPT_DIR%CompareExports.py"
) else (
    echo CompareExports.py not found; skipping file-compare step.
)

:: Print part assignments — run live so output appears in console immediately (no standalone runs required)
if exist "%SCRIPT_DIR%PrintPartAssignments.py" (
    echo Running PrintPartAssignments.py ^(live output^)...
    call "%PYTHON_CMD%" -u "%SCRIPT_DIR%PrintPartAssignments.py"
    echo PrintPartAssignments completed; report written to reports\PrintPartAssignments_*.txt
) else (
    echo PrintPartAssignments.py not found; skipping assignment diagnostic.
)

:: (Optional) Run RenameExportsByCentroid in dry-run if you want; disabled by default to avoid side-effects.
:: if exist "%SCRIPT_DIR%RenameExportsByCentroid.py" (
::     echo Running RenameExportsByCentroid.py (DRY-RUN)...
::     call "%PYTHON_CMD%" -u "%SCRIPT_DIR%RenameExportsByCentroid.py"
:: )

:finish
echo.
echo RunMagicaExport finished with RC=%RC%.
echo Press any key to close this window...
pause >nul
endlocal
exit /b %RC%