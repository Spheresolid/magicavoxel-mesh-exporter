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

:: Ensure deps folder is used for local installs
set "DEPS_DIR=%CD%\deps"

:: Prepend local deps to PYTHONPATH so installed packages are importable (must do this before running Ensure_deps)
set "PYTHONPATH=%DEPS_DIR%;%PYTHONPATH%"

:: Run dependency checker/installer (prompts). Pass --yes to auto-install without prompt.
if exist "Ensure_deps.py" (
    echo Checking local Python dependencies...
    "%PYTHON_CMD%" Ensure_deps.py
    if errorlevel 3 (
        echo Dependency installation failed. Please inspect pip output and retry.
        pause
        exit /b 1
    )
) else (
    echo Ensure_deps.py not found; skipping dependency auto-check.
)

:: Optional: run empty layer checker if present
if exist "CheckEmptyLayers.py" (
    echo Running CheckEmptyLayers.py...
    "%PYTHON_CMD%" CheckEmptyLayers.py
) else (
    echo CheckEmptyLayers.py not detected; skipping it.
)

:: Main exporter (exports .vox -> exported_meshes/<voxbasename>/)
echo Running ExportMeshes.py...
"%PYTHON_CMD%" ExportMeshes.py

:: Optional: compare expected vs actual exports
if exist "CompareExports.py" (
    echo Running CompareExports.py...
    "%PYTHON_CMD%" CompareExports.py
) else (
    echo CompareExports.py not found; skipping file-compare step.
)

:: Optional diagnostics / renamer if present
if exist "PrintPartAssignments.py" (
    echo Running PrintPartAssignments.py...
    "%PYTHON_CMD%" PrintPartAssignments.py
) else (
    echo PrintPartAssignments.py not found; skipping assignment diagnostic.
)

if exist "RenameExportsByCentroid.py" (
    echo Running RenameExportsByCentroid.py --commit...
    "%PYTHON_CMD%" RenameExportsByCentroid.py --commit
) else (
    echo RenameExportsByCentroid.py not found; skipping rename step.
)

if exist "FinalizeMapping.py" (
    echo Running FinalizeMapping.py...
    "%PYTHON_CMD%" FinalizeMapping.py
) else (
    echo FinalizeMapping.py not found; skipping finalization.
)

pause
endlocal
