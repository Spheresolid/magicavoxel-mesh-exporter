@echo off
setlocal ENABLEDELAYEDEXPANSION

:: Resolve script directory (folder where this .bat lives)
set "SCRIPT_DIR=%~dp0"

:: Locate Python
set "PYTHON_CMD="
for %%P in (python3 python) do (
    where %%P >nul 2>nul && set "PYTHON_CMD=%%P"
)

if not defined PYTHON_CMD (
    echo Python not found. Please install Python and add it to PATH.
    pause
    exit /b 1
)

:: Ensure deps folder is used for local installs and available to Python
set "DEPS_DIR=%SCRIPT_DIR%deps"
set "PYTHONPATH=%DEPS_DIR%;%PYTHONPATH%"

:: Run dependency checker/installer if present
if exist "%SCRIPT_DIR%Ensure_deps.py" (
    echo Checking local Python dependencies...
    "%PYTHON_CMD%" "%SCRIPT_DIR%Ensure_deps.py"
    set "RC=%ERRORLEVEL%"
    if "%RC%"=="3" (
        echo Dependency installation failed. Please inspect pip output and retry.
        pause
        exit /b 1
    ) else if "%RC%"=="2" (
        echo User declined to install missing packages into ./deps.
        set /p CONTINUE_WITHOUT=Continue without installing missing packages? [y/N] 
        if /I "%CONTINUE_WITHOUT%" NEQ "y" (
            echo Aborting per user request.
            pause
            exit /b 1
        ) else (
            echo Continuing without local installs. Note: scripts may fail if deps are missing.
        )
    ) else (
        echo Dependency check/installation complete.
    )
) else (
    echo Ensure_deps.py not found; skipping dependency auto-check.
)

:: Optional: run empty layer checker if present
if exist "%SCRIPT_DIR%CheckEmptyLayers.py" (
    echo Running CheckEmptyLayers.py...
    "%PYTHON_CMD%" "%SCRIPT_DIR%CheckEmptyLayers.py"
) else (
    echo CheckEmptyLayers.py not detected; skipping it.
)

:: Main exporter (exports .vox -> exported_meshes/<voxbasename>/)
echo Running ExportMeshes.py...
if exist "%SCRIPT_DIR%ExportMeshes.py" (
    "%PYTHON_CMD%" "%SCRIPT_DIR%ExportMeshes.py"
) else (
    echo ExportMeshes.py not found; aborting.
    pause
    exit /b 1
)

:: Optional: compare expected vs actual exports
if exist "%SCRIPT_DIR%CompareExports.py" (
    echo Running CompareExports.py...
    "%PYTHON_CMD%" "%SCRIPT_DIR%CompareExports.py"
) else (
    echo CompareExports.py not found; skipping file-compare step.
)

:: Optional diagnostics
if exist "%SCRIPT_DIR%PrintPartAssignments.py" (
    echo Running PrintPartAssignments.py...
    "%PYTHON_CMD%" "%SCRIPT_DIR%PrintPartAssignments.py"
) else (
    echo PrintPartAssignments.py not found; skipping assignment diagnostic.
)

:: Run renamer (global assignment) if present
if exist "%SCRIPT_DIR%RenameExportsByCentroid.py" (
    echo Running RenameExportsByCentroid.py --commit...
    "%PYTHON_CMD%" "%SCRIPT_DIR%RenameExportsByCentroid.py" --commit
) else (
    echo RenameExportsByCentroid.py not found; skipping rename step.
)

:: Finalize mapping if present
if exist "%SCRIPT_DIR%FinalizeMapping.py" (
    echo Running FinalizeMapping.py...
    "%PYTHON_CMD%" "%SCRIPT_DIR%FinalizeMapping.py"
) else (
    echo FinalizeMapping.py not found; skipping finalization.
)

:: Auto-generate and apply high-confidence overrides per .vox (dry-run then prompt)
pushd "%SCRIPT_DIR%"
if exist "AutoApplyOverrides.py" (
    for %%F in (*.vox) do (
        set "VOXNAME=%%~nF"
        echo.
        echo ===== Auto-suggest overrides for !VOXNAME! (dry-run, tiny->large heuristic enabled) =====
        "%PYTHON_CMD%" "AutoApplyOverrides.py" --vox "!VOXNAME!" --enable-tiny2large
        echo.

        :: If consumer sets the env var AUTOAPPLY=1 the script will auto-commit without prompting.
        if defined AUTOAPPLY (
            echo AUTOAPPLY detected; applying high-confidence overrides for !VOXNAME! now...
            "%PYTHON_CMD%" "AutoApplyOverrides.py" --vox "!VOXNAME!" --enable-tiny2large --auto
            if errorlevel 1 (
                echo Auto-commit failed for !VOXNAME!; check reports\ for details.
            ) else (
                echo Auto-commit successful for !VOXNAME!.
            )
        ) else (
            set /p APPLY_NOW=Apply high-confidence overrides for !VOXNAME! now? [y/N] 
            if /I "!APPLY_NOW!"=="y" (
                echo Applying high-confidence overrides for !VOXNAME!...
                "%PYTHON_CMD%" "AutoApplyOverrides.py" --vox "!VOXNAME!" --enable-tiny2large --auto
                if errorlevel 1 (
                    echo Auto-commit failed for !VOXNAME!; check reports\ for details.
                ) else (
                    echo Auto-commit successful for !VOXNAME!.
                )
            ) else (
                echo Skipping auto-commit for !VOXNAME!.
            )
        )
    )
) else (
    echo AutoApplyOverrides.py not found in %CD%; skipping auto-overrides step.
)
popd

pause
endlocal
