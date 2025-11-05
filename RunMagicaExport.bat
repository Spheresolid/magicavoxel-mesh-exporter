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

:: Print part assignments — run live so output appears in console immediately
if exist "%SCRIPT_DIR%PrintPartAssignments.py" (
    echo Running PrintPartAssignments.py ^(live output^)...
    
    :: Call the script once; it writes its own report.
    call "%PYTHON_CMD%" -u "%SCRIPT_DIR%PrintPartAssignments.py"
    
    if errorlevel 1 (
        echo [ERROR] PrintPartAssignments.py reported an error.
    ) else (
        echo PrintPartAssignments finished; report written to reports\PrintPartAssignments_*.txt
    )
) else (
    echo PrintPartAssignments.py not found; skipping assignment diagnostic.
)

:: --- CONTROL FLOW IS NOW HANDLED ---
:: Call the renamer subroutine (keeps renamer logic outside any prior parenthesized blocks)
call :RunRenamer
if errorlevel 1 set "RC=1"

:finish
echo.
echo RunMagicaExport finished with RC=%RC%.
echo Press any key to close this window...
pause >nul
endlocal
exit /b %RC%

:: ------------------------------
:: Subroutine: RunRenamer
:: Runs dry-run with debug then prompts for commit (keeps block scope separate)
:: ------------------------------
:RunRenamer
    setlocal
    if not exist "%SCRIPT_DIR%RenameExportsByCentroid.py" (
        echo [SKIP] RenameExportsByCentroid.py not found; skipping rename step.
        endlocal
        exit /b 0
    )
    echo.
    echo ----------------------------------------------------------------------
    echo Running RenameExportsByCentroid.py (DRY-RUN by default, with logical fix active)...
    call "%PYTHON_CMD%" -u "%SCRIPT_DIR%RenameExportsByCentroid.py" --vox Character --debug
    if errorlevel 1 (
        echo [ERROR] RenameExportsByCentroid.py failed during dry-run. Skipping commit prompt.
        endlocal
        exit /b 1
    )
    :: write a stable log copy (single run) so user can inspect later
    call "%PYTHON_CMD%" -u "%SCRIPT_DIR%RenameExportsByCentroid.py" --vox Character --debug > "%SCRIPT_DIR%reports\RenameDryrun_Character.log" 2>&1

    :: Prompt user to commit; use percent expansion (not delayed !) to avoid surprises
    set /p COMMIT_RENAME=RenameExportsByCentroid.py finished dry-run. Commit global renames now? [y/N] 
    if /I "%COMMIT_RENAME%"=="y" (
        echo Committing global assignment rename...
        call "%PYTHON_CMD%" -u "%SCRIPT_DIR%RenameExportsByCentroid.py" --vox Character --commit
        if errorlevel 1 (
            echo [ERROR] Global assignment commit failed.
            endlocal
            exit /b 1
        ) else (
            echo [SUCCESS] Global assignment commit successful.
        )
    ) else (
        echo Skipping global assignment commit.
    )

    echo ----------------------------------------------------------------------
    endlocal
    exit /b 0