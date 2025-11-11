@echo off
setlocal ENABLEDELAYEDEXPANSION

:: Force Python to run unbuffered so child scripts stream output immediately
set "PYTHONUNBUFFERED=1"

:: If double-clicked (no args) re-launch into an interactive child window that stays open.
if "%~1"=="" (
    start "" cmd /k "%~f0" --child
    exit /b 0
)

:: Interactive execution context
set "RC=0"
set "SCRIPT_DIR=%~dp0"

:: Defaults
if not defined SCALE_FACTOR set "SCALE_FACTOR=1.0"
if not defined MESH_SCALE set "MESH_SCALE=0.1"

:: RENAMER default (edit this one line to change default behavior)
if not defined DEFAULT_RENAME_ACTIVE set "DEFAULT_RENAME_ACTIVE=1"
if not defined RENAME_ACTIVE set "RENAME_ACTIVE=%DEFAULT_RENAME_ACTIVE%"

:: Parse CLI args and build PASS_ARGS (skip internal tokens like --child / --diagnose / --force-safe)
set "DIAG=0"
set "FORCE_SAFE=0"
set "PASS_ARGS="
for %%A in (%*) do (
    set "ARG=%%~A"
    if "!ARG!"=="" (
        rem skip empty
    ) else if /I "!ARG:~0,14!"=="--scale-factor=" (
        set "SCALE_FACTOR=!ARG:~14!"
    ) else if /I "!ARG:~0,3!"=="-s=" (
        set "SCALE_FACTOR=!ARG:~3!"
    ) else if /I "!ARG:~0,12!"=="--mesh-scale=" (
        set "MESH_SCALE=!ARG:~12!"
    ) else if /I "!ARG:~0,7!"=="--child" (
        rem skip internal child tokens (covers --child and --child=*)
    ) else if /I "!ARG:~0,9!"=="--diagnose" (
        set "DIAG=1"
    ) else if /I "!ARG:~0,11!"=="--force-safe" (
        set "FORCE_SAFE=1"
    ) else if /I "!ARG:~0,8!"=="--unsafe" (
        set "FORCE_SAFE=0"
    ) else if /I "!ARG!"=="--rename" (
        set "RENAME_ACTIVE=1"
    ) else if /I "!ARG!"=="--no-rename" (
        set "RENAME_ACTIVE=0"
    ) else if /I "!ARG:~0,15!"=="--rename-active=" (
        set "RENAME_ACTIVE=!ARG:~15!"
    ) else (
        set "PASS_ARGS=!PASS_ARGS! !ARG!"
    )
)
if defined PASS_ARGS set "PASS_ARGS=!PASS_ARGS:~1!"

:: Guard: prevent accidental collapse when MESH_SCALE==0
if "%MESH_SCALE%"=="0" (
    echo [WARN] MESH_SCALE=0 would collapse geometry. Resetting to 1.0
    set "MESH_SCALE=1.0"
)

:: EFFECTIVE_SCALE (explicit)
set "EFFECTIVE_SCALE=%SCALE_FACTOR%"
if "%EFFECTIVE_SCALE%"=="" set "EFFECTIVE_SCALE=1.0"

:: Locate Python early
set "PYTHON_CMD="
for %%P in (python3 python) do (
    where %%P >nul 2>nul && set "PYTHON_CMD=%%P"
)
if not defined PYTHON_CMD (
    echo Python not found. Please install Python and add it to PATH.
    set "RC=1"
    goto :finish
)

:: Ensure local deps visible to Python
set "DEPS_DIR=%SCRIPT_DIR%deps"
set "PYTHONPATH=%DEPS_DIR%;%PYTHONPATH%"

:: Diagnostics
echo Using SCALE_FACTOR=%SCALE_FACTOR%  MESH_SCALE=%MESH_SCALE%  EFFECTIVE_SCALE=%EFFECTIVE_SCALE%
echo Forwarding args:%PASS_ARGS%
echo [INFO] RENAME_ACTIVE=%RENAME_ACTIVE%  (DEFAULT_RENAME_ACTIVE=%DEFAULT_RENAME_ACTIVE%)

:: Write invocation audit (single-line echoes to avoid parser surprises)
if not exist "%SCRIPT_DIR%reports" mkdir "%SCRIPT_DIR%reports"
set "INV_LOG=%SCRIPT_DIR%reports\ExportRunArgs.txt"
echo Export run at: %DATE% %TIME% > "%INV_LOG%"
echo WorkingDir: %CD% >> "%INV_LOG%"
echo SCALE_FACTOR=%SCALE_FACTOR% >> "%INV_LOG%"
echo MESH_SCALE=%MESH_SCALE% >> "%INV_LOG%"
echo EFFECTIVE_SCALE=%EFFECTIVE_SCALE% >> "%INV_LOG%"
echo Forwarded args:%PASS_ARGS% >> "%INV_LOG%"
echo DIAG=%DIAG% >> "%INV_LOG%"
echo FORCE_SAFE=%FORCE_SAFE% >> "%INV_LOG%"
echo DEFAULT_RENAME_ACTIVE=%DEFAULT_RENAME_ACTIVE% >> "%INV_LOG%"
echo RENAME_ACTIVE=%RENAME_ACTIVE% >> "%INV_LOG%"
echo PYTHON_CMD=%PYTHON_CMD% >> "%INV_LOG%"
echo -- End -- >> "%INV_LOG%"
echo [INFO] Wrote invocation to reports\ExportRunArgs.txt

:: Run ExportMeshes.py (safe call with forwarded args)
echo Running ExportMeshes.py...
if exist "%SCRIPT_DIR%ExportMeshes.py" (
    call "%PYTHON_CMD%" -u "%SCRIPT_DIR%ExportMeshes.py" --to-maya --scale-factor %EFFECTIVE_SCALE% --mesh-scale %MESH_SCALE% %PASS_ARGS%
    if errorlevel 1 (
        echo ExportMeshes.py failed.
        set "RC=1"
        goto :finish
    )
    :: Optional fixup
    if exist "%SCRIPT_DIR%ExportMeshesFixup.py" (
        echo Running ExportMeshesFixup.py to validate/fix exported OBJs...
        call "%PYTHON_CMD%" -u "%SCRIPT_DIR%ExportMeshesFixup.py" --to-maya
        if errorlevel 1 echo ExportMeshesFixup.py reported issues. Check reports\OBJValidationReport.txt
    )
) else (
    echo ExportMeshes.py not found; aborting.
    set "RC=1"
    goto :finish
)

:: Optional post steps (safe single-line calls)
if exist "%SCRIPT_DIR%FinalizeMapping.py" call "%PYTHON_CMD%" -u "%SCRIPT_DIR%FinalizeMapping.py"
if exist "%SCRIPT_DIR%CompareExports.py" call "%PYTHON_CMD%" -u "%SCRIPT_DIR%CompareExports.py"
if exist "%SCRIPT_DIR%PrintPartAssignments.py" call "%PYTHON_CMD%" -u "%SCRIPT_DIR%PrintPartAssignments.py"

:: Read INTEGRITY_FAILS safely (temp file parse)
set "INTEGRITY_FAILS=0"
if exist "%INV_LOG%" (
    set "MV_TMP=%TEMP%\mv_integrity.tmp"
    if exist "%MV_TMP%" del "%MV_TMP%" >nul 2>nul
    findstr /B /I "INTEGRITY_FAILS=" "%INV_LOG%" > "%MV_TMP%" 2>nul || rem no match
    if exist "%MV_TMP%" (
        set /p INTEGRITY_LINE=<"%MV_TMP%"
        for /f "tokens=2 delims==" %%I in ("!INTEGRITY_LINE!") do set "INTEGRITY_FAILS=%%I"
        del "%MV_TMP%" >nul 2>nul
    )
)
for /f "tokens=* delims= " %%Z in ("%INTEGRITY_FAILS%") do set "INTEGRITY_FAILS=%%Z"
if "%INTEGRITY_FAILS%"=="" set "INTEGRITY_FAILS=0"
echo [INFO] INTEGRITY_FAILS=%INTEGRITY_FAILS%

:: Decide commit allowance
set "COMMIT_ALLOWED=0"
if "%INTEGRITY_FAILS%"=="0" set "COMMIT_ALLOWED=1"
if "%COMMIT_ALLOWED%"=="0" if defined ALLOW_RENAME_ON_INTEGRITY set "COMMIT_ALLOWED=1"

:: Normalize RENAME_ACTIVE to strict 0/1
for /f "tokens=* delims= " %%R in ("%RENAME_ACTIVE%") do set "RENAME_ACTIVE=%%R"
if /I "%RENAME_ACTIVE%"=="1" (
    set "RENAME_ACTIVE=1"
) else if /I "%RENAME_ACTIVE%"=="true" (
    set "RENAME_ACTIVE=1"
) else (
    set "RENAME_ACTIVE=0"
)
echo [DBG] Normalized RENAME_ACTIVE=%RENAME_ACTIVE%

:: Append audited flags
if exist "%INV_LOG%" (
    >> "%INV_LOG%" echo COMMIT_ALLOWED=%COMMIT_ALLOWED%
    >> "%INV_LOG%" echo INTEGRITY_FAILS=%INTEGRITY_FAILS%
    >> "%INV_LOG%" echo RENAME_ACTIVE=%RENAME_ACTIVE%
)

:: Run renamer at top-level (avoid nested-parenthesis parser issues)
if "%RENAME_ACTIVE%"=="1" (
    echo [INFO] Centroid renamer enabled; running dry-run.
    call :RunRenamer
    if errorlevel 1 set "RC=1"
) else (
    echo [INFO] Centroid renamer disabled; skipping renamer.
)

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
    echo Running RenameExportsByCentroid.py (DRY-RUN)...
    :: Export the batch decision so Python can see it via env
    set "RENAME_ACTIVE=%RENAME_ACTIVE%"
    :: Run Python renamer (fully quoted paths) and write dry-run log
    call "%PYTHON_CMD%" -u "%SCRIPT_DIR%RenameExportsByCentroid.py" --vox Character --debug > "%SCRIPT_DIR%reports\RenameDryrun_Character.log" 2>&1
    set "RNERR=%ERRORLEVEL%"
    echo ----- RenameExportsByCentroid.py dry-run output (tail) -----
    type "%SCRIPT_DIR%reports\RenameDryrun_Character.log" 2>nul
    echo ------------------------------------------------------------
    if "%RNERR%" NEQ "0" (
        echo [ERROR] RenameExportsByCentroid.py failed during dry-run. See %SCRIPT_DIR%reports\RenameDryrun_Character.log
        endlocal
        exit /b 1
    )
    :: Prompt user to commit (read from console so redirection doesn't break it)
    set "COMMIT_RENAME="
    <CON set /p COMMIT_RENAME=RenameExportsByCentroid.py finished dry-run. Commit global renames now? [y/N] 
    if /I "%COMMIT_RENAME%"=="y" (
        echo Committing global assignment rename...
        call "%PYTHON_CMD%" -u "%SCRIPT_DIR%RenameExportsByCentroid.py" --vox Character --commit > "%SCRIPT_DIR%reports\RenameCommit_Character.log" 2>&1
        set "RCERR=%ERRORLEVEL%"
        if "%RCERR%" NEQ "0" (
            echo [ERROR] Global assignment commit failed. See %SCRIPT_DIR%reports\RenameCommit_Character.log
            type "%SCRIPT_DIR%reports\RenameCommit_Character.log" 2>nul
            endlocal
            exit /b 1
        ) else (
            echo [SUCCESS] Global assignment commit successful. See %SCRIPT_DIR%reports\RenameCommit_Character.log
            type "%SCRIPT_DIR%reports\RenameCommit_Character.log" 2>nul
        )
    ) else (
        echo Skipping global assignment commit.
    )
    endlocal
    exit /b 0