@echo off
REM Streamlit DCA App Launcher for Windows
REM This script checks Python installation, sets up environment, and launches the Streamlit app

REM Always run from the directory that contains this script (prevents wrong working-dir issues)
setlocal EnableExtensions EnableDelayedExpansion
set "SCRIPT_DIR=%~dp0"
pushd "!SCRIPT_DIR!" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Failed to switch to launcher directory.
    echo Tip: if this folder path contains special characters like ^& you must run from File Explorer.
    pause
    exit /b 1
)

echo ========================================
echo Streamlit DCA Application Launcher
echo ========================================
echo.

REM Check Python installation
echo [1/6] Checking Python installation...
set "PYTHON_EXE=python"
"%PYTHON_EXE%" --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.8-3.11 and try again.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('"%PYTHON_EXE%" --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python version found: %PYTHON_VERSION%

REM Validate Python version (3.8-3.11 required for pycaret compatibility)
"%PYTHON_EXE%" -c "import sys; v=sys.version_info; raise SystemExit(0 if (v.major==3 and 8<=v.minor<=11) else 1)" >nul 2>&1
if errorlevel 1 (
    REM If a supported Python is already installed, prefer it automatically.
    call :find_existing_supported_python
    if not errorlevel 1 (
        goto :python_ok
    )

    echo WARNING: Python version %PYTHON_VERSION% is NOT supported for full functionality.
    echo Recommended: Python 3.8, 3.9, 3.10, or 3.11
    echo PyCaret requires Python 3.8-3.11 ^(not 3.12+^)
    echo.
    echo Choose an option:
    echo   [1] Continue anyway
    echo   [2] Auto-install Python 3.11 via winget
    echo   [3] Auto-install Python 3.10 via winget
    echo   [4] Exit and install manually
    echo.
    choice /c 1234 /n /m "Select 1-4: " 2>nul
    if errorlevel 255 goto :python_exit
    if errorlevel 4 goto :python_exit
    if errorlevel 3 goto :python_install_310
    if errorlevel 2 goto :python_install_311
    goto :python_continue
)
goto :python_ok

:python_install_311
call :install_python 311
goto :python_recheck

:python_install_310
call :install_python 310
goto :python_recheck

:python_recheck
REM Prefer the newly installed python if we can find it, otherwise ask user to re-run.
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
if exist "%ProgramFiles%\Python311\python.exe" set "PYTHON_EXE=%ProgramFiles%\Python311\python.exe"
if exist "%ProgramFiles%\Python310\python.exe" set "PYTHON_EXE=%ProgramFiles%\Python310\python.exe"

"%PYTHON_EXE%" --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Could not locate the newly installed Python automatically.
    echo Please close this window and run launch_app.bat again.
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('"%PYTHON_EXE%" --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python version now: %PYTHON_VERSION%
"%PYTHON_EXE%" -c "import sys; v=sys.version_info; raise SystemExit(0 if (v.major==3 and 8<=v.minor<=11) else 1)" >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Installed Python is still not in the supported range 3.8-3.11.
    pause
    exit /b 1
)
goto :python_ok

:python_continue
echo Continuing with current Python: %PYTHON_VERSION%
goto :python_ok

:python_exit
echo Please install Python 3.8-3.11 and re-run launch_app.bat.
echo Download from: https://www.python.org/downloads/
pause
exit /b 1

:python_ok
echo.

REM Use a SHORT per-user venv path to avoid Windows long path pip failures.
REM You may override by setting DCA_VENV_DIR before running this script.
if "%DCA_VENV_DIR%"=="" (
    set "DCA_VENV_DIR=%LOCALAPPDATA%\DCA_App\venv"
)
set "VENV_PY=%DCA_VENV_DIR%\Scripts\python.exe"

REM Check/create virtual environment
echo [2/6] Checking virtual environment...
if exist "%VENV_PY%" (
    echo Virtual environment found: %DCA_VENV_DIR%
) else (
    echo Virtual environment not found. Creating new one at:
    echo   %DCA_VENV_DIR%
    if not exist "%LOCALAPPDATA%\DCA_App" (
        mkdir "%LOCALAPPDATA%\DCA_App" >nul 2>&1
    )
    "%PYTHON_EXE%" -m venv "%DCA_VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
)
echo.

REM Check Windows long path support (warning only)
echo [3/6] Checking Windows Long Paths support...
if exist "src\dca_tools.py" (
    "%PYTHON_EXE%" src\dca_tools.py check-longpaths >nul 2>&1
    if errorlevel 1 (
        echo WARNING: Windows Long Paths appear to be disabled.
        echo This can cause pip install failures with deep package paths.
        echo We will continue, but if installation fails:
        echo   - Enable Long Paths ^(admin + reboot^), or
        echo   - Move this project to a shorter folder path
        echo.
    )
)
echo.

REM Validate requirements file before installation
echo [4/6] Validating requirements file...
if not exist "src\dca_tools.py" (
    echo WARNING: src\dca_tools.py not found. Skipping validation...
    echo This is okay if you're using an older version of the launcher.
    echo.
    goto :skip_validation
)
"%PYTHON_EXE%" src\dca_tools.py validate-requirements requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Requirements file validation failed!
    echo Please check requirements.txt for errors.
    echo Common issues:
    echo   - Typos in package names
    echo   - File encoding problems
    echo   - Corrupted file during transfer
    echo.
    echo If you received this file from someone else, ask them to:
    echo   1. Verify the file is not corrupted
    echo   2. Re-send the requirements.txt file
    echo   3. Ensure the file is UTF-8 encoded
    echo.
    pause
    exit /b 1
)
echo Requirements file validation passed.
echo.

REM Continue main flow (avoid falling into helper labels below)
goto :skip_validation

REM ------------------------------
REM Helpers
REM ------------------------------
:install_python
REM %1 is 311 or 310
REM First, check if that Python is already installed.
call :find_python_by_version %1
if not errorlevel 1 (
    echo.
    if "%1"=="311" (set "PY_DISP=3.11") else (set "PY_DISP=3.10")
    echo Python %PY_DISP% already appears to be installed at:
    echo   %PYTHON_EXE%
    exit /b 0
)

where winget >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: winget is not available on this system.
    echo Please install Python 3.8-3.11 manually and re-run launch_app.bat.
    echo Opening downloads page...
    start "" "https://www.python.org/downloads/windows/"
    exit /b 1
)

echo.
if "%1"=="311" (set "PY_DISP=3.11") else (set "PY_DISP=3.10")
echo Installing Python %PY_DISP% via winget ^(no-admin / per-user^), if possible...
if "%1"=="311" (
    winget install -e --id Python.Python.3.11 --scope user --silent --accept-package-agreements --accept-source-agreements
) else (
    winget install -e --id Python.Python.3.10 --scope user --silent --accept-package-agreements --accept-source-agreements
)
if errorlevel 1 (
    echo.
    echo ERROR: winget failed to install Python without admin approval.
    echo Please install Python 3.8-3.11 manually ^(per-user / \"Install Now\" / no admin^), then re-run launch_app.bat.
    echo Opening downloads page...
    start "" "https://www.python.org/downloads/windows/"
    exit /b 1
)
exit /b 0

:find_existing_supported_python
REM Finds an existing supported Python (3.11/3.10/3.9/3.8) and sets PYTHON_EXE + PYTHON_VERSION.
call :find_python_by_version 311
if not errorlevel 1 goto :found_supported
call :find_python_by_version 310
if not errorlevel 1 goto :found_supported
call :find_python_by_version 39
if not errorlevel 1 goto :found_supported
call :find_python_by_version 38
if not errorlevel 1 goto :found_supported
exit /b 1

:found_supported
for /f "tokens=2" %%i in ('"%PYTHON_EXE%" --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found supported Python: %PYTHON_VERSION%
echo Using Python at: %PYTHON_EXE%
exit /b 0

:find_python_by_version
REM %1 = 311, 310, 39, or 38
set "FOUND_PY="

REM Common per-user installs
if "%1"=="311" if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "FOUND_PY=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if "%1"=="310" if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" set "FOUND_PY=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
if "%1"=="39"  if exist "%LOCALAPPDATA%\Programs\Python\Python39\python.exe"  set "FOUND_PY=%LOCALAPPDATA%\Programs\Python\Python39\python.exe"
if "%1"=="38"  if exist "%LOCALAPPDATA%\Programs\Python\Python38\python.exe"  set "FOUND_PY=%LOCALAPPDATA%\Programs\Python\Python38\python.exe"

REM Common system installs
if "%FOUND_PY%"=="" (
    if "%1"=="311" if exist "%ProgramFiles%\Python311\python.exe" set "FOUND_PY=%ProgramFiles%\Python311\python.exe"
    if "%1"=="310" if exist "%ProgramFiles%\Python310\python.exe" set "FOUND_PY=%ProgramFiles%\Python310\python.exe"
    if "%1"=="39"  if exist "%ProgramFiles%\Python39\python.exe"  set "FOUND_PY=%ProgramFiles%\Python39\python.exe"
    if "%1"=="38"  if exist "%ProgramFiles%\Python38\python.exe"  set "FOUND_PY=%ProgramFiles%\Python38\python.exe"
)

REM Python Launcher (py.exe) if present
if "%FOUND_PY%"=="" (
    where py >nul 2>&1
    if not errorlevel 1 (
        REM Use /C: to avoid findstr treating leading '-' as an option
        if "%1"=="311" for /f "tokens=2*" %%A in ('py -0p 2^>nul ^| findstr /B /C:"-3.11"') do set "FOUND_PY=%%A"
        if "%1"=="310" for /f "tokens=2*" %%A in ('py -0p 2^>nul ^| findstr /B /C:"-3.10"') do set "FOUND_PY=%%A"
        if "%1"=="39"  for /f "tokens=2*" %%A in ('py -0p 2^>nul ^| findstr /B /C:"-3.9"') do set "FOUND_PY=%%A"
        if "%1"=="38"  for /f "tokens=2*" %%A in ('py -0p 2^>nul ^| findstr /B /C:"-3.8"') do set "FOUND_PY=%%A"
    )
)

if "%FOUND_PY%"=="" (
    exit /b 1
)

set "PYTHON_EXE=%FOUND_PY%"
exit /b 0

:skip_validation
REM Install/upgrade dependencies
echo [5/6] Installing/upgrading dependencies...
echo This may take a few minutes on first run...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel >nul 2>&1
if errorlevel 1 (
    echo WARNING: Failed to upgrade pip, but continuing...
)
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies.
    echo.
    echo If the error mentions deep paths or suggests enabling Long Paths:
    echo   - Enable Windows Long Paths ^(admin + reboot^), OR
    echo   - Move this project to a shorter folder ^(e.g., C:\DCA_App^)
    echo.
    echo Troubleshooting steps:
    echo   1. Check your internet connection
    echo   2. Verify Python version is 3.8-3.11
    echo   3. Try running: pip install --upgrade pip
    echo   4. Check the error message above for the specific package that failed
    echo   5. If a specific package fails, you can try installing it manually:
    echo      pip install ^<package_name^>
    echo.
    echo If the error mentions a package name that looks like a typo:
    echo   - The requirements file may have been corrupted during transfer
    echo   - Ask the person who shared the folder to verify requirements.txt
    echo   - Compare the file with the original to find any differences
    echo.
    pause
    exit /b 1
)
echo Dependencies installed successfully.
echo.

REM Launch Streamlit app
echo [6/6] Launching Streamlit application...
echo.
echo ========================================
echo The app will open in your default browser.
echo Press Ctrl+C to stop the application.
echo ========================================
echo.

REM Start Streamlit and open browser automatically
set "APP_PATH=%SCRIPT_DIR%streamlit_app\app.py"
if not exist "%APP_PATH%" (
    echo ERROR: Could not find Streamlit app entrypoint:
    echo   %APP_PATH%
    echo.
    echo This usually means you are running the launcher from the wrong folder.
    echo Please run launch_app.bat from the DCA-Application folder that contains streamlit_app\app.py.
    echo.
    pause
    exit /b 1
)

"%VENV_PY%" -m streamlit run "%APP_PATH%" --server.headless=false

REM If we get here, the app was closed
echo.
echo Application closed.
pause

popd >nul 2>&1
endlocal

