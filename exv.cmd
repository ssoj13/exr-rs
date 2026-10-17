@echo off
setlocal EnableDelayedExpansion

:: Resolve exr-rs binary (release -> debug -> PATH)
set "ROOT=%~dp0"
if exist "%ROOT%target\release\exr-rs.exe" (
    set "EXR_BIN=%ROOT%target\release\exr-rs.exe"
) else if exist "%ROOT%target\debug\exr-rs.exe" (
    set "EXR_BIN=%ROOT%target\debug\exr-rs.exe"
) else (
    set "EXR_BIN=exr-rs"
)

:: Default run with no arguments: convert the Robat example sequence if it exists, otherwise show usage
if "%~1"=="" (
    if exist "D:\_example_files\EXR\Robat\robat.0001.exr" (
        echo [exv.cmd] Converting D:\_example_files\EXR\Robat\robat.0001.exr -^> D:\_example_files\EXR\Robat\robat.exv ...
        "%EXR_BIN%" convert "D:\_example_files\EXR\Robat\robat.0001.exr" "D:\_example_files\EXR\Robat\robat.exv" --half --exv-codec bc6h-lossless
        if errorlevel 1 exit /b 1
        "%EXR_BIN%" info "D:\_example_files\EXR\Robat\robat.exv"
        exit /b 0
    )
    echo Usage:
    echo   exv.cmd ^<frame.0001.exr ^| frame.####.exr ^| folder^> [out.exv] [--half] [--exv-codec bc6h-lossless^|bc6h-lossy^|htj2k^|fast^|wave^|rt] [--gop N]
    echo   exv.cmd ^<clip.exv^>                                 ^(opens .exv clip in GPU viewer^)
    echo.
    echo Examples:
    echo   exv.cmd "D:\_example_files\EXR\Robat\robat.0001.exr"
    echo   exv.cmd "D:\_example_files\EXR\Robat\robat.0001.exr" "D:\_example_files\EXR\Robat\robat_lossy.exv" --exv-codec bc6h-lossy --gop 8
    exit /b 0
)

set "IN=%~1"
set "EXT=%~x1"

:: If the input is already an .exv file and no output is given, open it in the GPU viewer
if /I "%EXT%"==".exv" if "%~2"=="" (
    "%EXR_BIN%" view "%IN%"
    exit /b %ERRORLEVEL%
)

:: Determine output .exv path if not explicitly passed as 2nd positional arg (or if 2nd arg starts with '-')
set "OUT=%~2"
set "SHIFT_OUT=1"
if "%OUT%"=="" (
    set "SHIFT_OUT=0"
) else if "%OUT:~0,1%"=="-" (
    set "SHIFT_OUT=0"
)

if "%SHIFT_OUT%"=="0" (
    :: Strip frame number suffix from stem if present (e.g. robat.0001.exr -> robat.exv)
    set "DIR=%~dp1"
    set "STEM=%~n1"
    for %%A in ("!STEM!") do (
        set "BASE=%%~nA"
        set "TAIL=%%~xA"
    )
    if not "!TAIL!"=="" (
        set "OUT=!DIR!!BASE!.exv"
    ) else (
        set "OUT=!DIR!!STEM!.exv"
    )
    shift
) else (
    shift
    shift
)

:: Collect remaining flags
set "EXTRA="
:collect_args
if "%~1"=="" goto run_convert
set "EXTRA=!EXTRA! %1"
shift
goto collect_args

:run_convert
if "%EXTRA%"=="" set "EXTRA= --half"
echo [exv.cmd] "%EXR_BIN%" convert "%IN%" "%OUT%"%EXTRA%
"%EXR_BIN%" convert "%IN%" "%OUT%"%EXTRA%
if errorlevel 1 exit /b %ERRORLEVEL%
"%EXR_BIN%" info "%OUT%"
