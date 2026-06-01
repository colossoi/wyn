@echo off
REM Profile the wyn compiler with samply by compiling every testfile in
REM one process. Multi-input mode (`wyn compile a.wyn b.wyn ...`) means a
REM single binary invocation walks the full corpus, giving samply a long,
REM diverse run to sample from — no per-iteration startup noise, no
REM single-file overfitting.
REM
REM Usage (from the repo root):
REM   scripts\profile_compile.bat
REM
REM Prerequisites:
REM   cargo install samply
REM
REM The script:
REM   1. Builds wyn with the `profiling` cargo profile
REM      (release opts + full debuginfo so samply can resolve every
REM      function name including inlined frames).
REM   2. Collects every .wyn under testfiles\ and testfiles\playground\.
REM   3. Runs `samply record` against the binary, compiling all of them
REM      into a scratch output directory.
REM   4. samply opens the trace in the Firefox Profiler UI
REM      (https://profiler.firefox.com) when it finishes.
REM
REM The trace also lands on disk (samply prints the path); re-open later
REM with:  samply load ^<file^>

setlocal EnableDelayedExpansion

if "%CARGO_PROFILE%"=="" set CARGO_PROFILE=profiling

REM Resolve repo root (one level above this script).
set REPO_ROOT=%~dp0..
pushd "%REPO_ROOT%" || exit /b 1

REM Check samply is installed.
where samply >nul 2>&1
if errorlevel 1 (
    echo samply not found. Install it with:
    echo   cargo install samply
    popd
    exit /b 1
)

echo Building wyn with profile '%CARGO_PROFILE%'...
cargo build --profile %CARGO_PROFILE% --bin wyn
if errorlevel 1 (
    popd
    exit /b %errorlevel%
)

set BINARY=%CD%\target\%CARGO_PROFILE%\wyn.exe
if not exist "%BINARY%" (
    echo Built binary not found at %BINARY%
    popd
    exit /b 1
)

REM Collect every .wyn under testfiles\ and testfiles\playground\.
set FILES=
for %%f in (testfiles\*.wyn) do set FILES=!FILES! "%%f"
for %%f in (testfiles\playground\*.wyn) do set FILES=!FILES! "%%f"

if "!FILES!"=="" (
    echo No testfiles found under testfiles\ or testfiles\playground\
    popd
    exit /b 1
)

set OUTDIR=%TEMP%\wyn_profile_out
if not exist "%OUTDIR%" mkdir "%OUTDIR%"

echo Profiling compile of every testfile (output -^> %OUTDIR%)...
samply record -- "%BINARY%" compile !FILES! -o "%OUTDIR%"

set EXITCODE=%errorlevel%
popd
exit /b %EXITCODE%
