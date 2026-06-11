@echo off
rem Compile + validate the rng library demos, and (with --run) execute them on
rem the GPU via tephra. Hand-coded commands, one block per demo — when you add a
rem driver to lib\testfiles\, add its commands here.
rem
rem Usage:  scripts\run_tests.bat [--run]
cd /d "%~dp0.."

set RUN=0
if "%~1"=="--run" set RUN=1
set "WYN=target\debug\wyn.exe"

echo Building wyn...
cargo build -p wyn || exit /b 1

echo == compile + validate ==

"%WYN%" compile lib\testfiles\fasthash_demo.wyn -o "%TEMP%\fasthash_demo.spv" || exit /b 1
spirv-val "%TEMP%\fasthash_demo.spv" || exit /b 1
"%WYN%" compile lib\testfiles\fasthash_demo.wyn -t wgsl -o "%TEMP%\fasthash_demo.wgsl" || exit /b 1

"%WYN%" compile lib\testfiles\threefry_demo.wyn -o "%TEMP%\threefry_demo.spv" || exit /b 1
spirv-val "%TEMP%\threefry_demo.spv" || exit /b 1
"%WYN%" compile lib\testfiles\threefry_demo.wyn -t wgsl -o "%TEMP%\threefry_demo.wgsl" || exit /b 1

echo compile + validate: OK

if "%RUN%"=="0" (
    echo ^(pass --run to execute on the GPU via tephra^)
    exit /b 0
)

echo == tephra run ^(needs a Vulkan device^) ==
pushd extra\tephra
cargo build || ( popd & exit /b 1 )
popd
set "TEPHRA=extra\tephra\target\debug\tephra.exe"

"%TEPHRA%" run "%TEMP%\fasthash_demo.spv" --entry fasthash_fill -n 256 -w 64 --input iota
"%TEPHRA%" run "%TEMP%\threefry_demo.spv" --entry threefry_fill -n 256 -w 64 --input iota
