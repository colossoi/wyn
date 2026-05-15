@echo off
setlocal

cd /d "%~dp0\.."

echo Compiling scene.wyn...
cargo run --release --bin wyn -- compile testfiles\playground\acko_masthead\scene.wyn -o testfiles\playground\acko_masthead\scene.spv
if errorlevel 1 goto :fail

echo Validating...
spirv-val testfiles\playground\acko_masthead\scene.spv
if errorlevel 1 goto :fail

echo OK: scene.spv built and validated
echo.
echo To render:
echo   target\release\viz.exe vf testfiles\playground\acko_masthead\scene.spv ^
--topology triangle-list --shadertoy ^
--storage-dir testfiles\playground\acko_masthead ^
--index-buffer testfiles\playground\acko_masthead\indices.bin -v
goto :eof

:fail
echo FAILED
exit /b 1
