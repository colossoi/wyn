@echo off
setlocal

cd /d "%~dp0\.."

echo Compiling scene.wyn...
cargo run --release --bin wyn -- build testfiles\playground\acko_masthead\scene.wyn --graphics -o testfiles\playground\acko_masthead\scene.spv
if errorlevel 1 goto :fail

echo Validating...
spirv-val testfiles\playground\acko_masthead\scene.spv
if errorlevel 1 goto :fail

echo OK: scene.spv built and validated
echo.
echo To render:
echo   target\release\viz.exe pipeline testfiles\playground\acko_masthead\scene.spv ^
--storage-dir testfiles\playground\acko_masthead ^
--index-buffer testfiles\playground\acko_masthead\indices.bin -v
goto :eof

:fail
echo FAILED
exit /b 1
