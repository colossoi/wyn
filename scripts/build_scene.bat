@echo off
setlocal

cd /d "%~dp0\.."

echo Compiling scene.wyn...
cargo run --release --bin wyn -- build testfiles\acko_masthead --graphics -o testfiles\acko_masthead\scene.spv
if errorlevel 1 goto :fail

echo Validating...
spirv-val testfiles\acko_masthead\scene.spv
if errorlevel 1 goto :fail

echo OK: scene.spv built and validated
echo.
echo To render:
echo   extra\viz\target\release\viz.exe pipeline testfiles\acko_masthead\scene.spv ^
--storage-dir testfiles\acko_masthead ^
--index-buffer testfiles\acko_masthead\indices.bin -v
goto :eof

:fail
echo FAILED
exit /b 1
