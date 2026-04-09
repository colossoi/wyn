@echo off
setlocal

cd /d "%~dp0\.."

echo Compiling miner.wyn...
cargo run --release --bin wyn -- compile testfiles\miner.wyn -o %TEMP%\miner.spv
if errorlevel 1 goto :fail

echo Assembling sha256_compress...
spirv-as lib\sha256_compress.spvasm -o %TEMP%\sha256_compress.spv --target-env spv1.5
if errorlevel 1 goto :fail

echo Linking...
spirv-link %TEMP%\miner.spv %TEMP%\sha256_compress.spv -o testfiles\miner.spv
if errorlevel 1 goto :fail

echo Copying pipeline descriptor...
copy /y %TEMP%\miner.json testfiles\miner.json >nul

echo Validating...
spirv-val testfiles\miner.spv
if errorlevel 1 goto :fail

echo OK: testfiles\miner.spv validated
goto :eof

:fail
echo FAILED
exit /b 1
