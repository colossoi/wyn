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

"%WYN%" compile lib\testfiles\dist_demo.wyn -o "%TEMP%\dist_demo.spv" || exit /b 1
spirv-val "%TEMP%\dist_demo.spv" || exit /b 1
"%WYN%" compile lib\testfiles\dist_demo.wyn -t wgsl -o "%TEMP%\dist_demo.wgsl" || exit /b 1

"%WYN%" compile lib\testfiles\stats_normal.wyn -o "%TEMP%\stats_normal.spv" || exit /b 1
spirv-val "%TEMP%\stats_normal.spv" || exit /b 1
"%WYN%" compile lib\testfiles\stats_normal.wyn -t wgsl -o "%TEMP%\stats_normal.wgsl" || exit /b 1

"%WYN%" compile lib\testfiles\stats_uniform.wyn -o "%TEMP%\stats_uniform.spv" || exit /b 1
spirv-val "%TEMP%\stats_uniform.spv" || exit /b 1
"%WYN%" compile lib\testfiles\stats_uniform.wyn -t wgsl -o "%TEMP%\stats_uniform.wgsl" || exit /b 1

"%WYN%" compile lib\testfiles\stats_exponential.wyn -o "%TEMP%\stats_exponential.spv" || exit /b 1
spirv-val "%TEMP%\stats_exponential.spv" || exit /b 1
"%WYN%" compile lib\testfiles\stats_exponential.wyn -t wgsl -o "%TEMP%\stats_exponential.wgsl" || exit /b 1

"%WYN%" compile lib\testfiles\stats_uniform_int.wyn -o "%TEMP%\stats_uniform_int.spv" || exit /b 1
spirv-val "%TEMP%\stats_uniform_int.spv" || exit /b 1
"%WYN%" compile lib\testfiles\stats_uniform_int.wyn -t wgsl -o "%TEMP%\stats_uniform_int.wgsl" || exit /b 1

"%WYN%" compile lib\testfiles\noise_smoke.wyn -o "%TEMP%\noise_smoke.spv" || exit /b 1
spirv-val "%TEMP%\noise_smoke.spv" || exit /b 1
"%WYN%" compile lib\testfiles\noise_smoke.wyn -t wgsl -o "%TEMP%\noise_smoke.wgsl" || exit /b 1

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

"%TEPHRA%" run "%TEMP%\dist_demo.spv" --entry dist_normal_fill -n 256 -w 64

echo.
echo Stats slots: [count, mean, variance, stddev, min, max] over N=65536 draws.

echo.
echo --- stats_normal (standard normal) ---
"%TEPHRA%" run "%TEMP%\stats_normal.spv" --entry stats_normal -n 6 -w 64
echo Expect: mean 0, variance 1, stddev 1, extremes ~ +/- 4 to 5.

echo.
echo --- stats_uniform (uniform real [0,1)) ---
"%TEPHRA%" run "%TEMP%\stats_uniform.spv" --entry stats_uniform -n 6 -w 64
echo Expect: mean 0.5, variance 1/12 ~ 0.0833, stddev ~ 0.289, min ~ 0, max ~ 1.

echo.
echo --- stats_exponential (exponential rate 1) ---
"%TEPHRA%" run "%TEMP%\stats_exponential.spv" --entry stats_exponential -n 6 -w 64
echo Expect: mean 1, variance 1, stddev 1, min ~ 0, max ~ 10 to 12.

echo.
echo --- stats_uniform_int (uniform int [0,10), lifted to f32) ---
"%TEPHRA%" run "%TEMP%\stats_uniform_int.spv" --entry stats_uniform_int -n 6 -w 64
echo Expect: mean 4.5, variance 99/12 ~ 8.25, stddev ~ 2.872, min 0, max 9.

echo.
echo --- noise_smoke (one query at p=(3.5, 7.25), seed 0x9e3779b9) ---
"%TEPHRA%" run "%TEMP%\noise_smoke.spv" --entry noise_smoke -n 5 -w 64
echo Slots: [value2, perlin2, simplex2, worley2, fbm_perlin(6oct, lac=2, gain=0.5)].
echo Expect: first three in [-1, 1]; worley2 a small positive distance (~0..1.5);
echo         fbm_perlin a damped sum of octaves (no fixed range, but bounded).
