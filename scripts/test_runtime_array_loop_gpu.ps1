# Requires release builds of wyn and viz, spirv-val, and GPU access.
param([string]$Compiler)
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$outDir = Join-Path $workspace 'tmp/runtime-array-loop/gpu'
$null = New-Item -ItemType Directory -Force -Path $outDir
$wyn = if ($Compiler) { (Resolve-Path -LiteralPath $Compiler).Path } else { Join-Path $workspace 'target/release/wyn.exe' }
$viz = Join-Path $workspace 'extra/viz/target/release/viz.exe'
$inputFile = Join-Path $outDir 'input.bin'
$logFile = Join-Path $outDir 'run.log'
$cases = 0
foreach ($scenario in @('runtime_array_loop', 'runtime_array_branch', 'runtime_array_zip_map', 'allocated_array_loop', 'runtime_array_return_alias', 'runtime_array_four_inputs', 'runtime_array_updated_slice', 'runtime_array_three_outputs', 'runtime_array_rectangular_slice', 'runtime_array_shrinking_loop')) {
    $fixture = if ($scenario -in @('runtime_array_four_inputs', 'runtime_array_updated_slice', 'runtime_array_three_outputs', 'runtime_array_rectangular_slice', 'runtime_array_shrinking_loop')) { $scenario } else { "triangulation_$scenario" }
    $source = Join-Path $workspace "testfiles/regressions/$fixture.wyn"
    foreach ($target in @('wgsl', 'spirv')) {
        foreach ($optimize in @($false, $true)) {
            $extension = if ($target -eq 'spirv') { 'spv' } else { 'wgsl' }
            $shader = Join-Path $outDir "$scenario-$optimize.$extension"
            $buildArgs = @('build', $source, '--max-warnings', '0', '-t', $target, '-o', $shader)
            if ($optimize) { $buildArgs += '-O' }
            & $wyn @buildArgs
            if ($LASTEXITCODE -ne 0) { throw 'Runtime array compilation failed' }
            if ($target -eq 'spirv') {
                & spirv-val --target-env vulkan1.3 $shader
                if ($LASTEXITCODE -ne 0) { throw 'SPIR-V validation failed' }
            }
            foreach ($length in @(3, 0, 1, 10, 11, 65)) {
                $values = if ($length -eq 3) { @(3, 7, -2) } else {
                    @(for ($i = 0; $i -lt $length; $i++) { ($i % 11) - 5 })
                }
                $expected = @($values | ForEach-Object { $_ + 1 })
                $parameter = 'xs'
                $bytes = [byte[]]@($values | ForEach-Object { [BitConverter]::GetBytes([int]$_) })
                if ($scenario -in @('runtime_array_branch', 'runtime_array_return_alias', 'runtime_array_updated_slice', 'runtime_array_three_outputs')) {
                    $expected = if ($length -gt 10) { @() } else { @($values) }
                } elseif ($scenario -eq 'runtime_array_zip_map') {
                    $expected = if ($length -gt 10) { @() } else { @($values | ForEach-Object { $_ * 2 }) }
                } elseif ($scenario -eq 'allocated_array_loop') {
                    $parameter = 'n'
                    $bytes = [BitConverter]::GetBytes([int]$length)
                    $expected = @(for ($i = 0; $i -lt $length; $i++) { $i })
                } elseif ($scenario -eq 'runtime_array_four_inputs') {
                    $expected = @($values | ForEach-Object { $_ * 4 })
                } elseif ($scenario -eq 'runtime_array_rectangular_slice') {
                    $expected = @($values | Select-Object -Skip 1 | ForEach-Object { $_; $_+1; $_+2 })
                } elseif ($scenario -eq 'runtime_array_shrinking_loop') {
                    $parameter = 'n'
                    $bytes = [BitConverter]::GetBytes([int]$length)
                    $expected = @(for ($i = 2; $i -lt $length; $i++) { $i })
                }
                [IO.File]::WriteAllBytes($inputFile, $bytes)
                $expectedOutputs = @{ main = $expected }
                if ($scenario -eq 'runtime_array_return_alias') {
                    $expectedOutputs = @{ result_0 = $expected; result_1 = @($expected | Select-Object -Skip 1) }
                } elseif ($scenario -eq 'runtime_array_three_outputs') {
                    $expectedOutputs = @{ result_0 = $expected; result_1 = $expected; result_2 = @($expected | Select-Object -Skip 1) }
                } elseif ($scenario -eq 'runtime_array_updated_slice') {
                    $expected = @($expected | Select-Object -Skip 1)
                    if ($expected.Count) { $expected[0] = 99 }
                    $expectedOutputs = @{ main = $expected }
                }
                $runArgs = @('pipeline', $shader, '--entry', 'main', '--headless', '--no-config', '--input', "${parameter}:$inputFile")
                if ($scenario -eq 'runtime_array_four_inputs') {
                    foreach ($name in @('ys', 'zs', 'ws')) { $runArgs += @('--input', "${name}:$inputFile") }
                }
                foreach ($name in $expectedOutputs.Keys) {
                    $runArgs += @('--output', "${name}:$outDir/$name.json")
                }
                & $viz @runArgs *> $logFile
                if ($LASTEXITCODE -ne 0) {
                    throw "GPU execution failed: $scenario, $target, optimized=$optimize, n=$length; see $logFile"
                }
                if (Select-String -LiteralPath $logFile -Pattern 'Validation Error' -Quiet) {
                    $failureLog = Join-Path $outDir "$scenario-$target-$optimize-$length.log"
                    Copy-Item -LiteralPath $logFile -Destination $failureLog
                    throw "GPU validation failed: $scenario, $target, optimized=$optimize, n=$length; see $failureLog"
                }
                foreach ($name in $expectedOutputs.Keys) {
                    $expected = @($expectedOutputs[$name])
                    $result = Get-Content -LiteralPath "$outDir/$name.json" -Raw | ConvertFrom-Json
                    $actual = @($result.backing_buffer)
                    if ($scenario -eq 'runtime_array_rectangular_slice') {
                        $actual = @($actual | ForEach-Object { $_ })
                    }
                    if ($result.byte_length -ne 4 * $expected.Count -or $actual.Count -ne $expected.Count) {
                        throw "Wrong output size: $scenario, $name, $target, n=$length; got $($result.byte_length) bytes"
                    }
                    for ($i = 0; $i -lt $expected.Count; $i++) {
                        if ($actual[$i] -ne $expected[$i]) {
                            throw "Wrong output: $scenario, $name, $target, optimized=$optimize, n=$length, index=$i"
                        }
                    }
                }
                $cases++
            }
        }
    }
}
Write-Output "$cases GPU runtime array cases passed, including returned aliases, four inputs, and empty inputs."
