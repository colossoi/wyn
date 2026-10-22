# Requires release builds of wyn and viz, spirv-val, and GPU access.
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$outDir = Join-Path $workspace 'tmp/runtime-array-loop/gpu'
$null = New-Item -ItemType Directory -Force -Path $outDir
$wyn = Join-Path $workspace 'target/release/wyn.exe'
$viz = Join-Path $workspace 'extra/viz/target/release/viz.exe'
$source = Join-Path $workspace 'testfiles/regressions/triangulation_runtime_array_loop.wyn'
$inputFile = Join-Path $outDir 'input.bin'
$outputFile = Join-Path $outDir 'output.json'
$logFile = Join-Path $outDir 'run.log'
$cases = 0
$validationWarnings = 0
foreach ($target in @('wgsl', 'spirv')) {
    foreach ($optimize in @($false, $true)) {
        $extension = if ($target -eq 'spirv') { 'spv' } else { 'wgsl' }
        $shader = Join-Path $outDir "loop-$optimize.$extension"
        $buildArgs = @('build', $source, '--max-warnings', '0', '-t', $target, '-o', $shader)
        if ($optimize) { $buildArgs += '-O' }
        & $wyn @buildArgs
        if ($LASTEXITCODE -ne 0) { throw 'Runtime array loop compilation failed' }
        if ($target -eq 'spirv') {
            & spirv-val --target-env vulkan1.3 $shader
            if ($LASTEXITCODE -ne 0) { throw 'SPIR-V validation failed' }
        }
        foreach ($length in @(3, 0, 1, 64, 65, 257)) {
            $values = if ($length -eq 3) { @(3, 7, -2) } else {
                @(for ($i = 0; $i -lt $length; $i++) { ($i % 11) - 5 })
            }
            $bytes = [byte[]]@($values | ForEach-Object { [BitConverter]::GetBytes([int]$_) })
            [IO.File]::WriteAllBytes($inputFile, $bytes)
            & $viz pipeline $shader --entry main --headless --no-config --input "xs:$inputFile" `
                --output "main:$outputFile" *> $logFile
            if ($LASTEXITCODE -ne 0) {
                throw "GPU execution failed: $target, optimized=$optimize, n=$length; see $logFile"
            }
            if (Select-String -LiteralPath $logFile -Pattern 'Validation Error' -Quiet) {
                $validationWarnings++
                Copy-Item -LiteralPath $logFile -Destination (Join-Path $outDir "$target-$optimize-$length.log")
            }
            $result = Get-Content -LiteralPath $outputFile -Raw | ConvertFrom-Json
            $actual = @($result.backing_buffer)
            if ($result.byte_length -ne 4 * $length -or $actual.Count -ne $length) {
                throw "Wrong output size: $target, n=$length"
            }
            for ($i = 0; $i -lt $length; $i++) {
                if ($actual[$i] -ne $values[$i] + 1) {
                    throw "Wrong output: $target, optimized=$optimize, n=$length, index=$i"
                }
            }
            $cases++
        }
    }
}
Write-Output "$cases GPU runtime array loop cases passed, including [3, 7, -2] and empty inputs."
if ($validationWarnings) {
    Write-Warning "$validationWarnings runs emitted GPU validation errors; inspect logs in $outDir independently of the numeric results."
}
