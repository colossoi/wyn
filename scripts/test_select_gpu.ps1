# Compare eager choices and guarded division against a CPU reference on both targets.
# Requires a built wyn, viz, spirv-val, and GPU access. Artifacts stay in tmp/.
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$outDir = Join-Path $workspace 'tmp/select-lowering/gpu'
$null = New-Item -ItemType Directory -Force -Path $outDir
$wyn = Join-Path $workspace 'target/debug/wyn.exe'
$viz = Join-Path $workspace 'extra/viz/target/debug/viz.exe'
$source = Join-Path $workspace 'testfiles/select_lowering.wyn'
$inputFile = Join-Path $outDir 'input.bin'
$outputFile = Join-Path $outDir 'output.json'
$cases = 0
foreach ($target in @('spirv', 'wgsl')) {
    foreach ($optimize in @($false, $true)) {
        $extension = if ($target -eq 'spirv') { 'spv' } else { 'wgsl' }
        $shader = Join-Path $outDir "select-$optimize.$extension"
        $buildArgs = @('build', $source, '--max-warnings', '0', '--target', $target, '-o', $shader)
        if ($optimize) { $buildArgs += '-O' }
        & $wyn @buildArgs
        if ($LASTEXITCODE -ne 0) { throw 'Select compilation failed' }
        if ($target -eq 'spirv') {
            & spirv-val --target-env vulkan1.3 $shader
            if ($LASTEXITCODE -ne 0) { throw 'SPIR-V validation failed' }
        }
        foreach ($length in @(0, 1, 63, 64, 65, 257)) {
            $values = @(for ($i = 0; $i -lt $length; $i++) { ($i % 11) - 5 })
            $bytes = [byte[]]@($values | ForEach-Object { [BitConverter]::GetBytes([int]$_) })
            [IO.File]::WriteAllBytes($inputFile, $bytes)
            & $viz pipeline $shader --headless --no-config --input "xs:$inputFile" `
                --output "select_lowering:$outputFile" *> (Join-Path $outDir 'run.log')
            if ($LASTEXITCODE -ne 0) {
                throw "GPU execution failed: $target, optimized=$optimize, n=$length; see $outDir/run.log"
            }
            $result = Get-Content -LiteralPath $outputFile -Raw | ConvertFrom-Json
            $actual = @($result.backing_buffer)
            if ($result.byte_length -ne 4 * $length -or $actual.Count -ne $length) {
                throw "Wrong output size: $target, n=$length"
            }
            for ($i = 0; $i -lt $length; $i++) {
                $x = $values[$i]
                $chosen = if ($x -lt 0) { -$x } else { $x + 10 }
                $guarded = if ($x -ne 0) { [int][Math]::Truncate(120.0 / $x) } else { 7 }
                if ($actual[$i] -ne $chosen + $guarded) {
                    throw "Wrong output: $target, optimized=$optimize, n=$length, index=$i"
                }
            }
            $cases++
        }
    }
}
Write-Output "$cases GPU select cases passed, including guarded division at zero."
