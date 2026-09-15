# Check shared capture and live/dead array state against a CPU reference.
[CmdletBinding()]
param([switch]$Wgsl)
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$outDir = Join-Path $workspace 'tmp/tinyporto-codegen/gpu-state'
$null = New-Item -ItemType Directory -Force -Path $outDir
$wyn = Join-Path $workspace 'target/debug/wyn.exe'
$viz = Join-Path $workspace 'extra/viz/target/debug/viz.exe'
$extension = if ($Wgsl) { 'wgsl' } else { 'spv' }
$scalar = Get-Content (Join-Path $workspace 'testfiles/tinyporto_dead_array_state.wyn') -Raw
$sources = [ordered]@{
    handoff = Get-Content (Join-Path $workspace 'testfiles/tinyporto_capture_handoff.wyn') -Raw
    scalar = $scalar
    prior_array = $scalar.Replace('state + events[k]', 'state + last[k % 4]')
    returned_array = $scalar.Replace('[1]i32', '[4]i32').Replace('[state]', 'last')
}
$cases = 0
foreach ($name in $sources.Keys) {
    $sourceFile = Join-Path $outDir "$name.wyn"
    Set-Content -LiteralPath $sourceFile -Value $sources[$name]
    $shader = Join-Path $outDir "$name.$extension"
    $buildArgs = @('build', '--max-warnings', '0', $sourceFile, '-o', $shader)
    if ($Wgsl) { $buildArgs += @('--target', 'wgsl') }
    & $wyn @buildArgs
    if ($LASTEXITCODE -ne 0) { throw "$name compilation failed" }
    if (!$Wgsl) {
        & spirv-val --target-env vulkan1.3 $shader
        if ($LASTEXITCODE -ne 0) { throw "$name SPIR-V validation failed" }
    }
    $descriptor = Get-Content ([IO.Path]::ChangeExtension($shader, '.json')) -Raw | ConvertFrom-Json
    $outputs = @($descriptor.pipelines[0].bindings | Where-Object usage -eq 'output')
    foreach ($pattern in @('all', 'none', 'alternating')) {
        $events = @(for ($i = 0; $i -lt 32; $i++) {
            if ($pattern -eq 'all' -or ($pattern -eq 'alternating' -and $i % 2 -eq 0)) { $i + 1 } else { 0 }
        })
        $state = 0; $prior = 0; $last = @(0, 0, 0, 0)
        for ($i = 0; $i -lt 32; $i++) {
            if ($events[$i] -gt 0) {
                $state += $events[$i]; $prior += $last[$i % 4]; $last[$i % 4] = $events[$i]
            }
        }
        $expected = switch ($name) {
            'handoff' { @((0..63 | ForEach-Object { $last[$_ % 4] }), (0..127 | ForEach-Object { $_ + $state }), @($state)) }
            'scalar' { ,@(@($state)) }
            'prior_array' { ,@(@($prior)) }
            'returned_array' { ,@($last) }
        }
        # viz's JSON reader accepts f32 values; preserve i32 bits for this fixture.
        $encoded = @($events | ForEach-Object { [BitConverter]::UInt32BitsToSingle([uint32]$_) })
        $inputFile = Join-Path $outDir 'input.json'
        ConvertTo-Json -InputObject $encoded -Compress | Set-Content -LiteralPath $inputFile
        $runnerDescriptor = Join-Path $outDir "$name.runner.json"
        $runArgs = @('pipeline', $shader, '--pipeline', $runnerDescriptor, '--input', "events:$inputFile")
        for ($slot = 0; $slot -lt $outputs.Count; $slot++) {
            $runArgs += @('--output', "$($outputs[$slot].name):$(Join-Path $outDir "output-$slot.json")")
            # Headless viz sizes SameAsDispatch from the first stage. Give this
            # test's known output capacities explicitly; shader/dispatches stay intact.
            $outputs[$slot].length = [pscustomobject]@{ kind = 'fixed'; bytes = @($expected[$slot]).Count * 4 }
        }
        $descriptor | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath $runnerDescriptor
        & $viz @runArgs > (Join-Path $outDir 'run.log') 2>&1
        if ($LASTEXITCODE -ne 0) { throw "$name/$pattern GPU execution failed; see $outDir/run.log" }
        for ($slot = 0; $slot -lt $outputs.Count; $slot++) {
            $actual = @(Get-Content (Join-Path $outDir "output-$slot.json") -Raw | ConvertFrom-Json)
            $wanted = @($expected[$slot])
            if ($actual.Count -ne $wanted.Count) { throw "$name/$pattern output $slot has wrong length" }
            for ($i = 0; $i -lt $actual.Count; $i++) {
                $value = [BitConverter]::SingleToUInt32Bits([float]$actual[$i])
                if ($value -ne $wanted[$i]) { throw "$name/$pattern output $slot index $i got $value, expected $($wanted[$i])" }
            }
        }
        $cases++
    }
}
Write-Output "$cases $extension GPU state cases passed (shared, scalar, prior-array, and returned-array results)."
