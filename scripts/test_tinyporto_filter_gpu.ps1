# Execute compiler output and compare stable filtering with a CPU reference.
# Requires a built wyn, viz, spirv-val, and a working GPU. Outputs stay in tmp/.
[CmdletBinding()]
param([switch]$Wgsl)
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$outDir = Join-Path $workspace 'tmp/tinyporto-codegen/gpu'
$null = New-Item -ItemType Directory -Force -Path $outDir
$wyn = Join-Path $workspace 'target/debug/wyn.exe'
$viz = Join-Path $workspace 'extra/viz/target/debug/viz.exe'
$extension = if ($Wgsl) { 'wgsl' } else { 'spv' }
$shader = Join-Path $outDir "filter.$extension"
$source = Join-Path $workspace 'testfiles/tinyporto_filter_scan.wyn'
$buildArgs = @('build', '--max-warnings', '0', $source, '-o', $shader)
if ($Wgsl) { $buildArgs += @('--target', 'wgsl') }
& $wyn @buildArgs
if ($LASTEXITCODE -ne 0) { throw 'Filter compilation failed' }
if (!$Wgsl) {
    & spirv-val --target-env vulkan1.3 $shader
    if ($LASTEXITCODE -ne 0) { throw 'SPIR-V validation failed' }
}
$descriptor = Get-Content -LiteralPath ([IO.Path]::ChangeExtension($shader, '.json')) -Raw | ConvertFrom-Json
$bindings = $descriptor.pipelines[0].bindings
$outputName = ($bindings | Where-Object usage -eq 'output').name
$countName = ($bindings | Where-Object { $_.usage -eq 'intermediate' -and $_.length.bytes -eq 4 }).name
if (!$outputName -or !$countName) { throw 'Missing filter output or length binding' }
$inputFile = Join-Path $outDir 'input.json'
$outputFile = Join-Path $outDir 'output.json'
$countFile = Join-Path $outDir 'count.json'
$cases = 0
foreach ($length in @(0, 1, 63, 64, 65, 255, 256, 257, 4096, 39592)) {
    foreach ($pattern in @('all', 'none', 'alternating', 'sparse')) {
        $values = @(for ($i = 0; $i -lt [Math]::Max(1, $length); $i++) {
            $keep = switch ($pattern) {
                'all' { $true }
                'none' { $false }
                'alternating' { $i % 2 -eq 0 }
                'sparse' { ($i * 17 + 11) % 101 -lt 7 }
            }
            if ($keep) { $i + 1 } else { -($i + 1) }
        })
        $expected = @(for ($i = 0; $i -lt $length; $i++) { if ($values[$i] -gt 0) { $values[$i] } })
        ConvertTo-Json -InputObject $values -Compress | Set-Content -LiteralPath $inputFile
        & $viz pipeline $shader --input "xs:$inputFile" --push-constant "n:i32=$length" `
            --output "${outputName}:$outputFile" --output "${countName}:$countFile" `
            > (Join-Path $outDir 'run.log') 2>&1
        if ($LASTEXITCODE -ne 0) { throw "GPU execution failed: n=$length, pattern=$pattern; see $outDir/run.log" }
        $actual = @(Get-Content -LiteralPath $outputFile -Raw | ConvertFrom-Json)
        $countBits = @(Get-Content -LiteralPath $countFile -Raw | ConvertFrom-Json)
        $count = [BitConverter]::SingleToUInt32Bits([float]$countBits[0])
        if ($count -ne $expected.Count) { throw "Wrong count: n=$length, pattern=$pattern, got $count, expected $($expected.Count)" }
        for ($i = 0; $i -lt $count; $i++) {
            if ($actual[$i] -ne $expected[$i]) { throw "Wrong output: n=$length, pattern=$pattern, index=$i" }
        }
        $cases++
    }
}
Write-Output "$cases $extension GPU filter cases passed (counts and stable output order)."
