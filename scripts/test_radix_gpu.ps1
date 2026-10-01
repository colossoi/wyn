[CmdletBinding()]
param([string]$Compiler)
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$outDir = Join-Path $workspace 'target/radix-validation'
$null = New-Item -ItemType Directory -Force -Path $outDir
$wyn = if ($Compiler) { (Resolve-Path -LiteralPath $Compiler).Path } else { Join-Path $workspace 'target/release/wyn.exe' }
if (-not $env:WGPU_BACKEND) { $env:WGPU_BACKEND = 'vulkan' }
& $wyn build -O --target wgsl --target-double rust-wgpu --max-warnings 0 (Join-Path $workspace 'pkg/sort/test/radix_gpu.wyn') -o (Join-Path $outDir 'radix.wgsl')
if ($LASTEXITCODE -ne 0) { throw 'Radix compilation failed' }
foreach ($fixture in @(@('local_tuple_collectives', 'tuples'), @('nested_tuple_loop_state', 'nested'))) {
    & $wyn build -O --target wgsl --target-double rust-wgpu --max-warnings 0 (Join-Path $workspace "testfiles/regressions/$($fixture[0]).wyn") -o (Join-Path $outDir "$($fixture[1]).wgsl")
    if ($LASTEXITCODE -ne 0) { throw "Tuple-loop compilation failed: $($fixture[0])" }
}
Copy-Item -LiteralPath (Join-Path $workspace 'pkg/sort/test/radix_gpu.rs') -Destination (Join-Path $outDir 'lib.rs')
@'
[package]
name = "wyn-radix-gpu-tests"
version = "0.1.0"
edition = "2021"
[workspace]
[lib]
path = "lib.rs"
[dependencies]
wgpu = "27"
pollster = "0.3"
'@ | Set-Content -LiteralPath (Join-Path $outDir 'Cargo.toml')
& cargo test --release --manifest-path (Join-Path $outDir 'Cargo.toml') --target-dir (Join-Path $workspace 'target') --offline -- --nocapture --test-threads=1
if ($LASTEXITCODE -ne 0) { throw 'Radix GPU checks failed' }
