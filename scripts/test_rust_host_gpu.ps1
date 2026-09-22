# Compile and execute the generated Rust host APIs against a real GPU.
[CmdletBinding()]
param([string]$Compiler)
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$outDir = Join-Path $workspace 'tmp/rust-host-gpu'
$null = New-Item -ItemType Directory -Force -Path $outDir
$wyn = if ($Compiler) { (Resolve-Path -LiteralPath $Compiler).Path } else { Join-Path $workspace 'target/debug/wyn.exe' }
foreach ($fixture in @('batching', 'filter')) {
    $source = Join-Path $workspace "testfiles/rust_host_$fixture.wyn"
    foreach ($format in @('spirv', 'wgsl')) {
        $suffix = if ($format -eq 'spirv') { 'spv' } else { 'wgsl' }
        & $wyn build -O --target $format --target-double rust-wgpu --max-warnings 0 $source -o (Join-Path $outDir "$($fixture)_$suffix.$suffix")
        if ($LASTEXITCODE -ne 0) { throw "Host compilation failed: $fixture / $format" }
    }
}
Copy-Item -LiteralPath (Join-Path $workspace 'testfiles/rust_host_batching.rs') -Destination (Join-Path $outDir 'lib.rs')
@'
[package]
name = "wyn-rust-host-gpu-tests"
version = "0.1.0"
edition = "2021"
[workspace]
[lib]
path = "lib.rs"
[dependencies]
wgpu = { version = "27", features = ["spirv"] }
pollster = "0.3"
'@ | Set-Content -LiteralPath (Join-Path $outDir 'Cargo.toml')
& cargo test --manifest-path (Join-Path $outDir 'Cargo.toml') --offline -- --nocapture
if ($LASTEXITCODE -ne 0) { throw 'Generated Rust host GPU tests failed' }
