$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot

foreach ($crate in @('wyn-wasm', 'wyn-egir-viz-wasm')) {
    $manifestPath = Join-Path $repoRoot "$crate/Cargo.toml"
    cargo check --manifest-path $manifestPath --target wasm32-unknown-unknown --locked @args
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
