# Rebuild the compiler and compile one small program through egglog to WGSL.
# This checks compilation and emission, not GPU results or every rule family.
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$outDir = Join-Path $workspace 'target/egglog-smoke'
$null = New-Item -ItemType Directory -Force -Path $outDir
$shader = Join-Path $outDir 'smoke.wgsl'
Push-Location $workspace
try {
    if (Test-Path -LiteralPath $shader) { Remove-Item -LiteralPath $shader }
    & cargo run --quiet --bin wyn -- build testfiles/egglog_smoke.wyn -O --target wgsl -o $shader
    if ($LASTEXITCODE -ne 0) { throw 'Egglog smoke compilation failed' }
    if (!(Test-Path -LiteralPath $shader) -or !(Select-String -LiteralPath $shader -Pattern '@compute' -Quiet)) {
        throw 'Egglog smoke did not emit a compute shader'
    }
    Write-Host 'Egglog smoke passed.'
} finally {
    Pop-Location
}
