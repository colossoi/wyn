#!/usr/bin/env pwsh

<#
.SYNOPSIS
Compile and run one local playground example.

.DESCRIPTION
Builds Wyn and viz in release mode, compiles one source from
testfiles/playground into tmp/playground, and opens it with viz. Bare image
examples automatically receive scripts/playground_image_header.wyn; examples
with an explicit entry declaration compile unchanged.

.PARAMETER Example
The path to a .wyn playground source. Relative and absolute paths are accepted.
A bare name such as noise_demo remains a shortcut for
testfiles/playground/noise_demo.wyn.

.PARAMETER SkipBuild
Reuse the existing release binaries instead of building Wyn and viz first.

.PARAMETER OutputDirectory
Directory for the generated .spv and .json files. Relative paths resolve from
the caller's current directory. The default is tmp/playground in the repository.

.PARAMETER MaxFrames
Close viz after this many frames. The default, 0, runs until the window closes.

.EXAMPLE
pwsh -File scripts/play.ps1 testfiles/playground/noise_demo.wyn

.EXAMPLE
pwsh -File scripts/play.ps1 particles -SkipBuild -MaxFrames 60

.EXAMPLE
pwsh -File scripts/play.ps1 testfiles/playground/noise_demo.wyn -OutputDirectory tmp
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory, Position = 0)]
    [string]$Example,

    [switch]$SkipBuild,

    [string]$OutputDirectory,

    [ValidateRange(0, 2147483647)]
    [int]$MaxFrames = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory)]
        [string]$Program,

        [Parameter(Mandatory)]
        [string[]]$Arguments,

        [Parameter(Mandatory)]
        [string]$Description
    )

    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$playgroundDirectory = Join-Path $workspace 'testfiles/playground'
$playgroundHeader = Join-Path $workspace 'scripts/playground_image_header.wyn'
$artifactDirectory = if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    Join-Path $workspace 'tmp/playground'
} else {
    $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDirectory)
}
$isWindowsHost = [Environment]::OSVersion.Platform -eq [PlatformID]::Win32NT
$executableSuffix = if ($isWindowsHost) { '.exe' } else { '' }
$wynBinary = Join-Path $workspace "target/release/wyn$executableSuffix"
$vizBinary = Join-Path $workspace "extra/viz/target/release/viz$executableSuffix"

$sourceCandidate = $Example
if (-not (Test-Path -LiteralPath $sourceCandidate -PathType Leaf)) {
    $exampleFile = [IO.Path]::GetFileName($Example)
    if ($exampleFile -eq $Example) {
        if ([IO.Path]::GetExtension($exampleFile) -eq '') {
            $exampleFile += '.wyn'
        }
        $sourceCandidate = Join-Path $playgroundDirectory $exampleFile
    }
}
if (-not (Test-Path -LiteralPath $sourceCandidate -PathType Leaf)) {
    throw "Playground example not found: $Example"
}
$sourcePath = (Resolve-Path -LiteralPath $sourceCandidate).Path
if ([IO.Path]::GetExtension($sourcePath) -ine '.wyn') {
    throw "Playground examples must use the .wyn extension: $Example"
}

$name = [IO.Path]::GetFileNameWithoutExtension($sourcePath)
$spvPath = Join-Path $artifactDirectory "$name.spv"
$descriptorPath = Join-Path $artifactDirectory "$name.json"
$sourceDirectory = [IO.Path]::GetDirectoryName($sourcePath)
$vizConfig = Join-Path $sourceDirectory "$name.viz.json"

if (-not $env:RUST_MIN_STACK) {
    $env:RUST_MIN_STACK = '67108864'
}

Push-Location $workspace
try {
    if (-not $SkipBuild) {
        Write-Host 'Building Wyn and viz in release mode...'
        Invoke-NativeChecked cargo @(
            'build', '--release', '--package', 'wyn', '--bin', 'wyn'
        ) 'Wyn build'
        if ($isWindowsHost) {
            Invoke-NativeChecked cargo @(
                'rustc', '--release', '--manifest-path', 'extra/viz/Cargo.toml', '--bin', 'viz',
                '--', '-C', "link-arg=/STACK:$env:RUST_MIN_STACK"
            ) 'viz build'
        } else {
            Invoke-NativeChecked cargo @(
                'build', '--release', '--manifest-path', 'extra/viz/Cargo.toml'
            ) 'viz build'
        }
    } else {
        foreach ($binary in @($wynBinary, $vizBinary)) {
            if (-not (Test-Path -LiteralPath $binary -PathType Leaf)) {
                throw "Release binary not found: $binary (rerun without -SkipBuild)"
            }
        }
    }

    $null = New-Item -ItemType Directory -Force -Path $artifactDirectory
    $compileSource = $sourcePath
    $preparedSource = $null
    try {
        $hasExplicitEntry = [IO.File]::ReadLines($sourcePath) |
            Where-Object { $_ -match '^entry ' } |
            Select-Object -First 1
        if (-not $hasExplicitEntry) {
            $preparedSource = Join-Path $sourceDirectory ".run-$name-$([guid]::NewGuid()).wyn"
            $source = [IO.File]::ReadAllText($playgroundHeader) +
                [Environment]::NewLine + [IO.File]::ReadAllText($sourcePath)
            [IO.File]::WriteAllText($preparedSource, $source, [Text.UTF8Encoding]::new($false))
            $compileSource = $preparedSource
        }

        Write-Host "Compiling $sourcePath"
        Invoke-NativeChecked $wynBinary @(
            'compile', $compileSource, '--graphics', '--direct', '-o', $spvPath
        ) 'playground compilation'
    } finally {
        if ($null -ne $preparedSource) {
            Remove-Item -LiteralPath $preparedSource -Force -ErrorAction SilentlyContinue
        }
    }

    foreach ($artifact in @($spvPath, $descriptorPath)) {
        if (-not (Test-Path -LiteralPath $artifact -PathType Leaf)) {
            throw "Compiler did not produce expected artifact: $artifact"
        }
    }

    $vizArguments = @('pipeline', $spvPath)
    if (Test-Path -LiteralPath $vizConfig -PathType Leaf) {
        $vizArguments += @('--config', $vizConfig)
    }
    if ($MaxFrames -gt 0) {
        $vizArguments += "--max-frames=$MaxFrames"
    }

    Write-Host "SPIR-V:    $spvPath"
    Write-Host "Descriptor: $descriptorPath"
    Write-Host "Running $name..."
    Invoke-NativeChecked $vizBinary $vizArguments 'viz'
} finally {
    Pop-Location
}
