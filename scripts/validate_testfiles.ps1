#!/usr/bin/env pwsh

<#
.SYNOPSIS
Build Wyn, compile every top-level testfile, and validate the generated output.

.DESCRIPTION
PowerShell counterpart to validate_testfiles.nu. It visits *.wyn files directly
under testfiles/ and testfiles/playground/ (not recursively), compiles to SPIR-V
by default, and validates each result. With -Wgsl, it compiles to WGSL and uses
extra/viz for validation.

.PARAMETER Keep
Keep generated SPIR-V or WGSL files.

.PARAMETER OutDir
Write generated files to this directory. Supplying OutDir implies -Keep.

.PARAMETER Wgsl
Compile to WGSL and validate with viz instead of spirv-val.

.PARAMETER Release
Build and run release binaries instead of debug binaries.

.PARAMETER TrackedOnly
Validate only testfiles tracked by Git. Useful when a working tree contains
local experiments that should not participate in a repository gate.
#>
[CmdletBinding()]
param(
    [switch]$Keep,
    [string]$OutDir,
    [switch]$Wgsl,
    [switch]$Release,
    [switch]$TrackedOnly
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
function Invoke-NativeCaptured {
    param(
        [Parameter(Mandatory)]
        [string]$Program,

        [Parameter(Mandatory)]
        [string[]]$Arguments
    )

    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $output = & $Program @Arguments 2>&1
        $nativeExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    [pscustomobject]@{
        Output = $output
        ExitCode = $nativeExitCode
    }
}

$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$outDirWasGiven = $PSBoundParameters.ContainsKey('OutDir')
$keepOutputs = $Keep.IsPresent -or $outDirWasGiven
$outputDirectory = if ($outDirWasGiven) {
    $OutDir
} else {
    [IO.Path]::GetTempPath()
}
$null = New-Item -ItemType Directory -Force -Path $outputDirectory
$outputDirectory = (Resolve-Path -LiteralPath $outputDirectory).Path

$profile = if ($Release) { 'release' } else { 'debug' }

$isWindowsHost = [System.Environment]::OSVersion.Platform -eq [System.PlatformID]::Win32NT
$executableSuffix = if ($isWindowsHost) { '.exe' } else { '' }
$wynBinary = Join-Path $workspace "target/$profile/wyn$executableSuffix"
$vizBinary = Join-Path $workspace "extra/viz/target/$profile/viz$executableSuffix"
$playgroundDirectory = Join-Path $workspace 'testfiles/playground'
$playgroundHeader = Join-Path $workspace 'scripts/playground_image_header.wyn'

$exitCode = 0
Push-Location $workspace
try {
    Write-Host "Building wyn ($profile)..."
    $cargoArguments = @('build', '-p', 'wyn')
    if ($Release) {
        $cargoArguments = @('build', '--release', '-p', 'wyn')
    }
    Invoke-NativeChecked cargo $cargoArguments 'Wyn build'

    if ($Wgsl) {
        Write-Host "Building viz ($profile) for WGSL validation..."
        $vizArguments = @(
            'build',
            '--quiet',
            '--manifest-path',
            'extra/viz/Cargo.toml'
        )
        if ($Release) {
            $vizArguments = @(
                'build',
                '--release',
                '--quiet',
                '--manifest-path',
                'extra/viz/Cargo.toml'
            )
        }
        Invoke-NativeChecked cargo $vizArguments 'viz build'
    } elseif (-not (Get-Command spirv-val -ErrorAction SilentlyContinue)) {
        throw 'spirv-val is required for SPIR-V validation but was not found on PATH'
    }

    $files = if ($TrackedOnly) {
        $tracked = Invoke-NativeCaptured git @(
            '-C', $workspace, 'ls-files', '--', 'testfiles/*.wyn', 'testfiles/playground/*.wyn'
        )
        if ($tracked.ExitCode -ne 0) {
            throw "Git tracked-testfile discovery failed with exit code $($tracked.ExitCode)"
        }
        @(
            $tracked.Output |
                Where-Object {
                    $_ -is [string] -and
                    $_ -match '^(?:testfiles|testfiles/playground)/[^/]+\.wyn$'
                } |
                ForEach-Object { Get-Item -LiteralPath (Join-Path $workspace $_) -ErrorAction Stop }
        )
    } else {
        @(
            Get-ChildItem -LiteralPath (Join-Path $workspace 'testfiles') -File |
                Where-Object Extension -EQ '.wyn'
            Get-ChildItem -LiteralPath (Join-Path $workspace 'testfiles/playground') -File |
                Where-Object Extension -EQ '.wyn'
        )
    }
    $files = $files | Sort-Object FullName

    $passed = 0
    $failed = 0
    $skipped = 0

    foreach ($file in $files) {
        $base = $file.BaseName

        if ($Wgsl -and $base -in @('miner', 'sha256_test')) {
            Write-Host "Skipping $($file.FullName) (depends on linked SPIR-V helpers)"
            $skipped++
            continue
        }

        $extension = if ($Wgsl) { 'wgsl' } else { 'spv' }
        $outputPath = Join-Path $outputDirectory "$base.$extension"

        $compileSource = $file.FullName
        $preparedSource = $null
        $isPlaygroundSource = $file.DirectoryName -eq $playgroundDirectory
        $hasExplicitEntry = [IO.File]::ReadLines($file.FullName) |
            Where-Object { $_ -match '^entry ' } |
            Select-Object -First 1
        if ($isPlaygroundSource -and -not $hasExplicitEntry) {
            $preparedSource = Join-Path $playgroundDirectory ".validate-$base-$([guid]::NewGuid()).wyn"
            $source = [IO.File]::ReadAllText($playgroundHeader) +
                [Environment]::NewLine + [IO.File]::ReadAllText($file.FullName)
            [IO.File]::WriteAllText($preparedSource, $source, [Text.UTF8Encoding]::new($false))
            $compileSource = $preparedSource
        }

        if ($Wgsl) {
            Write-Host -NoNewline "Compiling $($file.FullName) -> WGSL... "
            $compile = Invoke-NativeCaptured $wynBinary @(
                'compile', $compileSource, '--graphics', '-t', 'wgsl', '-o', $outputPath
            )
        } else {
            Write-Host -NoNewline "Compiling $($file.FullName)... "
            $compile = Invoke-NativeCaptured $wynBinary @(
                'compile', $compileSource, '--graphics', '-o', $outputPath
            )
        }
        if ($null -ne $preparedSource) {
            Remove-Item -LiteralPath $preparedSource -Force -ErrorAction SilentlyContinue
        }

        if ($compile.ExitCode -ne 0) {
            Write-Host 'COMPILE FAILED'
            $compile.Output | ForEach-Object { Write-Host $_ }
            $failed++
            continue
        }

        Write-Host -NoNewline 'validating... '
        if ($Wgsl) {
            $validation = Invoke-NativeCaptured $vizBinary @('validate', $outputPath)
        } else {
            $validation = Invoke-NativeCaptured 'spirv-val' @($outputPath)
        }

        if ($validation.ExitCode -ne 0) {
            Write-Host 'VALIDATION FAILED'
            $validation.Output | ForEach-Object { Write-Host $_ }
            if (-not $keepOutputs) {
                Remove-Item -LiteralPath $outputPath -Force -ErrorAction SilentlyContinue
            }
            $failed++
            continue
        }

        if ($keepOutputs) {
            Write-Host "OK -> $outputPath"
        } else {
            Write-Host 'OK'
            Remove-Item -LiteralPath $outputPath -Force
        }
        $passed++
    }

    Write-Host ''
    Write-Host "Results: $passed passed, $failed failed, $skipped skipped"
    if ($failed -gt 0) {
        $exitCode = 1
    }
} catch {
    Write-Error $_ -ErrorAction Continue
    $exitCode = 1
} finally {
    Pop-Location
}

exit $exitCode
