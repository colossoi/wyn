#!/usr/bin/env pwsh

<#
.SYNOPSIS
Compile and briefly run every top-level playground example.

.DESCRIPTION
Builds Wyn and viz in release mode, compiles each
testfiles/playground/*.wyn file to tmp/playground, and runs its generated
pipeline descriptor through viz for 15 frames. Reports per-file failures
and a summary, then exits with a non-zero status if any example failed.

.PARAMETER Wait
Keep each viz window open until it is closed instead of limiting it to 15
frames. After the window closes, the script advances to the next example.
#>
[CmdletBinding()]
param(
    [switch]$Wait
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
$isWindowsHost = [System.Environment]::OSVersion.Platform -eq [System.PlatformID]::Win32NT
$executableSuffix = if ($isWindowsHost) { '.exe' } else { '' }
$wynBinary = Join-Path $workspace "target/release/wyn$executableSuffix"
$vizBinary = Join-Path $workspace "extra/viz/target/release/viz$executableSuffix"
$outputDirectory = Join-Path $workspace 'tmp/playground'
$exitCode = 0

Push-Location $workspace
try {
    Write-Host 'Building wyn + viz in release...'
    Invoke-NativeChecked cargo @(
        'build', '--release', '--package', 'wyn', '--bin', 'wyn'
    ) 'Wyn build'
    Invoke-NativeChecked cargo @(
        'build', '--release', '--manifest-path', 'extra/viz/Cargo.toml'
    ) 'viz build'

    $null = New-Item -ItemType Directory -Force -Path $outputDirectory
    $files = @(
        Get-ChildItem -LiteralPath (Join-Path $workspace 'testfiles/playground') -File |
            Where-Object Extension -EQ '.wyn' |
            Sort-Object FullName
    )

    $results = foreach ($file in $files) {
        $name = $file.BaseName
        $spv = Join-Path $outputDirectory "$name.spv"

        Write-Host ''
        Write-Host "=== $name ==="
        Write-Host "$wynBinary compile $($file.FullName) --graphics -o $spv"
        $compile = Invoke-NativeCaptured $wynBinary @(
            'compile', $file.FullName, '--graphics', '-o', $spv
        )

        if ($compile.ExitCode -ne 0) {
            $compile.Output | ForEach-Object { Write-Host $_ }
            [pscustomobject]@{ Name = $name; Stage = 'compile'; Ok = $false }
            continue
        }

        $vizArguments = @('pipeline', $spv)
        if (-not $Wait) {
            $vizArguments += '--max-frames=15'
        }

        Write-Host "$vizBinary $($vizArguments -join ' ')"
        $run = Invoke-NativeCaptured $vizBinary $vizArguments

        if ($run.ExitCode -ne 0) {
            $run.Output | ForEach-Object { Write-Host $_ }
            [pscustomobject]@{ Name = $name; Stage = 'run'; Ok = $false }
            continue
        }

        [pscustomobject]@{ Name = $name; Stage = 'ok'; Ok = $true }
    }

    $passed = @($results | Where-Object Ok).Count
    $compileFailed = @($results | Where-Object Stage -EQ 'compile').Count
    $runFailed = @($results | Where-Object Stage -EQ 'run').Count

    Write-Host ''
    Write-Host '=== summary ==='
    Write-Host ("pass:         {0}" -f $passed)
    Write-Host ("compile fail: {0}" -f $compileFailed)
    Write-Host ("run fail:     {0}" -f $runFailed)

    $failures = @($results | Where-Object { -not $_.Ok })
    if ($failures.Count -gt 0) {
        Write-Host ''
        Write-Host 'failures:'
        $failures | Select-Object Name, Stage | Format-Table | Out-Host
        $exitCode = 1
    }
} catch {
    Write-Error $_ -ErrorAction Continue
    $exitCode = 1
} finally {
    Pop-Location
}

exit $exitCode
