#!/usr/bin/env nu

# Compile and validate all .wyn test files. Matches the semantics of
# `scripts/validate_testfiles.sh`: walks both `testfiles/*.wyn`
# (compiler-feature tests) and `testfiles/playground/*.wyn` (shader
# demos), depth 1 only.
#
# Cross-platform: defaults the output directory to `$env.TEMP` on
# Windows and `/tmp` elsewhere, and uses the right binary suffix
# (`.exe` on Windows).

def main [
    --keep (-k)               # Keep generated SPIR-V / WGSL files
    --out-dir (-o): string    # Output directory (implies --keep)
    --wgsl                    # Compile to WGSL and validate via `viz validate`
    --release                 # Build wyn (and viz, with --wgsl) in --release
                              #   (default: debug — builds faster, runs slower)
] {
    let is_windows = $nu.os-info.name == "windows"
    let default_out = if $is_windows { $env.TEMP } else { "/tmp" }
    let out_dir_given = ($out_dir != null)
    let out_dir = if $out_dir_given { $out_dir } else { $default_out }
    let keep = $keep or $out_dir_given
    let exe = if $is_windows { ".exe" } else { "" }
    let profile = if $release { "release" } else { "debug" }
    let wyn_bin = $"./target/($profile)/wyn($exe)"
    let viz_bin = $"./extra/viz/target/($profile)/viz($exe)"
    let mode = if $wgsl { "wgsl" } else { "spirv" }

    print $"Building wyn \(($profile))..."
    if $release {
        ^cargo build --release -p wyn
    } else {
        ^cargo build -p wyn
    }

    if $wgsl {
        print $"Building viz \(($profile)) for WGSL validation..."
        if $release {
            ^cargo build --release --quiet --manifest-path extra/viz/Cargo.toml
        } else {
            ^cargo build --quiet --manifest-path extra/viz/Cargo.toml
        }
    }

    let files = (
        (glob testfiles/*.wyn) ++ (glob testfiles/playground/*.wyn)
    )

    mut pass = 0
    mut fail = 0
    mut skip = 0

    for f in $files {
        let base = ($f | path parse | get stem)

        if $mode == "wgsl" {
            # WGSL mode: compile + validate via viz (naga in-process).
            # Skip testfiles that depend on `impl_source`-linked SPIR-V
            # helpers — WGSL has no equivalent linkage path.
            if $base == "miner" or $base == "sha256_test" {
                print $"Skipping ($f) \(depends on linked SPIR-V helpers)"
                $skip = $skip + 1
                continue
            }

            let out_path = ($out_dir | path join $"($base).wgsl")
            print -n $"Compiling ($f) -> WGSL... "

            let compile = (do { ^$wyn_bin compile $f -t wgsl -o $out_path } | complete)
            if $compile.exit_code != 0 {
                print "COMPILE FAILED"
                print $compile.stderr
                $fail = $fail + 1
                continue
            }

            print -n "validating... "
            let val = (do { ^$viz_bin validate $out_path } | complete)
            if $val.exit_code != 0 {
                print "VALIDATION FAILED"
                print $val.stderr
                if not $keep { rm --force $out_path }
                $fail = $fail + 1
                continue
            }

            if $keep {
                print $"OK -> ($out_path)"
            } else {
                print "OK"
                rm --force $out_path
            }
            $pass = $pass + 1
        } else {
            # SPIR-V mode (default).
            let spv_path = ($out_dir | path join $"($base).spv")
            print -n $"Compiling ($f)... "

            let compile = (do { ^$wyn_bin compile $f -o $spv_path } | complete)
            if $compile.exit_code != 0 {
                print "COMPILE FAILED"
                print $compile.stderr
                $fail = $fail + 1
                continue
            }

            print -n "validating... "
            let val = (do { ^spirv-val $spv_path } | complete)
            if $val.exit_code != 0 {
                print "VALIDATION FAILED"
                print $val.stderr
                if not $keep { rm --force $spv_path }
                $fail = $fail + 1
                continue
            }

            if $keep {
                print $"OK -> ($spv_path)"
            } else {
                print "OK"
                rm --force $spv_path
            }
            $pass = $pass + 1
        }
    }

    print ""
    print $"Results: ($pass) passed, ($fail) failed, ($skip) skipped"
    if $fail > 0 {
        exit 1
    }
}
