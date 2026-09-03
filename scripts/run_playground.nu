#!/usr/bin/env nu
# Compile every testfiles/playground/*.wyn and run its generated pipeline
# descriptor through viz. By default, each example runs for 15 frames; pass
# --wait to keep each window open until it is closed. Reports per-file
# pass/fail and a summary at the end.

def main [
    --wait # Keep each Viz window open until it is closed.
] {
    cd ($env.FILE_PWD | path join "..")
    $env.RUST_MIN_STACK = ($env.RUST_MIN_STACK? | default "67108864")

    print "Building wyn + viz in release..."
    ^cargo build --release --package wyn --bin wyn
    if $nu.os-info.name == "windows" {
        ^cargo rustc --release --manifest-path extra/viz/Cargo.toml --bin viz -- -C $"link-arg=/STACK:($env.RUST_MIN_STACK)"
    } else {
        ^cargo build --release --manifest-path extra/viz/Cargo.toml
    }

    let wyn = if $nu.os-info.name == "windows" { "target/release/wyn.exe" } else { "target/release/wyn" }
    let viz = if $nu.os-info.name == "windows" { "extra/viz/target/release/viz.exe" } else { "extra/viz/target/release/viz" }
    let out_dir = "tmp/playground"
    let playground_header = "scripts/playground_image_header.wyn"
    mkdir $out_dir

    let results = (ls testfiles/playground/*.wyn | each { |f|
        let name = ($f.name | path parse | get stem)
        let src = $f.name
        let spv = ($out_dir | path join $"($name).spv")
        let viz_config = ($src | path dirname | path join $"($name).viz.json")
        let has_explicit_entry = (
            open $src | lines | any { |line| $line starts-with "entry " }
        )
        let prepared_source = if not $has_explicit_entry {
            let path = ($src | path dirname | path join $".run-($name)-(random uuid).wyn")
            [(open --raw $playground_header), "\n", (open --raw $src)] | str join | save $path
            $path
        } else {
            null
        }
        let compile_source = if $prepared_source != null { $prepared_source } else { $src }

        print $"=== ($name) ==="

        let compile_args = ["build", $compile_source, "--graphics", "--direct", "-o", $spv]
        print $"$ ($wyn) ($compile_args | str join ' ')"
        let compile = (do { ^$wyn ...$compile_args } | complete)
        if $prepared_source != null { rm --force $prepared_source }
        if $compile.exit_code != 0 {
            print $compile.stderr
            {name: $name, stage: "compile", ok: false}
        } else {
            let config_args = if ($viz_config | path exists) { ["--config", $viz_config] } else { [] }
            let run = if $wait {
                print $"$ ($viz) pipeline ($spv) ($config_args | str join ' ')"
                do { ^$viz pipeline $spv ...$config_args } | complete
            } else {
                print $"$ ($viz) pipeline ($spv) ($config_args | str join ' ') --max-frames=15"
                do { ^$viz pipeline $spv ...$config_args --max-frames=15 } | complete
            }
            {name: $name, stage: (if $run.exit_code == 0 { "ok" } else { "run" }), ok: ($run.exit_code == 0)}
        }
    })

    print ""
    print "=== summary ==="
    let pass = ($results | where ok | length)
    let compile_fail = ($results | where stage == "compile" | length)
    let run_fail = ($results | where stage == "run" | length)
    print $"pass:         ($pass)"
    print $"compile fail: ($compile_fail)"
    print $"run fail:     ($run_fail)"

    let fails = ($results | where not ok)
    if (not ($fails | is-empty)) {
        print ""
        print "failures:"
        $fails | select name stage | print
        exit 1
    }
}
