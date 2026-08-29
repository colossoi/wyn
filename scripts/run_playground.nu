#!/usr/bin/env nu
# Compile every testfiles/playground/*.wyn and run its generated pipeline
# descriptor through viz. By default, each example runs for 15 frames; pass
# --wait to keep each window open until it is closed. Reports per-file
# pass/fail and a summary at the end.

def main [
    --wait # Keep each Viz window open until it is closed.
] {
    cd ($env.FILE_PWD | path join "..")

    print "Building wyn + viz in release..."
    ^cargo build --release --package wyn --bin wyn
    ^cargo build --release --manifest-path extra/viz/Cargo.toml

    let wyn = if $nu.os-info.name == "windows" { "target/release/wyn.exe" } else { "target/release/wyn" }
    let viz = if $nu.os-info.name == "windows" { "extra/viz/target/release/viz.exe" } else { "extra/viz/target/release/viz" }
    let out_dir = "tmp/playground"
    mkdir $out_dir

    let results = (ls testfiles/playground/*.wyn | each { |f|
        let name = ($f.name | path parse | get stem)
        let src = $f.name
        let spv = ($out_dir | path join $"($name).spv")

        print $"=== ($name) ==="

        print $"$ ($wyn) compile ($src) --graphics -o ($spv)"
        let compile = (do { ^$wyn compile $src --graphics -o $spv } | complete)
        if $compile.exit_code != 0 {
            print $compile.stderr
            {name: $name, stage: "compile", ok: false}
        } else {
            let run = if $wait {
                print $"$ ($viz) pipeline ($spv)"
                do { ^$viz pipeline $spv } | complete
            } else {
                print $"$ ($viz) pipeline ($spv) --max-frames=15"
                do { ^$viz pipeline $spv --max-frames=15 } | complete
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
