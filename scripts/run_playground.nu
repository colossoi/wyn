#!/usr/bin/env nu
# Compile every testfiles/playground/*.wyn and run it through viz vf
# with --shadertoy --max-frames=15. Reports per-file pass/fail and a
# summary at the end.

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

    print $"$ ($wyn) compile ($src) -o ($spv)"
    let compile = (do { ^$wyn compile $src -o $spv } | complete)
    if $compile.exit_code != 0 {
        print $compile.stderr
        {name: $name, stage: "compile", ok: false}
    } else {
        print $"$ ($viz) vf ($spv) --shadertoy --max-frames=15"
        let run = (do { ^$viz vf $spv --shadertoy --max-frames=15 } | complete)
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
