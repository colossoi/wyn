#!/usr/bin/env nu

# Compile and validate all .wyn test files
def main [] {
    print "Building wyn (release)..."
    cargo build --release

    ls testfiles/*.wyn | each { |f|
        let base = ($f.name | path parse | get stem)
        let spv_path = $"/tmp/($base).spv"

        print -n $"Compiling ($f.name)... "

        let compile_result = do { ./target/release/wyn compile $f.name --partial-eval -o $spv_path } | complete

        if $compile_result.exit_code != 0 {
            print "COMPILE FAILED"
            print $compile_result.stderr
        } else {
            print -n "validating... "
            let val_result = do { spirv-val $spv_path } | complete

            if $val_result.exit_code != 0 {
                print "VALIDATION FAILED"
                print $val_result.stderr
            } else {
                print "OK"
            }
        }
    }
    null
}

main
