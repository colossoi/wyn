#!/usr/bin/env nu

# Compile and validate all .wyn test files
# Runs twice: once without partial-eval, once with partial-eval
def main [] {
    print "Building wyn (release)..."
    cargo build --release

    ls testfiles/*.wyn | each { |f|
        let base = ($f.name | path parse | get stem)

        # First pass: without --partial-eval
        let spv_path = $"/tmp/($base).spv"
        print -n $"Compiling ($f.name)... "

        let compile_result = do { ./target/release/wyn compile $f.name -o $spv_path } | complete

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

        # Second pass: with --partial-eval
        let spv_path_pe = $"/tmp/($base)_pe.spv"
        print -n $"Compiling ($f.name) (partial-eval)... "

        let compile_result_pe = do { ./target/release/wyn compile $f.name --partial-eval -o $spv_path_pe } | complete

        if $compile_result_pe.exit_code != 0 {
            print "COMPILE FAILED"
            print $compile_result_pe.stderr
        } else {
            print -n "validating... "
            let val_result_pe = do { spirv-val $spv_path_pe } | complete

            if $val_result_pe.exit_code != 0 {
                print "VALIDATION FAILED"
                print $val_result_pe.stderr
            } else {
                print "OK"
            }
        }
    }
    null
}

main
